#!/usr/bin/env python3
"""
preview_server.py — Visual cut review UI

Usage:
    python3 preview_server.py "video.mp4"
    python3 preview_server.py "video.mp4" --model small --port 8765
"""

import argparse
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
import threading
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, request, send_file, Response

sys.path.insert(0, str(Path(__file__).parent))
from video_editor import (
    FFMPEG, FFPROBE,
    extract_audio, transcribe, merge_cuts, cuts_to_keeps,
    build_srt, encode, get_duration,
    detect_repeated_takes, detect_repeated_takes_ai, detect_stutter_phrases, detect_cuts,
    FILLER_WORDS, FILLER_PAD, SILENCE_MAX, SILENCE_KEEP,
    RETAKE_PHRASES, RETAKE_SIGNALS, _take_start,
)

# ── Global state ──────────────────────────────────────────────────────────────

VIDEO_PATH    = None
THUMBS_DIR    = None
TEMP_DIR      = None
DURATION      = 0.0
WAVEFORM      = []
WORDS         = []
CUTS          = []     # [{id, start, end, label, type, active}]
THUMB_INTERVAL = 5.0
THUMB_COUNT    = 0
VIDEO_W        = 0     # source video pixel width
VIDEO_H        = 0     # source video pixel height
CROP           = None  # {x,y,w,h} in source pixels (16:9); None = no crop
# Colour adjust (like iMovie): brightness additive (-0.5..0.5), contrast/saturation multipliers (0..2).
ADJUST         = {"brightness": 0.0, "contrast": 1.0, "saturation": 1.0}
VIDEO_VOL      = 1.0   # volume multiplier for the main (camera/external) audio, iMovie-style (0..4)
MUTED_SEGS     = []    # [[start,end],...] ORIGINAL-time ranges where the main (video) audio is muted
# Text caption overlays. Each: {id, text, x, y, size, font, color, bold, italic, start, end}
# x,y are fractions (0..1) of the VISIBLE video frame (top-left of the text box); size is a
# fraction of frame HEIGHT so it scales across preview/export. Unlimited, may overlap freely.
CAPTIONS: list[dict] = []
# Image overlays. Each: {id, src (filename in project overlays/), x, y, w, start, end, opacity}
# x,y are fractions of the visible frame (top-left); w is a fraction of frame WIDTH.
IMAGE_OVERLAYS: list[dict] = []
# Audio layers (sound effects / music). Each: {id, src (filename in shared sound library),
# start (original-time seconds), dur, volume}. Uploaded sounds are also kept in a shared library.
AUDIO_LAYERS: list[dict] = []
SOUND_LIB = Path.home() / ".cache" / "video_editor" / "sound_library"
PROXY_PATH     = None  # low-res dense-keyframe proxy for smooth preview seeking
PROXY_READY    = False # True once the proxy for the CURRENT project is built
EDIT_PROXY_STATUS = {"state": "idle", "sig": None}  # rendered-edit smooth preview state
ENCODE_STATUS  = {"state": "idle", "message": "Ready"}
ENCODE_PROC    = None    # the running export ffmpeg Popen (so it can be cancelled)
ENCODE_CANCEL  = False   # set when the user cancels an export
PROCESS_STATUS = {"state": "idle", "message": ""}   # upload/re-process pipeline
SPLIT_POINTS: list[dict] = []   # [{id, pos}]  — visual split markers only
ORIGINAL_STEM      = None  # original filename stem (before any temp-copy rename)
STATE_FILE         = None  # path to persisted edits JSON
INITIAL_AUTO_CUTS  = []   # snapshot of auto-detected cuts before any user edits
UNDO_STACK_CACHE   = []   # mirror of client UNDO_STACK, persisted with edits
LAST_EXPORT_DIR    = None  # last directory user exported to (persisted in config.json)
MODEL_ARGS         = {"model": "base", "skip_fillers": False, "skip_bad_takes": False, "skip_silence": False}
EXTERNAL_AUDIO_PATH   = None   # path to separately recorded audio file
EXTERNAL_AUDIO_OFFSET = 0.0    # seconds: ext_audio_time = video_time - offset
SOURCE_VIDEOS      = []    # permanent uploaded video paths (for robust auto-resume)
SOURCE_EXT_AUDIO   = None  # permanent uploaded external-audio path
CURRENT_PROJECT_ID = None  # per-project id (keyed off video filename) — isolates state
RESUMING           = False # True only while reprocess_video is rebuilding CUTS


def sanitize_project_id(name: str) -> str:
    """Derive a safe, readable project id from an uploaded video's filename."""
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    if len(stem) > 60:
        stem = stem[:60].strip("._-")
    import time as _t
    return stem or ("project_" + _t.strftime("%Y%m%d_%H%M%S"))


# ── Config persistence (cross-session settings) ───────────────────────────────

CONFIG_PATH = Path.home() / ".cache" / "video_editor" / "config.json"

def load_config():
    global LAST_EXPORT_DIR
    try:
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text())
            LAST_EXPORT_DIR = data.get("last_export_dir")
            # Prefer the permanent uploads-dir source paths (survive temp cleanup)
            srcs = data.get("last_source_videos") or []
            srcs = [Path(p) for p in srcs if Path(p).exists()]
            if not srcs:
                lv = data.get("last_video_path")
                if lv and Path(lv).exists():
                    srcs = [Path(lv)]
            la = data.get("last_ext_audio_path")
            lo = data.get("last_ext_audio_offset", 0.0)
            pid = data.get("last_project_id")
            if srcs:
                return srcs, (Path(la) if la and Path(la).exists() else None), lo, pid
    except Exception:
        pass
    return [], None, 0.0, None

def save_config():
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text())
        if LAST_EXPORT_DIR:
            data["last_export_dir"] = LAST_EXPORT_DIR
        # Save PERMANENT upload paths (not the throwaway temp working copy)
        if SOURCE_VIDEOS:
            data["last_source_videos"] = [str(p) for p in SOURCE_VIDEOS]
            data["last_video_path"]    = str(SOURCE_VIDEOS[0])
        if CURRENT_PROJECT_ID:
            data["last_project_id"] = CURRENT_PROJECT_ID
        if SOURCE_EXT_AUDIO:
            data["last_ext_audio_path"]   = str(SOURCE_EXT_AUDIO)
            data["last_ext_audio_offset"] = EXTERNAL_AUDIO_OFFSET
        else:
            data.pop("last_ext_audio_path", None)
            data.pop("last_ext_audio_offset", None)
        CONFIG_PATH.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"  Warning: could not save config: {e}")


# ── Edit state persistence ────────────────────────────────────────────────────

def save_edits():
    """Write current cuts + split points to disk so restarts don't lose work."""
    if not STATE_FILE:
        return
    # GUARD: only block while the resume pipeline is rebuilding CUTS (which is
    # briefly empty). Normal editing — even during an audio sync — must always save.
    if RESUMING:
        return
    try:
        # Atomic write: write to a temp file then replace, so a crash mid-write
        # can never leave a truncated/corrupt state file.
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"cuts": CUTS, "split_points": SPLIT_POINTS,
                       "undo_stack": UNDO_STACK_CACHE, "crop": CROP,
                       "captions": CAPTIONS, "images": IMAGE_OVERLAYS, "adjust": ADJUST,
                       "audio": AUDIO_LAYERS, "video_volume": VIDEO_VOL,
                       "muted_segs": MUTED_SEGS}, f, indent=2)
        os.replace(tmp, STATE_FILE)
    except Exception as e:
        print(f"  Warning: could not save edits: {e}")


def transcript_cache_path(stem):
    return Path.home() / ".cache" / "video_editor" / f"{stem}_transcript.json"


def save_transcript(stem):
    """Cache transcription + waveform so future resumes skip Whisper + AI entirely."""
    try:
        path = transcript_cache_path(stem)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "words": WORDS, "waveform": WAVEFORM, "duration": DURATION,
                "thumb_interval": THUMB_INTERVAL, "thumb_count": THUMB_COUNT,
                "ext_audio_offset": EXTERNAL_AUDIO_OFFSET,
                "has_external_audio": EXTERNAL_AUDIO_PATH is not None,
            }, f)
        print(f"  Cached transcript ({len(WORDS)} words) for instant resume")
    except Exception as e:
        print(f"  Warning: could not cache transcript: {e}")


def load_transcript(stem):
    """Return cached transcript dict if it exists, else None."""
    try:
        path = transcript_cache_path(stem)
        if path.exists():
            return json.loads(path.read_text())
    except Exception as e:
        print(f"  Warning: could not load transcript cache: {e}")
    return None


def dedupe_cut_list(cuts):
    """Collapse cuts that cover the same region (within 20ms) — keeps the editor
    from accumulating thousands of identical cut copies. Preserves order/first seen."""
    seen, out = set(), []
    for c in cuts:
        key = (round(c.get("start", 0), 2), round(c.get("end", 0), 2), c.get("active", True))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def cut_exists(start, end, eps=0.02):
    """True if an active cut covering ~[start,end] already exists (avoid duplicates)."""
    return any(c.get("active", True) and abs(c["start"] - start) < eps and abs(c["end"] - end) < eps
               for c in CUTS)


def load_edits():
    """Restore previously saved cuts + split points if they exist."""
    global CUTS, SPLIT_POINTS, UNDO_STACK_CACHE, CROP, CAPTIONS, IMAGE_OVERLAYS, ADJUST, AUDIO_LAYERS, VIDEO_VOL, MUTED_SEGS
    if not STATE_FILE or not Path(STATE_FILE).exists():
        return
    try:
        data = json.loads(Path(STATE_FILE).read_text())
        saved_cuts   = data.get("cuts", [])
        saved_splits = data.get("split_points", [])
        saved_undo   = data.get("undo_stack", [])
        if data.get("crop"):
            CROP = data["crop"]
        CAPTIONS = data.get("captions", []) or []
        IMAGE_OVERLAYS = data.get("images", []) or []
        if data.get("adjust"): ADJUST = {**ADJUST, **data["adjust"]}
        AUDIO_LAYERS = data.get("audio", []) or []
        try: VIDEO_VOL = float(data.get("video_volume", 1.0))
        except (TypeError, ValueError): VIDEO_VOL = 1.0
        MUTED_SEGS = [list(m)[:2] for m in (data.get("muted_segs", []) or []) if isinstance(m, (list, tuple)) and len(m) >= 2]
        if saved_cuts or saved_splits:
            before = len(saved_cuts)
            saved_cuts = dedupe_cut_list(saved_cuts)
            if len(saved_cuts) != before:
                print(f"  Deduped cuts: {before} → {len(saved_cuts)}")
            CUTS              = saved_cuts
            SPLIT_POINTS      = saved_splits
            UNDO_STACK_CACHE  = saved_undo[-500:]   # cap runaway undo history
            print(f"  Restored {len(CUTS)} cuts + {len(SPLIT_POINTS)} splits + {len(UNDO_STACK_CACHE)} undo steps from saved state")
    except Exception as e:
        print(f"  Warning: could not load saved edits: {e}")


def save_learning_snapshot():
    """
    On export, write a snapshot comparing auto-detected cuts vs. final user edits.
    Accumulates across sessions in ~/.cache/video_editor/edit_style_log.jsonl
    so patterns in Helen's editing style can be learned over time.
    """
    try:
        log_path = Path.home() / ".cache" / "video_editor" / "edit_style_log.jsonl"
        auto_ids  = {c["id"] for c in INITIAL_AUTO_CUTS}
        final_ids = {c["id"] for c in CUTS if c["active"]}

        # Cuts that were auto-detected and kept by user
        kept_auto    = [c for c in INITIAL_AUTO_CUTS if c["id"] in final_ids]
        # Cuts that were auto-detected but user turned off
        rejected_auto = [c for c in INITIAL_AUTO_CUTS if c["id"] not in final_ids]
        # Cuts the user added manually (not in original auto set)
        added_manual = [c for c in CUTS if c["active"] and c["id"] not in auto_ids]

        record = {
            "video":          ORIGINAL_STEM,
            "timestamp":      __import__("datetime").datetime.now().isoformat(),
            "kept_auto":      [{"label": c["label"], "type": c["type"],
                                 "dur": round(c["end"] - c["start"], 2)} for c in kept_auto],
            "rejected_auto":  [{"label": c["label"], "type": c["type"],
                                 "dur": round(c["end"] - c["start"], 2)} for c in rejected_auto],
            "added_manual":   [{"label": c.get("label","manual"), "type": c.get("type","manual"),
                                 "dur": round(c["end"] - c["start"], 2)} for c in added_manual],
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  Learning snapshot saved ({len(kept_auto)} kept, {len(rejected_auto)} rejected, {len(added_manual)} manual)")
    except Exception as e:
        print(f"  Warning: could not save learning snapshot: {e}")


# ── Thumbnails ────────────────────────────────────────────────────────────────

def extract_thumbnails(video: Path, out_dir: Path, interval: float) -> int:
    print(f"Extracting thumbnails (1 per {interval:.0f}s)...")
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        FFMPEG, "-y", "-i", str(video),
        "-vf", f"fps=1/{interval},scale=160:-1",
        str(out_dir / "thumb%04d.jpg"),
    ], capture_output=True, check=True)
    count = len(list(out_dir.glob("thumb*.jpg")))
    print(f"  {count} thumbnails")
    return count


# ── Waveform ─────────────────────────────────────────────────────────────────

def extract_waveform(video: Path, samples_per_sec: float = 60.0) -> list:
    print("Extracting waveform...")
    target_rate = int(samples_per_sec * 512)
    result = subprocess.run([
        FFMPEG, "-i", str(video),
        "-vn", "-ac", "1", "-ar", str(target_rate),
        "-f", "f32le", "pipe:1",
    ], capture_output=True, check=True)

    data = result.stdout
    n = len(data) // 4
    floats = struct.unpack(f"{n}f", data[: n * 4])

    chunk = 512
    rms_vals = []
    for i in range(0, n, chunk):
        bucket = floats[i : i + chunk]
        rms = math.sqrt(sum(x * x for x in bucket) / len(bucket)) if bucket else 0
        rms_vals.append(rms)

    mx = max(rms_vals, default=1) or 1
    rms_vals = [round(v / mx, 4) for v in rms_vals]
    print(f"  {len(rms_vals)} waveform samples")
    return rms_vals


def peak_envelope(path, window_s=120.0, sps=50):
    """Peak-amplitude envelope of the first `window_s` seconds of an audio/video
    file at `sps` points/sec — used to draw waveforms for manual audio alignment."""
    target_rate = int(sps * 256)
    try:
        r = subprocess.run([FFMPEG, "-i", str(path), "-t", str(window_s),
                            "-vn", "-ac", "1", "-ar", str(target_rate),
                            "-f", "f32le", "pipe:1"], capture_output=True)
        data = r.stdout
        n = len(data) // 4
        fl = struct.unpack(f"{n}f", data[: n * 4])
        chunk = 256
        out = [max((abs(x) for x in fl[i:i + chunk]), default=0.0)
               for i in range(0, n, chunk)]
        mx = max(out, default=1.0) or 1.0
        return [round(v / mx, 3) for v in out]
    except Exception as e:
        print(f"  peak_envelope failed: {e}")
        return []


# ── Labeled cut detection ─────────────────────────────────────────────────────

def get_snippet(words: list, start: float, end: float, max_words: int = 8) -> str:
    w = [x for x in words if x["start"] >= start and x["end"] <= end]
    if not w:
        w = [x for x in words if x["start"] >= start and x["start"] <= end]
    if not w:
        return ""
    text = " ".join(x["raw"] for x in w[:max_words])
    return text + ("..." if len(w) > max_words else "")


def get_video_dims(video: Path):
    """Return (width, height) of the video's first stream, or (0,0) on failure."""
    try:
        r = subprocess.run([
            FFPROBE, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x", str(video),
        ], capture_output=True, text=True)
        w, h = r.stdout.strip().split("x")
        return int(w), int(h)
    except Exception:
        return 0, 0


def build_proxy(video: Path, out: Path):
    """Make a small, dense-keyframe proxy so the preview can seek instantly.
    540p, keyframe every 15 frames. Returns out on success, else None. Export
    never uses this — it's preview-only."""
    try:
        tmp = str(out) + ".building.mp4"
        r = subprocess.run([
            FFMPEG, "-y", "-i", str(video),
            "-vf", "scale=-2:540",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
            "-g", "15", "-keyint_min", "15", "-sc_threshold", "0",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            tmp,
        ], capture_output=True, text=True)
        if r.returncode == 0 and Path(tmp).exists():
            os.replace(tmp, out)
            return out
        print(f"  proxy build failed: {r.stderr[-400:]}")
    except Exception as e:
        print(f"  proxy build error: {e}")
    return None


def proxy_is_valid(path) -> bool:
    """A cached proxy is only reusable if it actually DECODES.

    A build that was interrupted (or written by an older code path) can leave a
    file that exists and is non-zero but has no usable video stream. Reusing it
    makes the preview silently black — the browser reports no error, it just
    never reaches a frame. Cheap ffprobe guard so that can't happen again."""
    try:
        r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width", "-of",
                            "default=noprint_wrappers=1:nokey=1", str(path)],
                           capture_output=True, text=True, timeout=20)
        return r.returncode == 0 and int(r.stdout.strip() or 0) > 0
    except Exception:
        return False


def _build_proxy_bg(video: Path, out: Path, pid: str):
    """Background proxy build; flips PROXY_READY only if its project is still open."""
    global PROXY_PATH, PROXY_READY
    print(f"  Building smooth-preview proxy for '{pid}'…")
    result = build_proxy(video, out)
    if result and CURRENT_PROJECT_ID == pid:
        PROXY_PATH  = result
        PROXY_READY = True
        print(f"  ✓ Proxy ready: {out.name}")


# ── Rendered smooth-preview (edited cut concatenated into one continuous file) ──

def compute_keeps():
    """The kept segments for the CURRENT edit — identical logic to the export."""
    active = sorted((c["start"], c["end"]) for c in CUTS if c.get("active"))
    merged = []
    for s, e in active:
        if merged and s <= merged[-1][1] + 0.15:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    merged = [(s, e) for s, e in merged]
    return cuts_to_keeps(merged, DURATION)


def keeps_signature(keeps):
    """Short stable signature of a keep list so we only re-render when the edit changes.
    Includes the audio source (external mic vs camera) so removing/attaching external audio
    yields a NEW signature → a fresh proxy URL (otherwise the browser replays the stale
    proxy with the old audio baked in)."""
    import hashlib
    raw = ";".join(f"{round(s,3)}-{round(e,3)}" for s, e in keeps)
    raw += "|ext=" + (Path(str(EXTERNAL_AUDIO_PATH)).name if EXTERNAL_AUDIO_PATH else "0")
    raw += f"@{round(float(EXTERNAL_AUDIO_OFFSET or 0), 3)}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def snap_keeps(keeps, fps):
    """Snap every keep boundary to the video frame grid (multiples of 1/fps).

    The video track quantizes each segment to whole frames; the audio track is
    sample-exact. If they're trimmed at different effective boundaries the two
    streams drift apart over many cuts (video creeps ahead — worst at the end).
    Snapping the boundaries first means the video trim and the audio trim describe
    the *same* frame-aligned span, so each segment's video and audio are identical
    lengths and stay locked start-to-finish. Cut points move by <½ frame (~20ms),
    inaudible. Used for BOTH the smooth preview and the export so they match."""
    if not fps or fps <= 0:
        return list(keeps)
    return [(round(s * fps) / fps, round(e * fps) / fps) for s, e in keeps]


def _probe_fps(path):
    try:
        r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=r_frame_rate", "-of",
                            "default=noprint_wrappers=1:nokey=1", str(path)],
                           capture_output=True, text=True)
        num, den = r.stdout.strip().split("/")
        return round(int(num) / int(den), 3)
    except Exception:
        return 30.0


def _stream_dur(path, stream):
    try:
        r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", stream,
                            "-show_entries", "stream=duration", "-of",
                            "default=noprint_wrappers=1:nokey=1", str(path)],
                           capture_output=True, text=True)
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def av_lock(src, out):
    """Final safety net: make the VIDEO exactly match the AUDIO duration.

    The audio is the accurate master — it's sample-exact and honors every cut.
    The video can still shed up to a frame per segment even after frame-snapping,
    so if any residual drift remains we stretch the *video* (setpts) to match the
    audio, never the other way around. The stretch is a fraction of a percent —
    imperceptible — and it preserves the audio untouched. Returns out.

    With snap_keeps() upstream the residual is usually below threshold and this
    becomes a no-op pass-through."""
    dv, da = _stream_dur(src, "v:0"), _stream_dur(src, "a:0")
    ratio = (da / dv) if dv > 0 else 1.0   # >1 → video is short, slow it to fit audio
    if dv <= 0 or da <= 0 or abs(ratio - 1.0) < 0.0005 or not (0.5 < ratio < 2.0):
        os.replace(src, out); return out   # negligible/uncorrectable → leave as-is
    # Preserve the source's colour tags (HDR/HLG) across the re-encode so it isn't
    # silently downgraded to bt709 (which would break the HLG-encoded captions).
    ctags = []
    try:
        cinfo = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                                "-show_entries", "stream=color_space,color_primaries,color_transfer",
                                "-of", "default=noprint_wrappers=1:nokey=1", str(src)],
                               capture_output=True, text=True).stdout.split()
        if len(cinfo) == 3 and "unknown" not in cinfo:
            ctags = ["-colorspace", cinfo[0], "-color_primaries", cinfo[1], "-color_trc", cinfo[2]]
    except Exception:
        pass
    r = subprocess.run([FFMPEG, "-y", "-i", str(src),
                        "-filter:v", f"setpts={ratio:.6f}*PTS",
                        "-c:v", "libx264", "-crf", "16", "-preset", "fast",
                        "-fps_mode", "cfr", "-pix_fmt", "yuv420p", *ctags,
                        "-c:a", "copy", "-movflags", "+faststart", str(out)],
                       capture_output=True, text=True)
    if r.returncode == 0 and Path(out).exists():
        Path(src).unlink(missing_ok=True)
        print(f"  A/V-locked video (×{ratio:.5f}, was {dv-da:+.3f}s short)")
        return out
    print(f"  av_lock failed, using raw: {r.stderr[-200:]}")
    os.replace(src, out); return out


def build_edit_proxy(keeps, out: Path):
    """Render the EDITED video (cuts removed) as one continuous 540p file from the
    proxy — so the preview can play it with zero seeks. Reuses the export's audio
    handling (camera or external). No crop (the preview applies crop via CSS)."""
    src = PROXY_PATH if (PROXY_PATH and Path(PROXY_PATH).exists()) else VIDEO_PATH
    n = len(keeps)
    if n == 0 or not src:
        return None
    # Frame-snap boundaries so video & audio trims describe the same span (no drift),
    # and force CFR per segment with an explicit fps filter so the concat can't drop frames.
    fps = _probe_fps(src)
    keeps = snap_keeps(keeps, fps)
    vt = "".join(f"[0:v]trim={s:.4f}:{e:.4f},fps={fps},setpts=PTS-STARTPTS[v{i}];"
                 for i, (s, e) in enumerate(keeps))
    vc = "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[vout]"
    tmp = str(out) + ".rendering.mp4"

    if EXTERNAL_AUDIO_PATH and Path(EXTERNAL_AUDIO_PATH).exists():
        ext_dur = get_duration(EXTERNAL_AUDIO_PATH)
        AFMT = "aformat=sample_rates=48000:channel_layouts=stereo"
        at = ""
        for i, (s, e) in enumerate(keeps):
            es = max(0.0, s - EXTERNAL_AUDIO_OFFSET)
            ee = min(ext_dur, e - EXTERNAL_AUDIO_OFFSET)
            if ee > es + 0.05:
                at += f"[1:a]atrim={es:.4f}:{ee:.4f},asetpts=PTS-STARTPTS,{AFMT}[a{i}];"
            else:
                at += (f"anullsrc=channel_layout=stereo:sample_rate=48000,"
                       f"atrim=0:{max(0.02, e-s):.4f},asetpts=PTS-STARTPTS[a{i}];")
        ac = "".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[aout]"
        cmd = [FFMPEG, "-y", "-i", str(src), "-i", str(EXTERNAL_AUDIO_PATH),
               "-filter_complex", vt + vc + ";" + at + ac]
    else:
        at = "".join(f"[0:a]atrim={s:.4f}:{e:.4f},asetpts=PTS-STARTPTS[a{i}];"
                     for i, (s, e) in enumerate(keeps))
        ac = "".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[aout]"
        cmd = [FFMPEG, "-y", "-i", str(src), "-filter_complex", vt + vc + ";" + at + ac]

    cmd += ["-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
            "-g", "15", "-keyint_min", "15", "-sc_threshold", "0",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", tmp]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and Path(tmp).exists():
            return av_lock(tmp, out)   # lock audio length to video length (no drift)
        print(f"  edit-proxy render failed: {r.stderr[-400:]}")
    except Exception as e:
        print(f"  edit-proxy render error: {e}")
    return None


def edit_proxy_path(pid, sig):
    return Path.home() / ".cache" / "video_editor" / "uploads" / pid / f"_editproxy_{sig}.mp4"


def _render_edit_proxy_bg(keeps, sig, pid):
    global EDIT_PROXY_STATUS
    out = edit_proxy_path(pid, sig)   # versioned by edit signature — never overwrites a playing file
    print(f"  Rendering smooth preview ({len(keeps)} segments, sig={sig})…")
    result = build_edit_proxy(keeps, out)
    if CURRENT_PROJECT_ID != pid:
        return
    if result:
        EDIT_PROXY_STATUS = {"state": "ready", "sig": sig}
        print(f"  ✓ Smooth preview ready (sig={sig})")
        # Clean up stale versions (keep only the current one)
        try:
            for f in out.parent.glob("_editproxy_*.mp4"):
                if f != out:
                    f.unlink(missing_ok=True)
        except Exception:
            pass
    elif EDIT_PROXY_STATUS.get("sig") == sig:
        EDIT_PROXY_STATUS = {"state": "idle", "sig": None}


def default_crop_169(w, h):
    """Largest centered 16:9 rectangle that fits inside a w×h frame."""
    if not w or not h:
        return None
    if w / h > 16 / 9:           # source wider than 16:9 → limit by height
        cw, ch = int(round(h * 16 / 9)), h
    else:                        # source taller/narrower → limit by width
        cw, ch = w, int(round(w * 9 / 16))
    return {"x": (w - cw) // 2, "y": (h - ch) // 2, "w": cw, "h": ch}


def sync_external_audio(video_path: Path, ext_audio_path: Path) -> float:
    """
    Cross-correlate the video's built-in audio with the external mic audio
    to find the time offset.  Returns offset in seconds such that:
        ext_audio_time = video_time - offset
    i.e. to trim the external audio for a kept video segment [s, e],
    use [s - offset, e - offset] from the ext audio file.
    """
    import wave, struct
    try:
        import numpy as np
    except ImportError:
        print("  numpy not found — skipping audio sync, offset=0")
        return 0.0

    SR = 8000   # 8 kHz mono is plenty for correlation
    LOOK = 60   # only use first 60s (plenty to find the sync point)

    tmp = Path(tempfile.mkdtemp(prefix="sync_"))
    vid_wav = tmp / "vid.wav"
    ext_wav = tmp / "ext.wav"

    subprocess.run([
        FFMPEG, "-y", "-i", str(video_path),
        "-ac", "1", "-ar", str(SR), "-t", str(LOOK),
        "-f", "wav", str(vid_wav),
    ], capture_output=True)
    subprocess.run([
        FFMPEG, "-y", "-i", str(ext_audio_path),
        "-ac", "1", "-ar", str(SR), "-t", str(LOOK),
        "-f", "wav", str(ext_wav),
    ], capture_output=True)

    def load_wav(path):
        with wave.open(str(path)) as w:
            raw = w.readframes(w.getnframes())
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        arr /= max(np.abs(arr).max(), 1)
        return arr

    vid = load_wav(vid_wav)
    ext = load_wav(ext_wav)

    # Zero-pad shorter array so correlate works across the full offset range
    max_len = max(len(vid), len(ext))
    vid = np.pad(vid, (0, max_len - len(vid)))
    ext = np.pad(ext, (0, max_len - len(ext)))

    corr  = np.correlate(vid, ext, mode="full")
    lag   = int(np.argmax(corr)) - (max_len - 1)   # positive = ext audio starts later
    offset = lag / SR   # video_time = ext_time + offset  →  ext_time = video_time - offset

    print(f"  Audio sync: lag={lag} samples, offset={offset:+.3f}s "
          f"({'ext starts {:.2f}s AFTER video'.format(offset) if offset > 0 else 'ext starts {:.2f}s BEFORE video'.format(-offset)})")
    import shutil; shutil.rmtree(tmp, ignore_errors=True)
    return offset


def build_labeled_cuts(words: list, duration: float,
                       skip_fillers=False, skip_bad_takes=False,
                       skip_silence=False) -> list:
    raw: list[dict] = []

    # ── Repeated takes (AI-based semantic detection) ─────────────────────────
    if not skip_bad_takes:
        # Short stutter phrases ("I get it… I get it… I get it…")
        for c in detect_stutter_phrases(words):
            raw.append({
                "start": c["start"], "end": c["end"],
                "label": c.get("label", "stutter"),
                "type":  c.get("type", "exact repeat"),
            })
        # AI semantic repeat take detection (longer rephrase restarts)
        for c in detect_repeated_takes_ai(words):
            raw.append({
                "start": c["start"], "end": c["end"],
                "label": c.get("label", "repeat take"),
                "type":  c.get("type", "rephrase"),
            })

        # ── Bad take phrases ──────────────────────────────────────────────────
        i, n = 0, len(words)
        while i < n:
            matched = False
            if words[i]["word"] in RETAKE_SIGNALS:
                ts = _take_start(words, i)
                te = words[i]["end"] + 0.3
                raw.append({"start": ts, "end": te,
                             "label": f'"{words[i]["raw"]}"',
                             "type":  "bad take signal"})
                i += 1
                matched = True
            if not matched:
                for phrase in RETAKE_PHRASES:
                    pw = len(phrase)
                    if i + pw <= n and [words[i + k]["word"] for k in range(pw)] == phrase:
                        ts = _take_start(words, i)
                        te = words[i + pw - 1]["end"] + 0.3
                        raw.append({"start": ts, "end": te,
                                     "label": " ".join(phrase),
                                     "type":  "restart phrase"})
                        i += pw
                        matched = True
                        break
            if not matched:
                i += 1

    # ── Filler words ──────────────────────────────────────────────────────────
    if not skip_fillers:
        for w in words:
            if w["word"] in FILLER_WORDS:
                raw.append({
                    "start": max(0, w["start"] - FILLER_PAD),
                    "end":   w["end"] + FILLER_PAD,
                    "label": f'"{w["raw"]}"',
                    "type":  "filler",
                })

    # ── Long silences ─────────────────────────────────────────────────────────
    if not skip_silence and words:
        if words[0]["start"] > SILENCE_MAX:
            raw.append({"start": SILENCE_KEEP, "end": words[0]["start"],
                        "label": "leading silence", "type": "silence"})
        for i in range(len(words) - 1):
            gap = words[i + 1]["start"] - words[i]["end"]
            if gap > SILENCE_MAX:
                raw.append({"start": words[i]["end"] + SILENCE_KEEP,
                             "end":   words[i + 1]["start"],
                             "label": f"pause ({gap:.1f}s)",
                             "type":  "silence"})
        trailing = duration - words[-1]["end"]
        if trailing > SILENCE_MAX:
            raw.append({"start": words[-1]["end"] + SILENCE_KEEP,
                        "end":   duration,
                        "label": "trailing silence", "type": "silence"})

    # ── Merge overlapping, assign IDs ─────────────────────────────────────────
    raw.sort(key=lambda c: c["start"])
    merged: list[dict] = []
    for c in raw:
        if merged and c["start"] < merged[-1]["end"] - 0.1:
            if c["end"] > merged[-1]["end"]:
                merged[-1]["end"] = c["end"]
        else:
            merged.append(dict(c))

    for i, c in enumerate(merged):
        c["id"] = f"cut_{i}"
        c["active"] = True

    print(f"  {len(merged)} cuts total")
    return merged


# ── High-quality export (no audio processing, matches original quality) ──────

def _ffmpeg_with_progress(cmd, total_dur, label="Encoding"):
    """Run an ffmpeg command, streaming live percent + ETA into ENCODE_STATUS so the
    export UI can show real progress instead of a flat 'encoding…'. `cmd` must end
    with the output path (we insert -progress before it). Returns (returncode, stderr)."""
    import time as _t, re as _re, tempfile as _tf
    cmd = cmd[:-1] + ["-progress", "pipe:1", "-nostats", cmd[-1]]
    errf = _tf.NamedTemporaryFile("w+", suffix=".log", delete=False)
    global ENCODE_PROC
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf,
                                text=True, bufsize=1)
        ENCODE_PROC = proc     # tracked so /api/cancel-encode can kill it
        start = _t.time()
        last = -1
        for line in proc.stdout:
            line = line.strip()
            if not line.startswith("out_time="):
                continue
            m = _re.match(r"(\d+):(\d+):(\d+(?:\.\d+)?)", line.split("=", 1)[1])
            if not (m and total_dur > 0):
                continue
            done = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
            pct = max(0.0, min(99.0, done / total_dur * 100))
            if int(pct) == last:
                continue
            last = int(pct)
            elapsed = _t.time() - start
            if pct > 1:
                eta = elapsed * (100 - pct) / pct
                eta_s = f"{int(eta // 60)}m {int(eta % 60):02d}s"
            else:
                eta_s = "estimating…"
            ENCODE_STATUS["message"] = f"{label} — {pct:.0f}% · ETA {eta_s}"
        proc.wait()
        errf.flush(); errf.seek(0); err = errf.read()
        return proc.returncode, err
    finally:
        ENCODE_PROC = None
        try: errf.close(); os.unlink(errf.name)
        except Exception: pass


# ── Text caption burn-in (this ffmpeg has no drawtext/libass → render with PIL, overlay) ──
CAPTION_FONT_FILES = {
    "Arial": ["/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
              "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf"],
    "Helvetica": ["/System/Library/Fonts/Helvetica.ttc"],
    "Avenir": ["/System/Library/Fonts/Avenir.ttc"],
}


def _load_caption_font(family, bold, italic, px):
    """Pick the face inside the family's files best matching bold/italic (Avenir &
    Helvetica are .ttc collections; Arial ships one file per style)."""
    from PIL import ImageFont
    cands = CAPTION_FONT_FILES.get(family) or CAPTION_FONT_FILES["Arial"]
    best = None
    for path in cands:
        if not os.path.exists(path):
            continue
        idx = 0
        while idx < 40:
            try:
                f = ImageFont.truetype(path, px, index=idx)
            except Exception:
                break
            try:
                style = (f.getname()[1] or "").lower()
            except Exception:
                style = ""
            isb = any(w in style for w in ("bold", "heavy", "black"))
            isi = ("italic" in style) or ("oblique" in style)
            score = int(isb == bool(bold)) + int(isi == bool(italic))
            if best is None or score > best[0]:
                best = (score, f)
            if score == 2:
                return f
            idx += 1
            if path.lower().endswith(".ttf"):
                break   # single-face file
    if best:
        return best[1]
    try:
        return ImageFont.truetype(cands[0], px)
    except Exception:
        return ImageFont.load_default()


APPLE_EMOJI_FONT = "/System/Library/Fonts/Apple Color Emoji.ttc"
# Runs of emoji (incl. ZWJ 200D, VS16 FE0F, keycap 20E3, skin-tone modifiers) kept together
_EMOJI_RE = None
def _emoji_re():
    global _EMOJI_RE
    if _EMOJI_RE is None:
        import re
        _EMOJI_RE = re.compile(
            "([\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\U00002600-\U000027BF"
            "\U00002B00-\U00002BFF\U0001F000-\U0001F02F\U0000FE0F\U0000200D\U000020E3]+)")
    return _EMOJI_RE


def _encode_graphics_for_hlg(img):
    """Re-encode an sRGB caption graphic into Rec.2020 + HLG so it shows the intended
    colour when burned into an HLG/HDR video. Without this, sRGB values are read through
    the HDR gamut and yellow skews orange/red. Video pixels are untouched — only the
    caption PNG is converted. Neutral colours (black/white) are preserved."""
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return img
    arr = np.asarray(img).astype(np.float32) / 255.0
    rgb, alpha = arr[..., :3], arr[..., 3:4]
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)  # sRGB→linear(709)
    M = np.array([[0.6274, 0.3293, 0.0433],      # Rec.709 → Rec.2020 (linear)
                  [0.0691, 0.9195, 0.0114],
                  [0.0164, 0.0880, 0.8956]], dtype=np.float32)
    lin = np.clip(lin @ M.T, 0.0, 1.0)
    a, b, c = 0.17883277, 0.28466892, 0.55991073  # HLG OETF (BT.2100)
    sig = np.where(lin <= 1.0 / 12.0, np.sqrt(3.0 * lin),
                   a * np.log(np.maximum(12.0 * lin - b, 1e-6)) + c)
    out = np.clip(np.concatenate([sig, alpha], axis=-1) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def _render_caption_png(cap, out_w, out_h, hdr_hlg=False):
    """Render one caption (multi-line + color emoji) to a transparent PNG.
    Returns (path, x_px, y_px) or (None, 0, 0). Lines split on '\\n', left-aligned to
    match the preview; emoji runs render from Apple Color Emoji and scale to the line.
    If hdr_hlg, the graphic is converted to the video's HLG/Rec.2020 space so colours match."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None, 0, 0
    text = str(cap.get("text", ""))
    if not text.strip():
        return None, 0, 0
    px = max(8, int(float(cap.get("size", 0.06)) * out_h))
    font = _load_caption_font(cap.get("font", "Avenir"), cap.get("bold", False),
                              cap.get("italic", False), px)
    color = cap.get("color", "#ffffff")
    # Outline thickness: `stroke` is a fraction of font size (default ≈ px/18). Pillow draws
    # it OUTWARD; the preview's -webkit-text-stroke is calibrated to match (see buildCapBox).
    _T = cap.get("stroke")
    _T = 0.0556 if _T is None else float(_T)
    stroke = max(0, int(round(_T * px)))
    pad = stroke + 4
    ls_px = int(round(float(cap.get("ls", 0) or 0) * out_h))   # letter spacing (matches preview)
    asc, desc = font.getmetrics()
    line_h = asc + desc
    _lh = cap.get("lh")
    if _lh is not None:
        line_gap = max(0, int(round(line_h * (float(_lh) - 1.0))))
    else:
        line_gap = max(2, int(line_h * 0.12))
    emoji_h = max(8, int(line_h * 0.92))
    try:
        emoji_font = ImageFont.truetype(APPLE_EMOJI_FONT, 160)
    except Exception:
        emoji_font = None
    measure = ImageDraw.Draw(Image.new("RGBA", (4, 4)))

    def render_emoji_run(s):
        if not emoji_font:
            return None
        big = Image.new("RGBA", (160 * (len(s) + 2), 220), (0, 0, 0, 0))
        try:
            ImageDraw.Draw(big).text((4, 4), s, font=emoji_font, embedded_color=True)
        except Exception:
            return None
        bb = big.getbbox()
        if not bb:
            return None
        crop = big.crop(bb)
        scale = emoji_h / crop.height
        return crop.resize((max(1, int(crop.width * scale)), emoji_h), Image.LANCZOS)

    # Optional fixed box width (fraction of frame) → word-wrap to that many pixels.
    import re as _re
    try:
        maxw_px = int(float(cap["w"]) * out_w) if cap.get("w") else None
    except (TypeError, ValueError):
        maxw_px = None

    # Break each source line into word/emoji atoms (so we can wrap between them).
    def atoms_of(line):
        atoms = []
        for i, part in enumerate(_emoji_re().split(line)):
            if not part:
                continue
            if i % 2 == 1:                       # emoji run
                im = render_emoji_run(part)
                if im is not None:
                    atoms.append(("emoji", im, im.width)); continue
                # emoji font unavailable → fall through, draw as text
            for tok in _re.findall(r"\s+|\S+", part):
                # width includes per-character letter spacing so wrapping + box size match
                atoms.append(("text", tok, measure.textlength(tok, font=font) + ls_px * len(tok)))
        return atoms

    lines_layout, max_w = [], 1
    for line in text.split("\n"):
        atoms = atoms_of(line)
        if maxw_px:
            cur, curw = [], 0
            for a in atoms:
                is_space = (a[0] == "text" and a[1].isspace())
                if cur and not is_space and (curw + a[2]) > maxw_px:
                    lines_layout.append(cur); max_w = max(max_w, curw); cur, curw = [], 0
                if not cur and is_space:
                    continue                     # trim leading space on a wrapped line
                cur.append(a); curw += a[2]
            if cur:
                lines_layout.append(cur); max_w = max(max_w, curw)
            if not atoms:
                lines_layout.append([])          # preserve blank lines
        else:
            segs = atoms
            lines_layout.append(segs)
            max_w = max(max_w, sum(a[2] for a in segs))

    n_lines = len(lines_layout)
    W = int(max_w) + 2 * pad
    H = n_lines * line_h + (n_lines - 1) * line_gap + 2 * pad
    img = Image.new("RGBA", (max(1, W), max(1, H)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    y = pad
    for segs in lines_layout:
        x = pad
        for kind, content, w in segs:
            if kind == "text":
                if ls_px:
                    cx = x
                    for ch in content:
                        draw.text((cx, y), ch, font=font, fill=color,
                                  stroke_width=stroke, stroke_fill=(0, 0, 0, 210))
                        cx += measure.textlength(ch, font=font) + ls_px
                else:
                    draw.text((x, y), content, font=font, fill=color,
                              stroke_width=stroke, stroke_fill=(0, 0, 0, 210))
            else:
                img.alpha_composite(content, (int(x), int(y + (line_h - content.height) // 2)))
            x += w
        y += line_h + line_gap

    _a = cap.get("_alpha")
    if _a is not None:
        try:
            a = max(0.0, min(1.0, float(_a)))
            if a < 0.999:
                alpha = img.split()[3].point(lambda v: int(v * a))
                img.putalpha(alpha)
        except Exception:
            pass
    if hdr_hlg:
        img = _encode_graphics_for_hlg(img)   # match the video's HLG/Rec.2020 colour space
    path = tempfile.mktemp(prefix="cap_", suffix=".png")
    img.save(path)
    x_px = int(float(cap.get("x", 0.1)) * out_w) - pad
    y_px = int(float(cap.get("y", 0.1)) * out_h) - pad
    # Center-based zoom: this PNG was rendered at font size*sc; shift it so the (smaller) render
    # stays centered on the same point as the full-size text, instead of anchored top-left.
    zs = cap.get("_zoom_sc")
    if zs is not None:
        try:
            zs = float(zs)
            if 0 < zs < 1:
                x_px += int(W * (1.0 / zs - 1.0) / 2.0)
                y_px += int(H * (1.0 / zs - 1.0) / 2.0)
        except Exception:
            pass
    return path, x_px, y_px


def _caption_output_intervals(cap, keeps):
    """Map a caption's [start,end] (ORIGINAL time) to output-time intervals, following
    the kept segments — so a caption over a cut region simply shows during what remains."""
    cs, ce = float(cap.get("start", 0)), float(cap.get("end", 0))
    out, cum = [], 0.0
    for (s, e) in keeps:
        a, b = max(cs, s), min(ce, e)
        if b > a + 0.001:
            out.append((cum + (a - s), cum + (b - s)))
        cum += (e - s)
    return out


def _caption_is_animated(cap):
    """True if the caption has a distinct end keyframe (position and/or size transform)."""
    x, y, sz = cap.get("x", 0.1), cap.get("y", 0.1), cap.get("size", 0.06)
    x2, y2, s2 = cap.get("x2"), cap.get("y2"), cap.get("size2")
    return ((x2 is not None and abs(float(x2) - float(x)) > 1e-4) or
            (y2 is not None and abs(float(y2) - float(y)) > 1e-4) or
            (s2 is not None and abs(float(s2) - float(sz)) > 1e-4))


def _caption_keyframe_steps(cap):
    """Expand an animated caption into static sub-captions (stepped keyframe animation for
    export). Position (x,y) and size interpolate linearly from the start to the end keyframe."""
    start, end = float(cap.get("start", 0)), float(cap.get("end", 0))
    dur = max(0.001, end - start)
    x1, y1, s1 = float(cap.get("x", 0.1)), float(cap.get("y", 0.1)), float(cap.get("size", 0.06))
    x2 = float(cap.get("x2", x1)); y2 = float(cap.get("y2", y1)); s2 = float(cap.get("size2", s1))
    n = max(2, min(90, int(dur / 0.06)))     # ~one step per 0.06s (≈16/s) for smooth motion
    steps = []
    for i in range(n):
        a, b = i / n, (i + 1) / n
        pm = (a + b) / 2.0                    # interpolate at the sub-interval midpoint
        sub = dict(cap)
        sub["start"] = start + a * dur
        sub["end"]   = start + b * dur
        sub["x"] = x1 + (x2 - x1) * pm
        sub["y"] = y1 + (y2 - y1) * pm
        sub["size"] = s1 + (s2 - s1) * pm
        for k in ("x2", "y2", "size2"):
            sub.pop(k, None)
        steps.append(sub)
    return steps


def _fx_charsPerSec(speed):   # typewriter chars/sec from a 0.1..1 speed
    return 4.0 + max(0.1, min(1.0, speed)) * 36.0

def _fx_duration(speed):      # fade/zoom in-or-out duration (s) from a 0.1..1 speed
    return max(0.1, 1.3 - max(0.1, min(1.0, speed)) * 1.1)

def _caption_fx_steps(cap):
    """Expand a text-effect caption (typewriter / fade / zoom) into static sub-captions with
    per-step text, alpha, or size. Returns None if the caption has no effect.
    Mirrors the live preview (fxCharsPerSec / fxDur in JS)."""
    fx = str(cap.get("fx", "none") or "none")
    if fx in ("none", ""):
        return None
    start, end = float(cap.get("start", 0)), float(cap.get("end", 0))
    total = max(0.05, end - start)
    speed = max(0.1, min(1.0, float(cap.get("fxSpeed", 0.5))))
    mode = str(cap.get("fxMode", "in") or "in")
    do_in, do_out = mode in ("in", "both"), mode in ("out", "both")
    text = str(cap.get("text", ""))
    base_size = float(cap.get("size", 0.06))

    def base(**over):
        d = dict(cap)
        for k in ("fx", "fxSpeed", "fxMode", "x2", "y2", "size2"):
            d.pop(k, None)
        d.update(over); return d

    steps = []
    if fx == "typewriter":
        n = len(text)
        if n == 0:
            return [base(start=start, end=end)]
        cps = _fx_charsPerSec(speed)
        stride = max(1, int(round(n / 40.0)))      # cap ~40 reveal steps
        c = stride
        while c < n:
            a = start + (c - stride) / cps
            b = start + c / cps
            if a >= end:
                break
            steps.append(base(text=text[:c], start=a, end=min(b, end)))
            c += stride
        # full text from when typing finishes → end
        full_at = start + n / cps
        steps.append(base(text=text, start=min(full_at, end - 0.001), end=end))
        return steps

    if fx in ("fade", "zoom"):
        D = min(total * 0.5, _fx_duration(speed))
        M = max(4, min(18, int(D / 0.05)))
        def emit(a, b, p):
            p = max(0.0, min(1.0, p))
            if fx == "fade":
                steps.append(base(_alpha=round(p, 3), start=a, end=b))
            else:
                sc = 0.6 + 0.4 * p
                # _zoom_sc lets _render_caption_png recenter the scaled PNG (center-based zoom)
                steps.append(base(size=base_size * sc, _alpha=round(0.4 + 0.6 * p, 3),
                                  _zoom_sc=round(sc, 4), start=a, end=b))
        if do_in:
            for k in range(M):
                emit(start + (k / M) * D, start + ((k + 1) / M) * D, (k + 0.5) / M)
        mid_s = start + (D if do_in else 0.0)
        mid_e = end - (D if do_out else 0.0)
        if mid_e > mid_s + 0.001:
            steps.append(base(start=mid_s, end=mid_e))
        if do_out:
            for k in range(M):
                emit(end - D + (k / M) * D, end - D + ((k + 1) / M) * D, 1.0 - (k + 0.5) / M)
        return steps
    return None


# ── Image overlays (export) — mirror captions: static PNG scaled/positioned, optional Ken Burns ──
def _image_is_animated(im):
    x, y, w = im.get("x", 0.25), im.get("y", 0.1), im.get("w", 0.4)
    x2, y2, w2 = im.get("x2"), im.get("y2"), im.get("w2")
    return ((x2 is not None and abs(float(x2) - float(x)) > 1e-4) or
            (y2 is not None and abs(float(y2) - float(y)) > 1e-4) or
            (w2 is not None and abs(float(w2) - float(w)) > 1e-4))


def _image_keyframe_steps(im):
    start, end = float(im.get("start", 0)), float(im.get("end", 0))
    dur = max(0.001, end - start)
    x1, y1, w1 = float(im.get("x", 0.25)), float(im.get("y", 0.1)), float(im.get("w", 0.4))
    x2 = float(im.get("x2", x1)); y2 = float(im.get("y2", y1)); w2 = float(im.get("w2", w1))
    n = max(2, min(90, int(dur / 0.06)))     # ≈16 steps/s for smooth motion
    steps = []
    for i in range(n):
        a, b = i / n, (i + 1) / n
        pm = (a + b) / 2.0
        sub = dict(im)
        sub["start"] = start + a * dur; sub["end"] = start + b * dur
        sub["x"] = x1 + (x2 - x1) * pm; sub["y"] = y1 + (y2 - y1) * pm; sub["w"] = w1 + (w2 - w1) * pm
        for k in ("x2", "y2", "w2"):
            sub.pop(k, None)
        steps.append(sub)
    return steps


def _render_image_overlay_png(im, out_w, out_h, hdr_hlg=False):
    """Scale the overlay image to w*out_w px (aspect-preserved), apply opacity, HDR-convert
    if needed. Returns (path, x_px, y_px) or (None, 0, 0)."""
    try:
        from PIL import Image
    except Exception:
        return None, 0, 0
    d = _project_overlay_dir()
    if d is None:
        return None, 0, 0
    src = d / str(im.get("src", ""))
    if not src.exists():
        return None, 0, 0
    try:
        pic = Image.open(str(src)).convert("RGBA")
    except Exception:
        return None, 0, 0
    tw = max(2, int(float(im.get("w", 0.4)) * out_w))
    th = max(2, int(tw * pic.height / pic.width))
    pic = pic.resize((tw, th), Image.LANCZOS)
    opacity = max(0.0, min(1.0, float(im.get("opacity", 1.0))))
    if opacity < 0.999:
        try:
            import numpy as np
            arr = np.asarray(pic).astype(np.float32)
            arr[..., 3] *= opacity
            pic = Image.fromarray(arr.clip(0, 255).astype("uint8"), "RGBA")
        except Exception:
            pass
    if hdr_hlg:
        pic = _encode_graphics_for_hlg(pic)
    path = tempfile.mktemp(prefix="imgov_", suffix=".png")
    pic.save(path)
    x_px = int(float(im.get("x", 0.25)) * out_w)
    y_px = int(float(im.get("y", 0.1)) * out_h)
    return path, x_px, y_px


def encode_hq(video: Path, keeps: list[tuple[float, float]], out: Path,
              ext_audio: Path | None = None, ext_offset: float = 0.0,
              crop: dict | None = None, captions: list | None = None,
              images: list | None = None, adjust: dict | None = None,
              audio: list | None = None, video_volume: float = 1.0,
              muted_segs: list | None = None) -> None:
    """
    Concatenate kept segments at full quality.
    If ext_audio is provided, the video track is muted and the external mic audio
    (shifted by ext_offset seconds) is used instead.
    If crop is provided ({x,y,w,h} source px), each frame is cropped + scaled to 16:9.
    """
    n = len(keeps)
    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = str(out) + ".raw.mp4"   # encode here first, then A/V-lock into `out`
    mode = "ext-audio" if ext_audio else "built-in audio"
    print(f"Encoding HQ ({n} segments, {mode}{', cropped' if crop else ''}) → {out}")

    fps_result = subprocess.run([
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video),
    ], capture_output=True, text=True)
    try:
        num, den = fps_result.stdout.strip().split("/")
        src_fps = round(int(num) / int(den), 3)
    except Exception:
        src_fps = 30.0

    # Is the source HLG HDR? If so, caption graphics must be converted to Rec.2020/HLG
    # (they're burned into an HLG-tagged output; plain sRGB would skew orange/red).
    try:
        trc = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                              "-show_entries", "stream=color_transfer",
                              "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
                             capture_output=True, text=True).stdout.strip()
        hdr_hlg = (trc == "arib-std-b67")
    except Exception:
        hdr_hlg = False

    # No tail buffer — export must match the preview exactly (boundaries are the raw cuts).
    # Frame-snap to the video grid so video & audio trims describe the same span and stay
    # locked across many cuts (identical to the smooth preview's snap_keeps).
    vid_dur = get_duration(video)
    buffered = snap_keeps([(s, min(e, vid_dur)) for s, e in keeps], src_fps)

    # Crop + scale-to-16:9 filter (applied per segment). Output 1080p (or 720p if small).
    if crop:
        ow = 1920 if crop["w"] >= 1280 else 1280
        oh = ow * 9 // 16
        crop_f = (f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']},"
                  f"scale={ow}:{oh},setsar=1,")
    else:
        crop_f = ""

    # ── Colour adjust (brightness/contrast/saturation), like iMovie ──
    adj = adjust or {}
    ab = float(adj.get("brightness", 0.0)); ac = float(adj.get("contrast", 1.0)); asat = float(adj.get("saturation", 1.0))
    adj_f = ""
    if abs(ab) > 1e-3 or abs(ac - 1.0) > 1e-3 or abs(asat - 1.0) > 1e-3:
        adj_f = f"eq=brightness={ab:.3f}:contrast={ac:.3f}:saturation={asat:.3f},"

    # ── Video segments (always from main video, no audio) ──
    vt = "".join(
        f"[0:v]trim={s:.4f}:{e:.4f},{crop_f}{adj_f}fps={src_fps},setpts=PTS-STARTPTS[v{i}];"
        for i, (s, e) in enumerate(buffered)
    )
    vc = "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[vout]"

    # Output frame size — maps caption fractions → pixels (same reference as the preview).
    # Must be the DISPLAY size: rotated phone videos store WxH swapped with a rotation flag,
    # and ffmpeg auto-applies the rotation on re-encode, so the output is the display size.
    if crop:
        out_w, out_h = ow, oh
    else:
        out_w, out_h = 1920, 1080
        try:
            pr = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                                 "-show_entries", "stream=width,height:stream_side_data=rotation",
                                 "-of", "json", str(video)], capture_output=True, text=True)
            st = json.loads(pr.stdout)["streams"][0]
            w, h = int(st["width"]), int(st["height"])
            rot = 0
            for sd in st.get("side_data_list", []):
                if "rotation" in sd:
                    rot = abs(int(sd["rotation"])) % 180
            out_w, out_h = (h, w) if rot == 90 else (w, h)   # swap for 90/270° rotation
        except Exception:
            pass

    if ext_audio:
        # ── External audio: shift each keep by -ext_offset to get ext timeline ──
        ext_dur = get_duration(ext_audio)
        AFMT = "aformat=sample_rates=48000:channel_layouts=stereo"
        at = ""
        for i, (s, e) in enumerate(buffered):
            es = max(0.0, s - ext_offset)
            ee = min(ext_dur, e - ext_offset)
            if ee > es + 0.05:
                at += f"[1:a]atrim={es:.4f}:{ee:.4f},asetpts=PTS-STARTPTS,{AFMT}[a{i}];"
            else:
                dur = max(0.02, e - s)
                at += (f"anullsrc=channel_layout=stereo:sample_rate=48000,"
                       f"atrim=0:{dur:.4f},asetpts=PTS-STARTPTS[a{i}];")
        ac = "".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[aout]"
        inputs = ["-i", str(video), "-i", str(ext_audio)]
        next_input = 2
    else:
        at = "".join(
            f"[0:a]atrim={s:.4f}:{e:.4f},asetpts=PTS-STARTPTS[a{i}];"
            for i, (s, e) in enumerate(buffered)
        )
        ac = "".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[aout]"
        inputs = ["-i", str(video)]
        next_input = 1

    # ── Caption + image overlays (burned in). Additive — no overlays → graph unchanged.
    caption_pngs = []
    overlay_str = ""
    video_map = "[vout]"
    cur = "[vout]"
    ov = 0

    def _add_overlay(png, px, py, intervals):
        nonlocal overlay_str, cur, next_input, ov
        caption_pngs.append(png)
        inputs.append("-i"); inputs.append(png)
        enable = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in intervals)
        nxt = f"[ov{ov}]"
        overlay_str += (f"{cur}[{next_input}:v]overlay="
                        f"x={int(px)}:y={int(py)}:enable='{enable}'{nxt};")
        cur = nxt; next_input += 1; ov += 1

    for cap in (captions or []):
        fx_steps = _caption_fx_steps(cap)
        if fx_steps is not None:
            subcaps = fx_steps
        elif _caption_is_animated(cap):
            subcaps = _caption_keyframe_steps(cap)
        else:
            subcaps = [cap]
        for cap_s in subcaps:
            intervals = _caption_output_intervals(cap_s, buffered)
            if not intervals:
                continue
            png, px, py = _render_caption_png(cap_s, out_w, out_h, hdr_hlg=hdr_hlg)
            if png:
                _add_overlay(png, px, py, intervals)

    for im in (images or []):
        subs = _image_keyframe_steps(im) if _image_is_animated(im) else [im]
        for im_s in subs:
            intervals = _caption_output_intervals(im_s, buffered)   # same start/end mapping
            if not intervals:
                continue
            png, px, py = _render_image_overlay_png(im_s, out_w, out_h, hdr_hlg=hdr_hlg)
            if png:
                _add_overlay(png, px, py, intervals)

    if overlay_str:
        video_map = cur

    # vt segments end with ';', vc has no trailing ';', overlay_str (if any) ends with ';'
    fc = vt + vc + ";" + overlay_str + at + ac

    # map an ORIGINAL-timeline instant onto the EDITED output timeline (cuts removed)
    def _edited_time(t):
        out = 0.0
        for (s, e) in buffered:
            if t >= e: out += (e - s)
            elif t > s: out += (t - s); break
            else: break
        return out

    # ── Main (video/camera or external-mic) audio volume ──
    main_audio = "[aout]"
    vv = max(0.0, min(4.0, float(video_volume)))
    if abs(vv - 1.0) > 1e-3:
        fc += f";[aout]volume={vv:.3f}[aoutv]"
        main_audio = "[aoutv]"

    # ── Per-segment mute of the main audio (does NOT touch the sfx layers) ──
    if muted_segs:
        exprs = []
        for m in muted_segs:
            try: s, e = float(m[0]), float(m[1])
            except (TypeError, ValueError, IndexError): continue
            es, ee = _edited_time(s), _edited_time(e)
            if ee > es + 0.001:
                exprs.append(f"volume=enable='between(t,{es:.3f},{ee:.3f})':volume=0")
        if exprs:
            fc += f";{main_audio}" + ",".join(exprs) + "[aoutm]"
            main_audio = "[aoutm]"

    # ── Audio layers (sound effects / music) mixed over the main audio ──
    audio_map = main_audio
    if audio:
        mix_parts = []
        labels = [main_audio]
        for k, a in enumerate(audio):
            src = SOUND_LIB / str(a.get("src", ""))
            if not src.exists():
                continue
            est = _edited_time(float(a.get("start", 0.0)))
            vol = max(0.0, min(3.0, float(a.get("volume", 1.0))))
            off = max(0.0, float(a.get("off", 0.0)))          # source offset (from splits)
            dur = max(0.05, float(a.get("dur", 0.0)))          # clip length
            dms = int(est * 1000)
            inputs += ["-i", str(src)]
            lbl = f"[sfx{k}]"
            # Trim the source to this clip's [off, off+dur] window, then delay to its edited start.
            mix_parts.append(
                f"[{next_input}:a]aformat=sample_rates=48000:channel_layouts=stereo,"
                f"atrim=start={off:.3f}:duration={dur:.3f},asetpts=PTS-STARTPTS,"
                f"volume={vol:.3f},adelay={dms}|{dms}{lbl};")
            labels.append(lbl)
            next_input += 1
        if len(labels) > 1:
            mix = "".join(mix_parts) + "".join(labels) + \
                  f"amix=inputs={len(labels)}:normalize=0:duration=first[afinal]"
            fc = fc + ";" + mix
            audio_map = "[afinal]"

    # Keep the HLG/Rec.2020 tags on the output so players interpret it (and the now
    # HLG-encoded captions) as HDR — matching the source.
    color_tags = (["-colorspace", "bt2020nc", "-color_primaries", "bt2020",
                   "-color_trc", "arib-std-b67"] if hdr_hlg else [])
    cmd = [FFMPEG, "-y", *inputs,
           "-filter_complex", fc,
           "-map", video_map, "-map", audio_map,
           "-c:v", "libx264", "-crf", "15", "-preset", "fast",
           "-pix_fmt", "yuv420p", "-fps_mode", "cfr",
           *color_tags,
           "-c:a", "aac", "-b:a", "256k",
           "-movflags", "+faststart",
           raw]

    try:
        total_dur = sum(e - s for s, e in buffered)   # output length → drives % + ETA
        returncode, stderr = _ffmpeg_with_progress(cmd, total_dur, f"Encoding {n} segments")
        if returncode != 0:
            print(stderr[-4000:])
            raise RuntimeError(f"FFmpeg failed:\n{stderr[-2000:]}")
    finally:
        for p in caption_pngs:
            try: os.unlink(p)
            except Exception: pass

    # Lock audio length to video length so multi-segment concat can't drift A/V
    ENCODE_STATUS["message"] = "Finalizing (locking audio/video)…"
    av_lock(raw, out)
    print(f"  Saved → {out}")


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/")
def index():
    # Never cache the UI — so every refresh (browser OR desktop app) loads the latest code.
    resp = Response(HTML_PAGE, mimetype="text/html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"]  = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/video")
def video():
    # Serve the smooth-preview proxy when it's ready (export still uses the original).
    if PROXY_READY and PROXY_PATH and Path(PROXY_PATH).exists():
        path = str(PROXY_PATH)
    else:
        path = str(VIDEO_PATH)
    size = os.path.getsize(path)
    range_header = request.headers.get("Range", None)

    if range_header:
        byte_range = range_header.strip().replace("bytes=", "")
        parts = byte_range.split("-")
        start = int(parts[0])
        end = int(parts[1]) if parts[1] else size - 1
        length = end - start + 1

        with open(path, "rb") as f:
            f.seek(start)
            data = f.read(length)

        resp = Response(data, 206, mimetype="video/mp4",
                        direct_passthrough=True)
        resp.headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        resp.headers["Accept-Ranges"] = "bytes"
        resp.headers["Content-Length"] = length
        return resp

    return send_file(path, mimetype="video/mp4", conditional=True)


@app.route("/ext-audio")
def ext_audio_stream():
    """Serve the external audio file (same range-request support as /video)."""
    if not EXTERNAL_AUDIO_PATH or not EXTERNAL_AUDIO_PATH.exists():
        return "", 404
    path = str(EXTERNAL_AUDIO_PATH)
    size = os.path.getsize(path)
    ext  = EXTERNAL_AUDIO_PATH.suffix.lower()
    mime = {"mp3": "audio/mpeg", "wav": "audio/wav", "m4a": "audio/mp4",
            ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
            ".aac": "audio/aac", ".flac": "audio/flac"}.get(ext, "audio/mpeg")
    range_header = request.headers.get("Range")
    if range_header:
        parts = range_header.strip().replace("bytes=", "").split("-")
        start = int(parts[0])
        end   = int(parts[1]) if parts[1] else size - 1
        length = end - start + 1
        with open(path, "rb") as f:
            f.seek(start); data = f.read(length)
        resp = Response(data, 206, mimetype=mime, direct_passthrough=True)
        resp.headers["Content-Range"]  = f"bytes {start}-{end}/{size}"
        resp.headers["Accept-Ranges"]  = "bytes"
        resp.headers["Content-Length"] = length
        return resp
    return send_file(path, mimetype=mime, conditional=True)


@app.route("/thumbs/<name>")
def thumb(name):
    p = THUMBS_DIR / name
    if p.exists():
        return send_file(str(p), mimetype="image/jpeg")
    return "", 404


@app.route("/api/state")
def api_state():
    if VIDEO_PATH is None:
        # No video loaded yet — return empty state so the client can show upload UI
        return jsonify({"no_video": True, "duration": 0, "cuts": [], "waveform": [],
                        "thumb_interval": 5, "thumb_count": 0, "split_points": [],
                        "words": [], "undo_stack": []}), 200
    return jsonify({
        "duration":           DURATION,
        "cuts":               CUTS,
        "waveform":           WAVEFORM,
        "thumb_interval":     THUMB_INTERVAL,
        "thumb_count":        THUMB_COUNT,
        "split_points":       SPLIT_POINTS,
        "words":              WORDS,
        "undo_stack":         UNDO_STACK_CACHE,
        "has_external_audio": EXTERNAL_AUDIO_PATH is not None,
        "ext_audio_offset":   EXTERNAL_AUDIO_OFFSET,
        "project_id":         CURRENT_PROJECT_ID,
        "project_name":       project_display_name(CURRENT_PROJECT_ID) if CURRENT_PROJECT_ID else None,
        "video_w":            VIDEO_W,
        "video_h":            VIDEO_H,
        "crop":               CROP,
        "captions":           CAPTIONS,
        "images":             IMAGE_OVERLAYS,
        "adjust":             ADJUST,
        "audio":              AUDIO_LAYERS,
        "video_volume":       VIDEO_VOL,
        "muted_segs":         MUTED_SEGS,
        "proxy_ready":        bool(PROXY_READY),
    })


@app.route("/api/proxy-status")
def api_proxy_status():
    return jsonify({"ready": bool(PROXY_READY and PROXY_PATH and Path(PROXY_PATH).exists())})


@app.route("/api/render-edit-proxy", methods=["POST"])
def api_render_edit_proxy():
    """Kick off (or confirm) a rendered smooth-preview of the CURRENT edit."""
    global EDIT_PROXY_STATUS
    keeps = compute_keeps()
    sig   = keeps_signature(keeps)
    out   = edit_proxy_path(CURRENT_PROJECT_ID or "p", sig)
    if EDIT_PROXY_STATUS.get("sig") == sig and EDIT_PROXY_STATUS.get("state") == "ready" and out.exists():
        return jsonify({"state": "ready", "sig": sig})
    if EDIT_PROXY_STATUS.get("state") == "rendering" and EDIT_PROXY_STATUS.get("sig") == sig:
        return jsonify({"state": "rendering", "sig": sig})
    EDIT_PROXY_STATUS = {"state": "rendering", "sig": sig}
    threading.Thread(target=_render_edit_proxy_bg,
                     args=(keeps, sig, CURRENT_PROJECT_ID), daemon=True).start()
    return jsonify({"state": "rendering", "sig": sig})


@app.route("/api/edit-proxy-status")
def api_edit_proxy_status():
    """Report render state plus the CURRENT edit signature so the client can tell
    if the rendered file is fresh (matches the current cuts) or stale."""
    cur_sig = keeps_signature(compute_keeps())
    # A versioned proxy is written atomically, so if the file for the CURRENT edit
    # exists on disk it's complete and fresh — even after a server restart or reload
    # (when EDIT_PROXY_STATUS has reset to idle). This lets the page re-enter smooth
    # mode instantly without re-encoding.
    out = edit_proxy_path(CURRENT_PROJECT_ID, cur_sig) if CURRENT_PROJECT_ID else None
    if out and out.exists():
        return jsonify({"state": "ready", "sig": cur_sig, "cur_sig": cur_sig, "fresh": True})
    return jsonify({
        "state":    EDIT_PROXY_STATUS.get("state", "idle"),
        "sig":      EDIT_PROXY_STATUS.get("sig"),
        "cur_sig":  cur_sig,
        "fresh":    False,
    })


@app.route("/api/align-data")
def api_align_data():
    """Waveform envelopes of the camera audio vs. the external audio, on a common
    time axis, for the manual drag-to-align tool."""
    if not (VIDEO_PATH and EXTERNAL_AUDIO_PATH and Path(EXTERNAL_AUDIO_PATH).exists()):
        return jsonify({"ok": False, "error": "no external audio loaded"})
    sps = 50
    window = 120.0
    vid = peak_envelope(VIDEO_PATH, window, sps)
    ext = peak_envelope(EXTERNAL_AUDIO_PATH, window, sps)
    return jsonify({"ok": True, "sps": sps, "window": window,
                    "video": vid, "ext": ext,
                    "offset": EXTERNAL_AUDIO_OFFSET,
                    "ext_duration": get_duration(EXTERNAL_AUDIO_PATH),
                    "video_duration": DURATION})


@app.route("/api/set-ext-offset", methods=["POST"])
def api_set_ext_offset():
    """Manually set the external-audio offset (from the drag-to-align tool),
    persist it, and re-render the smooth preview at the new alignment."""
    global EXTERNAL_AUDIO_OFFSET, EDIT_PROXY_STATUS
    try:
        off = float((request.get_json(force=True) or {}).get("offset"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "bad offset"}), 400
    EXTERNAL_AUDIO_OFFSET = off
    try:
        save_transcript(ORIGINAL_STEM)   # so resume reuses the manual offset
    except Exception as e:
        print(f"  (set-ext-offset) transcript cache update failed: {e}")
    save_config()
    print(f"  Manual ext-audio offset set: {off:+.4f}s — re-rendering smooth preview")
    keeps = compute_keeps()
    sig = keeps_signature(keeps)
    EDIT_PROXY_STATUS = {"state": "rendering", "sig": sig}
    threading.Thread(target=_render_edit_proxy_bg,
                     args=(keeps, sig, CURRENT_PROJECT_ID), daemon=True).start()
    return jsonify({"ok": True, "offset": off, "sig": sig})


@app.route("/api/edit-proxy")
def api_edit_proxy():
    """Serve the rendered edited-preview file (versioned by ?sig=, range requests)."""
    if not CURRENT_PROJECT_ID:
        return "", 404
    sig = request.args.get("sig") or EDIT_PROXY_STATUS.get("sig")
    if not sig:
        return "", 404
    path = edit_proxy_path(CURRENT_PROJECT_ID, sig)
    if not path.exists():
        return "", 404
    path = str(path)
    size = os.path.getsize(path)
    rng = request.headers.get("Range")
    if rng:
        parts = rng.strip().replace("bytes=", "").split("-")
        start = int(parts[0]); end = int(parts[1]) if parts[1] else size - 1
        length = end - start + 1
        with open(path, "rb") as f:
            f.seek(start); data = f.read(length)
        resp = Response(data, 206, mimetype="video/mp4", direct_passthrough=True)
        resp.headers["Content-Range"]  = f"bytes {start}-{end}/{size}"
        resp.headers["Accept-Ranges"]  = "bytes"
        resp.headers["Content-Length"] = length
        return resp
    return send_file(path, mimetype="video/mp4", conditional=True)


@app.route("/api/set-crop", methods=["POST"])
def api_set_crop():
    """Save the crop rectangle (source pixels, 16:9). Pass null to clear (no crop)."""
    global CROP
    body = request.json or {}
    c = body.get("crop")
    if c is None:
        CROP = None
    else:
        # Clamp to source bounds, keep ints
        x = max(0, min(int(c["x"]), VIDEO_W - 1))
        y = max(0, min(int(c["y"]), VIDEO_H - 1))
        w = max(16, min(int(c["w"]), VIDEO_W - x))
        h = max(9,  min(int(c["h"]), VIDEO_H - y))
        CROP = {"x": x, "y": y, "w": w, "h": h}
    save_edits()
    return jsonify({"ok": True, "crop": CROP})


@app.route("/api/set-adjust", methods=["POST"])
def api_set_adjust():
    """Save colour adjust (brightness/contrast/saturation)."""
    global ADJUST
    b = request.json or {}
    try:
        ADJUST = {
            "brightness": max(-0.5, min(0.5, float(b.get("brightness", 0.0)))),
            "contrast":   max(0.3, min(2.0, float(b.get("contrast", 1.0)))),
            "saturation": max(0.0, min(2.0, float(b.get("saturation", 1.0)))),
        }
    except (TypeError, ValueError):
        return jsonify({"ok": False}), 400
    save_edits()
    return jsonify({"ok": True, "adjust": ADJUST})


@app.route("/api/video-volume", methods=["POST"])
def api_video_volume():
    """Save the main (camera/external) audio volume multiplier (iMovie-style, 0..4)."""
    global VIDEO_VOL
    b = request.json or {}
    try:
        VIDEO_VOL = max(0.0, min(4.0, float(b.get("volume", 1.0))))
    except (TypeError, ValueError):
        return jsonify({"ok": False}), 400
    save_edits()
    return jsonify({"ok": True, "video_volume": VIDEO_VOL})


@app.route("/api/muted-segs", methods=["POST"])
def api_muted_segs():
    """Save the ORIGINAL-time ranges where the main (video) audio is muted."""
    global MUTED_SEGS
    b = request.json or {}
    clean = []
    for m in b.get("segs", []):
        try:
            s, e = float(m[0]), float(m[1])
            if e > s: clean.append([s, e])
        except (TypeError, ValueError, IndexError):
            continue
    MUTED_SEGS = clean
    save_edits()
    return jsonify({"ok": True, "muted_segs": MUTED_SEGS})


@app.route("/api/captions", methods=["POST"])
def api_captions():
    """Replace the full caption list. The editor sends the whole array on any change
    (add / edit / move / retime / delete) — captions overlap freely and are few, so a
    wholesale replace is simplest and race-free."""
    global CAPTIONS
    body = request.json or {}
    caps = body.get("captions", [])
    clean = []
    for c in caps:
        try:
            entry = {
                "id":     str(c.get("id") or f"cap_{len(clean)}"),
                "text":   str(c.get("text", "")),
                "x":      float(c.get("x", 0.1)),
                "y":      float(c.get("y", 0.1)),
                "size":   float(c.get("size", 0.06)),
                "font":   str(c.get("font", "Avenir")),
                "color":  str(c.get("color", "#ffffff")),
                "bold":   bool(c.get("bold", False)),
                "italic": bool(c.get("italic", False)),
                "start":  float(c.get("start", 0.0)),
                "end":    float(c.get("end", 0.0)),
            }
            if c.get("w"):                      # optional fixed box width (fraction) → wraps text
                entry["w"] = float(c["w"])
            if c.get("ls") is not None:          # letter spacing (fraction of frame height)
                entry["ls"] = float(c["ls"])
            if c.get("lh") is not None:          # line spacing (multiple of line height)
                entry["lh"] = float(c["lh"])
            if c.get("stroke") is not None:      # outline thickness (fraction of font size)
                entry["stroke"] = float(c["stroke"])
            if c.get("fx") and str(c["fx"]) != "none":   # text effect: typewriter / fade / zoom
                entry["fx"] = str(c["fx"])
                entry["fxSpeed"] = max(0.1, min(1.0, float(c.get("fxSpeed", 0.5))))
                entry["fxMode"] = str(c.get("fxMode", "in"))
            for k in ("x2", "y2", "size2"):      # optional end keyframe → animated transform
                if c.get(k) is not None:
                    entry[k] = float(c[k])
            clean.append(entry)
        except (TypeError, ValueError):
            continue
    # Safeguard: if this replace would wipe existing captions down to none, snapshot the
    # current edits file first so the captions can always be recovered.
    if CAPTIONS and not clean and STATE_FILE and Path(STATE_FILE).exists():
        try:
            import shutil, time
            bak = f"{STATE_FILE}.captions_bak_{int(time.time())}"
            shutil.copy2(STATE_FILE, bak)
            print(f"  Captions cleared ({len(CAPTIONS)}→0) — backed up to {Path(bak).name}")
        except Exception as e:
            print(f"  caption backup failed: {e}")
    CAPTIONS = clean
    save_edits()
    return jsonify({"ok": True, "count": len(CAPTIONS)})


def _project_overlay_dir():
    if not CURRENT_PROJECT_ID:
        return None
    d = Path.home() / ".cache" / "video_editor" / "uploads" / CURRENT_PROJECT_ID / "overlays"
    d.mkdir(parents=True, exist_ok=True)
    return d


@app.route("/api/upload-overlay-image", methods=["POST"])
def api_upload_overlay_image():
    """Save an uploaded image into the project's overlays/ dir (converted to PNG).
    Returns the stored filename to reference from an image overlay."""
    d = _project_overlay_dir()
    if d is None:
        return jsonify({"ok": False, "error": "no project"}), 400
    f = request.files.get("image")
    if not f:
        return jsonify({"ok": False, "error": "no file"}), 400
    try:
        from PIL import Image
        import io, time as _t
        im = Image.open(io.BytesIO(f.read())).convert("RGBA")
        name = f"ov_{int(_t.time()*1000)}.png"
        im.save(str(d / name))
        return jsonify({"ok": True, "src": name, "w": im.width, "h": im.height})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/overlay/<name>")
def serve_overlay(name):
    """Serve an overlay image for the preview."""
    d = _project_overlay_dir()
    if d is None:
        return "", 404
    p = (d / name)
    if p.exists() and p.parent == d:
        return send_file(str(p), mimetype="image/png")
    return "", 404


@app.route("/api/images", methods=["POST"])
def api_images():
    """Replace the full image-overlay list (whole array on any change, like captions)."""
    global IMAGE_OVERLAYS
    body = request.json or {}
    clean = []
    for c in body.get("images", []):
        try:
            e = {
                "id":      str(c.get("id") or f"img_{len(clean)}"),
                "src":     str(c.get("src", "")),
                "x":       float(c.get("x", 0.25)),
                "y":       float(c.get("y", 0.1)),
                "w":       float(c.get("w", 0.4)),
                "start":   float(c.get("start", 0.0)),
                "end":     float(c.get("end", 0.0)),
                "opacity": float(c.get("opacity", 1.0)),
            }
            if not e["src"]:
                continue
            for k in ("x2", "y2", "w2"):          # optional end keyframe (Ken Burns)
                if c.get(k) is not None:
                    e[k] = float(c[k])
            clean.append(e)
        except (TypeError, ValueError):
            continue
    IMAGE_OVERLAYS = clean
    save_edits()
    return jsonify({"ok": True, "count": len(IMAGE_OVERLAYS)})


# ── Sound effect library (shared across ALL projects) + audio layers ──
def _safe_sound_name(name):
    import re as _re
    base = _re.sub(r"[^A-Za-z0-9._ -]", "_", Path(name).name).strip() or "sound.mp3"
    if "." not in base:
        base += ".mp3"
    return base


@app.route("/api/upload-sound", methods=["POST"])
def api_upload_sound():
    """Save an uploaded audio file into the shared sound-effect library (kept forever,
    reusable in any project). If the name already exists with identical size, reuse it."""
    SOUND_LIB.mkdir(parents=True, exist_ok=True)
    f = request.files.get("audio")
    if not f:
        return jsonify({"ok": False, "error": "no file"}), 400
    data = f.read()
    name = _safe_sound_name(f.filename or "sound.mp3")
    dest = SOUND_LIB / name
    if dest.exists() and dest.stat().st_size == len(data):
        pass  # identical file already in the library — reuse
    else:
        if dest.exists():   # same name, different file → keep both
            stem, ext = dest.stem, dest.suffix
            i = 2
            while (SOUND_LIB / f"{stem}_{i}{ext}").exists():
                i += 1
            name = f"{stem}_{i}{ext}"; dest = SOUND_LIB / name
        dest.write_bytes(data)
    dur = get_duration(dest) or 0.0
    return jsonify({"ok": True, "src": name, "dur": round(dur, 3)})


@app.route("/api/sound-library")
def api_sound_library():
    """List every sound in the shared library (name + duration), newest first."""
    SOUND_LIB.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(SOUND_LIB.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if p.is_file() and p.suffix.lower() in (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"):
            out.append({"src": p.name, "dur": round(get_duration(p) or 0.0, 3)})
    return jsonify({"ok": True, "sounds": out})


@app.route("/sound/<name>")
def serve_sound(name):
    p = SOUND_LIB / Path(name).name
    if p.exists() and p.parent == SOUND_LIB:
        ext = p.suffix.lower()
        mime = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
                ".aac": "audio/aac", ".ogg": "audio/ogg", ".flac": "audio/flac"}.get(ext, "audio/mpeg")
        return send_file(str(p), mimetype=mime, conditional=True)
    return "", 404


# ── Sound trim/selection screen: stage an upload, then save a trimmed clip to the library ──
SOUND_TMP = Path.home() / ".cache" / "video_editor" / "_soundtmp"

@app.route("/api/upload-sound-temp", methods=["POST"])
def api_upload_sound_temp():
    """Stage an uploaded audio file for trimming (NOT yet in the library). Returns a temp id
    + duration + waveform peaks so the trim screen can render + scrub it."""
    import uuid, time as _time
    SOUND_TMP.mkdir(parents=True, exist_ok=True)
    # Housekeeping: drop any staged temps older than an hour (abandoned trim screens).
    try:
        for _p in SOUND_TMP.iterdir():
            if _p.is_file() and _time.time() - _p.stat().st_mtime > 3600:
                _p.unlink()
    except Exception:
        pass
    f = request.files.get("audio")
    if not f:
        return jsonify({"ok": False, "error": "no file"}), 400
    ext = (Path(f.filename or "sound.mp3").suffix or ".mp3").lower()
    tid = uuid.uuid4().hex + ext
    dest = SOUND_TMP / tid
    f.save(str(dest))
    dur = get_duration(dest) or 0.0
    peaks = _sound_peaks(dest)
    base = _safe_sound_name(f.filename or "sound.mp3")
    return jsonify({"ok": True, "temp_id": tid, "name": Path(base).stem,
                    "dur": round(dur, 3), "peaks": peaks})


@app.route("/sound-temp/<tid>")
def serve_sound_temp(tid):
    p = SOUND_TMP / Path(tid).name
    if p.exists() and p.parent == SOUND_TMP:
        return send_file(str(p), mimetype="audio/mpeg", conditional=True)
    return "", 404


@app.route("/api/save-trimmed-sound", methods=["POST"])
def api_save_trimmed_sound():
    """Trim a staged temp sound to [start,end] and save it into the shared library as <name>.mp3.
    Returns {src, dur} for the new library entry."""
    SOUND_LIB.mkdir(parents=True, exist_ok=True)
    b = request.json or {}
    src = SOUND_TMP / Path(str(b.get("temp_id", ""))).name
    if not src.exists() or src.parent != SOUND_TMP:
        return jsonify({"ok": False, "error": "temp not found"}), 404
    try:
        start = max(0.0, float(b.get("start", 0.0)))
        end = float(b.get("end", 0.0))
    except (TypeError, ValueError):
        return jsonify({"ok": False}), 400
    dur = max(0.05, end - start)
    # Edit-in-place: overwrite an existing library entry (re-trim). Else create a new unique name.
    overwrite = b.get("overwrite")
    if overwrite:
        name = Path(_safe_sound_name(overwrite)).name
        dest = SOUND_LIB / name
        if not (dest.exists() and dest.parent == SOUND_LIB):
            return jsonify({"ok": False, "error": "entry to overwrite not found"}), 404
    else:
        name = _safe_sound_name(b.get("name") or "sound")
        name = Path(name).stem + ".mp3"      # library entries are always .mp3
        dest = SOUND_LIB / name
        if dest.exists():
            stem = dest.stem; i = 2
            while (SOUND_LIB / f"{stem}_{i}.mp3").exists():
                i += 1
            name = f"{stem}_{i}.mp3"; dest = SOUND_LIB / name
    out_path = SOUND_LIB / (Path(name).stem + ".__editing.mp3") if overwrite else dest
    r = subprocess.run([FFMPEG, "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
                        "-i", str(src), "-c:a", "libmp3lame", "-q:a", "4", str(out_path)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not out_path.exists():
        return jsonify({"ok": False, "error": "trim failed: " + (r.stderr or "")[-200:]}), 500
    if overwrite:
        os.replace(str(out_path), str(dest))          # atomic in-place replace
        for pk in SOUND_LIB.glob(name + ".peaks*.json"):   # invalidate cached waveform
            try: pk.unlink()
            except Exception: pass
    try:
        src.unlink()   # staged temp no longer needed
    except Exception:
        pass
    outdur = get_duration(dest) or dur
    return jsonify({"ok": True, "src": name, "dur": round(outdur, 3)})


def _sound_peaks(path, n=None):
    """Peak-amplitude envelope of a sound file (0..1) for drawing its waveform.
    Resolution scales with duration (~220 pts/sec, capped) so short SFX and long
    music are both drawn precisely. Sampled at 16 kHz mono for accurate peaks."""
    try:
        ar = 16000
        r = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-vn", "-ac", "1",
                            "-ar", str(ar), "-f", "f32le", "pipe:1"], capture_output=True)
        total = len(r.stdout) // 4
        if total == 0:
            return []
        fl = struct.unpack(f"{total}f", r.stdout[: total * 4])
        if n is None:
            dur = total / ar
            n = int(min(8000, max(600, dur * 220)))
        bucket = max(1, total // n)
        peaks = [max((abs(x) for x in fl[i:i + bucket]), default=0.0) for i in range(0, total, bucket)]
        mx = max(peaks, default=1.0) or 1.0
        return [round(v / mx, 3) for v in peaks]
    except Exception:
        return []


@app.route("/api/sound-waveform")
def api_sound_waveform():
    """Return the waveform peaks for a library sound (cached next to the file)."""
    p = SOUND_LIB / Path(request.args.get("src", "")).name
    if not p.exists() or p.parent != SOUND_LIB:
        return jsonify({"ok": False}), 404
    cache = p.with_suffix(p.suffix + ".peaks2.json")   # v2 = higher-resolution envelope
    try:
        if cache.exists() and cache.stat().st_mtime >= p.stat().st_mtime:
            return jsonify({"ok": True, "peaks": json.loads(cache.read_text())})
    except Exception:
        pass
    peaks = _sound_peaks(p)
    try:
        cache.write_text(json.dumps(peaks))
    except Exception:
        pass
    return jsonify({"ok": True, "peaks": peaks})


@app.route("/api/audio-layers", methods=["POST"])
def api_audio_layers():
    """Replace the audio-layer list for the current project."""
    global AUDIO_LAYERS
    body = request.json or {}
    clean = []
    for c in body.get("audio", []):
        try:
            e = {
                "id":     str(c.get("id") or f"aud_{len(clean)}"),
                "src":    str(c.get("src", "")),
                "start":  float(c.get("start", 0.0)),
                "dur":    float(c.get("dur", 0.0)),
                "volume": max(0.0, min(3.0, float(c.get("volume", 1.0)))),
            }
            if e["src"]:
                clean.append(e)
        except (TypeError, ValueError):
            continue
    AUDIO_LAYERS = clean
    save_edits()
    return jsonify({"ok": True, "count": len(AUDIO_LAYERS)})


@app.route("/api/undo-stack", methods=["POST"])
def api_undo_stack():
    global UNDO_STACK_CACHE
    UNDO_STACK_CACHE = request.json.get("stack", [])
    save_edits()
    return jsonify({"ok": True})


@app.route("/api/cuts", methods=["POST"])
def api_cuts():
    body = request.json
    cut_id, active = body["id"], body["active"]
    for c in CUTS:
        if c["id"] == cut_id:
            c["active"] = active
            break
    save_edits()
    return jsonify({"ok": True})


@app.route("/api/split", methods=["POST"])
def api_split():
    global SPLIT_POINTS
    body = request.json
    SPLIT_POINTS.append({"id": body["id"], "pos": body["pos"]})
    save_edits()
    return jsonify({"ok": True})


@app.route("/api/delete-segment", methods=["POST"])
def api_delete_segment():
    body = request.json
    # Skip if an identical active cut already exists — stops runaway duplication
    if not cut_exists(body["start"], body["end"]):
        CUTS.append({
            "id":     body["id"],
            "start":  body["start"],
            "end":    body["end"],
            "label":  body.get("label", "manual delete"),
            "type":   "manual",
            "active": True,
        })
        save_edits()
    return jsonify({"ok": True})


@app.route("/api/remove-cut", methods=["POST"])
def api_remove_cut():
    body = request.json
    cut_id = body["id"]
    for i, c in enumerate(CUTS):
        if c["id"] == cut_id:
            CUTS.pop(i)
            break
    save_edits()
    return jsonify({"ok": True})


@app.route("/api/remove-split", methods=["POST"])
def api_remove_split():
    global SPLIT_POINTS
    body = request.json
    SPLIT_POINTS = [s for s in SPLIT_POINTS if s["id"] != body["id"]]
    save_edits()
    return jsonify({"ok": True})


@app.route("/api/replace-state", methods=["POST"])
def api_replace_state():
    """Wholesale-replace the cut + split state. Used by snapshot-based undo so a
    single undo restores the EXACT prior edit state (no fragile per-op inversion)."""
    global CUTS, SPLIT_POINTS
    body = request.json or {}
    CUTS = list(body.get("cuts", []))
    SPLIT_POINTS = list(body.get("split_points", []))
    save_edits()
    return jsonify({"ok": True, "cuts": len(CUTS), "splits": len(SPLIT_POINTS)})


def default_export_dir() -> Path:
    """Where to default the export folder, in priority order:
    1. This project's own remembered export dir
    2. The global last-used export dir
    3. ~/Downloads/yt vids
    (Browsers don't expose the upload's source folder, so we remember per project.)"""
    if CURRENT_PROJECT_ID:
        pd = read_project_meta(CURRENT_PROJECT_ID).get("export_dir")
        if pd and Path(pd).exists():
            return Path(pd)
    if LAST_EXPORT_DIR and Path(LAST_EXPORT_DIR).exists():
        return Path(LAST_EXPORT_DIR)
    return Path.home() / "Downloads" / "yt vids"


@app.route("/api/suggest-filename")
def api_suggest_filename():
    stem = ORIGINAL_STEM or VIDEO_PATH.stem
    stem = stem.replace("_combined", "").replace(" ", "_")
    return jsonify({"path": str(default_export_dir() / f"{stem}_edited.mp4")})


@app.route("/api/pick-save-path")
def api_pick_save_path():
    """Open a native macOS save dialog and return the chosen path."""
    global LAST_EXPORT_DIR
    suggested = request.args.get("name", "edited.mp4")
    out_dir = default_export_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    script = f'''
        set defaultFolder to POSIX file "{out_dir}" as alias
        set f to choose file name with prompt "Save edited video as:" ¬
            default name "{suggested}" ¬
            default location defaultFolder
        return POSIX path of f
    '''
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=120
        )
        path = result.stdout.strip()
        if path:
            if not path.lower().endswith(".mp4"):
                path += ".mp4"
            LAST_EXPORT_DIR = str(Path(path).parent)
            if CURRENT_PROJECT_ID:
                write_project_meta(CURRENT_PROJECT_ID, export_dir=LAST_EXPORT_DIR)
            save_config()
            return jsonify({"path": path})
    except Exception:
        pass
    return jsonify({"path": None})


def snap_keeps_to_words(keeps, words, merged_cuts, eps=0.02):
    """Prevent partial-word clipping in the export.

    A transcript word counts as 'kept' unless a cut FULLY contains it. But raw
    cut times can land mid-word, so a kept word's audio gets clipped at export
    even though the preview/transcript show it intact. This nudges each keep
    boundary out to the whole-word edge whenever a kept word straddles it.
    """
    if not words:
        return keeps

    def is_struck(w):
        return any(cs - 0.01 <= w["start"] and w["end"] <= ce + 0.01 for cs, ce in merged_cuts)

    snapped = []
    for ks, ke in keeps:
        for w in words:
            if is_struck(w):
                continue
            ws, we = w["start"], w["end"]
            if ws < ks - eps and we > ks + eps:   # word straddles keep START
                ks = min(ks, ws)
            if ws < ke - eps and we > ke + eps:   # word straddles keep END
                ke = max(ke, we)
        snapped.append([ks, ke])

    # Re-merge any keeps that now touch/overlap after widening
    snapped.sort()
    out = [snapped[0]]
    for s, e in snapped[1:]:
        if s <= out[-1][1] + eps:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


@app.route("/api/encode", methods=["POST"])
def api_encode():
    global ENCODE_STATUS, ENCODE_CANCEL
    ENCODE_CANCEL = False
    ENCODE_STATUS = {"state": "running", "message": "Starting encode..."}

    body = request.json or {}
    custom_path = body.get("out_path")

    def run():
        global ENCODE_STATUS
        out = None
        try:
            # Use the EXACT same merge logic the preview uses (getMergedCuts, gap 0.15)
            # so the export matches the preview frame-for-frame — no buffers, no snapping.
            active = sorted((c["start"], c["end"]) for c in CUTS if c["active"])
            merged = []
            for s, e in active:
                if merged and s <= merged[-1][1] + 0.15:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            merged = [(s, e) for s, e in merged]
            keeps  = cuts_to_keeps(merged, DURATION)

            if custom_path:
                out = Path(custom_path).expanduser().resolve()
                if out.suffix.lower() not in (".mp4", ".mov", ".mkv"):
                    out = out.with_suffix(".mp4")
            else:
                stem = ORIGINAL_STEM or VIDEO_PATH.stem
                stem = stem.replace("_combined", "").replace(" ", "_")
                out = Path.home() / "Downloads" / "yt vids" / f"{stem}_edited.mp4"

            out.parent.mkdir(parents=True, exist_ok=True)
            ENCODE_STATUS["message"] = f"Encoding {len(keeps)} segments → {out.name}..."
            encode_hq(VIDEO_PATH, keeps, out,
                      ext_audio=EXTERNAL_AUDIO_PATH,
                      ext_offset=EXTERNAL_AUDIO_OFFSET,
                      crop=CROP, captions=CAPTIONS, images=IMAGE_OVERLAYS, adjust=ADJUST,
                      audio=AUDIO_LAYERS, video_volume=VIDEO_VOL, muted_segs=MUTED_SEGS)
            save_learning_snapshot()
            # Remember the export folder — globally and for THIS project specifically
            global LAST_EXPORT_DIR
            LAST_EXPORT_DIR = str(out.parent)
            if CURRENT_PROJECT_ID:
                write_project_meta(CURRENT_PROJECT_ID, export_dir=LAST_EXPORT_DIR)
            save_config()
            ENCODE_STATUS = {"state": "done", "message": str(out)}
        except Exception as e:
            if ENCODE_CANCEL:
                ENCODE_STATUS = {"state": "cancelled", "message": "Export cancelled"}
            else:
                import traceback
                ENCODE_STATUS = {"state": "error", "message": str(e) + "\n" + traceback.format_exc()}
        finally:
            # Clean up the partial temp file from a cancelled encode (av_lock never ran,
            # so the final `out` doesn't exist yet — only the .raw.mp4 does).
            if ENCODE_CANCEL and out is not None:
                try:
                    raw = str(out) + ".raw.mp4"
                    if os.path.exists(raw):
                        os.unlink(raw)
                except Exception:
                    pass

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/cancel-encode", methods=["POST"])
def api_cancel_encode():
    """Stop a running export — kill the ffmpeg process; the encode thread then finishes
    with state 'cancelled' and removes any partial output."""
    global ENCODE_CANCEL
    if ENCODE_STATUS.get("state") != "running":
        return jsonify({"ok": False, "message": "no export running"})
    ENCODE_CANCEL = True
    p = ENCODE_PROC
    if p is not None:
        try: p.kill()
        except Exception: pass
    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    return jsonify(ENCODE_STATUS)


@app.route("/api/process-status")
def api_process_status():
    return jsonify(PROCESS_STATUS)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    global PROCESS_STATUS
    video_files = request.files.getlist("videos")
    audio_file  = request.files.get("ext_audio")   # optional separate audio

    if not video_files or all(f.filename == "" for f in video_files):
        return jsonify({"error": "No video files received"}), 400

    # Per-project folder keyed off the first video's filename.
    # Re-uploading the SAME file reopens that project (with its saved cuts). But generic
    # names (iMovie exports like "clip.mov", "My Movie.mp4") collide: a DIFFERENT video
    # with the same name would otherwise reopen the old project and reuse its cached proxy
    # → the preview shows the WRONG video. So if a project with this name already exists
    # but the incoming file differs (by size), give the new upload its own unique id.
    first_name  = next((f.filename for f in video_files if f.filename), "project")
    base_id     = sanitize_project_id(first_name)
    uploads_root = Path.home() / ".cache" / "video_editor" / "uploads"
    first_video = next((f for f in video_files if f.filename), None)
    incoming_size = None
    if first_video is not None:
        try:
            first_video.stream.seek(0, 2); incoming_size = first_video.stream.tell(); first_video.stream.seek(0)
        except Exception:
            incoming_size = None
    project_id = base_id
    existing_dir = uploads_root / base_id
    existing_clip = next(iter(existing_dir.glob("clip_000.*")), None) if existing_dir.exists() else None
    if existing_clip is not None and incoming_size is not None and existing_clip.stat().st_size != incoming_size:
        # Same name, different video → new distinct project (never reuse the old proxy/edits)
        n = 2
        while (uploads_root / f"{base_id}_{n}").exists():
            n += 1
        project_id = f"{base_id}_{n}"
        print(f"  Name collision '{base_id}' with a different video → new project '{project_id}'")
    upload_dir = uploads_root / project_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_videos = []
    for i, f in enumerate(video_files):
        ext = Path(f.filename).suffix or ".mov"
        dest = upload_dir / f"clip_{i:03d}{ext}"
        f.save(str(dest))
        saved_videos.append(dest)

    saved_audio = None
    if audio_file and audio_file.filename:
        ext = Path(audio_file.filename).suffix or ".wav"
        saved_audio = upload_dir / f"ext_audio{ext}"
        audio_file.save(str(saved_audio))

    # mode: "auto" = transcription + automatic cuts (default); "edit" = standard editing only
    skip_transcribe = (request.form.get("mode", "auto") == "edit")

    n = len(saved_videos)
    mode_txt = " (standard editing)" if skip_transcribe else ""
    msg = f"Received {n} video file{'s' if n>1 else ''}" + (" + external audio" if saved_audio else "") + f" → project '{project_id}'{mode_txt}…"
    PROCESS_STATUS = {"state": "processing", "message": msg}
    threading.Thread(target=reprocess_video, args=(saved_videos, saved_audio),
                     kwargs={"project_id": project_id, "skip_transcribe": skip_transcribe},
                     daemon=True).start()
    return jsonify({"ok": True})


# ── Project manager ───────────────────────────────────────────────────────────

def _project_meta_path(pid: str) -> Path:
    return Path.home() / ".cache" / "video_editor" / "uploads" / pid / "project.json"


def read_project_meta(pid: str) -> dict:
    """Read a project's metadata dict (name, export_dir, …) from project.json."""
    try:
        p = _project_meta_path(pid)
        if p.exists():
            return json.loads(p.read_text()) or {}
    except Exception:
        pass
    return {}


def write_project_meta(pid: str, **updates):
    """Merge-update a project's project.json metadata."""
    try:
        p = _project_meta_path(pid)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = read_project_meta(pid)
        meta.update({k: v for k, v in updates.items() if v is not None})
        p.write_text(json.dumps(meta))
    except Exception as e:
        print(f"  Warning: could not write project meta: {e}")


def project_display_name(pid: str) -> str:
    """A project's user-set display name (else its id)."""
    return read_project_meta(pid).get("name") or pid


def list_projects():
    """Scan the uploads dir and return all saved projects with metadata."""
    uploads = Path.home() / ".cache" / "video_editor" / "uploads"
    state_dir = Path.home() / ".cache" / "video_editor"
    projects = []
    if not uploads.exists():
        return projects
    for pdir in uploads.iterdir():
        if not pdir.is_dir():
            continue
        pid = pdir.name
        vids = sorted([p for p in pdir.glob("clip_*") if p.suffix.lower() in
                       (".mov", ".mp4", ".mkv", ".m4v", ".avi", ".webm")])
        if not vids:
            continue
        edits_path = state_dir / f"{pid}_edits.json"
        n_cuts = n_splits = 0
        if edits_path.exists():
            try:
                d = json.loads(edits_path.read_text())
                n_cuts   = len([c for c in d.get("cuts", []) if c.get("active")])
                n_splits = len(d.get("split_points", []))
            except Exception:
                pass
        has_audio = any(pdir.glob("ext_audio*"))
        has_cache = (state_dir / f"{pid}_transcript.json").exists()
        mtime = max((p.stat().st_mtime for p in [edits_path, *vids] if p.exists()), default=0)
        projects.append({
            "id": pid, "name": project_display_name(pid), "n_videos": len(vids),
            "n_cuts": n_cuts, "n_splits": n_splits,
            "has_ext_audio": has_audio, "cached": has_cache,
            "modified": mtime, "is_current": (pid == CURRENT_PROJECT_ID),
        })
    projects.sort(key=lambda p: p["modified"], reverse=True)
    return projects


@app.route("/api/projects")
def api_projects():
    return jsonify({"projects": list_projects(), "current": CURRENT_PROJECT_ID})


@app.route("/api/rename-project", methods=["POST"])
def api_rename_project():
    """Set a project's display name (stored in uploads/<id>/project.json). id unchanged."""
    body = request.json or {}
    pid  = body.get("project_id")
    name = (body.get("name") or "").strip()
    if not pid or not name:
        return jsonify({"error": "project_id and name required"}), 400
    pdir = Path.home() / ".cache" / "video_editor" / "uploads" / pid
    if not pdir.exists():
        return jsonify({"error": "Project not found"}), 404
    write_project_meta(pid, name=name)
    return jsonify({"ok": True, "name": name})


@app.route("/api/open-project", methods=["POST"])
def api_open_project():
    """Load an existing project into the running server (no restart)."""
    global PROCESS_STATUS
    pid = (request.json or {}).get("project_id")
    if not pid:
        return jsonify({"error": "No project_id"}), 400
    pdir = Path.home() / ".cache" / "video_editor" / "uploads" / pid
    if not pdir.exists():
        return jsonify({"error": "Project not found"}), 404
    vids = sorted([p for p in pdir.glob("clip_*") if p.suffix.lower() in
                   (".mov", ".mp4", ".mkv", ".m4v", ".avi", ".webm")])
    if not vids:
        return jsonify({"error": "No video in project"}), 404
    ext = next(iter(pdir.glob("ext_audio*")), None)
    PROCESS_STATUS = {"state": "processing", "message": f"Opening '{pid}'…"}
    threading.Thread(target=reprocess_video, args=(vids, ext),
                     kwargs={"is_resume": True, "project_id": pid}, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/exit-project", methods=["POST"])
def api_exit_project():
    """Unload the current project (back to the picker) WITHOUT deleting anything."""
    global VIDEO_PATH, CUTS, SPLIT_POINTS, WORDS, WAVEFORM, CURRENT_PROJECT_ID
    global EXTERNAL_AUDIO_PATH, EXTERNAL_AUDIO_OFFSET, SOURCE_VIDEOS, SOURCE_EXT_AUDIO, CAPTIONS, IMAGE_OVERLAYS, ADJUST, AUDIO_LAYERS, VIDEO_VOL, MUTED_SEGS
    save_edits()  # make sure latest edits are flushed first
    VIDEO_PATH = None
    CUTS = []; SPLIT_POINTS = []; WORDS = []; WAVEFORM = []; CAPTIONS = []; IMAGE_OVERLAYS = []
    ADJUST = {"brightness": 0.0, "contrast": 1.0, "saturation": 1.0}
    AUDIO_LAYERS = []; VIDEO_VOL = 1.0; MUTED_SEGS = []
    CURRENT_PROJECT_ID = None
    EXTERNAL_AUDIO_PATH = None; EXTERNAL_AUDIO_OFFSET = 0.0
    SOURCE_VIDEOS = []; SOURCE_EXT_AUDIO = None
    return jsonify({"ok": True})


@app.route("/api/delete-project", methods=["POST"])
def api_delete_project():
    """Permanently delete a project's files (video + edits + transcript)."""
    import shutil
    pid = (request.json or {}).get("project_id")
    if not pid or pid == CURRENT_PROJECT_ID:
        return jsonify({"error": "Cannot delete the open project"}), 400
    base = Path.home() / ".cache" / "video_editor"
    try:
        shutil.rmtree(base / "uploads" / pid, ignore_errors=True)
        (base / f"{pid}_edits.json").unlink(missing_ok=True)
        (base / f"{pid}_transcript.json").unlink(missing_ok=True)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/add-ext-audio", methods=["POST"])
def api_add_ext_audio():
    """Attach a separately-recorded audio track to the ALREADY-LOADED video —
    syncs + persists without re-uploading the video or re-transcribing. Cuts untouched."""
    global EXTERNAL_AUDIO_PATH, EXTERNAL_AUDIO_OFFSET, SOURCE_EXT_AUDIO, PROCESS_STATUS
    if VIDEO_PATH is None:
        return jsonify({"error": "No video loaded yet — upload a video first"}), 400
    audio_file = request.files.get("ext_audio")
    if not audio_file or not audio_file.filename:
        return jsonify({"error": "No audio file received"}), 400

    upload_dir = Path.home() / ".cache" / "video_editor" / "uploads" / (CURRENT_PROJECT_ID or "project")
    upload_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(audio_file.filename).suffix or ".wav"
    saved_audio = upload_dir / f"ext_audio{ext}"
    audio_file.save(str(saved_audio))

    def run():
        global EXTERNAL_AUDIO_PATH, EXTERNAL_AUDIO_OFFSET, SOURCE_EXT_AUDIO, PROCESS_STATUS, WAVEFORM
        try:
            PROCESS_STATUS = {"state": "processing", "message": "Syncing external audio…"}
            # Use the saved uploads file in place (no temp copy)
            offset = sync_external_audio(VIDEO_PATH, saved_audio)
            EXTERNAL_AUDIO_PATH   = saved_audio
            EXTERNAL_AUDIO_OFFSET = offset
            SOURCE_EXT_AUDIO      = saved_audio   # permanent path for resume
            # Refresh the cached transcript so it records the ext-audio offset
            save_transcript(ORIGINAL_STEM)
            save_config()
            print(f"  Added external audio: offset={offset:+.3f}s (cuts untouched)")
            PROCESS_STATUS = {"state": "done", "message": "External audio synced"}
        except Exception as e:
            import traceback
            PROCESS_STATUS = {"state": "error", "message": str(e) + "\n" + traceback.format_exc()}

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/remove-ext-audio", methods=["POST"])
def api_remove_ext_audio():
    """Undo external audio: revert to the camera's own audio. Clears the ext-audio state,
    re-extracts the waveform from the camera track, and rebuilds the smooth preview."""
    global EXTERNAL_AUDIO_PATH, EXTERNAL_AUDIO_OFFSET, SOURCE_EXT_AUDIO, WAVEFORM, EDIT_PROXY_STATUS
    if not EXTERNAL_AUDIO_PATH:
        return jsonify({"ok": False, "error": "no external audio attached"})
    # Rename the staged ext-audio file(s) so reopening the project doesn't re-attach them
    # (api_open_project globs "ext_audio*"). Kept on disk (prefixed) so it can be re-added.
    try:
        if CURRENT_PROJECT_ID:
            pdir = Path.home() / ".cache" / "video_editor" / "uploads" / CURRENT_PROJECT_ID
            for exf in pdir.glob("ext_audio*"):
                exf.rename(pdir / ("_disabled_" + exf.name))
    except Exception as e:
        print(f"  (remove-ext-audio) could not archive ext-audio file: {e}")
    EXTERNAL_AUDIO_PATH = None
    EXTERNAL_AUDIO_OFFSET = 0.0
    SOURCE_EXT_AUDIO = None
    try:
        save_transcript(ORIGINAL_STEM)   # cache now records has_external_audio = False
    except Exception as e:
        print(f"  (remove-ext-audio) transcript cache update failed: {e}")
    save_config()                        # drops last_ext_audio_path/offset

    def bg():
        global WAVEFORM, EDIT_PROXY_STATUS
        try:
            if VIDEO_PATH:
                WAVEFORM = extract_waveform(VIDEO_PATH)   # rebuild from camera audio
                save_transcript(ORIGINAL_STEM)
        except Exception as e:
            print(f"  (remove-ext-audio) waveform rebuild failed: {e}")
        keeps = compute_keeps(); sig = keeps_signature(keeps)
        EDIT_PROXY_STATUS = {"state": "rendering", "sig": sig}
        _render_edit_proxy_bg(keeps, sig, CURRENT_PROJECT_ID)   # rebuild preview w/ camera audio

    threading.Thread(target=bg, daemon=True).start()
    print("  Removed external audio — reverted to camera audio")
    return jsonify({"ok": True})


def reprocess_video(paths: list, ext_audio: Path | None = None, is_resume: bool = False,
                    project_id: str | None = None, skip_transcribe: bool = False):
    """Concatenate paths (if > 1), sync external audio if provided, run full pipeline.
    If is_resume and a transcript cache exists, skip Whisper + AI detection entirely.
    If skip_transcribe, load the video for STANDARD EDITING ONLY — no Whisper, no auto
    cuts, no transcript (for an already-cut video); still builds thumbnails + waveform.
    project_id keys all state (edits + transcript) so projects never collide."""
    import shutil
    global VIDEO_PATH, THUMBS_DIR, TEMP_DIR, DURATION, WAVEFORM, WORDS
    global CUTS, THUMB_INTERVAL, THUMB_COUNT, ORIGINAL_STEM, STATE_FILE
    global INITIAL_AUTO_CUTS, SPLIT_POINTS, PROCESS_STATUS, UNDO_STACK_CACHE, ENCODE_STATUS
    global EXTERNAL_AUDIO_PATH, EXTERNAL_AUDIO_OFFSET, SOURCE_VIDEOS, SOURCE_EXT_AUDIO
    global CURRENT_PROJECT_ID, RESUMING, VIDEO_W, VIDEO_H, CROP, PROXY_PATH, PROXY_READY
    global EDIT_PROXY_STATUS

    RESUMING = True   # block edit-saves until CUTS is fully rebuilt + restored
    PROXY_READY = False; PROXY_PATH = None   # reset for the project being loaded
    EDIT_PROXY_STATUS = {"state": "idle", "sig": None}

    # Remember the permanent source paths for robust auto-resume
    SOURCE_VIDEOS    = [Path(p) for p in paths]
    SOURCE_EXT_AUDIO = Path(ext_audio) if ext_audio else None
    # Derive project id from the uploads/<id>/ folder if not given (resume case)
    if project_id is None and paths:
        parent = Path(paths[0]).parent
        if parent.parent.name == "uploads":
            project_id = parent.name
    CURRENT_PROJECT_ID = project_id

    try:
        new_temp = Path(tempfile.mkdtemp(prefix="vp_"))   # small temp (audio.wav only)
        # Persistent per-project folder holds the video, thumbs, combined — survives restarts
        proj_dir = (Path.home() / ".cache" / "video_editor" / "uploads" / project_id) \
                   if project_id else new_temp
        proj_dir.mkdir(parents=True, exist_ok=True)

        if len(paths) == 1:
            # Use the uploaded file IN PLACE — no 780MB copy needed (read-only access)
            combined = Path(paths[0])
        else:
            # Concatenate once, cache the result in the project folder
            combined = proj_dir / "_combined.mp4"
            if not (is_resume and combined.exists()):
                PROCESS_STATUS = {"state": "processing", "message": "Concatenating videos…"}
                filelist = new_temp / "filelist.txt"
                with open(filelist, "w") as f:
                    for p in paths:
                        f.write(f"file '{p}'\n")
                result = subprocess.run(
                    [FFMPEG, "-y", "-f", "concat", "-safe", "0",
                     "-i", str(filelist), "-c", "copy", str(combined)],
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    PROCESS_STATUS["message"] = "Re-encoding to merge clips…"
                    subprocess.run(
                        [FFMPEG, "-y", "-f", "concat", "-safe", "0",
                         "-i", str(filelist),
                         "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                         "-c:a", "aac", "-b:a", "192k",
                         str(combined)],
                        check=True,
                    )

        # ── Reset all global state ──
        old_temp = TEMP_DIR
        VIDEO_PATH   = combined
        # State is keyed by project_id (filename-derived), NOT the temp working-copy
        # name — so each project has its own edits + transcript and never collides.
        ORIGINAL_STEM = project_id or combined.stem
        CUTS          = []
        SPLIT_POINTS  = []
        UNDO_STACK_CACHE = []
        ENCODE_STATUS = {"state": "idle", "message": "Ready"}
        EXTERNAL_AUDIO_PATH   = None
        EXTERNAL_AUDIO_OFFSET = 0.0

        # Source dimensions + default 16:9 centered crop (load_edits may override CROP)
        VIDEO_W, VIDEO_H = get_video_dims(VIDEO_PATH)
        CROP = default_crop_169(VIDEO_W, VIDEO_H)

        state_dir = Path.home() / ".cache" / "video_editor"
        state_dir.mkdir(parents=True, exist_ok=True)
        STATE_FILE = str(state_dir / f"{ORIGINAL_STEM}_edits.json")

        TEMP_DIR   = str(new_temp)
        THUMBS_DIR = proj_dir / "thumbs"   # persistent — reused across reopens

        # ── Try fast resume: skip Whisper + AI if transcript cache exists ──
        cache = load_transcript(ORIGINAL_STEM) if is_resume else None

        # ── External audio: use file in place; reuse cached offset (no re-sync) ──
        if ext_audio and Path(ext_audio).exists():
            EXTERNAL_AUDIO_PATH = Path(ext_audio)
            cached_off = cache.get("ext_audio_offset") if cache else None
            if cache and cache.get("has_external_audio") and cached_off is not None:
                EXTERNAL_AUDIO_OFFSET = cached_off   # already known — skip the slow sync
                print(f"  External audio: reusing cached offset={cached_off:+.3f}s (no re-sync)")
            else:
                PROCESS_STATUS = {"state": "processing", "message": "Syncing external audio…"}
                EXTERNAL_AUDIO_OFFSET = sync_external_audio(combined, EXTERNAL_AUDIO_PATH)
                print(f"  External audio synced: offset={EXTERNAL_AUDIO_OFFSET:+.3f}s")

        if cache:
            print(f"  Fast resume — cached transcript ({len(cache.get('words', []))} words), skipping Whisper + AI")
            PROCESS_STATUS = {"state": "processing", "message": "Restoring…"}
            DURATION       = cache.get("duration") or get_duration(VIDEO_PATH)
            THUMB_INTERVAL = cache.get("thumb_interval", 10.0 if DURATION > 300 else 5.0)
            WORDS          = cache.get("words", [])
            WAVEFORM       = cache.get("waveform", [])
            # Reuse persisted thumbnails if present — otherwise generate once
            existing = sorted(THUMBS_DIR.glob("thumb*.jpg")) if THUMBS_DIR.exists() else []
            if existing:
                THUMB_COUNT = len(existing)
                print(f"  Reusing {THUMB_COUNT} cached thumbnails")
            else:
                PROCESS_STATUS["message"] = "Building thumbnails…"
                THUMB_COUNT = extract_thumbnails(VIDEO_PATH, THUMBS_DIR, THUMB_INTERVAL)
            CUTS = []
            INITIAL_AUTO_CUTS = []
        else:
            # ── Full pipeline (first run / new upload) ──
            PROCESS_STATUS = {"state": "processing", "message": "Extracting thumbnails…"}
            DURATION = get_duration(VIDEO_PATH)
            THUMB_INTERVAL = 10.0 if DURATION > 300 else 5.0
            shutil.rmtree(THUMBS_DIR, ignore_errors=True)   # clear any stale thumbs
            THUMB_COUNT = extract_thumbnails(VIDEO_PATH, THUMBS_DIR, THUMB_INTERVAL)

            PROCESS_STATUS["message"] = "Extracting waveform…"
            WAVEFORM = extract_waveform(EXTERNAL_AUDIO_PATH if EXTERNAL_AUDIO_PATH else VIDEO_PATH)

            if skip_transcribe:
                # ── Standard-editing mode: no transcription, no auto cuts ──
                PROCESS_STATUS["message"] = "Preparing editor…"
                WORDS = []
                CUTS = []
                INITIAL_AUTO_CUTS = []
                save_transcript(ORIGINAL_STEM)   # cache (empty words) so resume stays instant
            else:
                audio_path = new_temp / "audio.wav"
                PROCESS_STATUS["message"] = "Extracting audio…"
                if EXTERNAL_AUDIO_PATH:
                    subprocess.run([
                        FFMPEG, "-y", "-i", str(EXTERNAL_AUDIO_PATH),
                        "-ss", str(max(0, -EXTERNAL_AUDIO_OFFSET)),
                        "-ac", "1", "-ar", "16000", str(audio_path),
                    ], capture_output=True)
                else:
                    extract_audio(VIDEO_PATH, audio_path)

                PROCESS_STATUS["message"] = "Transcribing…"
                WORDS = transcribe(audio_path, MODEL_ARGS["model"])

                PROCESS_STATUS["message"] = "Detecting cuts…"
                CUTS = build_labeled_cuts(
                    WORDS, DURATION,
                    skip_fillers=MODEL_ARGS["skip_fillers"],
                    skip_bad_takes=MODEL_ARGS["skip_bad_takes"],
                    skip_silence=MODEL_ARGS["skip_silence"],
                )
                INITIAL_AUTO_CUTS = [dict(c) for c in CUTS]
                # Cache transcript so future resumes are instant
                save_transcript(ORIGINAL_STEM)

        # CRITICAL: restore saved user edits FIRST (replaces auto cuts if any exist),
        # THEN persist. Doing save before load would clobber the user's saved work.
        load_edits()
        RESUMING = False   # CUTS is now fully rebuilt — saves are safe again
        PROCESS_STATUS = {"state": "finalizing", "message": "Restoring your cuts…"}
        save_edits()

        # Save paths to config so server restart can auto-resume
        save_config()

        # ── Smooth-preview proxy: reuse if cached, else build in the background ──
        PROXY_READY = False
        proxy_file  = proj_dir / "_proxy.mp4"
        PROXY_PATH  = proxy_file
        if proxy_file.exists() and proxy_is_valid(proxy_file):
            PROXY_READY = True
            print(f"  Reusing cached proxy ({proxy_file.name})")
        else:
            if proxy_file.exists():
                print(f"  Cached proxy is corrupt — discarding and rebuilding ({proxy_file.name})")
                proxy_file.unlink(missing_ok=True)
            threading.Thread(target=_build_proxy_bg,
                             args=(VIDEO_PATH, proxy_file, project_id), daemon=True).start()

        # Clean up old temp dir (never the persistent project folder)
        if old_temp and Path(old_temp) != new_temp and Path(old_temp) != proj_dir:
            shutil.rmtree(old_temp, ignore_errors=True)

        PROCESS_STATUS = {"state": "done", "message": f"Ready — {VIDEO_PATH.name}"}
    except Exception as exc:
        import traceback
        PROCESS_STATUS = {"state": "error", "message": str(exc) + "\n" + traceback.format_exc()}
    finally:
        RESUMING = False   # never leave saves blocked, even if resume failed


@app.route("/api/silence-regions")
def api_silence_regions():
    """Run ffmpeg silencedetect on the full video; return all silent segments."""
    try:
        cmd = [
            FFMPEG, "-i", str(VIDEO_PATH),
            "-af", "silencedetect=noise=-40dB:duration=0.15",
            "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        silences = []
        start = None
        for line in result.stderr.split("\n"):
            if "silence_start" in line:
                m = re.search(r"silence_start: ([\d.]+)", line)
                if m:
                    start = float(m.group(1))
            elif "silence_end" in line and start is not None:
                m = re.search(r"silence_end: ([\d.]+)", line)
                if m:
                    end = float(m.group(1))
                    if end - start >= 0.15:   # skip tiny blips
                        silences.append({"start": round(start, 4), "end": round(end, 4)})
                    start = None
        return jsonify({"silences": silences})
    except Exception as e:
        return jsonify({"silences": [], "error": str(e)})


# ── HTML/JS ───────────────────────────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Cut Review</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #1e1e1e; color: #fff; font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

/* ── Toolbar ── */
.toolbar { background: #2a2a2a; border-bottom: 1px solid #111; padding: 6px 12px; min-height: 44px; display: flex; align-items: center; gap: 8px; flex-shrink: 0; flex-wrap: wrap; row-gap: 6px; }
.toolbar .encode-btn { margin-left: auto; }   /* keep Export pushed to the right edge / end of the row */
.toolbar h1 { font-size: 13px; font-weight: 600; color: #ccc; letter-spacing: 0.2px; margin-right: 2px; }
.tb-sep { width: 1px; height: 22px; background: #444; margin: 0 2px; }
.play-btn { background: #3a3a3a; border: 1px solid #555; color: #fff; width: 32px; height: 32px; border-radius: 50%; cursor: pointer; font-size: 13px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.play-btn:hover { background: #505050; }
.time-display { font-size: 12px; font-weight: 600; font-variant-numeric: tabular-nums; color: #fff; background: #111; padding: 4px 10px; border-radius: 4px; border: 1px solid #3a3a3a; min-width: 88px; text-align: center; }
.tb-btn { background: #3a3a3a; border: 1px solid #555; color: #ddd; height: 28px; padding: 0 11px; border-radius: 5px; font-size: 12px; font-weight: 500; cursor: pointer; display: flex; align-items: center; gap: 5px; white-space: nowrap; }
.tb-btn:hover { background: #4a4a4a; }
.tb-btn.active { background: #1c3a5a; border-color: #3d9be9; color: #3d9be9; }
.badge-pill { background: #3d9be9; color: #fff; font-size: 10px; font-weight: 700; border-radius: 8px; padding: 1px 6px; margin-left: 2px; }
.badge-pill.zero { background: #444; color: #888; }
.spacer { flex: 1; }
.encode-btn { background: #1877d4; color: #fff; border: none; height: 30px; padding: 0 16px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; white-space: nowrap; }
.encode-btn:hover { background: #1565b8; }
.encode-btn:disabled { background: #333; color: #666; cursor: default; }

/* ── Workspace: [cuts sidebar] [video] [transcript] ── */
.workspace { display: flex; flex: 1; min-height: 0; overflow: hidden; border-bottom: 2px solid #111; }

/* Cuts sidebar — hidden by default, slides in */
.cuts-sidebar { width: 0; min-width: 0; overflow: hidden; transition: width 0.22s ease, min-width 0.22s ease; background: #252525; display: flex; flex-direction: column; flex-shrink: 0; }
.cuts-sidebar.open { width: 270px; min-width: 270px; border-right: 1px solid #111; }
.panel-header { background: #2a2a2a; border-bottom: 1px solid #111; padding: 8px 12px; display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
.panel-title { font-size: 11px; font-weight: 600; color: #999; text-transform: uppercase; letter-spacing: 0.6px; }
.panel-hint { font-size: 10px; color: #555; }
.cuts-list { flex: 1; overflow-y: auto; padding: 6px; display: flex; flex-direction: column; gap: 3px; }
.cuts-list::-webkit-scrollbar { width: 4px; }
.cuts-list::-webkit-scrollbar-thumb { background: #3a3a3a; border-radius: 3px; }

/* Cut list items */
.cut-item { background: #2e2e2e; border-radius: 5px; padding: 6px 8px; display: flex; align-items: center; gap: 6px; border: 1px solid #3a3a3a; cursor: pointer; transition: background 0.1s, opacity 0.15s; flex-shrink: 0; }
.cut-item:hover { background: #383838; }
.cut-item.off { opacity: 0.35; }
.ci-info { flex: 1; min-width: 0; }
.ci-time { font-size: 10px; color: #666; font-variant-numeric: tabular-nums; }
.ci-text { font-size: 11px; color: #ccc; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 1px; }
.ci-badge { font-size: 9px; font-weight: 700; padding: 2px 5px; border-radius: 7px; white-space: nowrap; flex-shrink: 0; text-transform: uppercase; letter-spacing: 0.3px; }
.badge-repeat  { color: #ff9f0a; background: rgba(255,159,10,0.12); }
.badge-restart { color: #ff6b6b; background: rgba(255,107,107,0.12); }
.badge-filler  { color: #bf5af2; background: rgba(191,90,242,0.12); }
.badge-silence { color: #666;    background: rgba(102,102,102,0.12); }
.badge-manual  { color: #3d9be9; background: rgba(61,155,233,0.12); }
.ci-actions { display: flex; gap: 3px; flex-shrink: 0; }
.ci-btn { border: none; padding: 3px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; cursor: pointer; white-space: nowrap; opacity: 0.35; transition: opacity 0.1s; }
.ci-btn.active { opacity: 1; }
.ci-btn:hover { opacity: 0.85; }
.ci-btn.accept { background: #c0392b; color: #fff; }
.ci-btn.reject { background: #2e7d32; color: #fff; }

/* Video panel */
.preview-panel { flex: 1; background: #000; display: flex; align-items: center; justify-content: center; min-width: 0; overflow: hidden; }
.video-stage { position: relative; overflow: hidden; background: #000; width: 100%; height: 100%; }
.preview-panel video { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; background: #000; display: block; opacity: 1; }
/* Buffer stays opacity:1 but BEHIND the active (fully occluded) so Chrome keeps its
   decoder running (opacity:0 / hidden video gets throttled → cold decoder → stall). */
.preview-panel video.bufhidden { z-index: 1; pointer-events: none; }
.preview-panel video.bufactive { z-index: 2; }

/* ── Image overlays ── */
#imageLayer { position: absolute; z-index: 4; pointer-events: none; overflow: visible; }
.img-ov { position: absolute; pointer-events: auto; cursor: move; user-select: none; display: block; }
.img-ov.sel { outline: 1.5px dashed #3d9be9; outline-offset: 2px; }
.img-ov .img-resize { position: absolute; right: -7px; bottom: -7px; width: 14px; height: 14px;
  background: #3d9be9; border: 2px solid #fff; border-radius: 3px; cursor: nwse-resize; }
.img-kf-label { position: absolute; left: 0; top: -16px; font-size: 10px; font-weight: 700; color: #fff;
  background: #3d9be9; padding: 0 5px; border-radius: 3px; line-height: 15px; pointer-events: none; }
.img-ov.kf-end { outline-style: dotted; outline-color: #e0a13a; }
.img-ov.kf-end .img-kf-label { background: #e0a13a; }
.img-block { position: absolute; height: 18px; background: #4a5a2d; border: 1px solid #7fae3d; border-radius: 3px;
  color: #eff; font-size: 10px; line-height: 16px; overflow: hidden; cursor: grab; box-sizing: border-box; }
.img-block.sel { background: #5c7a3a; border-color: #a5d47e; box-shadow: 0 0 0 1px #a5d47e; }
.img-block .cap-h { position: absolute; top: 0; width: 6px; height: 100%; cursor: ew-resize; }
.img-block .cap-h.l { left: 0; } .img-block .cap-h.r { right: 0; }
#imgPanel { display: none; align-items: center; gap: 9px; background: #232323; border-bottom: 1px solid #3a3a3a;
  padding: 0 12px; flex-shrink: 0; height: 42px; box-sizing: border-box; }
#imgPanel .cap-tgl { height: 26px; }
#audPanel .cap-tgl { height: 26px; }
.aud-block { position: absolute; height: 68px; background: #2f9e57; border: 1px solid #46c574; border-radius: 4px;
  color: #eafff0; font-size: 10px; overflow: hidden; cursor: grab; box-sizing: border-box; }
.aud-block.sel { border-color: #eaff00; box-shadow: 0 0 0 1.5px #eaff00; }
.aud-block .aud-wave { position: absolute; left: 0; top: 0; width: 100%; height: 100%; display: block; pointer-events: none; }
.aud-block .aud-lbl { position: absolute; left: 6px; top: 1px; right: 6px; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; pointer-events: none; text-shadow: 0 1px 2px rgba(0,0,0,.5); z-index: 1; }
.aud-block .cap-h { position: absolute; top: 0; width: 6px; height: 100%; cursor: ew-resize; z-index: 2; }
.aud-block .cap-h.l { left: 0; } .aud-block .cap-h.r { right: 0; }
/* iMovie-style draggable volume line (audio clips + video clip) */
.vol-line { position: absolute; left: 0; right: 0; height: 0; border-top: 1px solid #ffd23f;
  cursor: ns-resize; z-index: 6; pointer-events: auto; box-shadow: 0 0 1px rgba(0,0,0,.5); }
.vol-line:hover, .vol-line.dragging { border-top-color: #fff08a; box-shadow: 0 0 2px rgba(0,0,0,.6); }
.vol-line .vol-grip { position: absolute; right: 10px; top: -3px; width: 6px; height: 6px; border-radius: 50%;
  background: #ffd23f; border: 1px solid rgba(0,0,0,.4); }
.vol-line .vol-tag { position: absolute; right: 22px; top: -8px; font: 600 9px system-ui; color: #ffd23f;
  background: rgba(0,0,0,.55); padding: 0 4px; border-radius: 3px; opacity: 0; transition: opacity .1s; white-space: nowrap; }
.vol-line:hover .vol-tag, .vol-line.dragging .vol-tag { opacity: 1; }
.aud-block .vol-line { left: 7px; right: 7px; }   /* keep clear of the trim handles */
#vidVolBand { position: absolute; z-index: 6; pointer-events: none; }
.snd-row { display: flex; align-items: center; gap: 8px; background: #2a2a2a; border: 1px solid #3d3d3d;
  border-radius: 6px; padding: 7px 10px; cursor: pointer; }
.snd-row:hover { background: #34343a; border-color: #4a6a8a; }
.snd-row .snd-name { flex: 1; font-size: 12px; color: #ddd; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.snd-row .snd-dur { font-size: 11px; color: #888; }
.snd-row .snd-play { background: #333; border: 1px solid #555; color: #ccc; border-radius: 4px; width: 26px; height: 24px; cursor: pointer; }
.snd-trim-handle { position: absolute; top: 0; bottom: 0; width: 10px; margin-left: -5px; background: #46c574; cursor: ew-resize; z-index: 5; border-radius: 2px; }
.snd-trim-handle:hover { background: #5fe08c; }
.snd-trim-handle::after { content: ''; position: absolute; left: 4px; top: 50%; width: 2px; height: 22px; margin-top: -11px; background: rgba(0,0,0,0.45); }
/* ── Text caption overlay ── */
#captionLayer { position: absolute; z-index: 5; pointer-events: none; overflow: visible; }
.cap-box { position: absolute; pointer-events: auto; cursor: move; white-space: pre;
           line-height: 1.1; user-select: none;
           /* Crisp outline drawn BEHIND the fill (like the PIL export) so the fill colour
              stays pure — a blurry shadow was muting the white to grey. */
           -webkit-text-stroke: 0.055em rgba(0,0,0,0.82); paint-order: stroke fill;
           text-shadow: 0 1px 2px rgba(0,0,0,0.30); }
.cap-box.sel { outline: 1.5px dashed #3d9be9; outline-offset: 3px; }
.cap-box.kf-end { outline-style: dotted; outline-color: #e0a13a; }
.cap-kf-label { position: absolute; left: 0; top: -16px; font-size: 10px; font-weight: 700;
  color: #fff; background: #3d9be9; padding: 0 5px; border-radius: 3px; line-height: 15px;
  -webkit-text-stroke: 0; text-shadow: none; pointer-events: none; }
.cap-box.kf-end .cap-kf-label { background: #e0a13a; }
.cap-resize { position: absolute; right: -7px; top: 50%; transform: translateY(-50%);
  width: 11px; height: 26px; background: #3d9be9; border: 1.5px solid #fff; border-radius: 3px;
  cursor: ew-resize; pointer-events: auto; box-sizing: border-box; }
.cap-box.ghost { opacity: 0.45; }
/* Caption edit panel */
/* Docked caption edit bar: PERMANENT strip below the main toolbar, above the video.
   Always occupies its space (so selecting a caption never resizes the video) — it just
   swaps a hint for the edit controls. Never overlaps the video or the timeline. */
#capPanel { display: flex; align-items: center; gap: 7px; background: #232323;
            border-bottom: 1px solid #3a3a3a; padding: 0 12px; flex-shrink: 0;
            flex-wrap: nowrap; overflow-x: auto; overflow-y: hidden;
            height: 46px; box-sizing: border-box; }
#capControls { display: none; align-items: center; gap: 7px; flex-wrap: nowrap; }
#capPanel textarea { height: 30px !important; resize: none; overflow-y: auto; white-space: pre; }
#capPanel textarea { background: #1a1a1a; border: 1px solid #444; color: #eee; height: 30px; min-height: 30px;
            border-radius: 5px; padding: 5px 9px; font-size: 13px; width: 210px; resize: vertical;
            font-family: inherit; line-height: 1.3; vertical-align: middle; }
.cap-box.editing { outline: 1.5px solid #3d9be9 !important; background: rgba(0,0,0,0.12); }
#capPanel select { background: #1a1a1a; border: 1px solid #444; color: #eee; height: 28px; border-radius: 5px; font-size: 12px; }
#capPanel input[type=color] { width: 30px; height: 28px; border: 1px solid #444; border-radius: 5px; background: #1a1a1a; padding: 2px; cursor: pointer; }
#capPanel input[type=number] { background: #1a1a1a; border: 1px solid #444; color: #eee; height: 28px; border-radius: 5px; padding: 0 4px 0 7px; font-size: 12px; }
.cap-tgl { background: #3a3a3a; border: 1px solid #555; color: #ddd; height: 28px; min-width: 30px; padding: 0 9px;
           border-radius: 5px; font-size: 13px; cursor: pointer; }
.cap-tgl:hover { background: #4a4a4a; }
.cap-tgl.active { background: #1c3a5a; border-color: #3d9be9; color: #7ec4ff; }
.cap-panel-sep { width: 1px; height: 20px; background: #454545; margin: 0 2px; }
#capPanelGrip { cursor: move; color: #888; font-size: 15px; padding: 0 3px; user-select: none; line-height: 1; }
#capPanelGrip:hover { color: #ccc; }
/* Caption timeline track */
.cap-track { position: relative; width: 100%; min-height: 24px; background: #191919; border-top: 1px solid #111; }
.ov-track { min-height: 22px; background: #1d1d1d; border-top: none; border-bottom: 1px solid #0d0d0d; }
.ov-label { position: sticky; left: 0; z-index: 4; display: inline-block; font-size: 9px; font-weight: 700;
  color: #999; background: #262626; padding: 2px 7px; border-radius: 0 0 5px 0; pointer-events: none; }
.cap-block { position: absolute; height: 18px; background: #2d5a86; border: 1px solid #3d9be9; border-radius: 3px;
             color: #dff; font-size: 10px; line-height: 16px; overflow: hidden; cursor: grab; box-sizing: border-box; }
.cap-block.sel { background: #3d6fa5; border-color: #7ec4ff; box-shadow: 0 0 0 1px #7ec4ff; }
.cap-block .cap-lbl { position: absolute; left: 6px; right: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; pointer-events: none; }
.cap-block .cap-h { position: absolute; top: 0; width: 6px; height: 100%; cursor: ew-resize; }
.cap-block .cap-h.l { left: 0; } .cap-block .cap-h.r { right: 0; }
.capmenu-item { padding: 7px 12px; font-size: 13px; color: #ddd; border-radius: 4px; cursor: pointer; white-space: nowrap; }
.capmenu-item:hover { background: #3a3a3a; }

/* Crop editor modal */
.crop-modal { display:none; position:fixed; inset:0; background:rgba(0,0,0,0.85); z-index:7000; align-items:center; justify-content:center; flex-direction:column; gap:14px; }
.crop-modal.active { display:flex; }
.crop-modal-title { color:#fff; font-size:16px; font-weight:600; }
.crop-modal-hint { color:#888; font-size:12px; }
.crop-canvas-row { display:flex; gap:18px; align-items:flex-start; }
.crop-canvas-wrap { position:relative; line-height:0; box-shadow:0 0 0 1px #333; overflow:hidden; }
.crop-canvas-wrap video { display:block; max-width:62vw; max-height:60vh; }
.crop-result-col { display:flex; flex-direction:column; align-items:center; }
.crop-result-label { color:#888; font-size:11px; margin-bottom:6px; text-transform:uppercase; letter-spacing:0.5px; }
#cropResult { width:384px; height:216px; background:#000; border-radius:6px; box-shadow:0 0 0 1px #3d9be9; display:block; }
.crop-dim { position:absolute; inset:0; box-shadow:0 0 0 9999px rgba(0,0,0,0.55); pointer-events:none; } /* darken outside box */
.crop-box { position:absolute; border:2px solid #3d9be9; box-sizing:border-box; cursor:move; }
.crop-box::before { content:''; position:absolute; inset:0; box-shadow:0 0 0 9999px rgba(0,0,0,0.3); pointer-events:none; }
.crop-grid { position:absolute; inset:0; pointer-events:none; background-image:linear-gradient(rgba(255,255,255,0.25) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.25) 1px, transparent 1px); background-size:33.33% 33.33%; }
.crop-handle { position:absolute; width:14px; height:14px; background:#3d9be9; border:2px solid #fff; border-radius:50%; z-index:2; }
.crop-handle.tl { left:0; top:0; cursor:nwse-resize; }
.crop-handle.tr { right:0; top:0; cursor:nesw-resize; }
.crop-handle.bl { left:0; bottom:0; cursor:nesw-resize; }
.crop-handle.br { right:0; bottom:0; cursor:nwse-resize; }
.crop-actions { display:flex; gap:10px; margin-top:4px; }
.crop-actions button { border:none; border-radius:7px; padding:9px 18px; font-size:13px; font-weight:600; cursor:pointer; }
.crop-btn-apply { background:#1877d4; color:#fff; }
.crop-btn-secondary { background:#333; color:#ccc; }
.crop-btn-secondary:hover { background:#444; }

/* Transcript panel */
.transcript-panel { width: 42%; max-width: 600px; min-width: 260px; border-left: 1px solid #333; background: #1a1a1a; display: flex; flex-direction: column; overflow: hidden; flex-shrink: 0; }
.transcript-scroll { flex: 1; overflow-y: auto; padding: 16px 18px; line-height: 2; }
.transcript-scroll::-webkit-scrollbar { width: 5px; }
.transcript-scroll::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }

/* Word spans */
.word { cursor: pointer; border-radius: 2px; padding: 1px 0; color: #ccc; font-size: 14px; transition: background 0.08s; }
.word:hover { background: rgba(255,255,255,0.08); color: #fff; }
.word.hl { background: #1877d4; color: #fff; border-radius: 2px; }
.word.struck { text-decoration: line-through; color: #444 !important; background: none !important; }
.word.struck.hl { text-decoration: line-through; color: #444 !important; background: none !important; }
/* "Hide skipped" mode — hide deleted words in the transcript, and black out the deleted
   clip regions in the timeline so only the kept clips are visible. */
.hide-skipped .word.struck { display: none !important; }
.hide-skipped .cut-merged { background: #0a0a0a !important; border-left-color: #000 !important; border-right-color: #000 !important; z-index: 3 !important; }
.tl-paragraph { display: block; margin-bottom: 6px; }

/* ── Timeline ── */
.timeline-section { flex-shrink: 0; background: #1a1a1a; border-top: 1px solid #111; display: flex; flex-direction: column; }
.tl-scroll { overflow-x: auto; overflow-y: hidden; flex-shrink: 0; }
.tl-scroll::-webkit-scrollbar { height: 6px; background: #111; }
.tl-scroll::-webkit-scrollbar-thumb { background: #3a3a3a; border-radius: 4px; }
.tl-inner { position: relative; }

.ruler { height: 20px; position: relative; background: #222; border-bottom: 1px solid #111; }
.r-tick { position: absolute; bottom: 0; width: 1px; background: #3a3a3a; }
.r-tick.major { background: #666; height: 10px; }
.r-tick:not(.major) { height: 4px; }
.r-label { position: absolute; top: 2px; font-size: 10px; color: #666; transform: translateX(-50%); pointer-events: none; white-space: nowrap; }

.filmstrip { position: relative; height: 68px; background: transparent; overflow: hidden; border-bottom: 1px solid #111; cursor: crosshair; }

/* Hover scrub line on timeline */
.hover-line { position: absolute; top: 0; bottom: 0; width: 1px; background: rgba(255,255,255,0.85); pointer-events: none; z-index: 8; display: none; }
.hover-label { position: absolute; top: 4px; transform: translateX(-50%); background: rgba(0,0,0,0.82); color: #fff; font-size: 10px; font-weight: 600; font-variant-numeric: tabular-nums; padding: 2px 6px; border-radius: 4px; white-space: nowrap; pointer-events: none; letter-spacing: 0.2px; }
.thumb { position: absolute; top: 0; height: 68px; object-fit: cover; pointer-events: none; display: block; }

/* Merged deleted sections — iMovie-style gray overlay */
.cut-merged { position: absolute; top: 0; height: 68px; background: rgba(0,0,0,0.55); border-left: 2px solid #666; border-right: 2px solid #666; pointer-events: none; z-index: 2; }

/* Trim handles at clip edges — drag to extend clip into deleted area */
.trim-handle { position: absolute; top: 0; height: 68px; width: 14px; z-index: 7; cursor: col-resize; transform: translateX(-50%); display: flex; align-items: center; justify-content: center; }
.trim-handle::after { content: ''; width: 4px; height: 36px; background: rgba(255,255,255,0.55); border-radius: 2px; transition: background 0.1s, height 0.1s; }
.trim-handle:hover::after { background: #3d9be9; height: 44px; }
.trim-handle.dragging::after { background: #3d9be9; height: 44px; }

/* Silence regions on the waveform strip */
.silence-ov { position: absolute; height: 68px; background: rgba(241,196,15,0.30); border-left: 2px solid rgba(241,196,15,0.75); border-right: 2px solid rgba(241,196,15,0.75); z-index: 5; pointer-events: none; }
.silence-ov.hovered { background: rgba(241,196,15,0.55); border-color: #f1c40f; box-shadow: 0 0 0 1px rgba(241,196,15,0.4); }

.wf-canvas { display: block; background: #1e1e1e; }
.playhead { position: absolute; top: 0; bottom: 0; width: 2px; background: #3d9be9; pointer-events: none; z-index: 10; }
.ph-head { position: absolute; top: -1px; left: -6px; width: 0; height: 0; border-left: 7px solid transparent; border-right: 7px solid transparent; border-top: 9px solid #3d9be9; }
.split-line { position: absolute; top: 0; bottom: 0; width: 2px; background: #f1c40f; pointer-events: none; z-index: 5; }
.split-line::before { content: ''; position: absolute; top: -1px; left: -4px; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 7px solid #f1c40f; }

/* Selectable segment tiles (between splits) */
.seg-tile { position: absolute; top: 0; height: 68px; z-index: 4; cursor: pointer; border: 2px solid transparent; transition: border-color 0.1s, background 0.1s; }
.seg-tile:hover { border-color: rgba(61,155,233,0.55); background: rgba(61,155,233,0.06); }
.seg-tile.sel { border-color: #3d9be9; background: rgba(61,155,233,0.14); }
.seg-tile.sel::after { content: 'Delete'; position: absolute; top: 4px; right: 6px; font-size: 9px; font-weight: 700; color: #3d9be9; background: rgba(0,0,0,0.7); padding: 2px 6px; border-radius: 4px; pointer-events: none; }
.seg-tile.muted { background: rgba(0,0,0,0.42); }
.seg-tile.muted::before { content: '🔇'; position: absolute; top: 4px; left: 6px; font-size: 12px; pointer-events: none; text-shadow: 0 1px 3px rgba(0,0,0,.8); }

/* Context menu */
.ctx-menu { position: fixed; background: #2a2a2a; border: 1px solid #555; border-radius: 8px; padding: 4px; min-width: 175px; box-shadow: 0 8px 28px rgba(0,0,0,0.6); z-index: 1000; font-size: 12px; user-select: none; }
.ctx-item { padding: 7px 12px; border-radius: 5px; cursor: pointer; color: #ddd; display: flex; align-items: center; gap: 9px; white-space: nowrap; }
.ctx-item:hover { background: #3a3a3a; color: #fff; }
.ctx-item.danger { color: #e74c3c; }
.ctx-item.danger:hover { background: rgba(231,76,60,0.18); }
.ctx-item.restore { color: #27ae60; }
.ctx-item.restore:hover { background: rgba(39,174,96,0.15); }
.ctx-sep { height: 1px; background: #3a3a3a; margin: 3px 4px; }

/* Upload drag-over overlay */
.drop-overlay { display: none; position: fixed; inset: 0; background: rgba(61,155,233,0.18); border: 3px dashed #3d9be9; z-index: 5000; align-items: center; justify-content: center; flex-direction: column; gap: 12px; pointer-events: none; }
.drop-overlay.active { display: flex; }
.drop-overlay-icon { font-size: 56px; }
.drop-overlay-text { font-size: 20px; font-weight: 600; color: #3d9be9; }
/* Processing overlay */
.process-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.82); z-index: 6000; align-items: center; justify-content: center; flex-direction: column; gap: 16px; }
.process-overlay.active { display: flex; }
.process-spinner { width: 40px; height: 40px; border: 4px solid #333; border-top-color: #3d9be9; border-radius: 50%; animation: spin 0.8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.process-msg { font-size: 15px; color: #ccc; font-weight: 500; max-width: 420px; text-align: center; }
.process-sub { font-size: 11px; color: #666; }
/* Project picker */
.picker-overlay { display: none; position: fixed; inset: 0; background: #1a1a1a; z-index: 5500; flex-direction: column; padding: 0; }
.picker-overlay.active { display: flex; }
.picker-head { display: flex; align-items: center; justify-content: space-between; padding: 22px 32px; border-bottom: 1px solid #2a2a2a; }
.picker-title { font-size: 20px; font-weight: 700; color: #fff; }
.picker-title span { color: #666; font-weight: 400; font-size: 14px; margin-left: 8px; }
.picker-new { background: #1877d4; color: #fff; border: none; border-radius: 8px; padding: 11px 20px; font-size: 14px; font-weight: 600; cursor: pointer; }
.picker-new:hover { background: #2088e8; }
.picker-grid { flex: 1; overflow-y: auto; padding: 24px 32px; display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; align-content: start; }
.proj-card { background: #242424; border: 1px solid #333; border-radius: 10px; padding: 16px; cursor: pointer; transition: border-color 0.12s, transform 0.12s; position: relative; }
.proj-card:hover { border-color: #3d9be9; transform: translateY(-2px); }
.proj-card.current { border-color: #27ae60; }
.proj-card.current::after { content: 'OPEN'; position: absolute; top: 12px; right: 12px; font-size: 9px; font-weight: 700; color: #27ae60; background: rgba(39,174,96,0.15); padding: 2px 7px; border-radius: 5px; }
.proj-name { font-size: 14px; font-weight: 600; color: #fff; margin-bottom: 10px; word-break: break-word; padding-right: 40px; }
.proj-meta { font-size: 11px; color: #888; line-height: 1.7; }
.proj-badge { display: inline-block; font-size: 9px; font-weight: 700; padding: 2px 6px; border-radius: 4px; margin-right: 5px; }
.proj-badge.audio { background: rgba(39,174,96,0.18); color: #4cd080; }
.proj-badge.cached { background: rgba(61,155,233,0.18); color: #6bb6f0; }
.proj-del { position: absolute; bottom: 12px; right: 12px; background: none; border: none; color: #555; font-size: 13px; cursor: pointer; padding: 4px; opacity: 0; transition: opacity 0.12s; }
.proj-card:hover .proj-del { opacity: 1; }
.proj-del:hover { color: #e74c3c; }
.picker-empty { grid-column: 1/-1; text-align: center; color: #666; padding: 60px 20px; font-size: 14px; }
/* Status bar */
.statusbar { background: #1a1a1a; border-top: 1px solid #111; padding: 5px 12px; font-size: 10px; color: #666; flex-shrink: 0; display: flex; align-items: center; justify-content: space-between; }
.statusbar.enc  { color: #e67e22; }
.statusbar.done { color: #27ae60; }
.statusbar.err  { color: #e74c3c; }
.enc-spinner { width: 12px; height: 12px; border: 2px solid #e67e22; border-top-color: transparent;
               border-radius: 50%; animation: encspin 0.7s linear infinite; flex-shrink: 0; }
@keyframes encspin { to { transform: rotate(360deg); } }
#encCancelBtn { background: #3a2020; border: 1px solid #7a3a3a; color: #f0a; color: #ec8; height: 22px;
                padding: 0 11px; border-radius: 5px; font-size: 11px; font-weight: 600; cursor: pointer; }
#encCancelBtn:hover { background: #4a2626; }
.undo-hint { font-size: 10px; color: #3a3a3a; }
</style>
</head>
<body>

<!-- Drag-over drop zone -->
<div class="drop-overlay" id="dropOverlay">
  <div class="drop-overlay-icon">🎬</div>
  <div class="drop-overlay-text">Drop a video to load · image / MP3 onto the timeline to add a layer</div>
</div>

<!-- Processing overlay -->
<div class="process-overlay" id="processOverlay">
  <div class="process-spinner" id="processSpinner"></div>
  <div class="process-msg" id="processMsg">Processing…</div>
  <div class="process-sub" id="processSub">This takes 1–3 minutes. The editor will reload automatically.</div>
  <button id="processUploadBtn" onclick="document.getElementById('videoUploadInput').click()"
    style="display:none;margin-top:8px;background:#3d9be9;color:#fff;border:none;border-radius:8px;padding:12px 28px;font-size:15px;font-weight:600;cursor:pointer;">
    📁 Choose Video Files
  </button>
</div>

<!-- Project picker -->
<div class="picker-overlay" id="projectPicker">
  <div class="picker-head">
    <div class="picker-title">🎬 Projects <span id="pickerCount"></span></div>
    <button class="picker-new" onclick="document.getElementById('videoUploadInput').click()">+ New Project</button>
  </div>
  <div class="picker-grid" id="pickerGrid"></div>
</div>

<!-- Crop / reframe editor -->
<div class="crop-modal" id="cropModal">
  <div class="crop-modal-title">Edit Frame — 16:9</div>
  <div class="crop-modal-hint">Drag the box to reposition · drag a corner to zoom (always 16:9) · the right panel shows the final framing</div>
  <div class="crop-canvas-row">
    <div class="crop-canvas-wrap" id="cropCanvasWrap">
      <video id="cropVideo" preload="none" muted></video>
      <div class="crop-box" id="cropBox">
        <div class="crop-grid"></div>
        <div class="crop-handle tl" data-h="tl"></div>
        <div class="crop-handle tr" data-h="tr"></div>
        <div class="crop-handle bl" data-h="bl"></div>
        <div class="crop-handle br" data-h="br"></div>
      </div>
    </div>
    <div class="crop-result-col">
      <div class="crop-result-label">Output preview</div>
      <canvas id="cropResult" width="384" height="216"></canvas>
      <button class="crop-btn-secondary" id="cropPlayBtn" onclick="cropTogglePlay()" style="margin-top:8px;font-size:12px;padding:6px 14px;">▶ Play here</button>
    </div>
  </div>
  <div class="crop-actions">
    <button class="crop-btn-secondary" onclick="cropReset()">Center 16:9</button>
    <button class="crop-btn-secondary" onclick="cropRemove()">No crop</button>
    <button class="crop-btn-secondary" onclick="closeCropEditor()">Cancel</button>
    <button class="crop-btn-apply" onclick="cropApply()">Apply</button>
  </div>
</div>

<!-- Toolbar -->
<div class="toolbar">
  <h1>✂️ Cut Review</h1>
  <span id="projectName" style="font-size:11px;color:#888;font-weight:500;max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"></span>
  <div class="tb-sep"></div>
  <button class="tb-btn" onclick="exitToProjects()" title="Switch project / back to project list">← Projects</button>
  <div class="tb-sep"></div>
  <button class="play-btn" id="playBtn" onclick="togglePlay()">▶</button>
  <div class="time-display" id="timeDisplay">0:00:00.00</div>
  <div id="finalDuration" title="Final length after cuts (original length · time removed)"
       style="font-size:11px;font-weight:600;font-variant-numeric:tabular-nums;color:#27ae60;background:#111;padding:4px 10px;border-radius:4px;border:1px solid #2a4a32;white-space:nowrap;"></div>
  <div id="smoothBadge" title="Smooth-preview render status"
       style="display:none;font-size:11px;font-weight:600;color:#aaa;background:#111;padding:4px 10px;border-radius:4px;border:1px solid #333;white-space:nowrap;">⏳ rendering smooth…</div>
  <div class="tb-sep"></div>
  <button class="tb-btn" onclick="seekTo(0)">⏮</button>
  <div class="tb-sep"></div>
  <button class="tb-btn" id="cutsToggle" onclick="toggleCutsPanel()">
    ✂ Detected Cuts <span class="badge-pill zero" id="cutsBadge">0</span>
  </button>
  <button class="tb-btn" id="hideSkipBtn" onclick="toggleHideSkipped()" title="Hide the deleted/skipped parts so the transcript reads as the final cut">👁 Hide skipped</button>
  <div class="spacer"></div>
  <input type="file" id="videoUploadInput" accept="video/*" multiple style="display:none" onchange="promptUploadMode(this.files, null)">
  <input type="file" id="audioUploadInput" accept="audio/*,video/*" style="display:none" onchange="handleAudioPick(this.files)">
  <button class="tb-btn" onclick="document.getElementById('videoUploadInput').click()" title="Load a new video (or drop anywhere)">📁 Load Video</button>
  <button class="tb-btn" id="extAudioBtn" onclick="document.getElementById('audioUploadInput').click()" title="REPLACE the camera audio with a separately-recorded mic track (dual-system sound). NOT for sound effects — drop MP3s on the timeline or use 🔊 Sound for those.">🎙 Ext Audio</button>
  <button class="tb-btn" id="audioSrcBtn" onclick="toggleAudioSource()" title="Switch between external mic and original camera audio" style="display:none">🎙 Using: Ext</button>
  <button class="tb-btn" id="alignAudioBtn" onclick="openAlignModal()" title="Manually line up the external audio with the video" style="display:none">🎚 Align Audio</button>
  <button class="tb-btn" id="removeExtAudioBtn" onclick="removeExtAudio()" title="Undo external audio — revert to the camera's own audio" style="display:none;color:#e88;">✕ Remove Ext Audio</button>
  <div class="tb-sep"></div>
  <button class="tb-btn" id="cropBtn" onclick="openCropEditor()" title="Crop / reframe to 16:9">⛶ Edit Frame</button>
  <button class="tb-btn" id="colorBtn" onclick="toggleColorPanel()" title="Brightness / contrast / saturation">🎨 Color</button>
  <button class="tb-btn" id="addTextBtn" onclick="addCaption()" title="Add a text caption overlay">➕ Text</button>
  <input type="file" id="imgUploadInput" accept="image/*" style="display:none" onchange="handleImagePick(this.files)">
  <button class="tb-btn" id="addImageBtn" onclick="document.getElementById('imgUploadInput').click()" title="Add an image overlay">🖼 Image</button>
  <input type="file" id="soundUploadInput" accept="audio/*" style="display:none" onchange="handleSoundPick(this.files, 'lib')">
  <button class="tb-btn" id="addSoundBtn" onclick="openSoundLibrary()" title="Add a sound effect / music (saved to your library)">🔊 Sound</button>
  <div class="tb-sep"></div>
  <button class="encode-btn" id="encodeBtn" onclick="openExportModal()">▶ Export Final Video</button>
</div>

<!-- Caption edit panel (shown when a caption is selected) -->
<div id="capPanel">
  <span id="capHint" style="font-size:12px;color:#888;white-space:nowrap;">✎ Click a caption to edit it, or use <b style="color:#aaa;">➕ Text</b> to add one.</span>
  <span id="capControls" style="display:none;align-items:center;gap:7px;">
  <span style="font-size:11px;color:#777;font-weight:600;white-space:nowrap;">✎ Caption</span>
  <textarea id="capText" rows="1" placeholder="Caption text… (Enter = new line)" oninput="capEdit('text', this.value); capGrow(this);"></textarea>
  <select id="capFont" onchange="capEdit('font', this.value)" title="Font">
    <option value="Avenir">Avenir</option>
    <option value="Helvetica">Helvetica</option>
    <option value="Arial">Arial</option>
  </select>
  <select id="capSize" onchange="capSizeDropdown(this.value)" title="Size preset">
    <option value="custom">Custom</option>
    <option value="0.035">Small</option>
    <option value="0.055">Medium</option>
    <option value="0.08">Large</option>
    <option value="0.12">X-Large</option>
    <option value="0.17">Huge</option>
  </select>
  <input id="capSizeNum" type="number" min="6" max="400" step="1" title="Exact font size in px (at 1080p output)" oninput="capSizeNumInput(this.value)" style="width:54px;">
  <span style="font-size:11px;color:#888;margin-left:-3px;">px</span>
  <button id="capAnim" class="cap-tgl" onclick="toggleCapAnim()" title="Animate: transform from a Start to an End keyframe (position + size) over the caption">✨ Animate</button>
  <span id="capEndSize" style="display:none;align-items:center;gap:4px;font-size:11px;color:#888;">→ end
    <input id="capSizeNum2" type="number" min="6" max="400" step="1" title="End font size in px" oninput="capEndSizeInput(this.value)" style="width:50px;">px
    <button class="cap-tgl" onclick="swapCapKeyframes()" title="Swap start ↔ end (reverse the animation)" style="font-size:13px;">⇅</button></span>
  <button id="capBold" class="cap-tgl" onclick="capToggle('bold')" title="Bold" style="font-weight:800;">B</button>
  <button id="capItalic" class="cap-tgl" onclick="capToggle('italic')" title="Italic" style="font-style:italic;">I</button>
  <input id="capColor" type="color" value="#ffffff" onchange="capEdit('color', this.value)" title="Color">
  <span style="font-size:12px;color:#888;" title="Outline thickness">◌</span>
  <input id="capStroke" type="range" min="0" max="300" step="1" title="Outline thickness" oninput="capStrokeInput(this.value)" style="width:92px;vertical-align:middle;">
  <span id="capStrokeVal" style="font-size:11px;color:#aaa;width:26px;display:inline-block;">56</span>
  <span class="cap-panel-sep"></span>
  <span style="font-size:12px;color:#888;letter-spacing:1px;" title="Letter spacing">A↔</span>
  <input id="capLS" type="number" step="1" title="Letter spacing (px at 1080p)" oninput="capLSInput(this.value)" style="width:46px;">
  <span style="font-size:12px;color:#888;" title="Line spacing">≡</span>
  <input id="capLH" type="number" step="0.05" min="0.5" max="3" title="Line spacing (multiple of line height)" oninput="capLHInput(this.value)" style="width:46px;">
  <span class="cap-panel-sep"></span>
  <span style="font-size:11px;color:#888;">✨FX</span>
  <select id="capFx" onchange="capFxChange(this.value)" title="Text animation effect">
    <option value="none">None</option>
    <option value="typewriter">Typewriter</option>
    <option value="fade">Fade</option>
    <option value="zoom">Zoom</option>
  </select>
  <span id="capFxOpts" style="display:none;align-items:center;gap:6px;">
    <select id="capFxMode" onchange="capEditFx('fxMode', this.value)" title="When the effect plays">
      <option value="in">On enter</option>
      <option value="out">On exit</option>
      <option value="both">Both</option>
    </select>
    <span style="font-size:11px;color:#888;">Speed</span>
    <input id="capFxSpeed" type="range" min="0.1" max="1" step="0.05" value="0.5" oninput="capEditFx('fxSpeed', parseFloat(this.value))" style="width:78px;vertical-align:middle;" title="Effect speed">
  </span>
  <span class="cap-panel-sep"></span>
  <button class="cap-tgl" onclick="duplicateCaption(selectedCapId)" title="Duplicate (⌘C then ⌘V)">⧉</button>
  <button class="cap-tgl" onclick="capDelete()" title="Delete caption" style="color:#e66;">🗑</button>
  <button class="cap-tgl" onclick="deselectCaption()" title="Done editing">Done</button>
  </span>
</div>

<!-- Image overlay edit bar -->
<div id="imgPanel">
  <span style="font-size:11px;color:#777;font-weight:600;white-space:nowrap;">🖼 Image</span>
  <span style="font-size:11px;color:#888;">opacity</span>
  <input id="imgOpacity" type="range" min="0" max="100" value="100" oninput="imgEditOpacity(this.value)" style="width:110px;">
  <span id="imgOpacityVal" style="font-size:11px;color:#aaa;width:34px;">100%</span>
  <button id="imgAnim" class="cap-tgl" onclick="toggleImgAnim()" title="Ken Burns: animate position/size from a Start to an End keyframe">✨ Animate</button>
  <span class="cap-panel-sep"></span>
  <button class="cap-tgl" onclick="imgDelete()" title="Delete image overlay" style="color:#e66;">🗑</button>
  <button class="cap-tgl" onclick="deselectImg()" title="Done">Done</button>
</div>

<!-- Audio-layer edit bar -->
<div id="audPanel" style="display:none;align-items:center;gap:9px;background:#232323;border-bottom:1px solid #3a3a3a;padding:0 12px;flex-shrink:0;height:42px;box-sizing:border-box;">
  <span id="audName" style="font-size:11px;color:#aaa;font-weight:600;max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">🔊 sound</span>
  <span style="font-size:11px;color:#888;">volume</span>
  <input id="audVol" type="range" min="0" max="200" value="100" oninput="audEditVol(this.value)" style="width:120px;">
  <span id="audVolVal" style="font-size:11px;color:#aaa;width:38px;">100%</span>
  <button class="cap-tgl" onclick="audPreview()" title="Play this sound">▶</button>
  <span class="cap-panel-sep"></span>
  <button class="cap-tgl" onclick="audDelete()" title="Remove this sound" style="color:#e66;">🗑</button>
  <button class="cap-tgl" onclick="deselectAud()" title="Done">Done</button>
</div>

<!-- Sound library modal -->
<div id="soundLibModal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.7);z-index:10000;align-items:center;justify-content:center;">
  <div style="background:#1e1e1e;border:1px solid #444;border-radius:10px;padding:18px;width:min(480px,92vw);max-height:74vh;display:flex;flex-direction:column;">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
      <b style="font-size:15px;color:#eee;">🔊 Sound Library</b><span style="flex:1"></span>
      <button class="tb-btn" onclick="document.getElementById('soundUploadInput').click()">＋ Add sound</button>
      <button class="tb-btn" onclick="closeSoundLibrary()">Close</button>
    </div>
    <div id="soundLibHint" style="font-size:11px;color:#888;margin-bottom:10px;">Click a sound to drop it at the playhead. Every MP3 you add is saved here and reusable in any project.</div>
    <div id="soundLibList" style="overflow-y:auto;display:flex;flex-direction:column;gap:5px;"></div>
  </div>
</div>

<!-- Sound trim / selection screen -->
<div id="trimModal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.8);z-index:10003;align-items:center;justify-content:center;">
  <div style="background:#1e1e1e;border:1px solid #444;border-radius:12px;padding:20px;width:min(660px,94vw);box-shadow:0 16px 50px rgba(0,0,0,.6);">
    <div style="font-size:16px;font-weight:700;color:#eee;margin-bottom:3px;">✂️ Trim sound</div>
    <div id="trimSub" style="font-size:12px;color:#999;margin-bottom:14px;">Drag the green handles to select the part you want.</div>
    <div id="trimWaveWrap" style="position:relative;height:96px;background:#141414;border:1px solid #333;border-radius:6px;overflow:hidden;user-select:none;">
      <canvas id="trimWave" style="position:absolute;inset:0;width:100%;height:100%;"></canvas>
      <div id="trimDimL" style="position:absolute;top:0;bottom:0;left:0;background:rgba(0,0,0,0.55);pointer-events:none;"></div>
      <div id="trimDimR" style="position:absolute;top:0;bottom:0;right:0;background:rgba(0,0,0,0.55);pointer-events:none;"></div>
      <div id="trimHandleL" class="snd-trim-handle"></div>
      <div id="trimHandleR" class="snd-trim-handle"></div>
      <div id="trimPlayhead" style="position:absolute;top:0;bottom:0;width:2px;background:#fff;left:0;display:none;pointer-events:none;z-index:6;"></div>
    </div>
    <div style="display:flex;align-items:center;gap:12px;margin-top:12px;">
      <button class="tb-btn" id="trimPlayBtn" onclick="trimTogglePlay()">▶ Preview</button>
      <span id="trimRange" style="font-size:12px;color:#aaa;font-variant-numeric:tabular-nums;"></span>
      <span style="flex:1"></span>
      <span id="trimNameWrap" style="display:flex;align-items:center;gap:8px;">
        <span style="font-size:12px;color:#888;">Name</span>
        <input id="trimName" type="text" style="width:190px;padding:5px 8px;background:#2a2a2a;border:1px solid #444;border-radius:5px;color:#eee;" placeholder="sound name">
      </span>
    </div>
    <div style="display:flex;justify-content:flex-end;gap:10px;margin-top:16px;">
      <button class="tb-btn" onclick="closeTrim()">Cancel</button>
      <button class="tb-btn" id="trimLibBtn" onclick="confirmTrim('lib')">Save to library</button>
      <button class="encode-btn" id="trimSaveBtn" onclick="confirmTrim()">Save &amp; add to video</button>
    </div>
  </div>
</div>

<!-- Colour adjust popover -->
<div id="colorPanel" style="display:none;position:fixed;top:44px;right:20px;z-index:9600;background:#232323;border:1px solid #454545;border-radius:10px;padding:14px 16px;box-shadow:0 10px 34px rgba(0,0,0,.55);width:250px;">
  <div style="display:flex;align-items:center;margin-bottom:10px;"><b style="font-size:13px;color:#eee;">🎨 Color adjust</b><span style="flex:1"></span><button onclick="resetAdjust()" style="background:#333;border:1px solid #555;color:#ccc;border-radius:5px;font-size:11px;padding:3px 9px;cursor:pointer;">Reset</button></div>
  <label style="display:block;font-size:11px;color:#aaa;margin-bottom:2px;">Brightness <span id="adjBVal" style="color:#7ec4ff;float:right;"></span></label>
  <input id="adjBright" type="range" min="-50" max="50" value="0" oninput="setAdjust('brightness', this.value/100)" style="width:100%;margin-bottom:9px;">
  <label style="display:block;font-size:11px;color:#aaa;margin-bottom:2px;">Contrast <span id="adjCVal" style="color:#7ec4ff;float:right;"></span></label>
  <input id="adjContrast" type="range" min="50" max="180" value="100" oninput="setAdjust('contrast', this.value/100)" style="width:100%;margin-bottom:9px;">
  <label style="display:block;font-size:11px;color:#aaa;margin-bottom:2px;">Saturation <span id="adjSVal" style="color:#7ec4ff;float:right;"></span></label>
  <input id="adjSat" type="range" min="0" max="180" value="100" oninput="setAdjust('saturation', this.value/100)" style="width:100%;">
</div>

<!-- Caption right-click menu -->
<div id="capMenu" style="display:none;position:fixed;z-index:10000;background:#232323;border:1px solid #4a4a4a;border-radius:7px;padding:4px;box-shadow:0 6px 24px rgba(0,0,0,.5);min-width:160px;">
  <div class="capmenu-item" onclick="capMenuAction('split')">✂ Split at playhead</div>
  <div class="capmenu-item" onclick="capMenuAction('dup')">⧉ Duplicate</div>
  <div class="capmenu-item" onclick="capMenuAction('del')" style="color:#e77;">🗑 Delete</div>
</div>

<!-- Image / audio block right-click menu -->
<div id="ovMenu" style="display:none;position:fixed;z-index:10000;background:#232323;border:1px solid #4a4a4a;border-radius:7px;padding:4px;box-shadow:0 6px 24px rgba(0,0,0,.5);min-width:160px;">
  <div class="capmenu-item" id="ovMuteItem" onclick="ovMenuAction('mute')">🔇 Mute</div>
  <div class="capmenu-item" onclick="ovMenuAction('split')">✂ Split at playhead</div>
  <div class="capmenu-item" onclick="ovMenuAction('dup')">⧉ Duplicate</div>
  <div class="capmenu-item" onclick="ovMenuAction('del')" style="color:#e77;">🗑 Delete</div>
</div>

<!-- Video segment right-click menu -->
<div id="segMenu" style="display:none;position:fixed;z-index:10000;background:#232323;border:1px solid #4a4a4a;border-radius:7px;padding:4px;box-shadow:0 6px 24px rgba(0,0,0,.5);min-width:170px;">
  <div class="capmenu-item" id="segMuteItem" onclick="segMenuMute()">🔇 Mute segment</div>
</div>

<!-- Transcript word right-click menu -->
<div id="wordMenu" style="display:none;position:fixed;z-index:10000;background:#232323;border:1px solid #4a4a4a;border-radius:7px;padding:4px;box-shadow:0 6px 24px rgba(0,0,0,.5);min-width:190px;">
  <div class="capmenu-item" onclick="wordMenuAddSound()">🔊 Add sound effect here</div>
</div>

<!-- Upload-mode picker (auto-cut vs standard editing) -->
<div id="uploadModeModal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.8);z-index:10001;align-items:center;justify-content:center;">
  <div style="background:#1e1e1e;border:1px solid #444;border-radius:12px;padding:26px;width:min(560px,92vw);box-shadow:0 16px 50px rgba(0,0,0,.6);">
    <div style="font-size:18px;font-weight:700;color:#eee;margin-bottom:4px;">How should this video be loaded?</div>
    <div id="uploadModeSub" style="font-size:12px;color:#999;margin-bottom:20px;">Choose a mode.</div>
    <div style="display:flex;gap:14px;">
      <button onclick="chooseUploadMode('auto')" style="flex:1;text-align:left;background:#242424;border:1px solid #3d9be9;border-radius:10px;padding:16px;cursor:pointer;color:#eee;">
        <div style="font-size:15px;font-weight:700;margin-bottom:6px;">✂️ Auto cut + transcription</div>
        <div style="font-size:11.5px;color:#9bb;line-height:1.45;">Transcribe the audio and auto-detect filler words, silences, stutters & repeat takes. Takes 1–3 min.</div>
      </button>
      <button onclick="chooseUploadMode('edit')" style="flex:1;text-align:left;background:#242424;border:1px solid #46c574;border-radius:10px;padding:16px;cursor:pointer;color:#eee;">
        <div style="font-size:15px;font-weight:700;margin-bottom:6px;">🎬 Standard editing only</div>
        <div style="font-size:11.5px;color:#9c9;line-height:1.45;">Skip transcription & auto cuts — for an already-cut video. Loads fast; captions, images, sound, crop, volume all work.</div>
      </button>
    </div>
    <div style="text-align:right;margin-top:18px;">
      <button onclick="cancelUploadMode()" style="background:none;border:none;color:#888;font-size:12px;cursor:pointer;">Cancel</button>
    </div>
  </div>
</div>

<!-- Align-audio modal -->
<div id="alignModal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.78);z-index:10000;align-items:center;justify-content:center;">
  <div style="background:#1e1e1e;border:1px solid #444;border-radius:10px;padding:20px;width:min(1000px,94vw);box-shadow:0 12px 40px rgba(0,0,0,.6);">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
      <h2 style="margin:0;font-size:16px;color:#eee;">🎚 Align External Audio</h2>
      <span id="alignStatus" style="font-size:12px;color:#888;"></span>
      <div style="flex:1"></div>
      <button class="tb-btn" onclick="alignZoom(1.5)" title="Zoom in">＋</button>
      <button class="tb-btn" onclick="alignZoom(1/1.5)" title="Zoom out">－</button>
    </div>
    <p style="margin:2px 0 10px;font-size:12px;color:#aaa;">
      Drag the <b style="color:#3d9be9;">blue external-audio</b> track left/right until its speech peaks line up
      with the <b style="color:#bbb;">gray camera</b> track above it. Use <b>Test</b> to hear it, then <b>Apply</b>.
    </p>
    <div id="alignScroll" style="overflow-x:auto;overflow-y:hidden;background:#141414;border:1px solid #333;border-radius:6px;">
      <canvas id="alignCanvas" height="260" style="display:block;cursor:ew-resize;"></canvas>
    </div>
    <div style="display:flex;align-items:center;gap:8px;margin-top:14px;flex-wrap:wrap;">
      <span style="font-size:12px;color:#888;">Nudge:</span>
      <button class="tb-btn" onclick="alignNudge(-1)">⏪ −1s</button>
      <button class="tb-btn" onclick="alignNudge(-0.1)">◀ −0.1</button>
      <button class="tb-btn" onclick="alignNudge(-0.02)">‹ −0.02</button>
      <button class="tb-btn" onclick="alignNudge(0.02)">+0.02 ›</button>
      <button class="tb-btn" onclick="alignNudge(0.1)">+0.1 ▶</button>
      <button class="tb-btn" onclick="alignNudge(1)">+1s ⏩</button>
      <div style="flex:1"></div>
      <button class="tb-btn" onclick="alignAuto()" title="Re-run automatic detection as a starting point">🪄 Auto-detect</button>
      <button class="tb-btn" id="alignTestBtn" onclick="alignTest()">▶ Test 6s</button>
    </div>
    <div style="display:flex;align-items:center;gap:14px;margin-top:14px;">
      <div style="font-size:13px;color:#ccc;">
        Offset: <b id="alignOffsetVal" style="color:#3d9be9;font-size:15px;">0.00s</b>
        <span id="alignOffsetDesc" style="color:#888;margin-left:8px;"></span>
      </div>
      <div style="flex:1"></div>
      <button class="tb-btn" onclick="closeAlignModal(false)">Cancel</button>
      <button class="encode-btn" onclick="alignApply()">✓ Apply & Re-render</button>
    </div>
  </div>
</div>

<!-- Export modal -->
<div id="exportModal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.7);z-index:9999;align-items:center;justify-content:center;">
  <div style="background:#1e1e1e;border:1px solid #444;border-radius:10px;padding:24px;width:520px;max-width:90vw;">
    <div style="font-size:15px;font-weight:600;margin-bottom:16px;color:#fff;">Export Video</div>
    <div style="font-size:12px;color:#aaa;margin-bottom:6px;">Save to</div>
    <div style="display:flex;gap:8px;align-items:center;">
      <input id="exportPath" type="text" style="flex:1;box-sizing:border-box;background:#111;border:1px solid #555;border-radius:6px;color:#fff;font-size:12px;padding:8px 10px;outline:none;" />
      <button onclick="pickSaveLocation()" style="background:#333;color:#ccc;border:1px solid #555;border-radius:6px;padding:8px 12px;font-size:12px;cursor:pointer;white-space:nowrap;">📁 Browse</button>
    </div>
    <div style="margin-top:16px;display:flex;gap:10px;justify-content:flex-end;">
      <button onclick="closeExportModal()" style="background:#333;color:#ccc;border:none;border-radius:6px;padding:7px 16px;font-size:12px;cursor:pointer;">Cancel</button>
      <button onclick="doEncode()" id="exportConfirmBtn" style="background:#1877d4;color:#fff;border:none;border-radius:6px;padding:7px 16px;font-size:12px;font-weight:600;cursor:pointer;">Export</button>
    </div>
  </div>
</div>

<!-- Workspace: [cuts sidebar] [video] [transcript] -->
<div class="workspace">

  <!-- Cuts sidebar — hidden by default -->
  <div class="cuts-sidebar" id="cutsSidebar">
    <div class="panel-header">
      <span class="panel-title">Detected Cuts</span>
    </div>
    <div class="cuts-list" id="cutsList"></div>
  </div>

  <!-- Video -->
  <div class="preview-panel">
    <div class="video-stage" id="videoStage">
      <video id="playerA" preload="none"></video>
      <video id="playerB" preload="none"></video>
      <div id="imageLayer"></div>
      <div id="captionLayer"></div>
    </div>
  </div>

  <!-- Transcript -->
  <div class="transcript-panel">
    <div class="panel-header">
      <span class="panel-title">Transcript</span>
      <span class="panel-hint">Select text → Delete to cut</span>
    </div>
    <div class="transcript-scroll" id="transcriptScroll"></div>
  </div>

</div>

<!-- Timeline -->
<div class="timeline-section">
  <div class="tl-scroll" id="tlScroll">
    <div class="tl-inner" id="tlInner">
      <div class="ruler" id="ruler"></div>
      <!-- Overlay layers, sitting on TOP of the video row -->
      <div class="cap-track ov-track" id="imgTrack" title="Image overlays — drag to move, drag edges to retime"><span class="ov-label">🖼 Images</span></div>
      <div class="cap-track ov-track" id="capTrack" title="Text captions — drag to move, drag edges to retime"><span class="ov-label">T Text</span></div>
      <div class="filmstrip" id="filmstrip"></div>
      <canvas class="wf-canvas" id="wfCanvas" height="68"></canvas>
      <div class="cap-track ov-track" id="audioTrack" title="Sound effects / music — drop an MP3 here or use 🔊 Sound"><span class="ov-label">🔊 Sound</span></div>
      <!-- Playhead + hover line span the FULL timeline height (ruler → audio) -->
      <div class="playhead" id="playhead"><div class="ph-head"></div></div>
      <div class="hover-line" id="hoverLine"><div class="hover-label" id="hoverLabel"></div></div>
    </div>
  </div>
</div>

<div class="statusbar" id="status">
  <span style="display:flex;align-items:center;gap:8px;min-width:0;">
    <span class="enc-spinner" id="encSpinner" style="display:none;"></span>
    <span id="statusMsg" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">Loading...</span>
  </span>
  <span style="display:flex;align-items:center;gap:10px;flex-shrink:0;">
    <button id="encCancelBtn" onclick="cancelEncode()" style="display:none;">✕ Cancel Export</button>
    <span class="undo-hint" id="undoHint"></span>
  </span>
</div>

<script>
let S   = null;
let PPS = 80;
let BASE_PPS = 80;   // set at init, used as zoom anchor
let zoomScale = 1.0; // current zoom multiplier

/* ── Timeline coordinate layer ──────────────────────────────────────────────
 * Normally the timeline is drawn in ORIGINAL time (x = t * PPS) with the cut
 * regions blacked out in place. When "Hide skipped" is on, we COLLAPSE the cut
 * regions: every element is drawn in EDITED time (cuts contribute zero width),
 * so the timeline shows only the clips that survive into the final video.
 *   X(t)   original-seconds  -> pixels
 *   Xw(a,b) width in px of the ORIGINAL span [a,b] after collapse
 *   Tx(px) pixels -> original-seconds (inverse, for click/seek/hover)
 *   dispDur() total displayed length in seconds (edited length when collapsed)
 * With HIDE_SKIP off every function is the identity (t*PPS), so normal mode is
 * byte-for-byte unchanged. */
let HIDE_SKIP = false;
let KEEPS = [];        // [[s,e],...] kept (non-cut) segments, in original time
let DISPDUR = 0;       // sum of kept-segment lengths
function refreshCoords() {
  if (!S) { KEEPS = []; DISPDUR = 0; return; }
  const merged = getMergedCuts();
  KEEPS = []; let p = 0;
  for (const c of merged) { if (c.start > p + 1e-6) KEEPS.push([p, c.start]); p = Math.max(p, c.end); }
  if (p < (S.duration || 0) - 1e-6) KEEPS.push([p, S.duration || 0]);
  DISPDUR = 0; for (const [s, e] of KEEPS) DISPDUR += (e - s);
}
function editedSeconds(t) {           // original seconds -> edited seconds
  let acc = 0;
  for (const [s, e] of KEEPS) {
    if (t <= s) break;
    if (t >= e) acc += (e - s);
    else { acc += (t - s); break; }
  }
  return acc;
}
function origFromEdited(es) {          // edited seconds -> original seconds
  let acc = 0;
  for (const [s, e] of KEEPS) { const d = e - s; if (es <= acc + d) return s + (es - acc); acc += d; }
  return S ? (S.duration || 0) : 0;
}
// Leading offset so 0:00 sits away from the left edge (iMovie-style). Clicking in the empty
// space before 0:00 pins the ticker to 0:00; once scrolled past, content reaches the edge.
const TL_PAD = 120;
function X(t)     { return (HIDE_SKIP ? editedSeconds(t) : t) * PPS + TL_PAD; }
function Xw(a, b) { const w = (HIDE_SKIP ? Math.max(0, editedSeconds(b) - editedSeconds(a)) : (b - a)) * PPS; return w; }
function Tx(px)   { let es = (px - TL_PAD) / PPS; if (es < 0) es = 0; return HIDE_SKIP ? origFromEdited(es) : es; }
function dispDur() { return HIDE_SKIP ? DISPDUR : (S ? (S.duration || 0) : 0); }
function tlFullW() { return Math.ceil(dispDur() * PPS) + TL_PAD; }   // total timeline width incl. leading pad

const UNDO_STACK = [];
let wordSpans   = [];   // cached array of span elements, parallel to S.words
let lastHlIdx   = -1;  // index of currently highlighted word span
let selectedSeg = null; // {start, end} of currently selected filmstrip segment
let ALL_SILENCES = []; // all silence regions from server (unfiltered)
const SILENCE_BUFFER = 0.06; // 60ms buffer on each end when cutting silence
// Top offset of waveform canvas inside tl-inner: ruler(20) + filmstrip(68) + 1px border
const WF_TOP = 89;
// Silence trim threshold — scan forward from cut.end to find when audio actually begins
const AUDIO_THRESH = 0.04;  // amplitude 0–1; tune if too aggressive/permissive
const AUDIO_LOOK   = 0.5;   // max seconds to scan forward

/* ── Helpers ── */
function fmt(s) {
  const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), sec = s%60;
  return `${h}:${String(m).padStart(2,'0')}:${sec.toFixed(2).padStart(5,'0')}`;
}
function fmtS(s) {
  const m = Math.floor(s/60), sec = Math.floor(s%60);
  return `${m}:${String(sec).padStart(2,'0')}`;
}
function badgeClass(type) {
  if (type === 'manual') return 'badge-manual';
  if (type.includes('repeat') || type.includes('rephrase')) return 'badge-repeat';
  if (type.includes('restart') || type.includes('signal') || type.includes('bad')) return 'badge-restart';
  if (type === 'filler') return 'badge-filler';
  return 'badge-silence';
}
function setStatus(msg) { document.getElementById('statusMsg').textContent = msg; }
function setUndoHint(n) {
  document.getElementById('undoHint').textContent = n > 0 ? `⌘Z to undo (${n})` : '';
  fetch('/api/undo-stack', { method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ stack: UNDO_STACK }) });
}

// Snapshot-based undo: capture the FULL cut + split state BEFORE a user action, so a
// single ⌘Z restores exactly the prior state — regardless of silence-expansion,
// cut-merging, or remnant cuts. Call once at the top of each top-level user gesture.
function pushUndoSnapshot() {
  const snap = {
    cuts: S.cuts.map(c => ({...c})),
    split_points: (S.split_points || []).map(s => ({...s})),
  };
  const top = UNDO_STACK[UNDO_STACK.length - 1];
  // Skip if identical to the current top (guards against nested gestures double-pushing)
  if (top && top.cuts && JSON.stringify(top) === JSON.stringify(snap)) return;
  UNDO_STACK.push(snap);
  while (UNDO_STACK.length > 60) UNDO_STACK.shift();   // bound history + state-file size
  setUndoHint(UNDO_STACK.length);
}

/* ── Audio start finder: scan waveform forward from t until amplitude rises ── */
function findAudioStart(t) {
  if (!S || !S.waveform || !S.waveform.length) return t;
  const samps = S.waveform;
  const len   = samps.length;
  // Convert time to waveform index
  const i0   = Math.floor(t / S.duration * len);
  const iMax = Math.min(len - 1, Math.floor((t + AUDIO_LOOK) / S.duration * len));
  for (let i = i0; i <= iMax; i++) {
    if (samps[i] >= AUDIO_THRESH) {
      // Found audio — back up 60ms (a tiny buffer so we don't clip the attack)
      const audioT = i / len * S.duration;
      return Math.max(t, audioT - 0.06);
    }
  }
  return t; // no audio found within look-ahead — return original
}

// Scan BACKWARD from t to where audio last occurred (end of previous speech).
// Won't scan earlier than `floor`. Returns that audio-end time + 60ms tail buffer.
function findSilenceStartBefore(t, floor) {
  if (!S || !S.waveform || !S.waveform.length) return t;
  const samps = S.waveform;
  const len   = samps.length;
  const i0     = Math.min(len - 1, Math.floor(t / S.duration * len));
  const iFloor = Math.max(0, Math.floor(floor / S.duration * len));
  for (let i = i0; i >= iFloor; i--) {
    if (samps[i] >= AUDIO_THRESH) {
      const audioEndT = i / len * S.duration;
      return Math.min(t, audioEndT + 0.06);  // keep 60ms tail of prev word
    }
  }
  return floor;  // all silence back to floor
}

// Scan FORWARD from t to where audio next occurs (start of next speech).
// Won't scan later than `ceil`. Returns that audio-onset time − 60ms lead-in.
function findSilenceEndAfter(t, ceil) {
  if (!S || !S.waveform || !S.waveform.length) return t;
  const samps = S.waveform;
  const len   = samps.length;
  const i0    = Math.max(0, Math.floor(t / S.duration * len));
  const iCeil = Math.min(len - 1, Math.floor(ceil / S.duration * len));
  for (let i = i0; i <= iCeil; i++) {
    if (samps[i] >= AUDIO_THRESH) {
      const audioStartT = i / len * S.duration;
      return Math.max(t, audioStartT - 0.06);  // keep 60ms before next word
    }
  }
  return ceil;  // all silence up to ceil
}

/* ── Load & build ── */
async function load() {
  const r = await fetch('/api/state');
  S = await r.json();
  if (S.no_video) throw new Error('no_video');  // triggers upload UI in .catch()
  if (!S.split_points) S.split_points = [];
  if (!S.words) S.words = [];
  if (S.undo_stack && S.undo_stack.length) {
    // Keep only snapshot-format entries (have .cuts); drop legacy inverse-op entries.
    UNDO_STACK.push(...S.undo_stack.filter(e => e && e.cuts));
    setUndoHint(UNDO_STACK.length);
  }
  BASE_PPS = S.duration > 600 ? 35 : S.duration > 300 ? 55 : 80;
  PPS = BASE_PPS * zoomScale;
  try { HIDE_SKIP = localStorage.getItem('hideSkipped') === '1'; } catch (e) {}
  refreshCoords();
  const W = tlFullW();
  document.getElementById('tlInner').style.width = W + 'px';
  document.getElementById('filmstrip').style.width = W + 'px';
  buildRuler(W);
  buildFilmstrip(W);
  buildWaveform(W);
  buildOverlays();
  buildSplitLines();
  buildSegmentTiles();
  buildList();
  buildTranscript();
  updateCount();
  // Show which project is loaded (display name)
  const pn = document.getElementById('projectName');
  if (pn) pn.textContent = S.project_id ? '· ' + (S.project_name || S.project_id) : '';
  // Show external audio indicator if active
  const extBtn = document.getElementById('extAudioBtn');
  if (S.has_external_audio) {
    extBtn.textContent = '🎙 Ext Audio ✓';
    extBtn.style.color = '#27ae60';
    extBtn.title = `External mic active (offset: ${S.ext_audio_offset > 0 ? '+' : ''}${S.ext_audio_offset.toFixed(3)}s)`;
  } else {
    extBtn.textContent = '🎙 Ext Audio';
    extBtn.style.color = '';
  }
  setStatus('Click a word to seek · Select words + Delete to cut · Right-click timeline · ⌘Z undo'
            + (S.has_external_audio ? ' · 🎙 Ext mic active' : ''));
  initExtAudio();   // mute camera mic, sync external audio if present
  initSmoothOnLoad();  // engage the exact baked-audio preview on load (don't sit on the live fallback)
  initCaptions();      // load text-caption overlays for this project
  initImages();        // load image overlays for this project
  initAudio();         // load audio layers (sound effects / music)
  VIDEO_VOL = (S.video_volume != null ? Number(S.video_volume) : 1.0);
  MUTED = (S.muted_segs || []).map(m => ({ start: +m[0], end: +m[1] }));
  applyVideoVolume(); renderVidVolLine();   // main-audio volume line on the video clip
  restoreHideSkipped(); // apply the "hide skipped" transcript preference
  try { applyCropToPreview(); updateCropBtn(); applyAdjustToPreview(); } catch(e) { console.warn('crop preview:', e); }
  // Smooth-preview proxy: if not ready yet, poll and hot-swap when it finishes building;
  // if already cached, force both players to fully pre-buffer it for seamless swaps.
  if (!S.proxy_ready) pollProxy();
  else ensureProxyPreload();
  // Load silence regions in background (takes a few seconds — ffmpeg pass)
  loadSilences();
}

/* ── Smooth-preview proxy hot-swap ── */
function pollProxy() {
  if (!S || S.no_video) return;
  fetch('/api/proxy-status').then(r => r.json()).then(d => {
    if (d.ready) swapToProxy();
    else setTimeout(pollProxy, 4000);
  }).catch(() => setTimeout(pollProxy, 6000));
}
function swapToProxy() {
  if (_vidA.dataset.proxy === '1') return;        // already on proxy
  const t = video.currentTime, playing = !video.paused;
  _bufTarget = -1;
  // proxy is small (≈40MB) → fully pre-buffer BOTH players so forward playback after
  // a buffer-swap is instant (no fetch/decode ramp-up).
  [_vidA, _vidB].forEach(v => { v.dataset.proxy = '1'; v.preload = 'auto'; v.src = '/video?proxy=1'; });
  video.addEventListener('loadedmetadata', () => {
    try { video.currentTime = t; } catch(e){}
    if (playing) video.play().catch(() => {});
    try { applyCropToPreview(); } catch(e) {}
  }, { once: true });
  setStatus('Smooth preview ready ✓');
}

// If the proxy is already cached at load, point BOTH players at the decodable proxy
// (NOT the bare /video, which can serve the original .mov that Chrome's demuxer rejects)
// and fully pre-buffer it so the first frame shows and forward playback is instant.
function ensureProxyPreload() {
  if (!(S && S.proxy_ready)) return;
  [_vidA, _vidB].forEach(v => { v.dataset.proxy = '1'; v.preload = 'auto'; v.src = '/video?proxy=1'; v.load(); });
  // Nudge a frame onto the active element so it isn't black before first play.
  const showFrame = () => { try { video.currentTime = video.currentTime || 0.04; } catch(e){} try { applyCropToPreview(); } catch(e){} };
  if (video.readyState >= 1) showFrame();
  else video.addEventListener('loadedmetadata', showFrame, { once:true });
}

/* ── Timeline builders ── */
function buildRuler(W) {
  const el = document.getElementById('ruler');
  el.innerHTML = '';
  el.style.width = W + 'px';
  const total = dispDur();               // edited length when collapsed
  const major = total > 300 ? 30 : total > 120 ? 10 : 5;
  const minor = major / 5;
  // Axis is the DISPLAYED timeline: ticks are evenly spaced, labelled in final-cut time.
  for (let t = 0; t <= total + 0.01; t += minor) {
    const x = t * PPS + TL_PAD;
    const isMaj = Math.round(t / minor) % 5 === 0;
    const tick = document.createElement('div');
    tick.className = 'r-tick' + (isMaj ? ' major' : '');
    tick.style.left = x + 'px';
    el.appendChild(tick);
    if (isMaj) {
      const lbl = document.createElement('div');
      lbl.className = 'r-label';
      lbl.style.left = x + 'px';
      lbl.textContent = fmtS(t);
      el.appendChild(lbl);
    }
  }
}

function buildFilmstrip(W) {
  const strip = document.getElementById('filmstrip');
  strip.querySelectorAll('.thumb').forEach(t => t.remove());   // clear on re-render (zoom / hide toggle)
  const intv = S.thumb_interval;
  for (let i = 1; i <= S.thumb_count; i++) {
    const a = (i-1) * intv, b = i * intv;
    const left = X(a), w = HIDE_SKIP ? Xw(a, b) : Math.ceil(intv * PPS);
    if (HIDE_SKIP && w < 0.5) continue;   // tile lies entirely inside a cut → gone
    const img = document.createElement('img');
    img.className = 'thumb';
    img.src = `/thumbs/thumb${String(i).padStart(4,'0')}.jpg`;
    img.style.left = left + 'px';
    img.style.width = w + 'px';
    strip.appendChild(img);
  }
}

function buildWaveform(W) {
  // Draw at BASE resolution only — CSS width stretches it on zoom.
  // Avoids freezing the browser when W is huge (e.g. 500k px at high zoom).
  // Browsers cap a canvas dimension at ~32767px; a long video (e.g. 36 min) would
  // exceed that and the canvas fails to paint (renders as a broken/blank box). Cap
  // the INTERNAL width and let CSS stretch it to the displayed width.
  const MAX_CANVAS_W = 32000;   // just under the browser's ~32767px canvas limit — keeps it crisp
  const canvas = document.getElementById('wfCanvas');
  const total  = dispDur();
  const BASE_W = Math.max(1, Math.min(MAX_CANVAS_W, Math.ceil(total * BASE_PPS)));
  canvas.width  = BASE_W;
  canvas.height = 68;
  canvas.style.width  = Math.ceil(total * PPS) + 'px';   // content width (leading pad added via margin)
  canvas.style.marginLeft = TL_PAD + 'px';
  canvas.style.height = '68px';

  const ctx = canvas.getContext('2d');
  const H = 68, mid = H / 2;
  ctx.fillStyle = '#1e1e1e'; ctx.fillRect(0, 0, BASE_W, H);

  const samps = S.waveform, dur = S.duration || 1;
  ctx.fillStyle = '#5a8fcc';
  if (!HIDE_SKIP) {
    const samplesPerPx = samps.length / BASE_W;
    for (let x = 0; x < BASE_W; x++) {
      const i0 = Math.floor(x * samplesPerPx);
      const i1 = Math.max(i0 + 1, Math.ceil((x + 1) * samplesPerPx));
      let peak = 0;
      for (let i = i0; i < Math.min(i1, samps.length); i++) if (samps[i] > peak) peak = samps[i];
      const amp = peak * (mid - 3);
      ctx.fillRect(x, mid - amp, 1, amp * 2 || 1);
    }
  } else {
    // Collapsed: each output pixel maps to an EDITED-time slice; sample the original
    // waveform at the corresponding original time so cuts are skipped seamlessly.
    const secPerPx = total / BASE_W;
    for (let x = 0; x < BASE_W; x++) {
      const o0 = origFromEdited(x * secPerPx), o1 = origFromEdited((x + 1) * secPerPx);
      const i0 = Math.floor((o0 / dur) * samps.length);
      const i1 = Math.max(i0 + 1, Math.ceil((o1 / dur) * samps.length));
      let peak = 0;
      for (let i = i0; i < Math.min(i1, samps.length); i++) if (samps[i] > peak) peak = samps[i];
      const amp = peak * (mid - 3);
      ctx.fillRect(x, mid - amp, 1, amp * 2 || 1);
    }
  }
}

function updateWaveformWidth(W) {
  // On zoom: just CSS-stretch the existing canvas — no redraw.
  const canvas = document.getElementById('wfCanvas');
  canvas.style.width = Math.ceil(dispDur() * PPS) + 'px';   // content width; leading pad via margin
  canvas.style.marginLeft = TL_PAD + 'px';
}

/* ── Silence overlay helpers ── */
function silencesInKept() {
  // Return silence regions that fall within kept (non-cut) portions
  const mergedCuts = getMergedCuts();
  return ALL_SILENCES.filter(s => {
    // Skip if silence is fully inside an active cut region
    const inCut = mergedCuts.some(c => s.start >= c.start - 0.05 && s.end <= c.end + 0.05);
    if (inCut) return false;
    // Skip if no usable content remains after applying buffer
    return (s.end - s.start) > SILENCE_BUFFER * 2 + 0.05;
  });
}

function buildSilenceOverlays() {
  document.querySelectorAll('.silence-ov').forEach(e => e.remove());
  const inner = document.getElementById('tlInner');
  for (const s of silencesInKept()) {
    const el = document.createElement('div');
    el.className = 'silence-ov';
    el.style.top  = (document.getElementById('wfCanvas').offsetTop) + 'px';
    el.style.left = X(s.start) + 'px';
    el.style.width = Math.max(2, Xw(s.start, s.end)) + 'px';
    el.dataset.start = s.start;
    el.dataset.end   = s.end;
    inner.appendChild(el);
  }
}

function getSilenceAtTime(t) {
  // Return the silence region (if any) that contains time t
  return silencesInKept().find(s => t >= s.start && t <= s.end) || null;
}

async function loadSilences() {
  try {
    const r = await fetch('/api/silence-regions');
    const d = await r.json();
    ALL_SILENCES = d.silences || [];
    buildSilenceOverlays();
  } catch(e) { console.warn('Could not load silence regions:', e); }
}

function buildOverlays() {
  document.querySelectorAll('.cut-merged, .trim-handle').forEach(e => e.remove());
  const strip = document.getElementById('filmstrip');
  // Collapsed view: cut regions have zero width — don't draw the black overlays or
  // trim handles at all (that IS the "black spaces gone" behaviour).
  if (HIDE_SKIP) { buildSilenceOverlays(); return; }
  const merged = getMergedCuts();
  for (const mc of merged) {
    // Gray overlay covering the merged deleted region
    const el = document.createElement('div');
    el.className = 'cut-merged';
    el.style.left  = (mc.start * PPS) + 'px';
    el.style.width = Math.max(2, (mc.end - mc.start) * PPS) + 'px';
    strip.appendChild(el);

    // Left trim handle — sits at mc.start (right edge of left kept clip)
    // Drag RIGHT to extend left kept clip into deleted area
    const lh = document.createElement('div');
    lh.className = 'trim-handle handle-left';
    lh.style.left = (mc.start * PPS) + 'px';
    lh.dataset.side = 'left';
    lh.dataset.mcStart = mc.start;
    lh.dataset.mcEnd   = mc.end;
    lh.title = 'Drag right to extend clip';
    strip.appendChild(lh);

    // Right trim handle — sits at mc.end (left edge of right kept clip)
    // Drag LEFT to extend right kept clip into deleted area
    const rh = document.createElement('div');
    rh.className = 'trim-handle handle-right';
    rh.style.left = (mc.end * PPS) + 'px';
    rh.dataset.side = 'right';
    rh.dataset.mcStart = mc.start;
    rh.dataset.mcEnd   = mc.end;
    rh.title = 'Drag left to extend clip';
    strip.appendChild(rh);
  }
  buildSilenceOverlays();
}

function buildSplitLines() {
  document.querySelectorAll('.split-line').forEach(e => e.remove());
  const strip = document.getElementById('filmstrip');
  for (const sp of (S.split_points || [])) {
    const el = document.createElement('div');
    el.className = 'split-line';
    el.id = 'sp_' + sp.id;
    el.style.left = X(sp.pos) + 'px';
    strip.appendChild(el);
  }
}

/* Build clickable tiles for every kept segment (split-aware) */
function segBounds() {
  const bounds = new Set([0, S.duration]);
  for (const c of S.cuts) if (c.active) { bounds.add(c.start); bounds.add(c.end); }
  for (const sp of (S.split_points || [])) bounds.add(sp.pos);
  return [...bounds].sort((a, b) => a - b);
}

function buildSegmentTiles() {
  refreshCoords();   // cuts may have changed — recompute the collapse mapping
  document.querySelectorAll('.seg-tile').forEach(e => e.remove());
  selectedSeg = null;
  const strip = document.getElementById('filmstrip');
  const sorted = segBounds();
  for (let i = 0; i < sorted.length - 1; i++) {
    const s = sorted[i], e = sorted[i + 1];
    const isCut = S.cuts.some(c => c.active && c.start <= s + 0.05 && c.end >= e - 0.05);
    if (isCut) continue;  // don't place tiles over cut/grayed regions
    const tile = document.createElement('div');
    tile.className = 'seg-tile';
    tile.dataset.start = s;
    tile.dataset.end   = e;
    tile.style.left  = X(s) + 'px';
    tile.style.width = Xw(s, e) + 'px';
    if (typeof isSegMuted === 'function' && isSegMuted(s, e)) tile.classList.add('muted');
    tile.addEventListener('click', ev => {
      // Single-click: select this segment (also lets click bubble to seek)
      document.querySelectorAll('.seg-tile').forEach(t => t.classList.remove('sel'));
      if (selectedSeg && Math.abs(selectedSeg.start - s) < 0.05) {
        selectedSeg = null;  // click same tile again → deselect
      } else {
        selectedSeg = { start: s, end: e };
        tile.classList.add('sel');
      }
    });
    strip.appendChild(tile);   // right-click handled by tlScroll's contextmenu (full menu incl. Split/Delete/Mute)
  }
  reflowCollapsed();   // in hide-skipped mode, a cut change resizes the whole collapsed timeline
}

// When collapsed, a cut edit changes the total edited length → rebuild the width-dependent
// layers (ruler / filmstrip / waveform / overlay tracks). Guarded against re-entry.
let _reflowing = false;
function reflowCollapsed() {
  if (!HIDE_SKIP || _reflowing || !S) return;
  _reflowing = true;
  const W = tlFullW();
  document.getElementById('tlInner').style.width = W + 'px';
  document.getElementById('filmstrip').style.width = W + 'px';
  buildRuler(W); buildFilmstrip(W); buildWaveform(W);
  renderCapTrack(); renderImgTrack(); renderAudioTrack(); renderVidVolLine();
  _reflowing = false;
}

function clearSegSelection() {
  document.querySelectorAll('.seg-tile').forEach(t => t.classList.remove('sel'));
  selectedSeg = null;
}

function buildList() {
  const list = document.getElementById('cutsList');
  list.innerHTML = '';
  // Only show auto-detected cuts (not manual transcript cuts)
  const sorted = [...S.cuts].filter(c => c.type !== 'manual').sort((a,b) => a.start - b.start);
  for (const cut of sorted) {
    const dur = (cut.end - cut.start).toFixed(1);
    const div = document.createElement('div');
    div.className = 'cut-item' + (cut.active ? '' : ' off');
    div.id = 'li_' + cut.id;
    div.innerHTML = `
      <div class="ci-info">
        <div class="ci-time">${fmtS(cut.start)} → ${fmtS(cut.end)} · ${dur}s</div>
        <div class="ci-text">${cut.label || '—'}</div>
      </div>
      <span class="ci-badge ${badgeClass(cut.type)}">${cut.type}</span>
      <div class="ci-actions">
        <button class="ci-btn accept ${cut.active ? 'active' : ''}"
                onmousedown="event.stopPropagation();setCut('${cut.id}',true)">✂ Cut</button>
        <button class="ci-btn reject ${!cut.active ? 'active' : ''}"
                onmousedown="event.stopPropagation();setCut('${cut.id}',false)">✕ Keep</button>
      </div>`;
    div.onclick = () => seekTo(cut.start);
    list.appendChild(div);
  }
}

function updateCount() {
  const active = S.cuts.filter(c => c.active);
  const autoActive = active.filter(c => c.type !== 'manual');
  const secs = active.reduce((s,c) => s+(c.end-c.start), 0);
  const badge = document.getElementById('cutsBadge');
  badge.textContent = autoActive.length;
  badge.className = 'badge-pill' + (autoActive.length === 0 ? ' zero' : '');
  updateFinalDuration();
  if (typeof onEditChanged === 'function') onEditChanged();   // re-render smooth preview (debounced)
}

// Show the final edited length (original − removed). Uses merged cuts so
// overlapping/adjacent cuts aren't double-counted.
function fmtMS(s) {
  s = Math.max(0, Math.round(s));
  const m = Math.floor(s / 60), sec = s % 60;
  return `${m}:${String(sec).padStart(2,'0')}`;
}
function updateFinalDuration() {
  if (!S || !S.duration) return;
  let removed = 0;
  for (const mc of getMergedCuts()) removed += (mc.end - mc.start);
  const finalDur = Math.max(0, S.duration - removed);
  const el = document.getElementById('finalDuration');
  if (el) el.textContent = `✂ ${fmtMS(finalDur)} final  ·  ${fmtMS(S.duration)} orig  ·  −${fmtMS(removed)}`;
}

/* ── Cuts sidebar toggle ── */
function toggleCutsPanel() {
  const sidebar = document.getElementById('cutsSidebar');
  const btn = document.getElementById('cutsToggle');
  sidebar.classList.toggle('open');
  btn.classList.toggle('active', sidebar.classList.contains('open'));
}

// Hide/show the skipped (deleted) parts in the transcript so it reads as the final cut.
function applyHideSkipped(on) {
  const btn = document.getElementById('hideSkipBtn');
  document.body.classList.toggle('hide-skipped', on);   // covers transcript + timeline
  if (btn) { btn.classList.toggle('active', on); btn.textContent = on ? '👁 Show skipped' : '👁 Hide skipped'; }
  HIDE_SKIP = on;
  if (S) {
    refreshCoords();
    rebuildTimeline();          // redraw ruler/filmstrip/waveform/overlays/tracks in the new coord space
    buildWaveform(Math.ceil(dispDur() * PPS));   // waveform needs a real redraw (not just CSS stretch)
    // snap the playhead to the current position in the new coordinate space
    try { document.getElementById('playhead').style.left = X(_origTime || 0) + 'px'; } catch (e) {}
  }
}
function toggleHideSkipped() {
  const on = !document.body.classList.contains('hide-skipped');
  applyHideSkipped(on);
  try { localStorage.setItem('hideSkipped', on ? '1' : '0'); } catch (e) {}
}
function restoreHideSkipped() {
  let on = false;
  try { on = localStorage.getItem('hideSkipped') === '1'; } catch (e) {}
  applyHideSkipped(on);
}

/* ── Transcript ── */
function buildTranscript() {
  const scroll = document.getElementById('transcriptScroll');
  scroll.innerHTML = '';
  wordSpans = [];
  const words = S.words;
  if (!words.length) {
    scroll.textContent = '(No transcript available)';
    return;
  }

  // Group into paragraphs by long pauses (>1.5s)
  let paraEl = document.createElement('span');
  paraEl.className = 'tl-paragraph';
  scroll.appendChild(paraEl);

  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    const span = document.createElement('span');
    span.className = 'word';
    span.dataset.start = w.start;
    span.dataset.end   = w.end;
    span.dataset.idx   = i;
    span.textContent   = w.raw || w.word || '';
    span.addEventListener('click', () => { seekOriginal(w.start); scrollTimelineToTime(w.start); });
    span.addEventListener('contextmenu', (e) => showWordMenu(e, w));
    paraEl.appendChild(span);
    wordSpans.push(span);

    // Space or paragraph break
    const nextGap = (i < words.length - 1) ? (words[i+1].start - w.end) : 0;
    if (nextGap > 1.5 && i < words.length - 1) {
      // New paragraph
      paraEl = document.createElement('span');
      paraEl.className = 'tl-paragraph';
      scroll.appendChild(paraEl);
    } else if (i < words.length - 1) {
      paraEl.appendChild(document.createTextNode(' '));
    }
  }

  // Apply strikes for any existing cuts
  refreshTranscriptStrikes();
}

function refreshTranscriptStrikes() {
  for (let i = 0; i < wordSpans.length; i++) {
    const span = wordSpans[i];
    const wStart = parseFloat(span.dataset.start);
    const wEnd   = parseFloat(span.dataset.end);
    // A word is struck only if an active cut fully contains it (start ≤ wStart AND end ≥ wEnd)
    const inCut = S.cuts.some(c => c.active && c.start <= wStart + 0.01 && c.end >= wEnd - 0.01);
    span.classList.toggle('struck', inCut);
  }
}

/* Find exactly which word spans the user has selected.
   Uses range anchor/focus nodes so boundary words are not over-included. */
function getSelectedWordSpans(sel) {
  if (!sel || sel.isCollapsed) return [];
  const range = sel.getRangeAt(0);

  // Walk up from a node to find the nearest .word ancestor (or null)
  function closestWord(node) {
    let n = node;
    while (n && n !== document.getElementById('transcriptScroll')) {
      if (n.classList && n.classList.contains('word')) return n;
      n = n.parentNode;
    }
    return null;
  }

  let startIdx = -1, endIdx = -1;
  const startWord = closestWord(range.startContainer);
  const endWord   = closestWord(range.endContainer);

  if (startWord) startIdx = wordSpans.indexOf(startWord);
  if (endWord)   endIdx   = wordSpans.indexOf(endWord);

  // If selection starts in a space (text node between words), find next word inside range
  if (startIdx === -1) {
    for (let i = 0; i < wordSpans.length; i++) {
      if (range.intersectsNode(wordSpans[i])) { startIdx = i; break; }
    }
  }
  // If selection ends in a space, find prev word inside range
  if (endIdx === -1) {
    for (let i = wordSpans.length - 1; i >= 0; i--) {
      if (range.intersectsNode(wordSpans[i])) { endIdx = i; break; }
    }
  }

  if (startIdx === -1 || endIdx === -1) return [];
  const lo = Math.min(startIdx, endIdx);
  const hi = Math.max(startIdx, endIdx);
  return wordSpans.slice(lo, hi + 1);
}

/* Binary-search: find word index at time t */
function findWordAt(t) {
  const words = S.words;
  let lo = 0, hi = words.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (words[mid].end < t)        lo = mid + 1;
    else if (words[mid].start > t) hi = mid - 1;
    else return mid;
  }
  return -1;
}

function highlightWord(t) {
  const idx = findWordAt(t);
  if (idx === lastHlIdx) return;
  if (lastHlIdx >= 0 && wordSpans[lastHlIdx]) wordSpans[lastHlIdx].classList.remove('hl');
  lastHlIdx = idx;
  if (idx >= 0 && wordSpans[idx]) {
    const span = wordSpans[idx];
    if (!span.classList.contains('struck')) span.classList.add('hl');
    // Scroll transcript to keep current word in view
    span.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
}

/* ── Restore a time range: punch a hole in all overlapping active cuts ── */
async function restoreRange(selStart, selEnd) {
  pushUndoSnapshot();
  const newCuts = [];
  let anyFound = true; let iterations = 0;
  while (anyFound && iterations++ < 10) {
    const toRestore = S.cuts.filter(c =>
      c.active && c.start < selEnd + 0.05 && c.end > selStart - 0.05
    );
    if (!toRestore.length) { anyFound = false; break; }
    for (const cut of toRestore) {
      S.cuts = S.cuts.filter(c => c.id !== cut.id);
      fetch('/api/remove-cut', { method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({id: cut.id}) });
      if (cut.start < selStart - 0.05) {
        const nc = {id:'manual_'+Date.now()+'_L', start:cut.start, end:selStart,
                    label:cut.label, type:cut.type, active:true};
        newCuts.push(nc); S.cuts.push(nc);
      }
      if (cut.end > selEnd + 0.05) {
        const nc = {id:'manual_'+Date.now()+'_R', start:selEnd, end:cut.end,
                    label:cut.label, type:cut.type, active:true};
        newCuts.push(nc); S.cuts.push(nc);
      }
    }
  }
  for (const nc of newCuts) {
    await fetch('/api/delete-segment', { method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({id:nc.id, start:nc.start, end:nc.end, label:nc.label}) });
  }
  setUndoHint(UNDO_STACK.length);
  refreshTranscriptStrikes();
  buildOverlays();
  buildSegmentTiles();
  buildList();
  updateCount();
}

/* ── Trim handle drag — extend clip into deleted area ── */
let _trimDrag = null;

document.getElementById('filmstrip').addEventListener('mousedown', e => {
  const h = e.target.closest('.trim-handle');
  if (!h) return;
  e.preventDefault(); e.stopPropagation();

  const sc    = document.getElementById('tlScroll');
  const strip = document.getElementById('filmstrip');
  const mcStart = parseFloat(h.dataset.mcStart);
  const mcEnd   = parseFloat(h.dataset.mcEnd);
  const side    = h.dataset.side;   // 'left' | 'right'

  h.classList.add('dragging');

  // Visual preview line while dragging
  const preview = document.createElement('div');
  preview.style.cssText = 'position:absolute;top:0;height:68px;width:2px;background:#3d9be9;opacity:0.85;pointer-events:none;z-index:20;';
  preview.style.left = h.style.left;
  strip.appendChild(preview);

  _trimDrag = { side, mcStart, mcEnd, handle: h, previewEl: preview, currentT: null };

  // Min epsilon (in seconds) = ~1px at current zoom, so the clamp scales with zoom
  const EPS = 1 / PPS;

  function onMove(me) {
    if (!_trimDrag) return;
    const rect = sc.getBoundingClientRect();
    const x = me.clientX - rect.left + sc.scrollLeft;
    let t = Math.max(0, Math.min(x / PPS, S.duration));
    // Allow both directions — only prevent consuming the entire gray section
    if (side === 'left')  t = Math.min(t, mcEnd - EPS);    // can't eat past the right edge
    else                  t = Math.max(t, mcStart + EPS);   // can't eat past the left edge
    _trimDrag.currentT = t;
    preview.style.left = (t * PPS) + 'px';
  }

  async function onUp() {
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup', onUp);
    if (!_trimDrag) return;
    const drag = _trimDrag;
    _trimDrag = null;
    drag.handle.classList.remove('dragging');
    drag.previewEl.remove();
    if (drag.currentT == null) return;

    // Commit threshold = 2px worth of time at current zoom → precise when zoomed in,
    // still ignores accidental sub-pixel jitter when zoomed out.
    const MIN = 2 / PPS;
    pushUndoSnapshot();   // capture state before this trim gesture

    if (drag.side === 'left') {
      if (drag.currentT > drag.mcStart + MIN) {
        // Drag right into gray → restore [mcStart, currentT] (extend clip)
        await restoreRange(drag.mcStart, drag.currentT);
        setStatus(`Extended clip by ${(drag.currentT - drag.mcStart).toFixed(2)}s`);
      } else if (drag.currentT < drag.mcStart - MIN) {
        // Drag left out of gray → crop clip (add [currentT, mcStart] to deleted)
        const id = 'manual_' + Date.now();
        const cut = {id, start:drag.currentT, end:drag.mcStart, label:'trim', type:'manual', active:true};
        S.cuts.push(cut);
        await fetch('/api/delete-segment', {method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({id, start:drag.currentT, end:drag.mcStart, label:'trim'})});
        refreshTranscriptStrikes();
        buildOverlays(); buildSegmentTiles(); buildList(); updateCount();
        setStatus(`Cropped ${(drag.mcStart - drag.currentT).toFixed(2)}s`);
      }
    } else {
      if (drag.currentT < drag.mcEnd - MIN) {
        // Drag left into gray → restore [currentT, mcEnd] (extend clip)
        await restoreRange(drag.currentT, drag.mcEnd);
        setStatus(`Extended clip by ${(drag.mcEnd - drag.currentT).toFixed(2)}s`);
      } else if (drag.currentT > drag.mcEnd + MIN) {
        // Drag right out of gray → crop clip (add [mcEnd, currentT] to deleted)
        const id = 'manual_' + Date.now();
        const cut = {id, start:drag.mcEnd, end:drag.currentT, label:'trim', type:'manual', active:true};
        S.cuts.push(cut);
        await fetch('/api/delete-segment', {method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({id, start:drag.mcEnd, end:drag.currentT, label:'trim'})});
        refreshTranscriptStrikes();
        buildOverlays(); buildSegmentTiles(); buildList(); updateCount();
        setStatus(`Cropped ${(drag.currentT - drag.mcEnd).toFixed(2)}s`);
      }
    }
  }

  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', onUp);
});

/* ── Delete/restore selected transcript words ── */
// Re-apply a text selection spanning firstSpan..lastSpan so the user can
// immediately press Delete again to perform the opposite (toggle) action.
function reselectWords(firstSpan, lastSpan) {
  if (!firstSpan || !lastSpan) return;
  try {
    const range = document.createRange();
    range.setStartBefore(firstSpan);
    range.setEndAfter(lastSpan);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
  } catch (e) { /* selection failed — non-fatal */ }
}

function deleteSelectedWords() {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed) return;
  const transcript = document.getElementById('transcriptScroll');
  if (!transcript.contains(sel.anchorNode)) return;

  const selected = getSelectedWordSpans(sel);
  if (selected.length === 0) return;
  // Remember the selection bounds so we can re-highlight after the action
  const reselFirst = selected[0];
  const reselLast  = selected[selected.length - 1];
  sel.removeAllRanges();

  const allStruck = selected.every(sp => sp.classList.contains('struck'));

  if (allStruck) {
    // ── Restore: punch a hole in overlapping cuts for exactly the selected range ──
    // Mirror the same boundaries used when cutting
    const firstSelIdx = wordSpans.indexOf(selected[0]);
    const prevSelSp   = firstSelIdx > 0 ? wordSpans[firstSelIdx - 1] : null;
    const selStart    = prevSelSp ? parseFloat(prevSelSp.dataset.end) : parseFloat(selected[0].dataset.start);
    const lastSel  = selected[selected.length - 1];
    const lastIdx  = wordSpans.indexOf(lastSel);
    const nextSp   = wordSpans[lastIdx + 1];
    const selEnd   = nextSp ? parseFloat(nextSp.dataset.start) : S.duration;

    restoreRange(selStart, selEnd);
    setStatus(`Restored ${selected.length} word${selected.length > 1 ? 's' : ''} — press Delete again to re-cut`);
    reselectWords(reselFirst, reselLast);

  } else {
    // ── Cut: add a new manual cut for the selected range ──
    // Use the actual audio waveform to find the true silence gap on each side,
    // so we remove ALL silence between prev-word and next-word (Whisper word
    // timestamps are often early/late, which used to leave silence behind).
    const firstSpan = selected[0];
    const lastSpan  = selected[selected.length - 1];
    const firstIdx  = wordSpans.indexOf(firstSpan);
    const lastIdx   = wordSpans.indexOf(lastSpan);
    const prevSpan  = firstIdx > 0 ? wordSpans[firstIdx - 1] : null;
    const nextSpan  = wordSpans[lastIdx + 1];

    const aStart = parseFloat(firstSpan.dataset.start);
    const bEnd   = parseFloat(lastSpan.dataset.end);

    // START: if deleting the FIRST word(s), cut from 0 so ALL leading silence is
    // removed up to the first kept word. Otherwise, from where prev-word's audio ends.
    const floor = prevSpan ? parseFloat(prevSpan.dataset.start) : 0;
    const start = prevSpan ? findSilenceStartBefore(aStart, floor) : 0;

    // END: if deleting the LAST word(s), cut to the end (trailing silence too).
    // Otherwise, to where next-word's audio actually begins.
    const ceil = nextSpan ? parseFloat(nextSpan.dataset.end) : S.duration;
    const end  = nextSpan ? findSilenceEndAfter(bEnd, ceil) : S.duration;

    const label = selected.slice(0, 6).map(s => s.textContent).join(' ')
                  + (selected.length > 6 ? '...' : '');

    pushUndoSnapshot();
    const id  = 'manual_' + Date.now();
    const cut = { id, start, end, label, type: 'manual', active: true };
    S.cuts.push(cut);

    fetch('/api/delete-segment', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, start, end, label }),
    });

    refreshTranscriptStrikes();
    buildOverlays();
    updateCount();
    setStatus(`Cut ${selected.length} word${selected.length > 1 ? 's' : ''}: "${label}" — press Delete again to restore`);
    reselectWords(reselFirst, reselLast);
  }
}

/* ── Cut toggle (auto-detected cuts) ── */
async function setCut(id, active, skipUndo=false) {
  const cut = S.cuts.find(c => c.id === id);
  if (!cut || cut.active === active) return;
  if (!skipUndo) pushUndoSnapshot();   // snapshot BEFORE flipping the cut
  cut.active = active;
  await fetch('/api/cuts', { method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({id, active}) });
  const li = document.getElementById('li_' + id);
  if (li) {
    li.className = 'cut-item' + (active ? '' : ' off');
    li.querySelector('.ci-btn.accept').classList.toggle('active', active);
    li.querySelector('.ci-btn.reject').classList.toggle('active', !active);
  }
  refreshTranscriptStrikes();
  buildOverlays();
  buildSegmentTiles();
  updateCount();
}
function toggle(id) { const c = S.cuts.find(x=>x.id===id); if(c) setCut(id,!c.active); }

/* ── Context menu (timeline right-click) ── */
function tlPosFromEvent(e) {
  const sc = document.getElementById('tlScroll');
  return Tx(e.clientX - sc.getBoundingClientRect().left + sc.scrollLeft);
}
function findSegmentAt(pos) {
  const bounds = new Set([0, S.duration]);
  for (const c of S.cuts) if (c.active) { bounds.add(c.start); bounds.add(c.end); }
  for (const sp of (S.split_points||[])) bounds.add(sp.pos);
  const sorted = [...bounds].sort((a,b)=>a-b);
  for (let i=0; i<sorted.length-1; i++) {
    const s=sorted[i], e=sorted[i+1];
    if (pos >= s && pos <= e) {
      const isCut = S.cuts.some(c => c.active && c.start <= s+0.01 && c.end >= e-0.01);
      return {start:s, end:e, isKept:!isCut};
    }
  }
  return null;
}
function hideCtx() { document.getElementById('ctxMenu')?.remove(); }
function showCtxMenu(screenX, screenY, pos, hovSilence) {
  hideCtx();
  const seg = findSegmentAt(pos);
  const menu = document.createElement('div');
  menu.id = 'ctxMenu'; menu.className = 'ctx-menu';
  const vw = window.innerWidth, vh = window.innerHeight;
  menu.style.left = Math.min(screenX, vw-210) + 'px';
  menu.style.top  = Math.min(screenY, vh-140) + 'px';
  let html = '';
  if (hovSilence) {
    const cs = (hovSilence.start + SILENCE_BUFFER).toFixed(4);
    const ce = (hovSilence.end   - SILENCE_BUFFER).toFixed(4);
    html += `<div class="ctx-item danger" onclick="doCutSilence(${cs},${ce});hideCtx()">🔇&nbsp; Cut silence &nbsp;<span style="color:#555;font-size:10px">${fmtS(hovSilence.start)}–${fmtS(hovSilence.end)}</span></div>`;
  }
  if (seg && seg.isKept) {
    if (html) html += '<div class="ctx-sep"></div>';
    const muted = (typeof isSegMuted === 'function') && isSegMuted(seg.start, seg.end);
    html += `
      <div class="ctx-item" onclick="doSplit(${pos.toFixed(4)});hideCtx()">✂️&nbsp; Split here &nbsp;<span style="color:#555;font-size:10px">${fmtS(pos)}</span></div>
      <div class="ctx-item" onclick="toggleSegMute(${seg.start.toFixed(4)},${seg.end.toFixed(4)});hideCtx()">${muted ? '🔊&nbsp; Unmute segment' : '🔇&nbsp; Mute segment'}</div>
      <div class="ctx-sep"></div>
      <div class="ctx-item danger" onclick="doDelete(${seg.start.toFixed(4)},${seg.end.toFixed(4)});hideCtx()">🗑&nbsp; Delete segment &nbsp;<span style="color:#555;font-size:10px">${fmtS(seg.start)}–${fmtS(seg.end)}</span></div>`;
  } else if (seg && !seg.isKept) {
    if (html) html += '<div class="ctx-sep"></div>';
    html += `<div class="ctx-item restore" onclick="doRestoreSegment(${seg.start.toFixed(4)},${seg.end.toFixed(4)});hideCtx()">↩&nbsp; Restore segment</div>`;
  }
  if (!html) return;
  menu.innerHTML = html;
  menu.addEventListener('mousedown', e => e.stopPropagation());
  document.body.appendChild(menu);
  setTimeout(() => document.addEventListener('mousedown', hideCtx, {once:true}), 10);
}
document.getElementById('tlScroll').addEventListener('contextmenu', e => {
  e.preventDefault();
  const pos = tlPosFromEvent(e);
  // Use the silence already highlighted by mousemove (_hoveredSilence),
  // or fall back to a time-based lookup anywhere on the timeline
  const sil = _hoveredSilence || getSilenceAtTime(pos);
  showCtxMenu(e.clientX, e.clientY, pos, sil);
});

/* ── Cut silence (with buffer) ── */
async function doCutSilence(start, end) {
  pushUndoSnapshot();
  const id  = 'manual_' + Date.now();
  const cut = {id, start, end, label:'silence', type:'silence', active:true};
  S.cuts.push(cut);
  await fetch('/api/delete-segment', { method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({id, start, end, label:'silence'}) });
  refreshTranscriptStrikes();
  buildOverlays();
  buildSegmentTiles();
  updateCount();
  setStatus(`Cut silence ${fmtS(start)}–${fmtS(end)} (60ms buffer applied)`);
}

/* ── Split ── */
async function doSplit(pos) {
  pushUndoSnapshot();
  const id = 'split_' + Date.now();
  S.split_points.push({id, pos});
  await fetch('/api/split', { method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({id, pos}) });
  buildSplitLines();
  buildSegmentTiles();
  setStatus(`Split at ${fmtS(pos)} — click a segment to select, Delete to remove`);
}

/* ── Delete timeline segment (manual) ── */
async function doDelete(start, end) {
  pushUndoSnapshot();
  // Trim leading silence so next clip starts on actual audio
  const trimmedEnd = end < S.duration ? findAudioStart(end) : end;
  end = trimmedEnd;
  const id  = 'manual_' + Date.now();
  const cut = {id, start, end, label:'timeline delete', type:'manual', active:true};
  S.cuts.push(cut);
  await fetch('/api/delete-segment', { method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({id, start, end, label:'timeline delete'}) });
  refreshTranscriptStrikes();
  buildOverlays();
  buildSegmentTiles();
  updateCount();
  setStatus(`Deleted ${fmtS(start)}–${fmtS(end)}`);
}

function doRestoreSegment(start, end) {
  // Can't do exact-match: findSegmentAt slices by all cut/split boundaries,
  // so the visual sub-segment may not align with any single cut's start/end.
  // Find all active cuts that overlap [start, end] and split out remnants.
  const covering = S.cuts.filter(c => c.active && c.start < end - 0.05 && c.end > start + 0.05);
  if (covering.length === 0) return;
  pushUndoSnapshot();

  for (const cut of covering) {
    S.cuts = S.cuts.filter(c => c.id !== cut.id);
    fetch('/api/remove-cut', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: cut.id }),
    });

    if (cut.start < start - 0.05) {
      const lid = 'manual_' + Date.now() + '_L';
      const nc = { id: lid, start: cut.start, end: start, label: cut.label, type: cut.type, active: true };
      S.cuts.push(nc);
      fetch('/api/delete-segment', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: nc.id, start: nc.start, end: nc.end, label: nc.label }),
      });
    }

    if (cut.end > end + 0.05) {
      const rid = 'manual_' + Date.now() + '_R';
      const nc = { id: rid, start: end, end: cut.end, label: cut.label, type: cut.type, active: true };
      S.cuts.push(nc);
      fetch('/api/delete-segment', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: nc.id, start: nc.start, end: nc.end, label: nc.label }),
      });
    }
  }

  setUndoHint(UNDO_STACK.length);
  refreshTranscriptStrikes();
  buildOverlays();
  buildSegmentTiles();
  updateCount();
  setStatus(`Restored ${fmtS(start)}–${fmtS(end)}`);
}

/* ── Undo ── */
async function doUndo() {
  const snap = UNDO_STACK.pop();
  if (!snap) { setStatus('Nothing to undo'); return; }
  setUndoHint(UNDO_STACK.length);
  if (!snap.cuts) { setStatus('Undo'); return; }   // legacy op-format entry — skip safely
  // Restore the exact prior state.
  S.cuts = snap.cuts.map(c => ({...c}));
  S.split_points = (snap.split_points || []).map(s => ({...s}));
  await fetch('/api/replace-state', { method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ cuts: S.cuts, split_points: S.split_points }) });
  refreshTranscriptStrikes();
  buildOverlays();
  buildSplitLines();
  buildSegmentTiles();
  buildList();
  updateCount();
  setStatus('Undid last change');
}

/* ── Encode ── */
async function openExportModal() {
  const modal = document.getElementById('exportModal');
  modal.style.display = 'flex';
  // Fetch suggested filename from server
  const r = await fetch('/api/suggest-filename');
  const d = await r.json();
  document.getElementById('exportPath').value = d.path;
  document.getElementById('exportPath').focus();
}
function closeExportModal() {
  document.getElementById('exportModal').style.display = 'none';
}

async function pickSaveLocation() {
  const currentPath = document.getElementById('exportPath').value.trim();
  const suggestedName = currentPath ? currentPath.split('/').pop() : 'edited.mp4';
  const r = await fetch('/api/pick-save-path?name=' + encodeURIComponent(suggestedName));
  const d = await r.json();
  if (d.path) document.getElementById('exportPath').value = d.path;
}

// Close on backdrop click
document.getElementById('exportModal').addEventListener('click', e => {
  if (e.target === document.getElementById('exportModal')) closeExportModal();
});

function _encUI(running) {
  document.getElementById('encSpinner').style.display   = running ? '' : 'none';
  document.getElementById('encCancelBtn').style.display = running ? '' : 'none';
}
async function cancelEncode() {
  const btn = document.getElementById('encCancelBtn');
  btn.disabled = true; btn.textContent = 'Cancelling…';
  await fetch('/api/cancel-encode', { method:'POST' }).catch(()=>{});
}
async function doEncode() {
  const outPath = document.getElementById('exportPath').value.trim();
  closeExportModal();
  const btn = document.getElementById('encodeBtn');
  btn.disabled = true; btn.textContent = 'Encoding...';
  document.getElementById('status').className = 'statusbar enc';
  const cancelBtn = document.getElementById('encCancelBtn');
  cancelBtn.disabled = false; cancelBtn.textContent = '✕ Cancel Export';
  _encUI(true);
  await fetch('/api/encode', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ out_path: outPath }) });
  const iv = setInterval(async () => {
    const r = await fetch('/api/status');
    const s = await r.json();
    if (s.state === 'done') {
      clearInterval(iv); _encUI(false);
      document.getElementById('status').className = 'statusbar done';
      setStatus('✅ Done! Saved to: ' + s.message);
      btn.disabled = false; btn.textContent = '▶ Export Final Video';
    } else if (s.state === 'error') {
      clearInterval(iv); _encUI(false);
      document.getElementById('status').className = 'statusbar err';
      setStatus('❌ ' + s.message);
      btn.disabled = false; btn.textContent = '▶ Export Final Video';
    } else if (s.state === 'cancelled') {
      clearInterval(iv); _encUI(false);
      document.getElementById('status').className = 'statusbar';
      setStatus('Export cancelled.');
      btn.disabled = false; btn.textContent = '▶ Export Final Video';
    } else {
      setStatus(s.message);
      const m = (s.message || '').match(/(\d+)%/);
      btn.textContent = m ? ('Encoding ' + m[1] + '%') : 'Encoding...';
    }
  }, 800);
}

/* ══════════ Text caption overlays ══════════
   Captions are a free layer on top of the video: unlimited, may overlap, positioned as
   fractions of the visible frame, timed on the ORIGINAL timeline (playback maps to _origTime
   in both smooth and non-smooth modes). The live overlay is HTML/CSS; export burns them via PIL. */
let CAPS = [];
let selectedCapId = null;
let _capSaveTimer = null;
let _lastCapSig = '';
let _capDrag = null, _capBlk = null;
let _capEditing = null;   // id of the caption currently being edited inline (don't re-render it)

// The visible video rectangle inside the stage (letterboxed when uncropped; the whole
// stage when cropped) — the coordinate reference captions map to, same as the export.
function captionFrameRect() {
  const stage = document.getElementById('videoStage');
  const sr = stage.getBoundingClientRect();
  if (S && S.crop) return { left: 0, top: 0, w: sr.width, h: sr.height };
  const va = (S && S.video_w && S.video_h) ? S.video_w / S.video_h : 16 / 9;
  let w = sr.width, h = sr.width / va;
  if (h > sr.height) { h = sr.height; w = sr.height * va; }
  return { left: (sr.width - w) / 2, top: (sr.height - h) / 2, w, h };
}
function layoutCaptionLayer() {
  const L = document.getElementById('captionLayer'); if (!L) return;
  const r = captionFrameRect();
  L.style.left = r.left + 'px'; L.style.top = r.top + 'px';
  L.style.width = r.w + 'px'; L.style.height = r.h + 'px';
}
// Is this caption animated (has an end keyframe that differs from the start)?
function capAnimated(c) {
  return (c.x2 != null && Math.abs(c.x2 - c.x) > 1e-4) ||
         (c.y2 != null && Math.abs(c.y2 - c.y) > 1e-4) ||
         (c.size2 != null && Math.abs(c.size2 - c.size) > 1e-4);
}
// Interpolated position/size at time t (for playback of an animated caption).
function capValuesAt(c, t) {
  let x = c.x, y = c.y, size = c.size;
  if (capAnimated(c) && c.end > c.start) {
    let p = (t - c.start) / (c.end - c.start); p = Math.max(0, Math.min(1, p));
    if (c.x2 != null) x = c.x + (c.x2 - c.x) * p;
    if (c.y2 != null) y = c.y + (c.y2 - c.y) * p;
    if (c.size2 != null) size = c.size + (c.size2 - c.size) * p;
  }
  return { x, y, size };
}
// Build one caption box element for keyframe `kf` ('start'|'end') at the given fractions.
function buildCapBox(c, kf, x, y, size, r, opts) {
  const el = document.createElement('div');
  el.className = 'cap-box' + (opts.selected ? ' sel' : '') + (kf === 'end' ? ' kf-end' : '');
  el.style.left = (x * r.w) + 'px';
  el.style.top = (y * r.h) + 'px';
  el.style.fontFamily = '"' + c.font + '", Arial, sans-serif';
  el.style.fontSize = (size * r.h) + 'px';
  el.style.color = c.color;
  el.style.fontWeight = c.bold ? '700' : '400';
  el.style.fontStyle = c.italic ? 'italic' : 'normal';
  el.style.letterSpacing = ((c.ls || 0) * r.h) + 'px';   // scales with frame like size
  if (c.lh != null) el.style.lineHeight = String(c.lh);
  // Outline: `stroke` is the OUTWARD thickness as a fraction of font size (matches the
  // export's Pillow stroke). -webkit-text-stroke is centered and paint-order hides the
  // inner half, so double the width to make the visible outer stroke equal the export.
  const T = (c.stroke != null ? c.stroke : 0.0556);
  el.style.setProperty('-webkit-text-stroke', (T * 2).toFixed(3) + 'em rgba(0,0,0,0.85)');
  if (c.w) { el.style.width = (c.w * r.w) + 'px'; el.style.whiteSpace = 'pre-wrap'; el.style.overflowWrap = 'break-word'; }
  else { el.style.width = ''; el.style.whiteSpace = 'pre'; }
  el.textContent = c.text || ' ';
  el.dataset.id = c.id;
  el.dataset.kf = kf;
  if (opts.label) {
    const lb = document.createElement('div'); lb.className = 'cap-kf-label'; lb.textContent = opts.label;
    el.appendChild(lb);
  }
  el.title = 'Double-click to edit · drag to move · right-click for options';
  el.addEventListener('mousedown', capDragStart);
  el.addEventListener('contextmenu', ev => showCapMenu(ev, c.id));
  el.addEventListener('dblclick', ev => { ev.stopPropagation(); startCapEdit(el, c.id); });
  if (opts.selected) {
    const rh = document.createElement('div');
    rh.className = 'cap-resize';
    rh.title = 'Drag to set box width (wrap text) · double-click to reset to auto';
    rh.addEventListener('mousedown', capResizeStart);
    rh.addEventListener('dblclick', ev => {
      ev.stopPropagation();
      const cc = CAPS.find(x => x.id === c.id);
      if (cc) { delete cc.w; _lastCapSig = ''; renderCaptions(); renderCapTrack(); persistCaptions(); }
    });
    el.appendChild(rh);
  }
  document.getElementById('captionLayer').appendChild(el);
}
function renderCaptions() {
  const L = document.getElementById('captionLayer'); if (!L || !S) return;
  if (_capEditing) return;   // don't nuke the element being edited inline
  layoutCaptionLayer();
  const r = captionFrameRect();
  const t = _origTime || 0;
  L.innerHTML = '';
  for (const c of CAPS) {
    if (!(t >= c.start && t < c.end)) continue;   // only show captions on screen right now
    const selected = (c.id === selectedCapId);
    const animated = capAnimated(c);
    // Show the Start/End editing boxes only while PAUSED; during playback show the
    // interpolated position so the motion actually previews.
    if (selected && animated && video.paused) {
      // Show both keyframes for editing (iMovie-style Start/End).
      buildCapBox(c, 'start', c.x, c.y, c.size, r, { selected: true, label: 'Start' });
      buildCapBox(c, 'end', (c.x2 ?? c.x), (c.y2 ?? c.y), (c.size2 ?? c.size), r, { selected: true, label: 'End' });
    } else {
      const v = animated ? capValuesAt(c, t) : { x: c.x, y: c.y, size: c.size };
      buildCapBox(c, 'start', v.x, v.y, v.size, r, { selected });
    }
  }
}
// Drag the right-edge handle to set a caption's box width (wraps the text).
let _capResize = null;
function capResizeStart(e) {
  if (e.button !== 0) return;
  e.preventDefault(); e.stopPropagation();
  const c = CAPS.find(x => x.id === selectedCapId); if (!c) return;
  const r = captionFrameRect();
  const box = e.currentTarget.parentElement;
  const curW = c.w ? c.w : (box.getBoundingClientRect().width / r.w);
  _capResize = { id: c.id, sx: e.clientX, ow: curW, frameW: r.w };
  document.addEventListener('mousemove', capResizeMove);
  document.addEventListener('mouseup', capResizeEnd);
}
function capResizeMove(e) {
  if (!_capResize) return;
  const c = CAPS.find(x => x.id === _capResize.id); if (!c) return;
  let nw = _capResize.ow + (e.clientX - _capResize.sx) / _capResize.frameW;
  c.w = Math.max(0.05, Math.min(1.0, nw));
  _lastCapSig = ''; renderCaptions();
}
function capResizeEnd() {
  if (!_capResize) return;
  _capResize = null;
  document.removeEventListener('mousemove', capResizeMove);
  document.removeEventListener('mouseup', capResizeEnd);
  persistCaptions();
}
// Edit a caption's text inline, directly on the video (Enter = new line).
function startCapEdit(el, id, caretX, caretY) {
  if (_capEditing === id) return;   // already editing (e.g. dbl-click after click) → don't re-init
  const c = CAPS.find(x => x.id === id); if (!c) return;
  selectCaptionKeepEdit(id);
  _capEditing = id;
  // Remove the Start/End keyframe label + resize handle so they don't get pulled into the
  // editable text (innerText would otherwise capture "Start"/"End"). Rebuilt on blur.
  el.querySelectorAll('.cap-kf-label, .cap-resize').forEach(n => n.remove());
  el.contentEditable = 'true';
  el.style.whiteSpace = 'pre-wrap';
  el.style.cursor = 'text';
  el.classList.add('editing');
  el.focus();
  // Place the caret where the user clicked (normal-text-box feel); else select all (dbl-click).
  if (caretX != null && document.caretRangeFromPoint) {
    try {
      const rng = document.caretRangeFromPoint(caretX, caretY);
      if (rng) { const sel = document.getSelection(); sel.removeAllRanges(); sel.addRange(rng); }
      else document.getSelection().selectAllChildren(el);
    } catch (e) { try { document.getSelection().selectAllChildren(el); } catch (e2) {} }
  } else {
    try { document.getSelection().selectAllChildren(el); } catch (e) {}
  }
  const onInput = () => {
    c.text = el.innerText.replace(/ /g, ' ');
    const tb = document.getElementById('capText');
    if (tb && selectedCapId === id) tb.value = c.text;
    persistCaptions();
  };
  const onBlur = () => {
    el.removeEventListener('input', onInput);
    c.text = el.innerText.replace(/ /g, ' ');
    el.contentEditable = 'false'; el.style.cursor = 'move'; el.classList.remove('editing');
    _capEditing = null; _lastCapSig = '';
    renderCaptions(); renderCapTrack(); persistCaptions();
  };
  el.addEventListener('input', onInput);
  el.addEventListener('blur', onBlur, { once: true });
}
// Select without tearing down an in-progress inline edit.
function selectCaptionKeepEdit(id) {
  selectedCapId = id;
  const c = CAPS.find(x => x.id === id);
  if (!c) { hideCapControls(); return; }
  showCapControls();
  document.getElementById('capText').value = c.text;
  capGrow(document.getElementById('capText'));
  document.getElementById('capFont').value = c.font;
  setSizeUI(c);
  document.getElementById('capColor').value = c.color;
  document.getElementById('capBold').classList.toggle('active', !!c.bold);
  document.getElementById('capItalic').classList.toggle('active', !!c.italic);
}
// Cheap per-frame check: only rebuild the overlay when the visible set changes.
function capHasFx(c) { return c.fx && c.fx !== 'none'; }
function fxCharsPerSec(c) { return 4 + Math.max(0.1, Math.min(1, (c.fxSpeed != null ? c.fxSpeed : 0.5))) * 36; }
function fxDurFor(c) { return Math.max(0.1, 1.3 - Math.max(0.1, Math.min(1, (c.fxSpeed != null ? c.fxSpeed : 0.5))) * 1.1); }
function captionTick(t) {
  if (_capDrag || _capEditing || _capResize) return;   // active drag/edit/resize manages its own DOM
  let sig = selectedCapId + '|' + (video.paused ? 'p' : 'x') + '|', anyDyn = false;
  for (const c of CAPS) if (t >= c.start && t < c.end) { sig += c.id + ','; if (capAnimated(c) || capHasFx(c)) anyDyn = true; }
  if (sig !== _lastCapSig) { _lastCapSig = sig; renderCaptions(); }
  // Per-frame update of animated (Ken Burns) + text-effect captions during playback: no DOM
  // rebuild → smooth, no flicker. Runs right after a rebuild too, so there's no first-frame flash.
  if (anyDyn && !video.paused) updateDynamicCaptions(t);
}
function updateDynamicCaptions(t) {
  const L = document.getElementById('captionLayer'); if (!L) return;
  const r = captionFrameRect();
  for (const c of CAPS) {
    if (!(t >= c.start && t < c.end)) continue;
    const el = L.querySelector('.cap-box[data-id="' + c.id + '"][data-kf="start"]');
    if (!el) continue;
    if (capAnimated(c)) {
      const v = capValuesAt(c, t);
      el.style.left = (v.x * r.w) + 'px';
      el.style.top  = (v.y * r.h) + 'px';
      el.style.fontSize = (v.size * r.h) + 'px';
    }
    if (capHasFx(c)) applyCapFx(el, c, t);
  }
}
// Apply a text effect to the caption's live element at time t.
function applyCapFx(el, c, t) {
  const into = t - c.start, toEnd = c.end - t, mode = c.fxMode || 'in';
  const doIn = (mode === 'in' || mode === 'both'), doOut = (mode === 'out' || mode === 'both');
  if (c.fx === 'typewriter') {
    const cps = fxCharsPerSec(c), full = c.text || '';
    let n = full.length;
    if (doIn) n = Math.max(0, Math.min(full.length, Math.floor(into * cps)));
    if (doOut && (toEnd * cps) < full.length) n = Math.max(0, Math.min(n, Math.floor(toEnd * cps)));
    if (el.textContent !== full.slice(0, n)) el.textContent = full.slice(0, n);
    el.style.opacity = ''; el.style.transform = '';
  } else if (c.fx === 'fade' || c.fx === 'zoom') {
    const D = fxDurFor(c);
    let op = 1, sc = 1;
    if (doIn && into < D)  { const p = Math.max(0, into / D);  op = p;             sc = 0.6 + 0.4 * p; }
    if (doOut && toEnd < D){ const p = Math.max(0, toEnd / D); op = Math.min(op, p); sc = Math.min(sc, 0.6 + 0.4 * p); }
    if (c.fx === 'fade') { el.style.opacity = op; el.style.transform = ''; }
    else { el.style.opacity = 0.4 + 0.6 * ((sc - 0.6) / 0.4); el.style.transformOrigin = 'center center'; el.style.transform = 'scale(' + sc + ')'; }
  }
}
function persistCaptions() {
  if (S) S.captions = CAPS;
  clearTimeout(_capSaveTimer);
  _capSaveTimer = setTimeout(() => {
    fetch('/api/captions', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ captions: CAPS }) }).catch(() => {});
  }, 400);
}
function addCaption() {
  if (!S || S.no_video) { setStatus('Load a video first'); return; }
  const t = _origTime || 0, dur = S.duration || 0;
  const c = { id: 'cap_' + Date.now(), text: 'New text', x: 0.28, y: 0.44, size: 0.055,
    font: 'Avenir', color: '#ffffff', bold: true, italic: false,
    start: Math.max(0, t), end: Math.min(dur || t + 3, t + 3) };
  if (c.end <= c.start) c.end = c.start + 2;
  CAPS.push(c);
  selectCaption(c.id);
  _lastCapSig = '';
  renderCaptions(); renderCapTrack(); persistCaptions();
  const inp = document.getElementById('capText'); if (inp) { inp.focus(); inp.select(); }
}
function capGrow(ta) {   // no-op: bar is fixed height; multi-line scrolls inside the field
  // (full multi-line editing is on the video's inline box). Kept so callers stay valid.
}
// The docked bar is always present (fixed space); toggle its CONTENTS, not the bar,
// so selecting a caption never resizes the video.
function showCapControls() {
  document.getElementById('capHint').style.display = 'none';
  document.getElementById('capControls').style.display = 'flex';
}
function hideCapControls() {
  document.getElementById('capControls').style.display = 'none';
  document.getElementById('capHint').style.display = '';
}
function selectCaption(id) {
  selectedCapId = id;
  const c = CAPS.find(x => x.id === id);
  if (!c) { hideCapControls(); return; }
  showCapControls();
  document.getElementById('capText').value = c.text;
  capGrow(document.getElementById('capText'));
  document.getElementById('capFont').value = c.font;
  setSizeUI(c);
  document.getElementById('capColor').value = c.color;
  document.getElementById('capBold').classList.toggle('active', !!c.bold);
  document.getElementById('capItalic').classList.toggle('active', !!c.italic);
  // Out-of-range captions are hidden now — jump the playhead into this caption's window
  // so it's visible on the video and can be dragged/edited.
  if (_origTime < c.start || _origTime >= c.end) {
    const target = Math.min(c.end - 0.05, c.start + 0.1);
    _origTime = target;
    try { seekOriginal(target); } catch (e) {}
  }
  _lastCapSig = '';
  renderCaptions(); renderCapTrack();
}
function deselectCaption() {
  selectedCapId = null;
  hideCapControls();
  _lastCapSig = '';
  renderCaptions(); renderCapTrack();
}
function capEdit(field, val) {
  const c = CAPS.find(x => x.id === selectedCapId); if (!c) return;
  c[field] = val;
  _lastCapSig = '';
  renderCaptions(); renderCapTrack(); persistCaptions();
}
// Size is stored as a fraction of frame height; the number field shows px at 1080p output.
function setSizeUI(c) {
  const num = document.getElementById('capSizeNum');
  if (num) num.value = Math.round(c.size * 1080);
  const sel = document.getElementById('capSize');
  if (sel) {
    const m = [...sel.options].find(o => o.value !== 'custom' && Math.abs(parseFloat(o.value) - c.size) < 0.0015);
    sel.value = m ? m.value : 'custom';
  }
  // Animation UI state
  const anim = c.x2 != null;
  const ab = document.getElementById('capAnim'); if (ab) ab.classList.toggle('active', anim);
  const es = document.getElementById('capEndSize'); if (es) es.style.display = anim ? 'inline-flex' : 'none';
  const n2 = document.getElementById('capSizeNum2'); if (n2) n2.value = Math.round((c.size2 != null ? c.size2 : c.size) * 1080);
  const ls = document.getElementById('capLS'); if (ls) ls.value = Math.round((c.ls || 0) * 1080);
  const lh = document.getElementById('capLH'); if (lh) lh.value = (c.lh != null ? c.lh : 1.1);
  const skv = Math.round((c.stroke != null ? c.stroke : 0.0556) * 1000);
  const sk = document.getElementById('capStroke'); if (sk) sk.value = skv;
  const skl = document.getElementById('capStrokeVal'); if (skl) skl.textContent = skv;
  // Text-effect UI state
  const fx = c.fx || 'none';
  const fxSel = document.getElementById('capFx'); if (fxSel) fxSel.value = fx;
  const fxOpts = document.getElementById('capFxOpts'); if (fxOpts) fxOpts.style.display = (fx === 'none') ? 'none' : 'inline-flex';
  const fxMode = document.getElementById('capFxMode'); if (fxMode) fxMode.value = c.fxMode || 'in';
  const fxSpd = document.getElementById('capFxSpeed'); if (fxSpd) fxSpd.value = (c.fxSpeed != null ? c.fxSpeed : 0.5);
}
// Text effect (typewriter / fade / zoom)
function capFxChange(v) {
  const c = CAPS.find(x => x.id === selectedCapId); if (!c) return;
  if (v === 'none') { delete c.fx; }
  else { c.fx = v; if (c.fxSpeed == null) c.fxSpeed = 0.5; if (c.fxMode == null) c.fxMode = 'in'; }
  const fxOpts = document.getElementById('capFxOpts'); if (fxOpts) fxOpts.style.display = (v === 'none') ? 'none' : 'inline-flex';
  _lastCapSig = ''; renderCaptions(); persistCaptions();
  setStatus(v === 'none' ? 'Effect removed' : ('Effect: ' + v + ' — press play to preview'));
}
function capEditFx(field, val) {
  const c = CAPS.find(x => x.id === selectedCapId); if (!c) return;
  c[field] = val;
  _lastCapSig = ''; renderCaptions(); persistCaptions();
}
// Outline thickness — stored as a fraction of font size; slider is ×1000 (default ≈56, like Canva).
function capStrokeInput(v) {
  const c = CAPS.find(x => x.id === selectedCapId); if (!c) return;
  let n = parseFloat(v); if (isNaN(n)) return;
  n = Math.max(0, Math.min(300, n));
  c.stroke = n / 1000;
  const skl = document.getElementById('capStrokeVal'); if (skl) skl.textContent = Math.round(n);
  _lastCapSig = ''; renderCaptions(); persistCaptions();
}
// Letter spacing — stored as a fraction of frame height (px at 1080p in the UI), scales like size.
function capLSInput(v) {
  const c = CAPS.find(x => x.id === selectedCapId); if (!c) return;
  let px = parseFloat(v); if (isNaN(px)) return;
  c.ls = Math.max(-50, Math.min(200, px)) / 1080;
  _lastCapSig = ''; renderCaptions(); persistCaptions();
}
// Line spacing — multiple of the line height (like Canva's 1.0–1.4).
function capLHInput(v) {
  const c = CAPS.find(x => x.id === selectedCapId); if (!c) return;
  let m = parseFloat(v); if (isNaN(m)) return;
  c.lh = Math.max(0.5, Math.min(3, m));
  _lastCapSig = ''; renderCaptions(); persistCaptions();
}
// Toggle Ken-Burns-style animation for the selected caption (Start → End keyframe).
function toggleCapAnim() {
  const c = CAPS.find(x => x.id === selectedCapId); if (!c) return;
  if (c.x2 != null) {                     // turn OFF
    delete c.x2; delete c.y2; delete c.size2;
  } else {                                // turn ON — seed an End keyframe offset from Start
    c.x2 = Math.min(0.9, c.x + 0.06); c.y2 = c.y; c.size2 = c.size;
  }
  setSizeUI(c);
  _lastCapSig = ''; renderCaptions(); renderCapTrack(); persistCaptions();
}
function capEndSizeInput(v) {
  const c = CAPS.find(x => x.id === selectedCapId); if (!c) return;
  let px = parseFloat(v); if (isNaN(px)) return;
  c.size2 = Math.max(6, Math.min(400, px)) / 1080;
  _lastCapSig = ''; renderCaptions(); persistCaptions();
}
// Swap the Start and End keyframes (reverse the animation direction) — iMovie-style.
function swapCapKeyframes() {
  const c = CAPS.find(x => x.id === selectedCapId); if (!c || c.x2 == null) return;
  const sx = c.x, sy = c.y, ss = c.size;
  c.x = c.x2; c.y = (c.y2 != null ? c.y2 : c.y); c.size = (c.size2 != null ? c.size2 : c.size);
  c.x2 = sx; c.y2 = sy; c.size2 = ss;
  setSizeUI(c);
  _lastCapSig = ''; renderCaptions(); renderCapTrack(); persistCaptions();
  setStatus('Swapped start ↔ end');
}
function capSizeDropdown(v) {
  const f = parseFloat(v); if (isNaN(f)) return;   // "Custom" is a no-op label
  const c = CAPS.find(x => x.id === selectedCapId); if (!c) return;
  c.size = f;
  const num = document.getElementById('capSizeNum'); if (num) num.value = Math.round(f * 1080);
  _lastCapSig = ''; renderCaptions(); renderCapTrack(); persistCaptions();
}
function capSizeNumInput(v) {
  const c = CAPS.find(x => x.id === selectedCapId); if (!c) return;
  let px = parseFloat(v); if (isNaN(px)) return;   // allow mid-typing without clobbering
  px = Math.max(6, Math.min(400, px));
  c.size = px / 1080;
  const sel = document.getElementById('capSize');   // reflect on the preset dropdown only
  if (sel) {
    const m = [...sel.options].find(o => o.value !== 'custom' && Math.abs(parseFloat(o.value) - c.size) < 0.0015);
    sel.value = m ? m.value : 'custom';
  }
  _lastCapSig = ''; renderCaptions(); renderCapTrack(); persistCaptions();
}
function capToggle(field) {
  const c = CAPS.find(x => x.id === selectedCapId); if (!c) return;
  c[field] = !c[field];
  document.getElementById(field === 'bold' ? 'capBold' : 'capItalic').classList.toggle('active', c[field]);
  renderCaptions(); persistCaptions();
}
function capDelete() {
  CAPS = CAPS.filter(x => x.id !== selectedCapId);
  deselectCaption();
  renderCaptions(); renderCapTrack(); persistCaptions();
}
// ── Drag a caption around the frame ──
function capDragStart(e) {
  if (e.button !== 0) return;   // ignore right/middle click (context menu handles those)
  const el = e.currentTarget;
  const id = el.dataset.id;
  if (_capEditing === id) return;   // already editing this caption → let native text selection work
  const kf = el.dataset.kf || 'start';   // which keyframe this box edits
  const wasSelected = (selectedCapId === id);
  selectCaptionKeepEdit(id);
  el.classList.add('sel');
  const c = CAPS.find(x => x.id === id); if (!c) return;
  const r = captionFrameRect();
  const ox = (kf === 'end') ? (c.x2 ?? c.x) : c.x;
  const oy = (kf === 'end') ? (c.y2 ?? c.y) : c.y;
  _capDrag = { id, kf, el, wasSelected, moved: false,
               downX: e.clientX, downY: e.clientY, sx: e.clientX, sy: e.clientY, ox, oy, w: r.w, h: r.h };
  e.preventDefault(); e.stopPropagation();
  document.addEventListener('mousemove', capDragMove);
  document.addEventListener('mouseup', capDragEnd);
}
function capDragMove(e) {
  if (!_capDrag) return;
  // Don't treat it as a move until the pointer travels a few px — lets a click stay a click.
  if (!_capDrag.moved && Math.abs(e.clientX - _capDrag.downX) < 4 && Math.abs(e.clientY - _capDrag.downY) < 4) return;
  _capDrag.moved = true;
  const c = CAPS.find(x => x.id === _capDrag.id); if (!c) return;
  const nx = Math.max(0, Math.min(0.999, _capDrag.ox + (e.clientX - _capDrag.sx) / _capDrag.w));
  const ny = Math.max(0, Math.min(0.999, _capDrag.oy + (e.clientY - _capDrag.sy) / _capDrag.h));
  if (_capDrag.kf === 'end') { c.x2 = nx; c.y2 = ny; } else { c.x = nx; c.y = ny; }
  renderCaptions();
}
function capDragEnd() {
  if (!_capDrag) return;
  const d = _capDrag; _capDrag = null;
  document.removeEventListener('mousemove', capDragMove);
  document.removeEventListener('mouseup', capDragEnd);
  // A click (no drag) on an ALREADY-selected caption → edit the text inline, caret where clicked.
  if (!d.moved && d.wasSelected && d.kf === 'start' && d.el) {
    startCapEdit(d.el, d.id, d.downX, d.downY);
    return;
  }
  if (d.moved) persistCaptions();
}
// ── Caption timeline track (retime: drag body = move, drag edges = trim) ──
function renderCapTrack() {
  const track = document.getElementById('capTrack'); if (!track || !S) return;
  const W = tlFullW();
  track.style.width = W + 'px';
  track.innerHTML = '<span class="ov-label">T Text</span>';
  const laneEnds = [];
  const sorted = [...CAPS].sort((a, b) => a.start - b.start);
  for (const c of sorted) {
    let lane = 0;
    while (laneEnds[lane] !== undefined && laneEnds[lane] > c.start + 0.001) lane++;
    laneEnds[lane] = c.end;
    const el = document.createElement('div');
    el.className = 'cap-block' + (c.id === selectedCapId ? ' sel' : '');
    el.style.left = X(c.start) + 'px';
    el.style.width = Math.max(14, Xw(c.start, c.end)) + 'px';
    el.style.top = (2 + lane * 20) + 'px';
    el.dataset.id = c.id;
    el.innerHTML = '<div class="cap-h l"></div><span class="cap-lbl"></span><div class="cap-h r"></div>';
    el.querySelector('.cap-lbl').textContent = c.text || 'text';
    el.addEventListener('mousedown', capBlockDown);
    el.addEventListener('contextmenu', ev => showCapMenu(ev, c.id));
    track.appendChild(el);
  }
  track.style.height = Math.max(24, laneEnds.length * 20 + 6) + 'px';
}
function capBlockDown(e) {
  if (e.button !== 0) return;   // ignore right/middle click (context menu handles those)
  const el = e.currentTarget, id = el.dataset.id;
  selectCaption(id);
  const c = CAPS.find(x => x.id === id); if (!c) return;
  let mode = 'move';
  if (e.target.classList.contains('cap-h')) mode = e.target.classList.contains('l') ? 'l' : 'r';
  _capBlk = { id, mode, sx: e.clientX, os: c.start, oe: c.end };
  e.preventDefault(); e.stopPropagation();
  document.addEventListener('mousemove', capBlockMove);
  document.addEventListener('mouseup', capBlockUp);
}
function capBlockMove(e) {
  if (!_capBlk) return;
  const c = CAPS.find(x => x.id === _capBlk.id); if (!c) return;
  const dt = (e.clientX - _capBlk.sx) / PPS, dur = S.duration || 0;
  if (_capBlk.mode === 'move') {
    const len = _capBlk.oe - _capBlk.os;
    let ns = Math.max(0, Math.min(dur - len, _capBlk.os + dt));
    c.start = ns; c.end = ns + len;
  } else if (_capBlk.mode === 'l') {
    c.start = Math.max(0, Math.min(c.end - 0.1, _capBlk.os + dt));
  } else {
    c.end = Math.min(dur, Math.max(c.start + 0.1, _capBlk.oe + dt));
  }
  _lastCapSig = '';
  renderCapTrack(); renderCaptions();
}
function capBlockUp() {
  if (!_capBlk) return;
  _capBlk = null;
  document.removeEventListener('mousemove', capBlockMove);
  document.removeEventListener('mouseup', capBlockUp);
  persistCaptions();
}
// ── Duplicate (copy/paste), split, right-click menu ──
let _capClipboard = null, _capMenuId = null;
function duplicateCaption(id) {
  const src = CAPS.find(x => x.id === id); if (!src) return;
  const c = { ...src, id: 'cap_' + Date.now(),
    x: Math.min(0.95, src.x + 0.03), y: Math.min(0.95, src.y + 0.03) };
  CAPS.push(c); selectCaption(c.id); _lastCapSig = '';
  renderCaptions(); renderCapTrack(); persistCaptions();
  setStatus('Caption duplicated');
}
function pasteCaption() {
  if (!_capClipboard) return;
  const src = _capClipboard, dur = S.duration || 0;
  const len = Math.max(0.2, src.end - src.start);
  const start = Math.max(0, Math.min(dur - len, (_origTime || src.start)));
  const c = { ...src, id: 'cap_' + Date.now(), start, end: start + len,
    x: Math.min(0.95, src.x + 0.03), y: Math.min(0.95, src.y + 0.03) };
  CAPS.push(c); selectCaption(c.id); _lastCapSig = '';
  renderCaptions(); renderCapTrack(); persistCaptions();
  setStatus('Caption pasted at playhead');
}
function splitCaption(id) {
  const c = CAPS.find(x => x.id === id); if (!c) return;
  const t = _origTime || 0;
  if (t <= c.start + 0.05 || t >= c.end - 0.05) {
    setStatus('Move the playhead inside the caption first, then split'); return;
  }
  const second = { ...c, id: 'cap_' + Date.now(), start: t, end: c.end };
  c.end = t;
  CAPS.push(second); selectCaption(second.id); _lastCapSig = '';
  renderCaptions(); renderCapTrack(); persistCaptions();
  setStatus('Caption split at playhead');
}
function showCapMenu(e, id) {
  e.preventDefault(); e.stopPropagation();
  selectCaption(id);
  _capMenuId = id;
  const m = document.getElementById('capMenu');
  m.style.display = 'block';
  m.style.left = Math.min(e.clientX, window.innerWidth - 170) + 'px';
  m.style.top = Math.min(e.clientY, window.innerHeight - 130) + 'px';
}
function hideCapMenu() {
  const m = document.getElementById('capMenu');
  if (m) m.style.display = 'none';
  _capMenuId = null;
}
function capMenuAction(act) {
  const id = _capMenuId; hideCapMenu(); if (!id) return;
  if (act === 'split') splitCaption(id);
  else if (act === 'dup') duplicateCaption(id);
  else if (act === 'del') { selectedCapId = id; capDelete(); }
}
document.addEventListener('mousedown', (e) => {
  if (!e.target.closest('#capMenu')) hideCapMenu();
});

function initCaptions() {
  CAPS = (S && S.captions ? S.captions : []).map(c => ({ ...c }));
  selectedCapId = null;
  hideCapControls();
  _lastCapSig = '';
  renderCaptions(); renderCapTrack();
}
window.addEventListener('resize', () => { _lastCapSig = ''; renderCaptions(); });

/* ══════════ Image overlays ══════════ */
let IMGS = [], selectedImgId = null, _imgSaveTimer = null, _lastImgSig = '';
let _imgDrag = null, _imgResize = null, _imgBlk = null;
function imgAnimated(m){ return (m.x2!=null&&Math.abs(m.x2-m.x)>1e-4)||(m.y2!=null&&Math.abs(m.y2-m.y)>1e-4)||(m.w2!=null&&Math.abs(m.w2-m.w)>1e-4); }
function imgValuesAt(m,t){ let x=m.x,y=m.y,w=m.w; if(imgAnimated(m)&&m.end>m.start){let p=Math.max(0,Math.min(1,(t-m.start)/(m.end-m.start))); if(m.x2!=null)x=m.x+(m.x2-m.x)*p; if(m.y2!=null)y=m.y+(m.y2-m.y)*p; if(m.w2!=null)w=m.w+(m.w2-m.w)*p;} return {x,y,w}; }
function persistImages(){ if(S)S.images=IMGS; clearTimeout(_imgSaveTimer); _imgSaveTimer=setTimeout(()=>{ fetch('/api/images',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({images:IMGS})}).catch(()=>{}); },400); }
function buildImgBox(m, kf, x, y, w, r, opts){
  // Wrap the image in a DIV so it can hold the resize handle + label (an <img> can't have children).
  const box=document.createElement('div');
  box.className='img-ov'+(opts.selected?' sel':'')+(kf==='end'?' kf-end':'');
  box.style.left=(x*r.w)+'px'; box.style.top=(y*r.h)+'px'; box.style.width=(w*r.w)+'px'; box.style.height='auto';
  box.style.opacity=(m.opacity!=null?m.opacity:1);
  box.dataset.id=m.id; box.dataset.kf=kf;
  box.title='Drag to move · drag the corner to resize · right-click to delete';
  const img=document.createElement('img');
  img.src='/overlay/'+encodeURIComponent(m.src); img.draggable=false;
  img.style.cssText='display:block;width:100%;height:auto;pointer-events:none;user-select:none;';
  box.appendChild(img);
  if(opts.label){ const lb=document.createElement('div'); lb.className='img-kf-label'; lb.textContent=opts.label; box.appendChild(lb); }
  box.addEventListener('mousedown', imgDragStart);
  box.addEventListener('contextmenu', ev=>{ ev.preventDefault(); selectImg(m.id); if(confirm('Delete this image overlay?')) imgDelete(); });
  if(opts.selected){
    const h=document.createElement('div'); h.className='img-resize'; h.dataset.id=m.id; h.dataset.kf=kf;
    h.addEventListener('mousedown', imgResizeStart);
    box.appendChild(h);
  }
  document.getElementById('imageLayer').appendChild(box);
}
function renderImages(){
  const L=document.getElementById('imageLayer'); if(!L||!S) return;
  const r=captionFrameRect();
  L.style.left=r.left+'px'; L.style.top=r.top+'px'; L.style.width=r.w+'px'; L.style.height=r.h+'px';
  const t=_origTime||0; L.innerHTML='';
  for(const m of IMGS){
    if(!(t>=m.start&&t<m.end)) continue;
    const selected=(m.id===selectedImgId), animated=imgAnimated(m);
    if(selected&&animated){
      buildImgBox(m,'start',m.x,m.y,m.w,r,{selected:true,label:'Start'});
      buildImgBox(m,'end',(m.x2??m.x),(m.y2??m.y),(m.w2??m.w),r,{selected:true,label:'End'});
    } else {
      const v=animated?imgValuesAt(m,t):{x:m.x,y:m.y,w:m.w};
      buildImgBox(m,'start',v.x,v.y,v.w,r,{selected});
    }
  }
}
function imgTick(t){ if(_imgDrag||_imgResize) return; let sig=selectedImgId+'|'; for(const m of IMGS) if(t>=m.start&&t<m.end) sig+=m.id+','; if(sig!==_lastImgSig){ _lastImgSig=sig; renderImages(); } }
async function handleImagePick(files){
  if(!files||!files.length) return;
  const fd=new FormData(); fd.append('image', files[0]);
  document.getElementById('imgUploadInput').value='';
  const res=await fetch('/api/upload-overlay-image',{method:'POST',body:fd}).then(r=>r.json()).catch(()=>null);
  if(!res||!res.ok){ setStatus('Image upload failed'); return; }
  const t=_origTime||0, dur=S.duration||0;
  const ar=(res.h&&res.w)?res.h/res.w:0.6;
  const m={ id:'img_'+Date.now(), src:res.src, x:0.28, y:0.12, w:0.45, opacity:1,
    start:Math.max(0,t), end:Math.min(dur||t+4, t+4) };
  if(m.end<=m.start) m.end=m.start+3;
  IMGS.push(m); selectImg(m.id); _lastImgSig='';
  renderImages(); renderImgTrack(); persistImages();
  setStatus('Image overlay added — drag to place, resize from the corner');
}
function selectImg(id){
  selectedImgId=id; selectedCapId=null; hideCapControls();
  const m=IMGS.find(x=>x.id===id); const p=document.getElementById('imgPanel');
  if(!m){ p.style.display='none'; return; }
  p.style.display='flex';
  document.getElementById('imgOpacity').value=Math.round((m.opacity!=null?m.opacity:1)*100);
  document.getElementById('imgOpacityVal').textContent=Math.round((m.opacity!=null?m.opacity:1)*100)+'%';
  document.getElementById('imgAnim').classList.toggle('active', m.x2!=null);
  if(_origTime<m.start||_origTime>=m.end){ const tt=Math.min(m.end-0.05,m.start+0.1); _origTime=tt; try{seekOriginal(tt);}catch(e){} }
  _lastImgSig=''; renderImages(); renderImgTrack();
}
function deselectImg(){ selectedImgId=null; document.getElementById('imgPanel').style.display='none'; _lastImgSig=''; renderImages(); renderImgTrack(); }
function imgEditOpacity(v){ const m=IMGS.find(x=>x.id===selectedImgId); if(!m) return; m.opacity=Math.max(0,Math.min(1,v/100)); document.getElementById('imgOpacityVal').textContent=Math.round(v)+'%'; _lastImgSig=''; renderImages(); persistImages(); }
function toggleImgAnim(){ const m=IMGS.find(x=>x.id===selectedImgId); if(!m) return; if(m.x2!=null){ delete m.x2; delete m.y2; delete m.w2; } else { m.x2=Math.min(0.9,m.x+0.06); m.y2=m.y; m.w2=m.w; } document.getElementById('imgAnim').classList.toggle('active', m.x2!=null); _lastImgSig=''; renderImages(); renderImgTrack(); persistImages(); }
function imgDelete(){ IMGS=IMGS.filter(x=>x.id!==selectedImgId); deselectImg(); renderImages(); renderImgTrack(); persistImages(); }
function imgDragStart(e){ if(e.button!==0) return; const id=e.currentTarget.dataset.id, kf=e.currentTarget.dataset.kf||'start'; selectImg(id); const m=IMGS.find(x=>x.id===id); if(!m) return; const r=captionFrameRect(); const ox=(kf==='end')?(m.x2??m.x):m.x, oy=(kf==='end')?(m.y2??m.y):m.y; _imgDrag={id,kf,sx:e.clientX,sy:e.clientY,ox,oy,w:r.w,h:r.h}; e.preventDefault(); e.stopPropagation(); document.addEventListener('mousemove',imgDragMove); document.addEventListener('mouseup',imgDragEnd); }
function imgDragMove(e){ if(!_imgDrag) return; const m=IMGS.find(x=>x.id===_imgDrag.id); if(!m) return; const nx=Math.max(-0.3,Math.min(0.99,_imgDrag.ox+(e.clientX-_imgDrag.sx)/_imgDrag.w)); const ny=Math.max(-0.3,Math.min(0.99,_imgDrag.oy+(e.clientY-_imgDrag.sy)/_imgDrag.h)); if(_imgDrag.kf==='end'){m.x2=nx;m.y2=ny;}else{m.x=nx;m.y=ny;} renderImages(); }
function imgDragEnd(){ if(!_imgDrag) return; _imgDrag=null; document.removeEventListener('mousemove',imgDragMove); document.removeEventListener('mouseup',imgDragEnd); persistImages(); }
function imgResizeStart(e){ if(e.button!==0) return; e.preventDefault(); e.stopPropagation(); const id=e.currentTarget.dataset.id, kf=e.currentTarget.dataset.kf||'start'; const m=IMGS.find(x=>x.id===id); if(!m) return; const r=captionFrameRect(); const ow=(kf==='end')?(m.w2??m.w):m.w; _imgResize={id,kf,sx:e.clientX,ow,frameW:r.w}; document.addEventListener('mousemove',imgResizeMove); document.addEventListener('mouseup',imgResizeEnd); }
function imgResizeMove(e){ if(!_imgResize) return; const m=IMGS.find(x=>x.id===_imgResize.id); if(!m) return; let nw=Math.max(0.03,Math.min(1.5,_imgResize.ow+(e.clientX-_imgResize.sx)/_imgResize.frameW)); if(_imgResize.kf==='end')m.w2=nw; else m.w=nw; _lastImgSig=''; renderImages(); }
function imgResizeEnd(){ if(!_imgResize) return; _imgResize=null; document.removeEventListener('mousemove',imgResizeMove); document.removeEventListener('mouseup',imgResizeEnd); persistImages(); }
function renderImgTrack(){
  const track=document.getElementById('imgTrack'); if(!track||!S) return;
  const W=tlFullW(); track.style.width=W+'px'; track.innerHTML='<span class="ov-label">🖼 Images</span>';
  const lanes=[]; const sorted=[...IMGS].sort((a,b)=>a.start-b.start);
  for(const m of sorted){ let lane=0; while(lanes[lane]!==undefined&&lanes[lane]>m.start+0.001) lane++; lanes[lane]=m.end;
    const el=document.createElement('div'); el.className='img-block'+(m.id===selectedImgId?' sel':''); el.style.left=X(m.start)+'px'; el.style.width=Math.max(14,Xw(m.start,m.end))+'px'; el.style.top=(2+lane*20)+'px'; el.dataset.id=m.id;
    el.innerHTML='<div class="cap-h l"></div><span style="position:absolute;left:6px;right:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;pointer-events:none;">🖼 '+(m.src||'image')+'</span><div class="cap-h r"></div>';
    el.addEventListener('mousedown', imgBlockDown); el.addEventListener('contextmenu', ev=>showOvMenu(ev,'img',m.id)); track.appendChild(el);
  }
  track.style.height=Math.max(24,lanes.length*20+6)+'px';
}
function imgBlockDown(e){ const el=e.currentTarget,id=el.dataset.id; selectImg(id); const m=IMGS.find(x=>x.id===id); if(!m) return; let mode='move'; if(e.target.classList.contains('cap-h')) mode=e.target.classList.contains('l')?'l':'r'; _imgBlk={id,mode,sx:e.clientX,os:m.start,oe:m.end}; e.preventDefault(); e.stopPropagation(); document.addEventListener('mousemove',imgBlockMove); document.addEventListener('mouseup',imgBlockUp); }
function imgBlockMove(e){ if(!_imgBlk) return; const m=IMGS.find(x=>x.id===_imgBlk.id); if(!m) return; const dt=(e.clientX-_imgBlk.sx)/PPS, dur=S.duration||0; if(_imgBlk.mode==='move'){ const len=_imgBlk.oe-_imgBlk.os; let ns=Math.max(0,Math.min(dur-len,_imgBlk.os+dt)); m.start=ns; m.end=ns+len; } else if(_imgBlk.mode==='l'){ m.start=Math.max(0,Math.min(m.end-0.1,_imgBlk.os+dt)); } else { m.end=Math.min(dur,Math.max(m.start+0.1,_imgBlk.oe+dt)); } _lastImgSig=''; renderImgTrack(); renderImages(); }
function imgBlockUp(){ if(!_imgBlk) return; _imgBlk=null; document.removeEventListener('mousemove',imgBlockMove); document.removeEventListener('mouseup',imgBlockUp); persistImages(); }
function initImages(){ IMGS=(S&&S.images?S.images:[]).map(m=>({...m})); selectedImgId=null; document.getElementById('imgPanel').style.display='none'; _lastImgSig=''; renderImages(); renderImgTrack(); }
window.addEventListener('resize', () => { _lastImgSig = ''; renderImages(); });

/* ══════════ Audio layers (sound effects / music) ══════════ */
let AUD = [], selectedAudId = null, _audEls = {}, _audBlk = null, _audSaveTimer = null, _libPreviewEl = null;
function persistAudio(){ if(S)S.audio=AUD; clearTimeout(_audSaveTimer); _audSaveTimer=setTimeout(()=>{ fetch('/api/audio-layers',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({audio:AUD})}).catch(()=>{}); },400); }
function initAudio(){ AUD=(S&&S.audio?S.audio:[]).map(a=>({...a})); selectedAudId=null; for(const k in _audEls){try{_audEls[k].pause();}catch(e){}} _audEls={}; document.getElementById('audPanel').style.display='none'; renderAudioTrack(); }
function _audEl(a){ if(!_audEls[a.id]){ const el=new Audio('/sound/'+encodeURIComponent(a.src)); el.preload='auto'; _audEls[a.id]=el; } return _audEls[a.id]; }
// Keep each active sound playing in sync with the playhead during preview playback.
function audioTick(t){
  for(const a of AUD){
    const el=_audEl(a); const active=(t>=a.start && t<a.start+a.dur);
    if(active && !video.paused){
      el.volume=Math.max(0,Math.min(1,(a.volume!=null?a.volume:1)));
      const target=(t-a.start)+(a.off||0);
      if(Math.abs(el.currentTime-target)>0.3){ try{el.currentTime=target;}catch(e){} }
      if(el.paused) el.play().catch(()=>{});
    } else { if(!el.paused) el.pause(); }
  }
}
const _soundPeaks = {};   // src -> peaks array (cached client-side)
function getSoundPeaks(src, cb){
  if(_soundPeaks[src]){ cb(_soundPeaks[src]); return; }
  fetch('/api/sound-waveform?src='+encodeURIComponent(src)).then(r=>r.json()).then(d=>{ if(d&&d.ok){ _soundPeaks[src]=d.peaks||[]; cb(_soundPeaks[src]); } }).catch(()=>{});
}
function drawAudWave(canvas, peaks){
  if(!canvas||!peaks||!peaks.length) return;
  // draw at device-pixel resolution for crisp bars
  const dpr=Math.min(2, window.devicePixelRatio||1);
  const w=Math.max(1, canvas.clientWidth||canvas.width), h=Math.max(1, canvas.clientHeight||canvas.height);
  const pxW=Math.min(32000, Math.round(w*dpr)), sx=pxW/w;   // cap internal width (browser canvas limit)
  canvas.width=pxW; canvas.height=Math.round(h*dpr);
  const ctx=canvas.getContext('2d'); ctx.setTransform(sx,0,0,dpr,0,0); ctx.clearRect(0,0,w,h);
  ctx.fillStyle='rgba(6,38,18,0.9)';
  const mid=h/2, N=peaks.length;
  // For each pixel column take the TRUE MAX over the span of peaks it covers,
  // so downsampling never smooths away transients — the envelope stays precise.
  for(let x=0;x<w;x++){
    const a=Math.floor(x/w*N), b=Math.max(a+1, Math.floor((x+1)/w*N));
    let p=0; for(let i=a;i<b&&i<N;i++){ if(peaks[i]>p) p=peaks[i]; }
    const amp=p*(h*0.47);
    ctx.fillRect(x, mid-amp, 1, Math.max(1, amp*2));
  }
}
function renderAudioTrack(){
  const track=document.getElementById('audioTrack'); if(!track||!S) return;
  const W=tlFullW(); track.style.width=W+'px'; track.innerHTML='<span class="ov-label">🔊 Sound</span>';
  const lanes=[]; const sorted=[...AUD].sort((a,b)=>a.start-b.start);
  for(const a of sorted){ let lane=0; const end=a.start+a.dur; while(lanes[lane]!==undefined&&lanes[lane]>a.start+0.001) lane++; lanes[lane]=end;
    // Audio plays for its full real duration in the output regardless of video cuts,
    // so its width is dur*PPS (NOT the collapsed video span Xw, which would shrink it to
    // a sliver when it sits over a cut-heavy stretch in hide-skipped mode).
    const bw=Math.max(14, a.dur*PPS);
    const el=document.createElement('div'); el.className='aud-block'+(a.id===selectedAudId?' sel':''); el.style.left=X(a.start)+'px'; el.style.width=bw+'px'; el.style.top=(2+lane*72)+'px'; el.dataset.id=a.id;
    el.innerHTML='<canvas class="aud-wave"></canvas><div class="cap-h l"></div><span class="aud-lbl">🔊 '+(a.src||'sound')+'</span><div class="cap-h r"></div>';
    el.addEventListener('mousedown', audBlockDown); el.addEventListener('contextmenu', ev=>showOvMenu(ev,'aud',a.id)); track.appendChild(el);
    const canvas=el.querySelector('.aud-wave');
    getSoundPeaks(a.src, peaks=>{ drawAudWave(canvas, peaks); });
    // iMovie-style volume line across this clip
    const aid=a.id;
    const vl=makeVolLine((a.volume!=null?a.volume:1), 68,
      nv=>{ const x=AUD.find(z=>z.id===aid); if(x){ x.volume=nv; if(_audEls[aid])_audEls[aid].volume=Math.min(1,nv);
             if(selectedAudId===aid){ const s=document.getElementById('audVol'); if(s){ s.value=Math.round(nv*100); document.getElementById('audVolVal').textContent=Math.round(nv*100)+'%'; } } } },
      ()=>persistAudio());
    el.appendChild(vl);
  }
  track.style.height=Math.max(74,lanes.length*72+6)+'px';
}
function audBlockDown(e){ const el=e.currentTarget,id=el.dataset.id; selectAud(id); const a=AUD.find(x=>x.id===id); if(!a) return; let mode='move'; if(e.target.classList.contains('cap-h')) mode=e.target.classList.contains('l')?'l':'r'; _audBlk={id,mode,sx:e.clientX,os:a.start,od:a.dur}; e.preventDefault(); e.stopPropagation(); document.addEventListener('mousemove',audBlockMove); document.addEventListener('mouseup',audBlockUp); }
function audBlockMove(e){ if(!_audBlk) return; const a=AUD.find(x=>x.id===_audBlk.id); if(!a) return; const dt=(e.clientX-_audBlk.sx)/PPS, dur=S.duration||0; if(_audBlk.mode==='move'){ a.start=Math.max(0,Math.min(dur,_audBlk.os+dt)); } else if(_audBlk.mode==='r'){ a.dur=Math.max(0.2,_audBlk.od+dt); } else { a.start=Math.max(0,_audBlk.os+dt); } renderAudioTrack(); }
function audBlockUp(){ if(!_audBlk) return; _audBlk=null; document.removeEventListener('mousemove',audBlockMove); document.removeEventListener('mouseup',audBlockUp); persistAudio(); }
function selectAud(id){ selectedAudId=id; const a=AUD.find(x=>x.id===id); const p=document.getElementById('audPanel'); if(!a){p.style.display='none'; return;} p.style.display='flex';
  document.getElementById('audName').textContent='🔊 '+a.src;
  document.getElementById('audVol').value=Math.round((a.volume!=null?a.volume:1)*100);
  document.getElementById('audVolVal').textContent=Math.round((a.volume!=null?a.volume:1)*100)+'%';
  if(_origTime<a.start||_origTime>=a.start+a.dur){ try{seekOriginal(a.start+0.02);}catch(e){} _origTime=a.start+0.02; }
  renderAudioTrack(); }
function deselectAud(){ selectedAudId=null; document.getElementById('audPanel').style.display='none'; if(_libPreviewEl){try{_libPreviewEl.pause();}catch(e){}} renderAudioTrack(); }
function audEditVol(v){ const a=AUD.find(x=>x.id===selectedAudId); if(!a) return; a.volume=Math.max(0,Math.min(2,v/100)); document.getElementById('audVolVal').textContent=Math.round(v)+'%'; if(_audEls[a.id])_audEls[a.id].volume=Math.min(1,a.volume); persistAudio(); }
function audPreview(){ const a=AUD.find(x=>x.id===selectedAudId); if(!a) return; const el=_audEl(a); el.currentTime=0; el.volume=Math.min(1,(a.volume!=null?a.volume:1)); el.play().catch(()=>{}); }
function audDelete(){ const a=AUD.find(x=>x.id===selectedAudId); if(a&&_audEls[a.id]){try{_audEls[a.id].pause();}catch(e){} delete _audEls[a.id];} AUD=AUD.filter(x=>x.id!==selectedAudId); deselectAud(); renderAudioTrack(); persistAudio(); }
function addSoundClip(src, dur, atTime){ const start=(atTime!=null?atTime:(_origTime||0)); const a={id:'aud_'+Date.now(), src, start:Math.max(0,start), dur:Math.max(0.3, dur||3), volume:1, off:0}; AUD.push(a); selectAud(a.id); renderAudioTrack(); persistAudio(); setStatus('Added sound "'+src+'" at '+fmt(a.start)); }

/* ── Right-click menu for image + audio timeline blocks (split / duplicate / delete) ── */
let _ovMenu = { kind:null, id:null };
function showOvMenu(e, kind, id){
  e.preventDefault(); e.stopPropagation();
  if(kind==='img') selectImg(id); else selectAud(id);
  _ovMenu = { kind, id };
  const m=document.getElementById('ovMenu');
  // Mute item only applies to audio clips (images have no sound)
  const mute=document.getElementById('ovMuteItem');
  if(kind==='aud'){ const a=AUD.find(x=>x.id===id); mute.style.display='block';
    mute.textContent = (a && a.volume===0) ? '🔊 Unmute' : '🔇 Mute'; }
  else { mute.style.display='none'; }
  m.style.display='block';
  m.style.left=Math.min(e.clientX, window.innerWidth-180)+'px';
  m.style.top =Math.min(e.clientY, window.innerHeight-150)+'px';
}
function hideOvMenu(){ const m=document.getElementById('ovMenu'); if(m) m.style.display='none'; _ovMenu={kind:null,id:null}; }
function ovMenuAction(act){
  const {kind,id}=_ovMenu; hideOvMenu(); if(!id) return;
  if(kind==='img'){
    if(act==='split') splitImage(id);
    else if(act==='dup') duplicateImage(id);
    else if(act==='del'){ selectedImgId=id; imgDelete(); }
  } else {
    if(act==='mute') toggleAudMute(id);
    else if(act==='split') splitAudio(id);
    else if(act==='dup') duplicateAudio(id);
    else if(act==='del'){ selectedAudId=id; audDelete(); }
  }
}
function toggleAudMute(id){
  const a=AUD.find(x=>x.id===id); if(!a) return;
  if(a.volume===0){ a.volume=(a._premute!=null?a._premute:1); delete a._premute; setStatus('Sound un-muted'); }
  else { a._premute=(a.volume!=null?a.volume:1); a.volume=0; setStatus('Sound muted 🔇'); }
  if(_audEls[id]) _audEls[id].volume=Math.min(1,a.volume);
  if(selectedAudId===id){ const s=document.getElementById('audVol'); if(s){ s.value=Math.round(a.volume*100); document.getElementById('audVolVal').textContent=Math.round(a.volume*100)+'%'; } }
  renderAudioTrack(); persistAudio();
}
document.addEventListener('mousedown', (e)=>{ if(!e.target.closest('#ovMenu')) hideOvMenu(); });
function splitImage(id){
  const m=IMGS.find(x=>x.id===id); if(!m) return; const t=_origTime||0;
  if(t<=m.start+0.05 || t>=m.end-0.05){ setStatus('Move the playhead inside the image clip first, then split'); return; }
  const second={...m, id:'img_'+Date.now(), start:t, end:m.end}; m.end=t;
  IMGS.push(second); selectImg(second.id); _lastImgSig=''; renderImages(); renderImgTrack(); persistImages();
  setStatus('Image clip split at playhead');
}
function duplicateImage(id){
  const src=IMGS.find(x=>x.id===id); if(!src) return;
  const c={...src, id:'img_'+Date.now()}; IMGS.push(c); selectImg(c.id); _lastImgSig='';
  renderImages(); renderImgTrack(); persistImages(); setStatus('Image duplicated');
}
function splitAudio(id){
  const a=AUD.find(x=>x.id===id); if(!a) return; const t=_origTime||0, end=a.start+a.dur;
  if(t<=a.start+0.05 || t>=end-0.05){ setStatus('Move the playhead inside the sound clip first, then split'); return; }
  const off=(a.off||0);
  const second={...a, id:'aud_'+Date.now(), start:t, dur:end-t, off: off+(t-a.start)};
  a.dur=t-a.start;
  AUD.push(second); selectAud(second.id); renderAudioTrack(); persistAudio();
  setStatus('Sound clip split at playhead');
}
function duplicateAudio(id){
  const src=AUD.find(x=>x.id===id); if(!src) return;
  const c={...src, id:'aud_'+Date.now()}; AUD.push(c); selectAud(c.id); renderAudioTrack(); persistAudio();
  setStatus('Sound duplicated');
}

/* ══════════ iMovie-style volume lines (drag up/down to set clip volume) ══════════
 * A horizontal line across each clip; its vertical position = volume. Range 0..2×
 * (100% sits at the middle of the clip; drag up to boost, down to quiet/mute). */
let VIDEO_VOL = 1.0;                 // main (camera/external) audio volume multiplier
const VOL_MAX = 2.0;
function mainVol(){ return Math.min(1, Math.max(0, VIDEO_VOL)); }   // HTMLMediaElement can't exceed 1
function volToY(v, H){ return H * (1 - Math.max(0, Math.min(VOL_MAX, v)) / VOL_MAX); }
function yToVol(y, H){ return Math.max(0, Math.min(VOL_MAX, VOL_MAX * (1 - y / H))); }
function makeVolLine(v, H, onDrag, onEnd){
  const vl=document.createElement('div'); vl.className='vol-line';
  vl.style.top=volToY(v, H)+'px';
  vl.innerHTML='<div class="vol-tag">'+Math.round(v*100)+'%</div><div class="vol-grip"></div>';
  vl.addEventListener('mousedown', e=>{
    e.preventDefault(); e.stopPropagation();
    const sy=e.clientY, startTop=parseFloat(vl.style.top)||0; vl.classList.add('dragging');
    const mv=ev=>{ const ny=Math.max(0, Math.min(H, startTop+(ev.clientY-sy))); const nv=yToVol(ny, H);
      vl.style.top=ny+'px'; vl.querySelector('.vol-tag').textContent=Math.round(nv*100)+'%'; onDrag(nv); };
    const up=()=>{ vl.classList.remove('dragging'); document.removeEventListener('mousemove',mv); document.removeEventListener('mouseup',up); onEnd(); };
    document.addEventListener('mousemove',mv); document.addEventListener('mouseup',up);
  });
  return vl;
}
function applyVideoVolume(){
  const v=mainVol();
  try{ _vidA.volume=v; }catch(e){} try{ _vidB.volume=v; }catch(e){}
  if(typeof _extAudio!=='undefined' && _extAudio){ try{ _extAudio.volume=v; }catch(e){} }
}
let _vidVolTimer=null;
function persistVideoVol(){ if(S)S.video_volume=VIDEO_VOL; clearTimeout(_vidVolTimer);
  _vidVolTimer=setTimeout(()=>{ fetch('/api/video-volume',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({volume:VIDEO_VOL})}).catch(()=>{}); },300); }
function renderVidVolLine(){
  const inner=document.getElementById('tlInner'); const wf=document.getElementById('wfCanvas'); if(!inner||!wf) return;
  const old=document.getElementById('vidVolBand'); if(old) old.remove();
  const H=wf.offsetHeight||40;
  const band=document.createElement('div'); band.id='vidVolBand';
  band.style.top=wf.offsetTop+'px'; band.style.height=H+'px'; band.style.left='0'; band.style.width=inner.style.width||'100%';
  band.title='Video volume — drag up/down';
  const vl=makeVolLine(VIDEO_VOL, H, nv=>{ VIDEO_VOL=nv; applyVideoVolume(); }, ()=>persistVideoVol());
  band.appendChild(vl); inner.appendChild(band);
}

/* ══════════ Per-segment mute of the main (video) audio ══════════
 * MUTED = original-time ranges where the camera/external audio is silenced.
 * Muting a video segment does NOT affect the MP3 sound layers, and vice-versa. */
let MUTED = [];
function persistMuted(){ if(S)S.muted_segs=MUTED.map(m=>[m.start,m.end]);
  fetch('/api/muted-segs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({segs:MUTED.map(m=>[m.start,m.end])})}).catch(()=>{}); }
function isMainMuted(t){ for(const m of MUTED){ if(t>=m.start-0.001 && t<m.end-0.001) return true; } return false; }
function mainVolAt(t){ return isMainMuted(t) ? 0 : mainVol(); }
function applyMainVolAt(t){ const v=mainVolAt(t); try{ video.volume=v; }catch(e){}
  if(typeof _extAudio!=='undefined' && _extAudio){ try{ _extAudio.volume=v; }catch(e){} } }
function isSegMuted(s,e){ return MUTED.some(m=>Math.abs(m.start-s)<0.03 && Math.abs(m.end-e)<0.03); }
function toggleSegMute(s,e){
  const i=MUTED.findIndex(m=>Math.abs(m.start-s)<0.03 && Math.abs(m.end-e)<0.03);
  if(i>=0){ MUTED.splice(i,1); setStatus('Segment un-muted'); }
  else { MUTED.push({start:s,end:e}); setStatus('Segment muted 🔇'); }
  persistMuted(); buildSegmentTiles(); applyMainVolAt(_origTime||0);
}
/* Right-click menu for a video segment (seg-tile) */
let _segMenu = null;
function showSegMenu(e, s, e2){
  e.preventDefault(); e.stopPropagation();
  _segMenu = { s, e: e2 };
  const m=document.getElementById('segMenu');
  m.querySelector('#segMuteItem').textContent = isSegMuted(s,e2) ? '🔊 Unmute segment' : '🔇 Mute segment';
  m.style.display='block';
  m.style.left=Math.min(e.clientX, window.innerWidth-190)+'px';
  m.style.top =Math.min(e.clientY, window.innerHeight-90)+'px';
}
function hideSegMenu(){ const m=document.getElementById('segMenu'); if(m) m.style.display='none'; _segMenu=null; }
function segMenuMute(){ const d=_segMenu; hideSegMenu(); if(d) toggleSegMute(d.s, d.e); }
document.addEventListener('mousedown', (e)=>{ if(!e.target.closest('#segMenu')) hideSegMenu(); });

/* Sound library modal */
let _soundPickTarget = null;   // when set, clicking a library sound drops it at THIS time (not the ticker)
function openSoundLibrary(){ _soundPickTarget = null; setSoundLibHint(''); const m=document.getElementById('soundLibModal'); m.style.display='flex'; refreshSoundLib(); }
// Open the library to place a sound at a specific time (e.g. from a transcript word).
function openSoundLibraryAt(time, label){ _soundPickTarget = time; setSoundLibHint(label ? ('Click a sound to add at “'+label+'”.') : ''); document.getElementById('soundLibModal').style.display='flex'; refreshSoundLib(); }
function setSoundLibHint(extra){ const h=document.getElementById('soundLibHint'); if(h) h.textContent = extra || 'Click a sound to drop it at the playhead. Every MP3 you add is saved here and reusable in any project.'; }
/* Transcript word right-click → add a sound effect at that word's time */
let _wordMenu = null;
function showWordMenu(e, w){
  e.preventDefault(); e.stopPropagation();
  _wordMenu = { start: (w.start||0), label: (w.raw||w.word||'').trim() };
  const m=document.getElementById('wordMenu'); m.style.display='block';
  m.style.left=Math.min(e.clientX, window.innerWidth-200)+'px';
  m.style.top =Math.min(e.clientY, window.innerHeight-70)+'px';
}
function hideWordMenu(){ const m=document.getElementById('wordMenu'); if(m) m.style.display='none'; _wordMenu=null; }
function wordMenuAddSound(){ const w=_wordMenu; hideWordMenu(); if(!w) return; openSoundLibraryAt(w.start, w.label); }
document.addEventListener('mousedown', (e)=>{ if(!e.target.closest('#wordMenu')) hideWordMenu(); });
function closeSoundLibrary(){ document.getElementById('soundLibModal').style.display='none'; if(_libPreviewEl){try{_libPreviewEl.pause();}catch(e){}} }
async function refreshSoundLib(){
  const list=document.getElementById('soundLibList'); list.innerHTML='<div style="color:#888;font-size:12px;">Loading…</div>';
  const d=await fetch('/api/sound-library').then(r=>r.json()).catch(()=>null);
  list.innerHTML='';
  if(!d||!d.sounds||!d.sounds.length){ list.innerHTML='<div style="color:#888;font-size:12px;">No sounds yet — upload an MP3 to start your library.</div>'; return; }
  for(const s of d.sounds){
    const row=document.createElement('div'); row.className='snd-row';
    row.innerHTML='<button class="snd-play" title="Preview">▶</button><span class="snd-name">'+s.src+'</span><span class="snd-dur">'+fmtDur(s.dur)+'</span><button class="snd-play snd-edit" title="Edit trim">✏️</button>';
    row.querySelector('.snd-play').addEventListener('click', ev=>{ ev.stopPropagation(); if(_libPreviewEl){try{_libPreviewEl.pause();}catch(e){}} _libPreviewEl=new Audio('/sound/'+encodeURIComponent(s.src)); _libPreviewEl.play().catch(()=>{}); });
    row.querySelector('.snd-edit').addEventListener('click', ev=>{ ev.stopPropagation(); editLibrarySound(s.src); });
    row.addEventListener('click', ()=>{ const at=_soundPickTarget; _soundPickTarget=null; addSoundClip(s.src, s.dur, at); closeSoundLibrary(); });
    list.appendChild(row);
  }
}
function fmtDur(s){ s=Math.round(s||0); return Math.floor(s/60)+':'+String(s%60).padStart(2,'0'); }
async function handleSoundPick(files, mode){
  if(!files||!files.length) return;
  const f = files[0];   // capture BEFORE clearing the input (this.files is live — value='' empties it)
  document.getElementById('soundUploadInput').value='';
  if(!f) return;
  // Route through the trim / selection screen. mode 'add' (default) also drops it into the video
  // at the playhead; mode 'lib' just saves the trimmed sound to the library.
  stageSoundForTrim(f, mode || 'add');
}

/* ══════════ Sound trim / selection screen ══════════ */
let _trim = null;   // {tempId, dur, peaks, selStart, selEnd, mode, audio, playing, rafId}
async function stageSoundForTrim(file, mode){
  const fd=new FormData(); fd.append('audio', file);
  setStatus('Loading “'+file.name+'”…');
  const r=await fetch('/api/upload-sound-temp',{method:'POST',body:fd}).then(x=>x.json()).catch(()=>null);
  if(!r||!r.ok){ setStatus('Could not load that sound'); return; }
  openTrimScreen(r, mode||'add');
}
async function editLibrarySound(src){
  // Re-trim an existing library sound in place. Stage a copy of it, open the trim screen in 'edit' mode.
  setStatus('Loading “'+src+'” to edit…');
  const blob = await fetch('/sound/'+encodeURIComponent(src)).then(r=>r.blob()).catch(()=>null);
  if(!blob){ setStatus('Could not load that sound'); return; }
  const file = new File([blob], src, {type:'audio/mpeg'});
  const fd=new FormData(); fd.append('audio', file);
  const r=await fetch('/api/upload-sound-temp',{method:'POST',body:fd}).then(x=>x.json()).catch(()=>null);
  if(!r||!r.ok){ setStatus('Could not load that sound'); return; }
  openTrimScreen(r, 'edit', src);
}
function openTrimScreen(d, mode, overwriteSrc){
  const dur=d.dur||0;
  const isEdit = (mode==='edit');
  const selEnd = (dur>0 && !isEdit) ? Math.min(1, (dur>12 ? 8/dur : 1)) : 1;   // edit: start with whole clip
  _trim = { tempId:d.temp_id, dur, peaks:d.peaks||[], selStart:0, selEnd, mode:mode||'add',
            overwrite: overwriteSrc || null,
            audio:new Audio('/sound-temp/'+encodeURIComponent(d.temp_id)), playing:false, rafId:0 };
  document.getElementById('trimName').value = d.name || 'sound';
  document.getElementById('trimNameWrap').style.display = isEdit ? 'none' : 'flex';   // name fixed when editing
  document.getElementById('trimSub').textContent =
    isEdit ? 'Re-trim “'+overwriteSrc+'” — drag the handles and save your changes.'
    : 'Drag the green handles to pick the part you want.';
  // Edit mode: single "Save changes" (overwrite). Otherwise offer both save options.
  document.getElementById('trimSaveBtn').textContent = isEdit ? 'Save changes' : 'Save & add to video';
  document.getElementById('trimSaveBtn').disabled = false;
  document.getElementById('trimLibBtn').style.display = isEdit ? 'none' : '';
  document.getElementById('trimHandleL').onmousedown = (e)=>_trimHandleDrag('L', e);
  document.getElementById('trimHandleR').onmousedown = (e)=>_trimHandleDrag('R', e);
  document.getElementById('trimModal').style.display='flex';
  requestAnimationFrame(()=>{ drawTrimWave(); renderTrimSel(); });
}
function closeTrim(){
  if(_trim){ try{_trim.audio.pause();}catch(e){} if(_trim.rafId) cancelAnimationFrame(_trim.rafId); }
  _trim=null;
  document.getElementById('trimModal').style.display='none';
  document.getElementById('trimPlayhead').style.display='none';
}
function drawTrimWave(){
  const cv=document.getElementById('trimWave'); if(!cv||!_trim) return;
  const w=cv.clientWidth||600, h=cv.clientHeight||96, dpr=Math.min(2,window.devicePixelRatio||1);
  cv.width=Math.round(w*dpr); cv.height=Math.round(h*dpr);
  const ctx=cv.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,w,h);
  const peaks=_trim.peaks, N=peaks.length||1, mid=h/2;
  ctx.fillStyle='#4a7fb0';
  for(let x=0;x<w;x++){ const a=Math.floor(x/w*N), b=Math.max(a+1,Math.floor((x+1)/w*N)); let p=0; for(let i=a;i<b&&i<N;i++) if(peaks[i]>p)p=peaks[i]; const amp=p*(h*0.46); ctx.fillRect(x, mid-amp, 1, Math.max(1,amp*2)); }
}
function renderTrimSel(){
  if(!_trim) return;
  const W=document.getElementById('trimWaveWrap').clientWidth;
  const a=_trim.selStart, b=_trim.selEnd;
  document.getElementById('trimHandleL').style.left=(a*W)+'px';
  document.getElementById('trimHandleR').style.left=(b*W)+'px';
  document.getElementById('trimDimL').style.width=(a*W)+'px';
  document.getElementById('trimDimR').style.width=((1-b)*W)+'px';
  const s=a*_trim.dur, e=b*_trim.dur;
  document.getElementById('trimRange').textContent = fmt(s)+' – '+fmt(e)+'  ('+(e-s).toFixed(2)+'s)';
}
function _trimHandleDrag(which, e){
  if(!_trim) return;
  e.preventDefault(); e.stopPropagation();
  const rect=document.getElementById('trimWaveWrap').getBoundingClientRect();
  const mv=ev=>{ let f=(ev.clientX-rect.left)/rect.width; f=Math.max(0,Math.min(1,f));
    if(which==='L') _trim.selStart=Math.min(f, _trim.selEnd-0.005);
    else _trim.selEnd=Math.max(f, _trim.selStart+0.005);
    renderTrimSel(); };
  const up=()=>{ document.removeEventListener('mousemove',mv); document.removeEventListener('mouseup',up); };
  document.addEventListener('mousemove',mv); document.addEventListener('mouseup',up);
}
function trimTogglePlay(){
  if(!_trim) return;
  const btn=document.getElementById('trimPlayBtn'), ph=document.getElementById('trimPlayhead');
  const W=document.getElementById('trimWaveWrap').clientWidth;
  if(_trim.playing){ _trim.audio.pause(); _trim.playing=false; btn.textContent='▶ Preview'; if(_trim.rafId)cancelAnimationFrame(_trim.rafId); ph.style.display='none'; return; }
  const s=_trim.selStart*_trim.dur, e=_trim.selEnd*_trim.dur;
  try{ _trim.audio.currentTime=s; }catch(err){}
  _trim.audio.play().catch(()=>{});
  _trim.playing=true; btn.textContent='⏸ Stop'; ph.style.display='block';
  const loop=()=>{ if(!_trim||!_trim.playing) return; const t=_trim.audio.currentTime;
    if(t>=e){ _trim.audio.pause(); _trim.playing=false; btn.textContent='▶ Preview'; ph.style.display='none'; return; }
    ph.style.left=((t/_trim.dur)*W)+'px'; _trim.rafId=requestAnimationFrame(loop); };
  _trim.rafId=requestAnimationFrame(loop);
}
async function confirmTrim(modeOverride){
  if(!_trim) return;
  const name=(document.getElementById('trimName').value||'sound').trim()||'sound';
  // 'edit' keeps its overwrite behaviour; otherwise the button chooses add-to-video vs library-only.
  const mode = (_trim.mode==='edit') ? 'edit' : (modeOverride || 'add');
  const s=_trim.selStart*_trim.dur, e=_trim.selEnd*_trim.dur, tempId=_trim.tempId, overwrite=_trim.overwrite;
  const sb=document.getElementById('trimSaveBtn'); sb.disabled=true; sb.textContent='Saving…';
  const body={temp_id:tempId, start:s, end:e, name};
  if(overwrite) body.overwrite=overwrite;
  const r=await fetch('/api/save-trimmed-sound',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)}).then(x=>x.json()).catch(()=>null);
  sb.disabled=false;
  if(!r||!r.ok){ setStatus('Trim failed'); sb.textContent='Save'; return; }
  closeTrim();
  if(mode==='edit'){
    delete _soundPeaks[r.src];                 // waveform changed → drop cached peaks
    // update any placed clips that reference this sound so their duration/waveform refresh
    for(const a of AUD){ if(a.src===r.src){ a.dur=r.dur; a.off=0; } }
    renderAudioTrack(); persistAudio();
    if(document.getElementById('soundLibModal').style.display==='flex') refreshSoundLib();
    setStatus('Updated “'+r.src+'” ✓');
    return;
  }
  if(document.getElementById('soundLibModal').style.display==='flex') refreshSoundLib();
  if(mode==='add') addSoundClip(r.src, r.dur);
  else setStatus('Saved “'+r.src+'” to your sound library ✓');
}
function initAudioOnLoad(){ initAudio(); }

/* ── Diagnostic: window._DIAG=true logs currentTime vs wall-clock after each cut ── */
let _diagN = 0, _diagWall0 = 0, _diagPrevCt = 0;
function _diagStart(label) {
  if (!window._DIAG) return;
  _diagN = 30; _diagWall0 = performance.now(); _diagPrevCt = video.currentTime;
  console.log('[diag] === ' + label + ' | bufRS=' + _buf.readyState + ' bufCT=' + _buf.currentTime.toFixed(3) + ' bufPaused=' + _buf.paused + ' ===');
}
function _diagTick(t) {
  if (!window._DIAG || _diagN <= 0) return;
  const wall = (performance.now() - _diagWall0) / 1000;
  console.log('[diag] wall=' + wall.toFixed(3) + ' ct=' + t.toFixed(3) + ' Δct=' + (t - _diagPrevCt).toFixed(4)
    + ' rs=' + video.readyState + ' seeking=' + (video.seeking?1:0) + ' paused=' + (video.paused?1:0));
  _diagPrevCt = t; _diagN--;
}

/* ── Video playback (double-buffered for seamless cut transitions) ── */
const _vidA = document.getElementById('playerA');
const _vidB = document.getElementById('playerB');
let video = _vidA;   // the ACTIVE element (visible + playing). swaps with the buffer.
let _buf  = _vidB;   // the BACKGROUND element pre-decoding the next clip.
let _bufTarget = -1; // time _buf is currently pre-seeked to (-1 = none)

// Self-heal a failed media load (e.g. the proxy was still building, or a bad range
// hit the original .mov). Without this, one bad load leaves the preview black forever.
[_vidA, _vidB].forEach(v => v.addEventListener('error', () => {
  // If a smooth-mode (edit-proxy) load failed, fall back to the seek-based proxy.
  if (_smoothMode && v === video) { try { exitSmoothMode('load-error'); } catch(e){} return; }
  setTimeout(() => {
    if (!v.error) return;
    const t = v.currentTime || 0;
    v.dataset.proxy = '1'; v.preload = 'auto';
    v.src = '/video?proxy=1&r=' + Date.now();   // cache-bust to force a clean reload
    v.load();
    v.addEventListener('loadedmetadata', () => { try { v.currentTime = t; } catch(e){} }, { once:true });
  }, 1200);
}));

function _setActiveClasses() {
  video.className = 'bufactive';
  _buf.className  = 'bufhidden';
}
_setActiveClasses();
_buf.muted = true;

// PHASE-LOCK the buffer: it plays the upcoming content shifted forward by the cut
// length, so it *arrives* at nextStart exactly as the active reaches the cut. One
// initial seek to set the phase, then it free-runs at 1x (decoder stays warm, no
// repeated flushing). `desired` = where the buffer should be right now.
function _armBufferPhase(nextStart, desired) {
  _bufTarget = nextStart;
  _buf.muted = true;
  try { _buf.currentTime = Math.max(0, desired); } catch(e){}
  _buf.play().catch(() => {});
}
function _phaseLockBuffer(desired) {
  if (_bufTarget < 0) return;
  // Correct only on LARGE drift — a small drift is fine, and re-seeking would flush
  // the decoder (the very thing we're avoiding).
  if (Math.abs(_buf.currentTime - desired) > 0.07) { try { _buf.currentTime = Math.max(0, desired); } catch(e){} }
}

// Hand off to the buffer, which is already playing warm AND already positioned at
// ~targetT (from phase-locking) — so NO seek is needed → no decoder flush → no stall.
function _swapBuffers(targetT) {
  if (Math.abs(_buf.currentTime - targetT) > 0.25) { try { _buf.currentTime = targetT; } catch(e){} }
  const old = video;
  video = _buf; _buf = old;
  _bufTarget = -1;
  _setActiveClasses();
  applyCropToPreview();                 // keep crop styling on the now-active element
  video.muted = intendedVideoMuted();   // camera audio resumes; ext stays muted
  if (video.paused) video.play().catch(() => {});
  old.pause(); old.muted = true;        // retire the old element (becomes the buffer)
  if (_extAudio) _extSeek(video.currentTime - _extOffset);
  _diagStart('SWAP→' + targetT.toFixed(3));
}

function togglePlay() { video.paused ? video.play() : video.pause(); }
function seekTo(t)    { _bufTarget = -1; seekOriginal(t); }

function scrollTimelineToTime(t) {
  const sc = document.getElementById('tlScroll');
  sc.scrollLeft = X(t) - sc.clientWidth / 2;
}

/* ── Rendered smooth-preview (plays the edited cut as one continuous file) ── */
let _smoothMode  = false;   // playing the rendered edit proxy (gapless)
let _editMap     = null;    // [{oStart,oEnd,eStart}] mapping edited↔original time
let _editTotal   = 0;       // total edited duration
let _origTime    = 0;       // current ORIGINAL-timeline position
let _renderTimer = null;

// Build the edited↔original time map from the current kept segments.
function buildEditMap() {
  const keeps = getKeepSegments();
  let acc = 0;
  _editMap = keeps.map(k => { const seg = {oStart:k.start, oEnd:k.end, eStart:acc}; acc += (k.end-k.start); return seg; });
  _editTotal = acc;
}
function editedToOriginal(te) {
  if (!_editMap || !_editMap.length) return te;
  for (const s of _editMap) { if (te <= s.eStart + (s.oEnd - s.oStart) + 0.0005) return s.oStart + (te - s.eStart); }
  const last = _editMap[_editMap.length-1]; return last.oEnd;
}
function originalToEdited(to) {
  if (!_editMap || !_editMap.length) return to;
  for (const s of _editMap) { if (to < s.oStart) return s.eStart; if (to <= s.oEnd) return s.eStart + (to - s.oStart); }
  return _editTotal;
}
// Seek by ORIGINAL-timeline time, regardless of which mode we're in.
function seekOriginal(to) {
  _origTime = to;
  if (_smoothMode) video.currentTime = originalToEdited(to);
  else video.currentTime = to;
}

function enterSmoothMode(sig) {
  if (_smoothMode || !sig) return;
  buildEditMap();
  const to = video.currentTime;
  _origTime = to;
  const playing = !video.paused;
  _smoothMode = true;
  if (_buf) { try { _buf.pause(); } catch(e){} }
  if (_extAudio) { try { _extAudio.pause(); } catch(e){} }
  video.muted = false;                         // edit proxy has the right audio baked in
  video.src = '/api/edit-proxy?sig=' + encodeURIComponent(sig);
  video.addEventListener('loadedmetadata', () => {
    try { video.currentTime = originalToEdited(to); } catch(e){}
    if (playing) video.play().catch(()=>{});
    applyCropToPreview();
  }, { once: true });
  const b = document.getElementById('smoothBadge');
  b.style.display = ''; b.textContent = '✓ smooth'; b.style.color = '#27ae60'; b.style.borderColor = '#2a4a32';
  setStatus('Smooth preview ✓ (rendered)');
}
function exitSmoothMode(reason) {
  if (!_smoothMode) return;
  const to = editedToOriginal(video.currentTime);
  const playing = !video.paused;
  _smoothMode = false;
  _origTime = to;
  video.src = '/video?proxy=1';                // back to the seek-based original proxy
  video.muted = intendedVideoMuted();
  video.addEventListener('loadedmetadata', () => {
    try { video.currentTime = to; } catch(e){}
    if (playing) video.play().catch(()=>{});
    applyCropToPreview();
    if (_extAudio && _audioMode === 'ext' && playing) _extAudio.play().catch(()=>{});
  }, { once: true });
}

// Called after every cut change: invalidate smooth mode and (debounced) re-render.
function onEditChanged() {
  if (_smoothMode) exitSmoothMode('edited');
  const b = document.getElementById('smoothBadge');
  b.style.display=''; b.textContent='⏳ rendering smooth…'; b.style.color='#aaa'; b.style.borderColor='#333';
  clearTimeout(_renderTimer);
  _renderTimer = setTimeout(requestEditRender, 1800);
}
function requestEditRender() {
  if (!S || S.no_video) return;
  fetch('/api/render-edit-proxy', { method:'POST' }).then(r=>r.json()).then(d=>{
    if (d.state === 'ready') maybeEnterSmooth();
    else pollEditRender();
  }).catch(()=>{});
}
function pollEditRender() {
  fetch('/api/edit-proxy-status').then(r=>r.json()).then(d=>{
    if (d.fresh) maybeEnterSmooth();
    else if (d.state === 'rendering') setTimeout(pollEditRender, 1200);
    // else: a newer edit superseded it; onEditChanged's debounce will re-request
  }).catch(()=> setTimeout(pollEditRender, 2000));
}
// On page load, get onto the EXACT baked-audio preview instead of the imprecise
// live <audio> fallback. If a rendered proxy for the current edit already exists,
// enter smooth mode instantly; otherwise (and only when external audio is in play,
// where exact sync matters) kick a render and swap in when it's ready.
function initSmoothOnLoad() {
  if (!S || S.no_video) return;
  fetch('/api/edit-proxy-status').then(r => r.json()).then(d => {
    if (d.fresh) { enterSmoothMode(d.sig); return; }
    if (S.has_external_audio) {
      const b = document.getElementById('smoothBadge');
      if (b) { b.style.display = ''; b.textContent = '⏳ rendering smooth…'; b.style.color = '#aaa'; b.style.borderColor = '#333'; }
      requestEditRender();
    }
  }).catch(() => {});
}
function maybeEnterSmooth() {
  // Only switch if the render still matches the current edit
  fetch('/api/edit-proxy-status').then(r=>r.json()).then(d=>{
    if (d.fresh) enterSmoothMode(d.sig);
  });
}

/* ── Crop / reframe (16:9) ── */
// Apply the saved crop to the MAIN preview as a live "crop to fill" view.
function applyCropToPreview() {
  const panel = document.querySelector('.preview-panel');
  const stage = document.getElementById('videoStage');
  if (!stage || !panel) return;
  const vids = [_vidA, _vidB];   // style BOTH buffers identically
  const c = S && S.crop, vw = S && S.video_w, vh = S && S.video_h;
  if (!c || !vw || !vh) {
    // No crop → show the full frame contained
    stage.classList.remove('cropping');
    stage.style.width = '100%'; stage.style.height = '100%';
    vids.forEach(v => { v.style.width='100%'; v.style.height='100%'; v.style.left='0'; v.style.top='0'; v.style.objectFit='contain'; });
    try { _lastCapSig = ''; renderCaptions(); } catch(e) {}
    return;
  }
  // Largest 16:9 stage that fits the panel
  const pw = panel.clientWidth, ph = panel.clientHeight;
  let sw = pw, sh = pw * 9 / 16;
  if (sh > ph) { sh = ph; sw = ph * 16 / 9; }
  stage.classList.add('cropping');
  stage.style.width = sw + 'px'; stage.style.height = sh + 'px';
  const scale = sw / c.w;             // display px per source px
  vids.forEach(v => {
    v.style.objectFit = 'fill';
    v.style.width  = (vw * scale) + 'px';
    v.style.height = (vh * scale) + 'px';
    v.style.left   = (-c.x * scale) + 'px';
    v.style.top    = (-c.y * scale) + 'px';
  });
  try { _lastCapSig = ''; renderCaptions(); } catch(e) {}
}
window.addEventListener('resize', applyCropToPreview);

/* ── Colour adjust (brightness / contrast / saturation) ── */
function applyAdjustToPreview() {
  const a = (S && S.adjust) ? S.adjust : { brightness: 0, contrast: 1, saturation: 1 };
  const f = `brightness(${(1 + (a.brightness || 0)).toFixed(3)}) contrast(${(a.contrast || 1).toFixed(3)}) saturate(${(a.saturation || 1).toFixed(3)})`;
  [_vidA, _vidB].forEach(v => { if (v) v.style.filter = f; });
}
function updateColorUI() {
  const a = (S && S.adjust) ? S.adjust : { brightness: 0, contrast: 1, saturation: 1 };
  document.getElementById('adjBright').value = Math.round((a.brightness || 0) * 100);
  document.getElementById('adjContrast').value = Math.round((a.contrast || 1) * 100);
  document.getElementById('adjSat').value = Math.round((a.saturation || 1) * 100);
  document.getElementById('adjBVal').textContent = ((a.brightness || 0) > 0 ? '+' : '') + Math.round((a.brightness || 0) * 100) + '%';
  document.getElementById('adjCVal').textContent = Math.round((a.contrast || 1) * 100) + '%';
  document.getElementById('adjSVal').textContent = Math.round((a.saturation || 1) * 100) + '%';
  const changed = (a.brightness || 0) !== 0 || (a.contrast || 1) !== 1 || (a.saturation || 1) !== 1;
  document.getElementById('colorBtn').classList.toggle('active', changed);
}
function toggleColorPanel() {
  const p = document.getElementById('colorPanel');
  const show = p.style.display === 'none';
  p.style.display = show ? 'block' : 'none';
  if (show) updateColorUI();
}
let _adjSaveTimer = null;
function setAdjust(field, val) {
  if (!S) return;
  if (!S.adjust) S.adjust = { brightness: 0, contrast: 1, saturation: 1 };
  S.adjust[field] = parseFloat(val);
  applyAdjustToPreview(); updateColorUI();
  clearTimeout(_adjSaveTimer);
  _adjSaveTimer = setTimeout(() => {
    fetch('/api/set-adjust', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(S.adjust) }).catch(() => {});
  }, 300);
}
function resetAdjust() {
  if (!S) return;
  S.adjust = { brightness: 0, contrast: 1, saturation: 1 };
  applyAdjustToPreview(); updateColorUI();
  fetch('/api/set-adjust', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(S.adjust) });
}

/* ── Crop editor modal ── */
let _cropRect = null;      // working crop {x,y,w,h} in SOURCE px
let _cropDisp = {x:0,y:0,w:0,h:0};  // displayed video rect (px) in the modal
let _cropRO = null;        // ResizeObserver: re-map the box when the video lays out/resizes

function openCropEditor() {
  if (!S || !S.video_w) { setStatus('No video loaded'); return; }
  const modal = document.getElementById('cropModal');
  const cv = document.getElementById('cropVideo');
  modal.classList.add('active');
  const startT = video.currentTime || 1;
  _cropRect = S.crop ? {...S.crop} : default169(S.video_w, S.video_h);
  const sync = () => { if (cv.videoWidth) { measureCropDisp(); drawCropBox(); updateCropResult(); } };
  const onReady = () => {
    try { cv.currentTime = startT; } catch(e){}
    // Measure only AFTER the element has laid out at its real (video) size.
    requestAnimationFrame(() => requestAnimationFrame(sync));
  };
  // preload="auto" + load() so the FRAME and real dimensions are available immediately
  // (with preload="none" the element stays 300×150 and black until played — which broke
  // both the box mapping and made the frame invisible).
  if (!cv.src) { cv.preload = 'auto'; cv.src = '/video?proxy=1'; cv.load(); }
  if (cv.readyState >= 2 && cv.videoWidth) onReady();
  else cv.addEventListener('loadeddata', onReady, {once:true});
  // Redraw the output preview whenever a new frame is ready (scrub or play)
  cv.addEventListener('seeked', updateCropResult);
  cv.addEventListener('timeupdate', updateCropResult);
  // Re-map the box if the video element resizes (late layout, window resize)
  if (!_cropRO) {
    _cropRO = new ResizeObserver(() => { if (modal.classList.contains('active')) sync(); });
    _cropRO.observe(cv);
  }
  setTimeout(sync, 150);
  setTimeout(sync, 450);
}
function closeCropEditor() {
  const cv = document.getElementById('cropVideo');
  cv.pause();
  document.getElementById('cropModal').classList.remove('active');
}

// Play/pause the clip inside the crop editor so you can preview motion while framing
function cropTogglePlay() {
  const cv = document.getElementById('cropVideo');
  const btn = document.getElementById('cropPlayBtn');
  if (cv.paused) { cv.play().catch(()=>{}); btn.textContent = '⏸ Pause'; _cropRAF(); }
  else { cv.pause(); btn.textContent = '▶ Play here'; }
}
function _cropRAF() {
  const cv = document.getElementById('cropVideo');
  if (cv.paused) return;
  updateCropResult();
  requestAnimationFrame(_cropRAF);
}

// Draw the cropped region scaled to fill the 16:9 output-preview canvas
function updateCropResult() {
  const cv = document.getElementById('cropVideo');
  const cn = document.getElementById('cropResult');
  if (!cv || !cn || !cv.videoWidth || !_cropRect) return;
  const ctx = cn.getContext('2d');
  const f = cv.videoWidth / S.video_w;   // cropVideo native px per source px
  try {
    ctx.drawImage(cv, _cropRect.x*f, _cropRect.y*f, _cropRect.w*f, _cropRect.h*f,
                  0, 0, cn.width, cn.height);
  } catch(e) {}
}

function default169(w, h) {
  let cw, ch;
  if (w / h > 16/9) { ch = h; cw = Math.round(h * 16/9); }
  else { cw = w; ch = Math.round(w * 9/16); }
  return { x: Math.round((w-cw)/2), y: Math.round((h-ch)/2), w: cw, h: ch };
}

function measureCropDisp() {
  const cv = document.getElementById('cropVideo');
  const r = cv.getBoundingClientRect();   // actual rendered video rect (px), post-layout
  _cropDisp = { x:0, y:0, w: r.width, h: r.height };
}
function src2disp() {
  // display px per source px — guard against a not-yet-laid-out element (would shrink the box)
  return (_cropDisp.w && S.video_w) ? _cropDisp.w / S.video_w : 1;
}

function drawCropBox() {
  const box = document.getElementById('cropBox');
  const k = src2disp();
  box.style.left   = (_cropRect.x * k) + 'px';
  box.style.top    = (_cropRect.y * k) + 'px';
  box.style.width  = (_cropRect.w * k) + 'px';
  box.style.height = (_cropRect.h * k) + 'px';
  updateCropResult();   // keep the output preview in sync with the box
}

function clampCrop() {
  // keep 16:9 and within source bounds
  _cropRect.w = Math.max(160, Math.min(_cropRect.w, S.video_w));
  _cropRect.h = Math.round(_cropRect.w * 9 / 16);
  if (_cropRect.h > S.video_h) { _cropRect.h = S.video_h; _cropRect.w = Math.round(_cropRect.h * 16/9); }
  _cropRect.x = Math.max(0, Math.min(_cropRect.x, S.video_w - _cropRect.w));
  _cropRect.y = Math.max(0, Math.min(_cropRect.y, S.video_h - _cropRect.h));
}

// Drag body (pan) + corner handles (resize, 16:9 locked)
(function initCropDrag(){
  const box = document.getElementById('cropBox');
  let mode=null, startX=0, startY=0, orig=null;
  function down(e){
    const handle = e.target.closest('.crop-handle');
    mode = handle ? handle.dataset.h : 'move';
    startX = e.clientX; startY = e.clientY; orig = {..._cropRect};
    e.preventDefault(); e.stopPropagation();
    document.addEventListener('mousemove', move);
    document.addEventListener('mouseup', up);
  }
  function move(e){
    if (!mode) return;
    const k = src2disp();
    const dx = (e.clientX - startX) / k, dy = (e.clientY - startY) / k;  // source px
    if (mode === 'move') {
      _cropRect.x = orig.x + dx; _cropRect.y = orig.y + dy;
    } else {
      // resize from a corner, keep 16:9; anchor the opposite corner
      let nw = orig.w + (mode==='tr'||mode==='br' ? dx : -dx);
      nw = Math.max(160, nw);
      let nh = nw * 9/16;
      let nx = orig.x, ny = orig.y;
      if (mode==='tl') { nx = orig.x + (orig.w - nw); ny = orig.y + (orig.h - nh); }
      if (mode==='tr') { ny = orig.y + (orig.h - nh); }
      if (mode==='bl') { nx = orig.x + (orig.w - nw); }
      _cropRect = {x:nx, y:ny, w:nw, h:nh};
    }
    clampCrop(); drawCropBox();
  }
  function up(){ mode=null; document.removeEventListener('mousemove', move); document.removeEventListener('mouseup', up); }
  box.addEventListener('mousedown', down);
})();

function cropReset() { _cropRect = default169(S.video_w, S.video_h); clampCrop(); drawCropBox(); }
function cropRemove() { _cropRect = null; cropSave(null); closeCropEditor(); }
async function cropApply() {
  clampCrop();
  const r = {x:Math.round(_cropRect.x), y:Math.round(_cropRect.y), w:Math.round(_cropRect.w), h:Math.round(_cropRect.h)};
  await cropSave(r);
  closeCropEditor();
}
async function cropSave(rect) {
  const resp = await fetch('/api/set-crop', { method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ crop: rect }) });
  const d = await resp.json();
  S.crop = d.crop;
  applyCropToPreview();
  updateCropBtn();
  setStatus(rect ? 'Crop applied (16:9)' : 'Crop removed');
}
function updateCropBtn() {
  const btn = document.getElementById('cropBtn');
  if (!btn) return;
  const cropped = S && S.crop && S.video_w && (S.crop.w < S.video_w || S.crop.h < S.video_h || S.crop.x>0 || S.crop.y>0);
  btn.style.color = cropped ? '#27ae60' : '';
  btn.textContent = cropped ? '⛶ Frame ✓' : '⛶ Edit Frame';
}

/* ── External audio sync ── */
let _extAudio   = null;   // HTMLAudioElement for external mic
let _extOffset  = 0;      // seconds: extTime = videoTime - _extOffset
let _extSeeking = false;  // true while a (muted) seek is in flight
let _audioMode  = 'ext';  // 'ext' (external mic) | 'original' (camera audio)

function initExtAudio() {
  if (!S || !S.has_external_audio) {
    document.getElementById('audioSrcBtn').style.display = 'none';
    document.getElementById('alignAudioBtn').style.display = 'none';
    document.getElementById('removeExtAudioBtn').style.display = 'none';
    return;
  }
  _extOffset = S.ext_audio_offset || 0;
  _extAudio  = new Audio('/ext-audio');
  _extAudio.preload = 'auto';
  _audioMode = 'ext';
  // Restore volume once a seek settles — hides the seek pop/click
  _extAudio.addEventListener('seeked', () => { _extAudio.volume = mainVolAt(_origTime||0); _extSeeking = false; });
  // Show the source toggle and apply the default (external)
  document.getElementById('audioSrcBtn').style.display = '';
  document.getElementById('alignAudioBtn').style.display = '';
  document.getElementById('removeExtAudioBtn').style.display = '';
  applyAudioMode();
}
async function removeExtAudio() {
  if (!confirm('Remove the external audio and go back to the camera’s own audio? Your cuts and captions are kept.')) return;
  setStatus('Removing external audio, restoring camera audio…');
  // stop any external-audio playback and reset the audio system
  try { if (_extAudio) _extAudio.pause(); } catch (e) {}
  _extAudio = null; _audioMode = 'original'; if (typeof video !== 'undefined') video.muted = false;
  const r = await fetch('/api/remove-ext-audio', { method: 'POST' }).then(r => r.json()).catch(() => null);
  if (!r || !r.ok) { setStatus('Could not remove external audio'); return; }
  document.getElementById('audioSrcBtn').style.display = 'none';
  document.getElementById('alignAudioBtn').style.display = 'none';
  document.getElementById('removeExtAudioBtn').style.display = 'none';
  const eb = document.getElementById('extAudioBtn'); if (eb) { eb.textContent = '🎙 Ext Audio'; eb.style.color = ''; }
  setStatus('External audio removed — camera audio restored. Reloading…');
  // Reload so the player fetches the NEW proxy (fresh signature = camera audio, not the
  // stale external-audio proxy) and rebuilds all audio state cleanly.
  setTimeout(() => location.reload(), 700);
}

/* ── Manual drag-to-align external audio ── */
let _alignData = null, _alignOffset = 0, _alignPPS = 18, _alignDrag = null, _alignOrigOffset = 0;

async function openAlignModal() {
  const m = document.getElementById('alignModal');
  m.style.display = 'flex';
  document.getElementById('alignStatus').textContent = 'Loading waveforms…';
  if (_smoothMode) exitSmoothMode('align');      // drop to the proxy + <audio> path for live testing
  if (_extAudio) try { _extAudio.pause(); } catch (e) {}
  _alignOrigOffset = _extOffset;
  try {
    const d = await fetch('/api/align-data').then(r => r.json());
    if (!d.ok) { document.getElementById('alignStatus').textContent = d.error || 'No external audio loaded.'; return; }
    _alignData = d;
    _alignOffset = d.offset || 0;
    drawAlign(); updateAlignReadout();
    document.getElementById('alignStatus').textContent = 'Drag the blue track to line up the peaks.';
  } catch (e) {
    document.getElementById('alignStatus').textContent = 'Failed to load: ' + e;
  }
}
function closeAlignModal(applied) {
  document.getElementById('alignModal').style.display = 'none';
  _alignDrag = null;
  if (_extAudio) try { _extAudio.pause(); } catch (e) {}
  if (!applied) { _extOffset = _alignOrigOffset; }   // discard the trial offset
  video.pause();
}
function alignZoom(f) { _alignPPS = Math.max(4, Math.min(80, _alignPPS * f)); drawAlign(); }
function alignNudge(d) { _alignOffset += d; drawAlign(); updateAlignReadout(); }

function _drawWave(ctx, arr, sps, pps, offsetSec, baseY, amp, color) {
  ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.beginPath();
  for (let i = 0; i < arr.length; i++) {
    const x = (i / sps + offsetSec) * pps;       // video-time axis: video_t = ext_t + offset
    if (x < 0) continue;
    const h = arr[i] * amp;
    ctx.moveTo(x, baseY - h); ctx.lineTo(x, baseY + h);
  }
  ctx.stroke();
}
function drawAlign() {
  const d = _alignData; if (!d) return;
  const cv = document.getElementById('alignCanvas'), pps = _alignPPS;
  const totalSec = d.window + 12;                // a little slack for positive offsets
  cv.width = Math.round(totalSec * pps); cv.height = 260;
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#141414'; ctx.fillRect(0, 0, cv.width, cv.height);
  // time grid
  ctx.font = '10px sans-serif';
  for (let s = 0; s <= totalSec; s += 5) {
    const x = s * pps;
    ctx.strokeStyle = (s % 10 === 0) ? '#333' : '#242424';
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, cv.height); ctx.stroke();
    ctx.fillStyle = '#666'; ctx.fillText(s + 's', x + 2, cv.height - 4);
  }
  ctx.strokeStyle = '#444'; ctx.beginPath(); ctx.moveTo(0, 128); ctx.lineTo(cv.width, 128); ctx.stroke();
  _drawWave(ctx, d.video, d.sps, pps, 0, 70, 55, '#999');           // camera (fixed)
  _drawWave(ctx, d.ext, d.sps, pps, _alignOffset, 186, 55, '#3d9be9'); // external (draggable)
  ctx.fillStyle = '#bbb'; ctx.fillText('📹 camera audio (fixed reference)', 6, 16);
  ctx.fillStyle = '#3d9be9'; ctx.fillText('🎙 external audio — drag to align', 6, 142);
}
function updateAlignReadout() {
  const o = _alignOffset;
  document.getElementById('alignOffsetVal').textContent = o.toFixed(2) + 's';
  document.getElementById('alignOffsetDesc').textContent =
    o < -0.005 ? `external recording started ${(-o).toFixed(2)}s before the video`
    : o > 0.005 ? `external recording started ${o.toFixed(2)}s after the video`
    : 'aligned at zero';
}
function alignAuto() {
  if (_alignData && typeof _alignData.offset === 'number') {
    _alignOffset = _alignData.offset; drawAlign(); updateAlignReadout();
    document.getElementById('alignStatus').textContent = 'Reset to auto-detected offset — fine-tune by dragging.';
  }
}
function alignTest() {
  if (!_alignData) return;
  const T = Math.min(20, (_alignData.video_duration || 30) / 3);
  _extOffset = _alignOffset;          // apply trial offset to the live <audio> sync
  _audioMode = 'ext';
  video.muted = true;
  video.currentTime = T;
  video.play().catch(() => {});
  if (_extAudio) { _extSeek(T - _extOffset); _extAudio.play().catch(() => {}); }
  document.getElementById('alignStatus').textContent = 'Playing 6s at this offset…';
  clearTimeout(window._alignTestTimer);
  window._alignTestTimer = setTimeout(() => {
    video.pause(); if (_extAudio) _extAudio.pause();
    document.getElementById('alignStatus').textContent = 'Drag to adjust, or Apply when it lines up.';
  }, 6000);
}
async function alignApply() {
  document.getElementById('alignStatus').textContent = 'Applying & re-rendering…';
  try {
    const d = await fetch('/api/set-ext-offset', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ offset: _alignOffset }),
    }).then(r => r.json());
    if (!d.ok) { document.getElementById('alignStatus').textContent = 'Failed: ' + (d.error || '?'); return; }
    _extOffset = _alignOffset;
    if (S) S.ext_audio_offset = _alignOffset;
    closeAlignModal(true);
    setStatus('Audio offset applied — re-rendering smooth preview…');
    const b = document.getElementById('smoothBadge');
    if (b) { b.style.display = ''; b.textContent = '⏳ rendering smooth…'; b.style.color = '#aaa'; b.style.borderColor = '#333'; }
    pollEditRender();
  } catch (e) {
    document.getElementById('alignStatus').textContent = 'Failed: ' + e;
  }
}
// Drag handling on the canvas
(function () {
  const attach = () => {
    const cv = document.getElementById('alignCanvas');
    if (!cv) { setTimeout(attach, 300); return; }
    cv.addEventListener('mousedown', e => { _alignDrag = { x: e.clientX, off: _alignOffset }; e.preventDefault(); });
    window.addEventListener('mousemove', e => {
      if (!_alignDrag) return;
      _alignOffset = _alignDrag.off + (e.clientX - _alignDrag.x) / _alignPPS;
      drawAlign(); updateAlignReadout();
    });
    window.addEventListener('mouseup', () => { _alignDrag = null; });
  };
  attach();
})();

// Switch between external mic and original camera audio (A/B compare)
function toggleAudioSource() {
  _audioMode = (_audioMode === 'ext') ? 'original' : 'ext';
  applyAudioMode();
  setStatus(_audioMode === 'ext' ? 'Audio: external mic' : 'Audio: original camera');
}

function applyAudioMode() {
  const btn = document.getElementById('audioSrcBtn');
  if (_audioMode === 'original') {
    video.muted = false;                  // play camera audio
    if (_extAudio) _extAudio.pause();      // silence external
    if (btn) { btn.textContent = '📹 Using: Original'; btn.style.color = '#e67e22'; }
  } else {
    video.muted = true;                   // silence camera
    if (btn) { btn.textContent = '🎙 Using: Ext'; btn.style.color = '#27ae60'; }
    if (_extAudio && !video.paused) {     // resume external in sync
      const extT = video.currentTime - _extOffset;
      if (extT >= 0) { _extSeek(extT); _extAudio.play().catch(() => {}); }
    }
  }
}

// Seek the external audio with a brief mute so the jump doesn't pop/click.
// Idempotent: skips if already within 30ms of the target.
function _extSeek(target) {
  if (!_extAudio) return;
  target = Math.max(0, target);
  if (Math.abs(_extAudio.currentTime - target) < 0.03) return;
  _extAudio.volume = 0;        // mute to hide the discontinuity
  _extSeeking = true;
  _extAudio.currentTime = target;
}

function _extSync() {
  if (!_extAudio || _audioMode !== 'ext') return;   // 'original' mode → camera audio only
  const extT = video.currentTime - _extOffset;
  if (extT < 0) { _extAudio.pause(); return; }
  // While the video is SEEKING (skipping a cut), hold the audio so it doesn't run
  // ahead of the frozen frame. Resumes the instant the new frame is ready.
  if (video.seeking) { if (!_extAudio.paused) _extAudio.pause(); return; }
  // Only hard-correct LARGE drift (e.g. tab throttling). Small drift is inaudible.
  if (!_extSeeking && Math.abs(_extAudio.currentTime - extT) > 0.35) _extSeek(extT);
  if (!video.paused && _extAudio.paused) _extAudio.play().catch(() => {});
  if (video.paused  && !_extAudio.paused) _extAudio.pause();
  // Safety: if somehow left muted while not actively seeking, restore volume
  if (!_extSeeking && _extAudio.volume === 0 && !isMainMuted(_origTime||0)) _extAudio.volume = mainVol();
}

// The video should only be audible when we're using the camera's own audio
// (i.e. NOT using an external track in 'ext' mode).
function intendedVideoMuted() { return !!(_extAudio && _audioMode === 'ext'); }

// Attach a listener to BOTH buffer elements; `onActive` runs only for whichever
// element is currently active, so background-buffer events don't cause side effects.
function onActiveVideo(evt, onActive) {
  const h = function(e) { if (e.currentTarget === video) onActive(e); };
  _vidA.addEventListener(evt, h);
  _vidB.addEventListener(evt, h);
}

onActiveVideo('play', () => {
  document.getElementById('playBtn').textContent = '⏸';
  // In smooth mode the audio is BAKED into the proxy — the separate <audio> element
  // must stay out of it (it would double up, and currentTime here is EDITED time).
  if (_smoothMode) { video.muted = false; return; }
  if (_extAudio && _audioMode === 'ext') {
    const extT = video.currentTime - _extOffset;
    if (extT >= 0) { _extSeek(extT); _extAudio.play().catch(() => {}); }
  }
});
onActiveVideo('pause', () => {
  document.getElementById('playBtn').textContent = '▶';
  if (_smoothMode) return;
  if (_extAudio) _extAudio.pause();
});
// Mute the active element during a direct seek so a blip can't leak; restore after.
// Smooth mode keeps the baked audio audible and never touches the <audio> element.
onActiveVideo('seeking', () => { if (!_smoothMode) video.muted = true; });
onActiveVideo('seeked', () => {
  if (_smoothMode) { video.muted = false; return; }
  video.muted = intendedVideoMuted();
  if (_extAudio) _extSeek(video.currentTime - _extOffset);
});

// Pre-compute merged cut ranges (gap 0.15 — same as export)
function getMergedCuts() {
  const active = S.cuts.filter(c => c.active).sort((a,b) => a.start - b.start);
  const merged = [];
  for (const c of active) {
    if (merged.length && c.start <= merged[merged.length-1].end + 0.15) {
      merged[merged.length-1].end = Math.max(merged[merged.length-1].end, c.end);
    } else {
      merged.push({start: c.start, end: c.end});
    }
  }
  return merged;
}

// Kept segments = complement of merged cuts. This is IDENTICAL to what the
// export keeps (cuts_to_keeps with the same 0.15 merge + 0.01 min-gap), so the
// preview plays exactly what the export produces — no more, no less.
function getKeepSegments() {
  const merged = getMergedCuts();
  const keeps = [];
  let pos = 0;
  for (const mc of merged) {
    if (pos < mc.start - 0.01) keeps.push({start: pos, end: mc.start});
    pos = Math.max(pos, mc.end);
  }
  if (pos < S.duration - 0.01) keeps.push({start: pos, end: S.duration});
  return keeps;
}

// rAF playback driver with double-buffering: while the active element plays the
// current kept segment, the buffer element pre-decodes the NEXT segment's first
// frame. At the cut we swap to the buffer — instant, no seek/freeze, no audio bleed.
let _lastSkipTime = -1;
function _playbackLoop() {
  if (!S || !S.cuts) { requestAnimationFrame(_playbackLoop); return; }

  // ── SMOOTH MODE: playing the rendered edit proxy continuously (no cut-skipping).
  // currentTime is EDITED time; map it back to the original timeline for the UI.
  if (_smoothMode) {
    const to = editedToOriginal(video.currentTime);
    _origTime = to;
    document.getElementById('timeDisplay').textContent = fmt(to);
    const x = X(to);
    document.getElementById('playhead').style.left = x + 'px';
    const sc = document.getElementById('tlScroll');
    const L = sc.scrollLeft, W = sc.clientWidth;
    if (!video.paused && (x < L + 60 || x > L + W - 60)) sc.scrollLeft = x - W / 2;
    highlightWord(to);
    applyMainVolAt(to);   // per-segment mute of the main audio
    captionTick(to); imgTick(to); audioTick(to);
    requestAnimationFrame(_playbackLoop);
    return;
  }

  const t = video.currentTime;
  _diagTick(t);
  _origTime = t;

  if (!video.paused) {
    const keeps = getKeepSegments();
    // Which kept segment are we in?
    let ci = -1;
    for (let i = 0; i < keeps.length; i++) {
      if (t >= keeps[i].start - 0.02 && t < keeps[i].end) { ci = i; break; }
    }

    if (ci >= 0) {
      _lastSkipTime = -1;
      const cur  = keeps[ci];
      const next = keeps[ci + 1] || null;
      // In the final ~0.7s before the cut, run the buffer phase-locked: shifted ahead
      // by the cut length so it reaches next.start right as the active reaches cur.end.
      if (next && (cur.end - t) < 0.7) {
        const gap = next.start - cur.end;     // length of the cut being skipped
        const desired = t + gap;              // buffer should be `gap` ahead of the active
        if (_bufTarget !== next.start) _armBufferPhase(next.start, desired);
        else _phaseLockBuffer(desired);
      }
      // Reached the cut → hand off to the warm, already-positioned buffer (no seek).
      if (t >= cur.end - 0.006) {
        if (next) { _swapBuffers(next.start); requestAnimationFrame(_playbackLoop); return; }
        else { video.pause(); }   // last segment ended
      }
    } else if (keeps.length) {
      // Not inside any keep (scrubbed into a cut, or started mid-cut). One-time direct
      // jump to the next keep — phase-locking only applies to continuous playback.
      let next = null;
      for (const k of keeps) { if (k.start >= t - 0.001) { next = k; break; } }
      if (next && next.start > t + 0.001) {
        if (_lastSkipTime !== next.start) {
          _lastSkipTime = next.start;
          _bufTarget = -1;
          video.muted = true;
          video.currentTime = next.start;
          _extSeek(next.start - _extOffset);
        }
        requestAnimationFrame(_playbackLoop);
        return;
      } else if (!next) {
        video.pause();
      }
    }
  }

  document.getElementById('timeDisplay').textContent = fmt(t);
  const x = X(t);
  document.getElementById('playhead').style.left = x + 'px';
  const sc = document.getElementById('tlScroll');
  const L = sc.scrollLeft, W = sc.clientWidth;
  if (!video.paused && (x < L + 60 || x > L + W - 60)) sc.scrollLeft = x - W / 2;
  highlightWord(t);
  applyMainVolAt(t);   // per-segment mute of the main audio
  captionTick(t); imgTick(t); audioTick(t);
  _extSync();   // keep external audio drifted < 150ms

  requestAnimationFrame(_playbackLoop);
}
requestAnimationFrame(_playbackLoop);

onActiveVideo('timeupdate', () => {
  // Keep time display + playhead in sync even when paused / scrubbing
  if (video.paused) {
    const to = _smoothMode ? editedToOriginal(video.currentTime) : video.currentTime;
    _origTime = to;
    document.getElementById('timeDisplay').textContent = fmt(to);
    document.getElementById('playhead').style.left = X(to) + 'px';
    highlightWord(to);
  }
});

/* ── Timeline left-click to seek ── */
document.getElementById('tlScroll').addEventListener('click', e => {
  if (e.target.closest('.trim-handle')) return;
  const sc = document.getElementById('tlScroll');
  seekOriginal(Tx(e.clientX - sc.getBoundingClientRect().left + sc.scrollLeft));
});

/* ── Timeline hover: scrub line + live preview frame ── */
const hoverLine  = document.getElementById('hoverLine');
const hoverLabel = document.getElementById('hoverLabel');
let scrubbing = false;

let _hoveredSilence = null;

document.getElementById('tlScroll').addEventListener('mousemove', e => {
  const sc   = document.getElementById('tlScroll');
  const rect = sc.getBoundingClientRect();
  const x    = e.clientX - rect.left + sc.scrollLeft;
  const t    = Math.max(0, Math.min(Tx(x), S.duration));

  // Snap to silence anywhere on the timeline (filmstrip or waveform)
  const sil = getSilenceAtTime(t);

  // Highlight / un-highlight silence overlays
  if (sil !== _hoveredSilence) {
    document.querySelectorAll('.silence-ov').forEach(el => el.classList.remove('hovered'));
    if (sil) {
      document.querySelectorAll('.silence-ov').forEach(el => {
        if (Math.abs(parseFloat(el.dataset.start) - sil.start) < 0.01) el.classList.add('hovered');
      });
    }
    _hoveredSilence = sil;
  }

  // Snap hover line to nearest silence boundary if over a silence
  let displayT = t;
  if (sil) {
    const distStart = Math.abs(t - sil.start);
    const distEnd   = Math.abs(t - sil.end);
    displayT = distStart < distEnd ? sil.start : sil.end;
  }
  const displayX = X(displayT);

  hoverLine.style.left    = displayX + 'px';
  hoverLine.style.display = 'block';
  hoverLabel.textContent  = fmtS(displayT) + '.' + String(Math.floor((displayT % 1) * 100)).padStart(2,'0')
                            + (sil ? ' 🔇' : '');
  // NOTE: hovering only shows the guide line — it no longer moves the playhead. The
  // ticker/preview move only when you CLICK the timeline, so hovering over other UI
  // won't shift the time (and won't make the selected caption disappear).
});

document.getElementById('tlScroll').addEventListener('mouseleave', () => {
  hoverLine.style.display = 'none';
  scrubbing = false;
  document.querySelectorAll('.silence-ov').forEach(el => el.classList.remove('hovered'));
  _hoveredSilence = null;
});

/* ── Timeline zoom (pinch on trackpad, or Ctrl+scroll) ── */
function rebuildTimeline() {
  PPS = BASE_PPS * zoomScale;
  refreshCoords();
  const W = tlFullW();
  document.getElementById('tlInner').style.width = W + 'px';
  document.getElementById('filmstrip').style.width = W + 'px';
  buildRuler(W);
  buildFilmstrip(W);
  updateWaveformWidth(W);  // CSS-only stretch — no canvas redraw (collapsed redraw handled by caller)
  buildOverlays();  // also calls buildSilenceOverlays internally
  buildSplitLines();
  buildSegmentTiles();
  renderCapTrack();
  renderImgTrack();
  renderAudioTrack();
  renderVidVolLine();
}

document.getElementById('tlScroll').addEventListener('wheel', e => {
  if (!e.ctrlKey) return;  // only handle pinch/ctrl+scroll
  e.preventDefault();

  const sc = document.getElementById('tlScroll');
  // Anchor zoom to the point under the cursor
  const mouseX = e.clientX - sc.getBoundingClientRect().left + sc.scrollLeft;
  const tAtCursor = Tx(mouseX);

  // deltaY < 0 = spread (zoom in), deltaY > 0 = pinch (zoom out)
  const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
  zoomScale = Math.max(0.2, Math.min(20, zoomScale * factor));

  rebuildTimeline();

  // Keep the time-under-cursor pinned to the same screen position
  sc.scrollLeft = X(tAtCursor) - (e.clientX - sc.getBoundingClientRect().left);
}, { passive: false });

/* ── Keyboard ── */
document.addEventListener('keydown', e => {
  const typing = e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA'
                 || e.target.isContentEditable;
  if ((e.key === 'Delete' || e.key === 'Backspace') && !typing) {
    // Priority 1: transcript text selection
    const sel = window.getSelection();
    if (sel && !sel.isCollapsed) {
      const transcript = document.getElementById('transcriptScroll');
      if (transcript.contains(sel.anchorNode)) {
        e.preventDefault();
        deleteSelectedWords();
        return;
      }
    }
    // Priority 2: selected filmstrip segment
    if (selectedSeg) {
      e.preventDefault();
      const { start, end } = selectedSeg;
      clearSegSelection();
      doDelete(start, end);
      return;
    }
    // Priority 3: selected caption
    if (selectedCapId) {
      e.preventDefault();
      capDelete();
      return;
    }
  }
  if (typing) return;
  // Caption copy / paste (duplicate). Skip if the user has a transcript text selection.
  const hasTextSel = window.getSelection && !window.getSelection().isCollapsed;
  if ((e.metaKey||e.ctrlKey) && (e.key==='c'||e.key==='C') && selectedCapId && !hasTextSel) {
    const c = CAPS.find(x => x.id === selectedCapId);
    if (c) { _capClipboard = { ...c }; e.preventDefault(); setStatus('Caption copied — ⌘V to paste'); }
    return;
  }
  if ((e.metaKey||e.ctrlKey) && (e.key==='v'||e.key==='V') && _capClipboard) {
    e.preventDefault(); pasteCaption(); return;
  }
  if ((e.metaKey||e.ctrlKey) && e.key==='z') { e.preventDefault(); doUndo(); return; }
  if (e.code === 'Space')      { e.preventDefault(); togglePlay(); }
  if (e.code === 'ArrowLeft')  seekOriginal(Math.max(0, _origTime - 5));
  if (e.code === 'ArrowRight') seekOriginal(Math.min(S.duration||0, _origTime + 5));
});

// Clicking outside any seg-tile clears selection
document.getElementById('tlScroll').addEventListener('mousedown', e => {
  if (!e.target.closest('.seg-tile')) clearSegSelection();
});

/* ── Video upload & re-processing ── */
let _dragCounter = 0;

document.addEventListener('dragenter', e => {
  if (!e.dataTransfer.types.includes('Files')) return;
  _dragCounter++;
  document.getElementById('dropOverlay').classList.add('active');
});
document.addEventListener('dragleave', () => {
  _dragCounter--;
  if (_dragCounter <= 0) { _dragCounter = 0; document.getElementById('dropOverlay').classList.remove('active'); }
});
document.addEventListener('dragover', e => { e.preventDefault(); });
document.addEventListener('drop', e => {
  e.preventDefault();
  _dragCounter = 0;
  document.getElementById('dropOverlay').classList.remove('active');
  const all = [...e.dataTransfer.files];
  const videos = all.filter(f => f.type.startsWith('video/') || /\.(mov|mp4|mkv|m4v|avi|webm)$/i.test(f.name));
  const audios = all.filter(f => f.type.startsWith('audio/') || /\.(mp3|wav|m4a|aac|ogg|flac)$/i.test(f.name));
  const images = all.filter(f => f.type.startsWith('image/') || /\.(png|jpe?g|gif|webp|bmp|heic)$/i.test(f.name));
  // If dropped onto the timeline, use the drop X as the placement time.
  const tl = document.getElementById('tlScroll');
  if (tl && S && (images.length || audios.length)) {
    const r = tl.getBoundingClientRect();
    if (e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top - 60 && e.clientY <= r.bottom) {
      _origTime = Math.max(0, Math.min(Tx(e.clientX - r.left + tl.scrollLeft), S.duration || 0));
    }
  }
  if (videos.length) { promptUploadMode(videos, null); return; }
  if (audios.length) handleSoundPick(audios);   // saved to library + added as an audio layer
  if (images.length) handleImagePick(images);   // added as an image overlay
});

let _pendingExtAudio = null;  // held until user uploads video

async function handleAudioPick(fileList) {
  const f = fileList[0];
  if (!f) return;

  // If a video is ALREADY loaded, attach the audio to the current session
  // (sync only — no video re-upload, no re-transcribe, cuts untouched).
  if (S && !S.no_video && S.duration > 0) {
    document.getElementById('processSpinner').style.display = '';
    document.getElementById('processUploadBtn').style.display = 'none';
    document.getElementById('processSub').textContent = 'Syncing audio to your current video — your cuts are kept.';
    document.getElementById('processMsg').textContent = 'Uploading & syncing external audio…';
    document.getElementById('processOverlay').classList.add('active');
    const form = new FormData();
    form.append('ext_audio', f);
    try {
      const r = await fetch('/api/add-ext-audio', { method:'POST', body: form });
      if (!r.ok) { const e = await r.json().catch(()=>({})); alert('Add audio failed: ' + (e.error||'')); document.getElementById('processOverlay').classList.remove('active'); return; }
      pollProcessStatus();   // on 'done' → reloads page (cuts intact, ext audio active)
    } catch(err) {
      alert('Add audio error: ' + err);
      document.getElementById('processOverlay').classList.remove('active');
    }
    return;
  }

  // No video yet — queue the audio to be uploaded together with the video
  _pendingExtAudio = f;
  const btn = document.getElementById('extAudioBtn');
  btn.textContent = '🎙 ' + f.name.slice(0, 20) + (f.name.length > 20 ? '…' : '');
  btn.style.color = '#27ae60';
  setStatus(`External audio queued: ${f.name} — now load your video`);
}

/* Ask which pipeline to run when a video is loaded: auto-cut vs standard editing. */
let _pendingUpload = null;
function promptUploadMode(fileList, extAudio) {
  const files = [...fileList];
  if (!files.length) return;
  _pendingUpload = { files, ext: extAudio || _pendingExtAudio };
  const n = files.length;
  const el = document.getElementById('uploadModeSub');
  if (el) el.textContent = `${n} video${n>1?'s':''} ready${_pendingUpload.ext ? ' (+ external audio)' : ''}. How should it be loaded?`;
  document.getElementById('uploadModeModal').style.display = 'flex';
}
function chooseUploadMode(mode) {
  document.getElementById('uploadModeModal').style.display = 'none';
  if (!_pendingUpload) return;
  const u = _pendingUpload; _pendingUpload = null;
  handleVideoUpload(u.files, u.ext, mode);
}
function cancelUploadMode() {
  document.getElementById('uploadModeModal').style.display = 'none';
  _pendingUpload = null;
}

async function handleVideoUpload(fileList, extAudio, mode) {
  const files = [...fileList];
  if (!files.length) return;
  const ext = extAudio || _pendingExtAudio;
  if (!mode) mode = 'auto';

  document.getElementById('processSpinner').style.display = '';
  document.getElementById('processUploadBtn').style.display = 'none';
  document.getElementById('processSub').textContent = (mode === 'edit')
    ? 'Preparing the editor — no transcription. Reloads automatically.'
    : 'This takes 1–3 minutes. The editor will reload automatically.';
  document.getElementById('processMsg').textContent = `Uploading ${files.length} video${files.length > 1 ? 's' : ''}${ext ? ' + external audio' : ''}…`;
  document.getElementById('processOverlay').classList.add('active');

  const form = new FormData();
  for (const f of files) form.append('videos', f);
  if (ext) form.append('ext_audio', ext);
  form.append('mode', mode);
  _pendingExtAudio = null;

  try {
    const r = await fetch('/api/upload', { method: 'POST', body: form });
    if (!r.ok) { alert('Upload failed'); document.getElementById('processOverlay').classList.remove('active'); return; }
    pollProcessStatus();
  } catch(err) {
    alert('Upload error: ' + err);
    document.getElementById('processOverlay').classList.remove('active');
  }
}

let _pollSawProcessing = false;   // only reload after a real processing→done transition
function pollProcessStatus() {
  fetch('/api/process-status').then(r => r.json()).then(d => {
    if (d.state === 'idle') {
      // Waiting for upload — keep polling quietly, never reload
      setTimeout(pollProcessStatus, 2000);
      return;
    }
    if (d.state === 'processing' || d.state === 'finalizing') {
      _pollSawProcessing = true;
      document.getElementById('processMsg').textContent = d.message || 'Processing…';
      document.getElementById('processSpinner').style.display = '';
      document.getElementById('processUploadBtn').style.display = 'none';
      document.getElementById('processSub').textContent = 'This takes a moment. The editor will reload automatically.';
      document.getElementById('processOverlay').classList.add('active');
      setTimeout(pollProcessStatus, 1200);
      return;
    }
    if (d.state === 'done') {
      document.getElementById('processOverlay').classList.remove('active');
      // ONLY reload if we actually watched processing happen — prevents reload loops
      if (_pollSawProcessing) { _pollSawProcessing = false; location.reload(); }
      return;   // otherwise we're already on the right view; stop polling
    }
    if (d.state === 'error') {
      document.getElementById('processOverlay').classList.remove('active');
      alert('Processing failed:\\n' + d.message);
      return;
    }
    setTimeout(pollProcessStatus, 1500);
  }).catch(() => setTimeout(pollProcessStatus, 2000));
}

// On load, check if we're waiting for an upload (no video loaded yet)
async function checkInitialState() {
  const r = await fetch('/api/process-status');
  const d = await r.json();
  if (d.state === 'processing') {
    document.getElementById('processMsg').textContent = d.message || 'Processing…';
    document.getElementById('processOverlay').classList.add('active');
    pollProcessStatus();
  }
}

/* ── Project manager ── */
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function startRename(nameEl, id, current) {
  const input = document.createElement('input');
  input.type = 'text';
  input.value = current;
  input.style.cssText = 'width:100%;box-sizing:border-box;background:#111;border:1px solid #3d9be9;border-radius:5px;color:#fff;font-size:14px;font-weight:600;padding:4px 6px;outline:none;';
  nameEl.replaceWith(input);
  input.focus(); input.select();
  let done = false;
  const commit = async (save) => {
    if (done) return; done = true;
    const newName = input.value.trim();
    if (save && newName && newName !== current) {
      await fetch('/api/rename-project', { method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ project_id: id, name: newName }) });
    }
    showPicker();  // re-render with the new name (or revert)
  };
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); commit(true); }
    else if (e.key === 'Escape') { e.preventDefault(); commit(false); }
  });
  input.addEventListener('blur', () => commit(true));
}

function fmtAgo(ts) {
  if (!ts) return '';
  const s = Date.now()/1000 - ts;
  if (s < 60) return 'just now';
  if (s < 3600) return Math.floor(s/60) + 'm ago';
  if (s < 86400) return Math.floor(s/3600) + 'h ago';
  return Math.floor(s/86400) + 'd ago';
}

async function showPicker() {
  const grid = document.getElementById('pickerGrid');
  grid.innerHTML = '<div class="picker-empty">Loading…</div>';
  document.getElementById('projectPicker').classList.add('active');
  const r = await fetch('/api/projects');
  const d = await r.json();
  const projs = d.projects || [];
  document.getElementById('pickerCount').textContent = projs.length ? `(${projs.length})` : '';
  if (!projs.length) {
    grid.innerHTML = '<div class="picker-empty">No projects yet.<br>Click <b>+ New Project</b> or drop a video to begin.</div>';
    return;
  }
  grid.innerHTML = '';
  for (const p of projs) {
    const card = document.createElement('div');
    card.className = 'proj-card' + (p.is_current ? ' current' : '');
    const badges = (p.has_ext_audio ? '<span class="proj-badge audio">🎙 audio</span>' : '')
                 + (p.cached ? '<span class="proj-badge cached">⚡ cached</span>' : '');
    card.innerHTML = `
      <div class="proj-name" title="Double-click to rename">${escapeHtml(p.name)}</div>
      <div class="proj-meta">
        ${badges}${badges ? '<br>' : ''}
        ✂ ${p.n_cuts} cuts · ${p.n_splits} splits${p.n_videos > 1 ? ' · ' + p.n_videos + ' clips' : ''}<br>
        ${fmtAgo(p.modified)}
      </div>
      <button class="proj-del" title="Delete project" onclick="event.stopPropagation();deleteProject('${p.id}')">🗑</button>`;
    card.onclick = () => openProject(p.id);
    // Name: single-click opens (after a short delay), double-click renames.
    // The delay lets us cancel the "open" when a 2nd click arrives.
    const nameEl = card.querySelector('.proj-name');
    let nameClickTimer = null;
    nameEl.addEventListener('click', ev => {
      ev.stopPropagation();              // don't let the card's onclick fire
      if (nameClickTimer) return;        // second click handled by dblclick
      nameClickTimer = setTimeout(() => { nameClickTimer = null; openProject(p.id); }, 230);
    });
    nameEl.addEventListener('dblclick', ev => {
      ev.stopPropagation();
      if (nameClickTimer) { clearTimeout(nameClickTimer); nameClickTimer = null; }
      startRename(nameEl, p.id, p.name);
    });
    grid.appendChild(card);
  }
}

async function openProject(id) {
  document.getElementById('projectPicker').classList.remove('active');
  document.getElementById('processSpinner').style.display = '';
  document.getElementById('processUploadBtn').style.display = 'none';
  document.getElementById('processSub').textContent = 'Loading project — your cuts are restored from cache.';
  document.getElementById('processMsg').textContent = 'Opening project…';
  document.getElementById('processOverlay').classList.add('active');
  try {
    const r = await fetch('/api/open-project', { method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ project_id: id }) });
    if (!r.ok) { alert('Could not open project'); document.getElementById('processOverlay').classList.remove('active'); showPicker(); return; }
    pollProcessStatus();   // reloads page when done → editor with that project
  } catch(e) {
    alert('Open error: ' + e);
    document.getElementById('processOverlay').classList.remove('active');
    showPicker();
  }
}

async function exitToProjects() {
  try { await fetch('/api/exit-project', { method:'POST' }); } catch(e) {}
  showPicker();
}

async function deleteProject(id) {
  if (!confirm('Delete project "' + id + '"? This removes its video, cuts and transcript permanently.')) return;
  const r = await fetch('/api/delete-project', { method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ project_id: id }) });
  if (!r.ok) { const e = await r.json().catch(()=>({})); alert('Delete failed: ' + (e.error||'')); return; }
  showPicker();
}

load().then(checkInitialState).catch((err) => {
  // Only show the picker when there is genuinely no project loaded.
  // A render error in a loaded project must NOT bounce us to the picker (caused flicker).
  if (err && err.message === 'no_video') {
    showPicker();
    pollProcessStatus();
  } else {
    console.error('load() error (staying in editor):', err);
  }
});
</script>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global VIDEO_PATH, THUMBS_DIR, TEMP_DIR, DURATION, WAVEFORM, WORDS
    global CUTS, THUMB_INTERVAL, THUMB_COUNT, ORIGINAL_STEM, STATE_FILE, INITIAL_AUTO_CUTS
    global MODEL_ARGS

    parser = argparse.ArgumentParser(description="Visual cut review UI")
    parser.add_argument("input", nargs="?", help="Input video file (optional — can upload via browser)")
    parser.add_argument("--model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--skip-fillers",   action="store_true")
    parser.add_argument("--skip-bad-takes", action="store_true")
    parser.add_argument("--skip-silence",   action="store_true")
    args = parser.parse_args()

    # Store model args globally so re-processing after upload uses same settings
    MODEL_ARGS = {
        "model":          args.model,
        "skip_fillers":   args.skip_fillers,
        "skip_bad_takes": args.skip_bad_takes,
        "skip_silence":   args.skip_silence,
    }

    # Load cross-session config (last export dir, last video(s), etc.)
    last_videos, last_ext_audio, last_ext_offset, last_project_id = load_config()

    if not args.input:
        url = f"http://localhost:{args.port}"
        if last_videos:
            # Auto-resume last session — kick off reprocess before Flask blocks
            print("\n" + "="*50)
            print(f"Bad Take Trimmer — resuming project: {last_project_id or last_videos[0].name}")
            print("="*50)
            PROCESS_STATUS["state"]   = "processing"
            PROCESS_STATUS["message"] = f"Resuming {last_project_id or last_videos[0].name}…"
            threading.Thread(target=reprocess_video,
                             args=(last_videos, last_ext_audio),
                             kwargs={"is_resume": True, "project_id": last_project_id},
                             daemon=True).start()
            threading.Timer(1.2, lambda: webbrowser.open(url)).start()
            app.run(host="localhost", port=args.port, debug=False, use_reloader=False, threaded=True)
            return
        else:
            # No video specified and no previous session — wait for upload
            print("\n" + "="*50)
            print("Bad Take Trimmer — waiting for video upload")
            print("="*50)
            print(f"\nOpen {url} and upload a video to begin.\n")
            threading.Timer(1.2, lambda: webbrowser.open(url)).start()
            app.run(host="localhost", port=args.port, debug=False, use_reloader=False, threaded=True)
            return

    VIDEO_PATH = Path(args.input).expanduser().resolve()
    ORIGINAL_STEM = VIDEO_PATH.stem  # save before any rename
    if not VIDEO_PATH.exists():
        sys.exit(f"File not found: {VIDEO_PATH}")

    # Set up persistent state file keyed to this video
    state_dir = Path.home() / ".cache" / "video_editor"
    state_dir.mkdir(parents=True, exist_ok=True)
    STATE_FILE = str(state_dir / f"{ORIGINAL_STEM}_edits.json")

    TEMP_DIR  = tempfile.mkdtemp(prefix="vp_")
    THUMBS_DIR = Path(TEMP_DIR) / "thumbs"

    print(f"\n{'='*50}")
    print(f"Video: {VIDEO_PATH.name}")
    print(f"{'='*50}\n")

    DURATION = get_duration(VIDEO_PATH)
    THUMB_INTERVAL = 10.0 if DURATION > 300 else 5.0
    THUMB_COUNT    = extract_thumbnails(VIDEO_PATH, THUMBS_DIR, THUMB_INTERVAL)
    WAVEFORM       = extract_waveform(VIDEO_PATH)

    audio_path = Path(TEMP_DIR) / "audio.wav"
    extract_audio(VIDEO_PATH, audio_path)
    WORDS = transcribe(audio_path, args.model)

    print("\nDetecting cuts...")
    CUTS = build_labeled_cuts(
        WORDS, DURATION,
        skip_fillers=args.skip_fillers,
        skip_bad_takes=args.skip_bad_takes,
        skip_silence=args.skip_silence,
    )

    # Snapshot auto-detected cuts before user edits overwrite them
    INITIAL_AUTO_CUTS = [dict(c) for c in CUTS]

    # Restore saved edits (replaces auto-detected cuts if saved state exists)
    load_edits()

    url = f"http://localhost:{args.port}"
    print(f"\nOpening preview at {url}")
    print("Press Ctrl+C to stop.\n")

    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host="localhost", port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
