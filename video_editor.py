#!/usr/bin/env python3
"""
video_editor.py — Automated YouTube video editor

Removes: filler words, bad takes, long silences
Adds:    captions, audio cleanup (compression, loudnorm)

Usage:
    python3 video_editor.py input.mp4
    python3 video_editor.py input.mp4 --model small
    python3 video_editor.py input.mp4 --skip-captions --skip-bad-takes
"""

import argparse
from difflib import SequenceMatcher
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional


# ── Config ────────────────────────────────────────────────────────────────────

FILLER_WORDS = {"um", "uh", "uhh", "umm", "hmm", "er", "ah", "eh"}

# Multi-word restart phrases to detect (order matters — longer first)
RETAKE_PHRASES = [
    ["let", "me", "try", "that", "again"],
    ["let", "me", "do", "that", "again"],
    ["let", "me", "start", "over"],
    ["let", "me", "try", "again"],
    ["let", "me", "redo", "that"],
    ["okay", "let", "me", "try"],
    ["alright", "let", "me", "try"],
    ["from", "the", "beginning"],
    ["from", "the", "top"],
    ["starting", "over"],
    ["start", "over"],
    ["take", "that", "again"],
]

# Single-word hard signals — almost always mean "cut here"
RETAKE_SIGNALS = {"cut", "redo", "retake"}

SILENCE_MAX  = 0.8   # trim silences longer than this (seconds)
SILENCE_KEEP = 0.25  # keep this much silence after trimming
FILLER_PAD   = 0.05  # buffer to add around each filler cut

# Fallback ffmpeg paths for macOS homebrew
FFMPEG_PATHS  = [shutil.which("ffmpeg"),  "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]
FFPROBE_PATHS = [shutil.which("ffprobe"), "/opt/homebrew/bin/ffprobe", "/usr/local/bin/ffprobe"]


def find_bin(candidates: list) -> str:
    for p in candidates:
        if p and Path(p).exists():
            return p
    sys.exit("ffmpeg not found. Install via: brew install ffmpeg")


FFMPEG  = find_bin(FFMPEG_PATHS)
FFPROBE = find_bin(FFPROBE_PATHS)


# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def get_duration(path: Path) -> float:
    r = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(r.stdout)["format"]["duration"])


# ── Step 1: Extract audio ─────────────────────────────────────────────────────

def extract_audio(video: Path, out: Path) -> None:
    print("Extracting audio...")
    run([FFMPEG, "-y", "-i", str(video), "-vn", "-ar", "16000", "-ac", "1", str(out)])


# ── Step 2: Transcribe ────────────────────────────────────────────────────────

def transcribe(audio: Path, model_size: str) -> list[dict]:
    print(f"Transcribing ({model_size} model) — first run downloads ~100MB–1.5GB...")
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(audio), word_timestamps=True, language="en")
    words = []
    for seg in segments:
        for w in (seg.words or []):
            words.append({
                "word": re.sub(r"[^a-z]", "", w.word.lower()),
                "raw":  w.word.strip(),
                "start": round(w.start, 4),
                "end":   round(w.end,   4),
            })
    print(f"  {len(words)} words transcribed")
    return words


# ── Step 3: Detect cuts ───────────────────────────────────────────────────────

def _take_start(words: list[dict], idx: int) -> float:
    """Walk backward to find the last pause >1s — that's where this take began."""
    for j in range(idx - 1, 0, -1):
        if words[j]["start"] - words[j - 1]["end"] > 1.0:
            return words[j]["start"]
    return words[0]["start"] if words else 0.0


def detect_repeated_takes(words: list[dict]) -> list[tuple[float, float]]:
    """
    Detects when the same sentence is said multiple times — even if rephrased.
    Keeps only the FINAL version, cuts all earlier attempts.

    Two strategies:
      1. Exact: first 5 words are identical (works for true word-for-word repeats)
      2. Fuzzy: same first word + SequenceMatcher ratio >= 0.25, within 20s
                catches rephrases like "today I'm going to..." → "today I will..."
    """
    if not words:
        return []

    PHRASE_PAUSE     = 0.4   # pause that separates phrases (seconds)
    MAX_RETAKE_GAP   = 300   # don't look for repeats further apart than 5 min
    CLOSE_GAP        = 20    # fuzzy match only applies within this window (seconds)
    EXACT_WORDS      = 5     # leading words needed for exact match
    FUZZY_MIN_WORDS  = 3     # min phrase length to attempt fuzzy match
    FUZZY_RATIO      = 0.25  # SequenceMatcher threshold — low because phrases are short

    # Common "I" contractions — normalize so "I'm" and "I'll" both count as same opener
    I_FORMS = {"i", "im", "ill", "ive", "id"}

    def first_words_match(a: str, b: str) -> bool:
        if a == b:
            return True
        if a in I_FORMS and b in I_FORMS:
            return True
        return False

    # Segment words into phrases at pauses
    phrases: list[tuple[int, list[dict]]] = []
    cur_start, cur = 0, []
    for i, w in enumerate(words):
        if not cur:
            cur_start = i
        cur.append(w)
        next_gap = (words[i + 1]["start"] - w["end"]) if i < len(words) - 1 else 999
        if next_gap > PHRASE_PAUSE or i == len(words) - 1:
            phrases.append((cur_start, cur[:]))
            cur = []

    def fp(ws: list[dict]) -> list[str]:
        return [w["word"] for w in ws if w["word"]]

    cuts: list[tuple[float, float]] = []
    cut_set: set[int] = set()

    for i in range(len(phrases)):
        if i in cut_set:
            continue
        idx_i, pw_i = phrases[i]
        fp_i = fp(pw_i)
        if len(fp_i) < FUZZY_MIN_WORDS:
            continue
        t_i = pw_i[0]["start"]

        for j in range(i + 1, len(phrases)):
            idx_j, pw_j = phrases[j]
            t_j = pw_j[0]["start"]
            gap = t_j - t_i

            if gap > MAX_RETAKE_GAP:
                break

            fp_j = fp(pw_j)
            if len(fp_j) < FUZZY_MIN_WORDS:
                continue

            # Must start with the same word (or both be I-contractions)
            if not first_words_match(fp_i[0], fp_j[0]):
                continue

            # Strategy 1 — exact leading words
            leading = sum(1 for a, b in zip(fp_i, fp_j) if a == b)
            # stop counting at first mismatch
            exact_leading = 0
            for a, b in zip(fp_i, fp_j):
                if a == b:
                    exact_leading += 1
                else:
                    break
            exact_hit = exact_leading >= min(EXACT_WORDS, min(len(fp_i), len(fp_j)))

            # Strategy 2 — fuzzy match for close rephrases
            # Normalize I-contractions so "I'm"→"i", "I'll"→"i" etc. compare equal
            def norm(ws: list[str]) -> list[str]:
                return ["i" if w in I_FORMS else w for w in ws]

            fuzzy_hit = False
            if gap <= CLOSE_GAP:
                ratio = SequenceMatcher(None, norm(fp_i), norm(fp_j)).ratio()
                fuzzy_hit = ratio >= FUZZY_RATIO

            if exact_hit or fuzzy_hit:
                take_start = _take_start(words, idx_i)
                cuts.append((take_start, t_j))
                cut_set.add(i)
                mode = "exact" if exact_hit else "fuzzy rephrase"
                snippet = " ".join(fp_i[:6])
                print(f"    repeated take [{mode}] ['{snippet}...']  {take_start:.1f}s – {t_j:.1f}s")
                break

    print(f"  Repeated takes: {len(cuts)}")
    return cuts


def detect_stutter_phrases(words: list[dict]) -> list[dict]:
    """
    Catches repeated opener phrases (2-5 words) that appear 2+ times in a tight burst.

    Strategy:
    - 2-word phrases: only match at phrase boundaries (after ≥0.4s pause), max 15s gap.
      Prevents "going to", "to be", "want to" etc. from flooding results.
    - 3-5 word phrases: word-level scan, max 20s gap.
      Catches "I get it…" even when not separated by a long pause.

    Returns list of {start, end, label, type} dicts.
    """
    if len(words) < 4:
        return []

    PAUSE    = 0.4   # gap that marks a phrase boundary
    MIN_LEN  = 2
    MAX_LEN  = 5
    MAX_GAP  = {2: 15.0, 3: 20.0, 4: 25.0, 5: 30.0}

    normed = [w["word"] for w in words]

    # Build phrase-start indices (word index after a ≥PAUSE gap, or start of transcript)
    phrase_starts = {0}
    for i in range(1, len(words)):
        if words[i]["start"] - words[i - 1]["end"] >= PAUSE:
            phrase_starts.add(i)

    cuts = []
    covered = set()

    # Scan shorter triggers FIRST so "I get it" (n=3) captures all 4 occurrences
    # before "I get it. The" (n=5) can steal individual pairs.
    # ALL lengths require phrase-boundary starts — prevents mid-sentence content phrases
    # like "people, they've probably been through" from matching inside both a bad take
    # and the good take that follows.
    for n in range(MIN_LEN, MAX_LEN + 1):
        max_gap = MAX_GAP[n]

        for i in sorted(phrase_starts):
            if i in covered or i + n > len(words):
                continue
            trigger = normed[i:i + n]
            if any(t == "" for t in trigger):
                continue

            # Find all phrase-boundary occurrences within the time window
            occurrences = [i]
            for j in sorted(phrase_starts):
                if j <= i:
                    continue
                if words[j]["start"] - words[i]["start"] > max_gap:
                    break
                if j + n > len(words):
                    continue
                if normed[j:j + n] == trigger:
                    occurrences.append(j)

            if len(occurrences) < 2:
                continue

            cut_start = words[occurrences[0]]["start"]
            cut_end   = words[occurrences[-1]]["start"]
            if cut_end - cut_start < 1.0:
                continue

            raw_label = " ".join(words[k]["raw"] for k in range(i, i + n))
            label = f"{raw_label}… (×{len(occurrences)})"
            cuts.append({"start": cut_start, "end": cut_end,
                         "label": label, "type": "exact repeat"})
            print(f"    stutter: {cut_start:.1f}s – {cut_end:.1f}s  ({label})")

            # Mark all occurrence positions so longer sub-patterns don't split them
            for occ in occurrences:
                for k in range(occ, occ + n):
                    covered.add(k)

    # Sort and merge overlapping/adjacent cuts
    cuts.sort(key=lambda c: c["start"])
    merged = []
    for c in cuts:
        if merged and c["start"] <= merged[-1]["end"] + 0.2:
            merged[-1]["end"] = max(merged[-1]["end"], c["end"])
        else:
            merged.append(c)

    return merged


def detect_repeated_takes_ai(words: list[dict]) -> list[dict]:
    """
    Use Claude (Sonnet) in sliding 75-second windows to find repeat takes.
    A repeat take = speaker starts a sentence, stops, restarts same sentence/thought
    (possibly different wording) within the next ~20 seconds.
    Windowed approach prevents confusing thematic repetition with actual stuttering.
    """
    if not words:
        return []
    try:
        import anthropic, json, re, os
        from pathlib import Path as _Path

        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            env = _Path(__file__).parent.parent / ".env"
            if env.exists():
                for line in env.read_text().splitlines():
                    if line.startswith("ANTHROPIC_API_KEY="):
                        key = line.split("=", 1)[1].strip()
                        break
        if not key:
            print("  AI take detection: no API key, skipping")
            return []

        # Group words into phrases at pauses ≥ 0.4s; annotate long pauses
        PAUSE = 0.4
        phrases, cur, cur_start = [], [], words[0]["start"]
        for i, w in enumerate(words):
            cur.append(w)
            gap = (words[i+1]["start"] - w["end"]) if i < len(words)-1 else 999
            if gap >= PAUSE or i == len(words)-1:
                pause_note = f" [pause {gap:.1f}s]" if gap >= 1.5 and i < len(words)-1 else ""
                phrases.append({"start": cur_start, "end": w["end"],
                                 "text": " ".join(x["raw"] for x in cur) + pause_note})
                cur = []
                if i < len(words)-1:
                    cur_start = words[i+1]["start"]

        def fmt_ts(sec):
            m, s = divmod(sec, 60)
            return f"{int(m)}:{s:05.2f}"

        client = anthropic.Anthropic(api_key=key)
        print("  Detecting repeat takes with AI (sliding windows)...")

        WINDOW     = 75   # seconds per window
        STEP       = 60   # advance each iteration (15s overlap)
        MAX_CUT    = 40   # reject any flagged cut longer than this

        all_results: list[dict] = []
        seen_starts: set[float] = set()
        duration = words[-1]["end"]
        win_start = 0.0

        while win_start < duration:
            win_end = win_start + WINDOW
            win_phrases = [p for p in phrases if p["start"] >= win_start and p["start"] < win_end]
            if not win_phrases:
                win_start += STEP
                continue

            transcript = "\n".join(f"[{fmt_ts(p['start'])}] {p['text']}" for p in win_phrases)

            resp = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                messages=[{"role": "user", "content": f"""You are a video editor reviewing a raw YouTube recording.

Find ONLY "repeat takes" — where the speaker:
1. Starts a sentence or thought
2. Stops (mid-sentence OR trails off) within a few seconds
3. Immediately restarts THE SAME sentence/thought (possibly different wording)

NOT a repeat take:
- Speaker finishes a thought, then later revisits the same topic (normal content)
- Normal transitions ("also", "another thing is", "so anyway")
- Any two phrases more than ~20 seconds apart

The false start and restart MUST be close together (within ~20 seconds).

Transcript ({fmt_ts(win_start)} – {fmt_ts(min(win_end, duration))}):
{transcript}

Return ONLY a JSON array. For each repeat take:
{{"start": <seconds the false start begins>, "end": <seconds the good take begins>, "label": "<brief description>"}}

Constraints:
- end - start must be under {MAX_CUT} seconds
- timestamps must appear in the transcript above
- if no repeat takes, return []

JSON only."""
                }]
            )

            text = resp.content[0].text.strip()
            m = re.search(r'\[.*?\]', text, re.DOTALL)
            if m:
                try:
                    for c in json.loads(m.group()):
                        if not isinstance(c, dict):
                            continue
                        s, e = float(c.get("start", 0)), float(c.get("end", 0))
                        dur = e - s
                        # Only accept cuts whose START falls in the non-overlap zone
                        in_zone = win_start <= s < win_start + STEP
                        if in_zone and 1.0 <= dur <= MAX_CUT and s not in seen_starts:
                            seen_starts.add(s)
                            all_results.append({"start": s, "end": e,
                                                 "label": c.get("label", "repeat take"),
                                                 "type": "rephrase"})
                            print(f"    AI repeat take: {s:.1f}s – {e:.1f}s  ({c.get('label','')})")
                except json.JSONDecodeError:
                    pass

            win_start += STEP

        print(f"  AI repeat takes: {len(all_results)}")
        return all_results

    except Exception as exc:
        print(f"  AI take detection failed: {exc}")
        return []


def detect_cuts(
    words: list[dict],
    duration: float,
    skip_fillers: bool = False,
    skip_bad_takes: bool = False,
    skip_silence: bool = False,
) -> list[tuple[float, float]]:

    cuts: list[tuple[float, float]] = []

    # Filler words
    if not skip_fillers:
        count = 0
        for w in words:
            if w["word"] in FILLER_WORDS:
                cuts.append((max(0, w["start"] - FILLER_PAD), w["end"] + FILLER_PAD))
                count += 1
        print(f"  Filler words:  {count}")

    # Repeated takes (no explicit signal — same sentence said multiple times)
    if not skip_bad_takes:
        cuts.extend(detect_repeated_takes(words))

    # Bad takes (explicit restart phrases / signals)
    if not skip_bad_takes:
        count = 0
        i, n = 0, len(words)
        while i < n:
            matched = False

            # Single-word hard signal
            if words[i]["word"] in RETAKE_SIGNALS:
                ts = _take_start(words, i)
                te = words[i]["end"] + 0.3
                cuts.append((ts, te))
                print(f"    bad take [{words[i]['word']}]  {ts:.1f}s – {te:.1f}s")
                count += 1
                i += 1
                matched = True

            # Multi-word phrases
            if not matched:
                for phrase in RETAKE_PHRASES:
                    pw = len(phrase)
                    if i + pw <= n and [words[i + j]["word"] for j in range(pw)] == phrase:
                        ts = _take_start(words, i)
                        te = words[i + pw - 1]["end"] + 0.3
                        cuts.append((ts, te))
                        print(f"    bad take ['{' '.join(phrase)}']  {ts:.1f}s – {te:.1f}s")
                        count += 1
                        i += pw
                        matched = True
                        break

            if not matched:
                i += 1

        print(f"  Bad takes:     {count}")

    # Long silences
    if not skip_silence and words:
        count = 0
        if words[0]["start"] > SILENCE_MAX:
            cuts.append((SILENCE_KEEP, words[0]["start"]))
            count += 1
        for i in range(len(words) - 1):
            gap = words[i + 1]["start"] - words[i]["end"]
            if gap > SILENCE_MAX:
                cuts.append((words[i]["end"] + SILENCE_KEEP, words[i + 1]["start"]))
                count += 1
        if duration - words[-1]["end"] > SILENCE_MAX:
            cuts.append((words[-1]["end"] + SILENCE_KEEP, duration))
            count += 1
        print(f"  Silence gaps:  {count}")

    return cuts


def merge_cuts(cuts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not cuts:
        return []
    merged = [list(sorted(cuts)[0])]
    for s, e in sorted(cuts)[1:]:
        if s <= merged[-1][1] + 0.1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def cuts_to_keeps(cuts: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    keeps, pos = [], 0.0
    for cs, ce in cuts:
        if pos < cs - 0.01:
            keeps.append((pos, cs))
        pos = max(pos, ce)
    if pos < duration - 0.01:
        keeps.append((pos, duration))
    return keeps


# ── Step 4: Build captions ────────────────────────────────────────────────────

def build_srt(words: list[dict], keeps: list[tuple[float, float]], out: Path) -> None:
    print("Building captions...")

    def to_out(t: float) -> Optional[float]:
        offset = 0.0
        for ks, ke in keeps:
            if t < ks:
                return None
            if t <= ke:
                return offset + (t - ks)
            offset += ke - ks
        return None

    def fmt(s: float) -> str:
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{sec:06.3f}".replace(".", ",")

    kept = []
    for w in words:
        s, e = to_out(w["start"]), to_out(w["end"])
        if s is not None and e is not None and to_out((w["start"] + w["end"]) / 2) is not None:
            kept.append({**w, "os": s, "oe": e})

    lines, idx = [], 1
    for i in range(0, len(kept), 6):
        chunk = kept[i : i + 6]
        text = " ".join(w["raw"] for w in chunk).strip()
        if text:
            lines.append(f"{idx}\n{fmt(chunk[0]['os'])} --> {fmt(chunk[-1]['oe'])}\n{text}\n")
            idx += 1

    out.write_text("\n".join(lines))
    print(f"  {idx - 1} subtitle entries")


# ── Step 5: Encode ────────────────────────────────────────────────────────────

def encode(
    video: Path,
    keeps: list[tuple[float, float]],
    srt: Optional[Path],
    out: Path,
) -> None:
    print(f"Encoding ({len(keeps)} segments)...")
    n = len(keeps)
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build per-segment trim filters
    vt = "".join(f"[0:v]trim={s:.4f}:{e:.4f},setpts=PTS-STARTPTS[v{i}];" for i, (s, e) in enumerate(keeps))
    at = "".join(f"[0:a]atrim={s:.4f}:{e:.4f},asetpts=PTS-STARTPTS[a{i}];" for i, (s, e) in enumerate(keeps))

    # Concatenate
    vc = "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[vcat];"
    ac = "".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[araw];"

    # Audio cleanup chain
    afilt = (
        "[araw]"
        "highpass=f=80,"
        "lowpass=f=14000,"
        "acompressor=threshold=-18dB:ratio=3:attack=5:release=50,"
        "loudnorm=I=-16:TP=-1.5:LRA=11"
        "[aout]"
    )

    fc = vt + at + vc + ac + afilt
    video_out = "[vcat]"

    result = subprocess.run([
        FFMPEG, "-y", "-i", str(video),
        "-filter_complex", fc,
        "-map", video_out, "-map", "[aout]",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        str(out),
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(result.stderr[-3000:])
        raise RuntimeError(f"FFmpeg encode failed (exit {result.returncode})")

    # Copy SRT alongside output so it can be uploaded to YouTube
    if srt and srt.exists():
        srt_out = out.with_suffix(".srt")
        srt_out.write_text(srt.read_text())
        print(f"  Captions saved → {srt_out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Automated YouTube video editor")
    parser.add_argument("input", help="Input .mp4 file")
    parser.add_argument("--model", default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--output", help="Output path (default: <input>_edited.mp4)")
    parser.add_argument("--skip-captions",  action="store_true")
    parser.add_argument("--skip-fillers",   action="store_true")
    parser.add_argument("--skip-bad-takes", action="store_true")
    parser.add_argument("--skip-silence",   action="store_true")
    args = parser.parse_args()

    video = Path(args.input).expanduser().resolve()
    if not video.exists():
        sys.exit(f"File not found: {video}")

    out = Path(args.output).expanduser().resolve() if args.output else video.parent / (video.stem + "_edited.mp4")

    print(f"\n{'='*50}")
    print(f"Input:  {video}")
    print(f"Output: {out}")
    print(f"{'='*50}\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        audio_path = tmp / "audio.wav"
        srt_path   = tmp / "captions.srt"

        duration = get_duration(video)
        extract_audio(video, audio_path)
        words = transcribe(audio_path, args.model)

        print("\nDetecting cuts...")
        raw_cuts = detect_cuts(
            words, duration,
            skip_fillers=args.skip_fillers,
            skip_bad_takes=args.skip_bad_takes,
            skip_silence=args.skip_silence,
        )
        merged = merge_cuts(raw_cuts)
        keeps  = cuts_to_keeps(merged, duration)

        cut_secs = sum(e - s for s, e in merged)
        print(f"\n  Total removed: {cut_secs:.1f}s ({cut_secs / 60:.1f} min)")
        print(f"  Segments kept: {len(keeps)}\n")

        if not args.skip_captions:
            build_srt(words, keeps, srt_path)

        encode(video, keeps, srt_path if not args.skip_captions else None, out)

    print(f"\nDone! → {out}\n")


if __name__ == "__main__":
    main()
