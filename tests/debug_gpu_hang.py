import os
import sys
import signal
import threading
import time

os.environ["CT2_VERBOSE"] = "3"

from pathlib import Path
rocm_base = Path(r'C:\Program Files\AMD\ROCm')
if rocm_base.exists():
    for version_dir in sorted(rocm_base.iterdir(), reverse=True):
        rocm_bin = version_dir / 'bin'
        if rocm_bin.exists() and (rocm_bin / 'amdhip64_6.dll').exists():
            os.add_dll_directory(str(rocm_bin))
            print(f"Added ROCm DLL path: {rocm_bin}")
            break

import numpy as np
import wave
from faster_whisper import WhisperModel

wav_path = sys.argv[1] if len(sys.argv) > 1 else None
if not wav_path:
    print("Usage: python debug_gpu_hang.py <path_to_wav>")
    sys.exit(1)

with wave.open(wav_path, 'rb') as wf:
    sr = wf.getframerate()
    frames = wf.readframes(wf.getnframes())
    audio_full = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    duration = len(audio_full) / sr
    print(f"Loaded: {wav_path} ({duration:.1f}s, {sr}Hz)")

model = WhisperModel("tiny", device="cuda", compute_type="float32")
print("Model loaded on GPU\n")

TIMEOUT = 15


def transcribe_with_timeout(label, audio, timeout=TIMEOUT, **kwargs):
    result = {"text": None, "info": None, "error": None, "timed_out": False}

    def run():
        try:
            segments, info = model.transcribe(audio, **kwargs)
            tokens = []
            for i, s in enumerate(segments):
                tokens.append(s.text)
                print(f"  [{label}] segment {i}: '{s.text}' ({s.start:.1f}-{s.end:.1f}s)")
            result["text"] = "".join(tokens).strip()
            result["info"] = info
        except Exception as e:
            result["error"] = str(e)

    t = threading.Thread(target=run, daemon=True)
    start = time.time()
    t.start()
    t.join(timeout=timeout)
    elapsed = time.time() - start

    if t.is_alive():
        result["timed_out"] = True
        print(f"  [{label}] TIMED OUT after {elapsed:.1f}s")
    elif result["error"]:
        print(f"  [{label}] ERROR: {result['error']} ({elapsed:.1f}s)")
    else:
        detected = ""
        if result["info"]:
            detected = f" detected={result['info'].language}({result['info'].language_probability:.2f})"
        print(f"  [{label}] OK: '{result['text']}'{detected} ({elapsed:.1f}s)")

    return result


print("=" * 60)
print("TEST 1: Random noise (sanity check - should work)")
print("=" * 60)
noise = np.random.randn(16000).astype(np.float32) * 0.01
transcribe_with_timeout("noise", noise, beam_size=1, language="en")

print()
print("=" * 60)
print("TEST 2: Real speech - vanilla (the failing case)")
print("=" * 60)
transcribe_with_timeout("vanilla", audio_full, beam_size=1, language="en")

print()
print("=" * 60)
print("TEST 3: Real speech - disable no_speech detection")
print("  (if model wrongly detects 'no speech', this forces transcription)")
print("=" * 60)
transcribe_with_timeout(
    "no_speech=0", audio_full,
    beam_size=1, language="en",
    no_speech_threshold=0.0,
)

print()
print("=" * 60)
print("TEST 4: Real speech - disable timestamps")
print("  (timestamps use extra decoder passes that could hang)")
print("=" * 60)
transcribe_with_timeout(
    "no_timestamps", audio_full,
    beam_size=1, language="en",
    without_timestamps=True,
)

print()
print("=" * 60)
print("TEST 5: Real speech - short clip (first 2 seconds only)")
print("=" * 60)
audio_short = audio_full[:sr * 2]
print(f"  Using first 2s ({len(audio_short)} samples)")
transcribe_with_timeout("2s_clip", audio_short, beam_size=1, language="en")

print()
print("=" * 60)
print("TEST 6: Real speech - very short clip (first 1 second)")
print("=" * 60)
audio_1s = audio_full[:sr * 1]
print(f"  Using first 1s ({len(audio_1s)} samples)")
transcribe_with_timeout("1s_clip", audio_1s, beam_size=1, language="en")

print()
print("=" * 60)
print("TEST 7: Real speech - max_new_tokens=1 (force early stop)")
print("  (if this hangs, the problem is before token generation)")
print("=" * 60)
transcribe_with_timeout(
    "max1tok", audio_full,
    beam_size=1, language="en",
    without_timestamps=True,
    max_new_tokens=1,
)

print()
print("=" * 60)
print("TEST 8: Encoder-only test (language detection, no decode)")
print("=" * 60)
start = time.time()
try:
    lang, prob, all_probs = model.detect_language(audio_full)
    elapsed = time.time() - start
    print(f"  Language: {lang} (prob={prob:.3f}) in {elapsed:.1f}s")
    top5 = sorted(all_probs, key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top 5: {top5}")
except Exception as e:
    print(f"  detect_language FAILED: {e}")

print()
print("=" * 60)
print("DONE - check which tests hung vs completed")
print("=" * 60)
