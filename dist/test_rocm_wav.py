import os
import sys

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
from faster_whisper import WhisperModel

wav_path = sys.argv[1] if len(sys.argv) > 1 else None
if not wav_path:
    print("Usage: python test_rocm_wav.py <path_to_wav>")
    sys.exit(1)

import wave
with wave.open(wav_path, 'rb') as wf:
    sr = wf.getframerate()
    frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    print(f"Loaded: {wav_path} ({len(audio)/sr:.1f}s, {sr}Hz)")

model = WhisperModel("tiny", device="cuda", compute_type="float32")
print("Model loaded on GPU\n")

tests = [
    {"beam_size": 1, "language": "en"},
    {"beam_size": 5, "language": "en"},
    {"beam_size": 1, "language": None},
    {"beam_size": 5, "language": None},
]

for t in tests:
    label = f"beam={t['beam_size']}, lang={t['language'] or 'auto'}"
    try:
        segments, info = model.transcribe(audio, **t, condition_on_previous_text=False)
        text = "".join(s.text for s in segments).strip()
        detected = f"{info.language}({info.language_probability:.2f})"
        print(f"[{label}] detected={detected} text='{text}'")
    except Exception as e:
        print(f"[{label}] FAILED: {e}")
