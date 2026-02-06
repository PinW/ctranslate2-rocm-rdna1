# GPU Transcription Hang — Investigation

*Status: OPEN — decoder hangs on real speech, works on random noise*

## Problem

CTranslate2 built from source for ROCm 6.2 / gfx1010 loads models and runs GPU inference without errors, but:

1. **Random noise → produces output** (gibberish, as expected). Completes in ~2s.
2. **Real speech → hangs indefinitely**. The `model.transcribe()` call never returns. No error, no crash — just blocks forever.
3. **App shows empty transcriptions** when it doesn't fully hang — "Transcription completed in 3.2 seconds" but no text. Likely the same root cause with shorter audio or simpler content.

## What works

- Model loads on GPU: `WhisperModel("tiny", device="cuda", compute_type="float32")`
- GPU detected: `AMD Radeon RX 5700 XT (CC=10.1)`
- Language detection works: `en (confidence: 1.00)`
- Random noise transcription with `beam_size=1, language="en"` completes

## What fails

- `compute_type: float16` → `cuBLAS UNKNOWN` error (Tensile kernel gap)
- `compute_type: int8` → rejected (no MKL/oneDNN in this build)
- Real speech with any beam_size/language combo → hang or empty output
- `Ctrl+C` doesn't cleanly shut down in GPU mode — process must be killed

## Likely cause

The community rocBLAS fallback Tensile kernels for gfx1010 produce results that are numerically close enough to not trigger errors, but accumulate precision differences during the autoregressive decoder loop. With random noise the model quickly outputs gibberish and stops. With real speech, the decoder tries to produce meaningful tokens and either:
- Gets stuck in an infinite loop (never reaches end-of-sequence)
- Produces empty/whitespace tokens that get stripped away

This is consistent with reports of "gibberish output" from other users running unsupported GPU architectures with fallback Tensile kernels.

## Not yet tested

- ~~**CPU float32 with this build**~~ — TESTED: fails with `No SGEMM backend on CPU`. This build has no CPU math backend at all. To test CPU, need to reinstall the stock PyPI ctranslate2 wheel temporarily.
- **Larger models** (base, small) — different model sizes use different GEMM shapes, might hit different Tensile kernels.
- **beam_size=1 with real speech** in isolation — the test script hung on the first test, but the app managed to return (empty) in ~3s. Could be audio-length dependent.

## Test files

- `dist/test_rocm_wav.py` — takes a WAV file, tries 4 combinations of beam_size (1/5) and language (en/auto)
- `documentation/temp/test_rocm_transcribe.py` — original test with random noise (this one works)
- `documentation/temp/test.wav` — real speech recording for testing

## Related docs

- `docs/ctranslate2-rocm6-build.md` — the build process
- `docs/rocm-gfx1010-build-plan.md` — full journey including DLL shim diagnosis
- `README.md` — known limitations section

## Environment

- GPU: AMD RX 5700 XT (gfx1010, RDNA 1)
- CTranslate2: 4.7.1 (built from source, ROCm 6.2, gfx1010)
- Community rocBLAS: likelovewant/ROCmLibs v0.6.2.4 (fallback Tensile kernels)
- Python: 3.13, faster-whisper, device=cuda, compute_type=float32

## Next steps

1. Test CPU float32 to confirm the build math is correct when GPU isn't involved
2. Research if other gfx1010 users have seen similar decoder hangs
3. Consider testing with optimized (not fallback) Tensile kernels if any exist for gfx1010
4. Try `HSA_OVERRIDE_GFX_VERSION=10.3.0` to use gfx1030 kernels instead of fallback — risky but would test if kernel quality is the issue
