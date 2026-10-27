# native-16bit packages: changelog

## native-16bit-diagnostic-1 (stage 1: diagnostic)

- Added `tools/native16_diagnostics.py`. Symbols mode reports the environment,
  CPU feature flags, library loading and a dlsym check of BNNS / BLAS /
  vImage candidates, and calls no Accelerate function. Probe mode (opt-in)
  calls documented BNNS FC / MatMul / Copy and vImage functions on tiny
  matrices, for FP32/FP16/BF16 weight, input and output combinations, with
  and without `BNNSFlagsUseClientPtr`. It records refusal or success, error
  against float64, allocation bytes (via the `alloc_memory` hook) and
  whether the filter still reads the caller's weights. Crash breadcrumbs
  skip a crashing step, and its whole dtype group, on later runs. A
  self-test step isolates the ctypes allocation callback, so a crash there
  disables the callback instead of every probe.
- Added host tests and a fake BNNS (`tests/native16_fake_bnns.c`) that
  exercise the ctypes plumbing without Accelerate.
- Added `docs/native16_feasibility.md` and `docs/native16_diagnostic.md`.
- Installs `install_native_16bit.py` and `rollback_native_16bit.py` at the
  repository root. Later stages then need only their ZIP, and a package can
  update those two scripts in place.
- No production file is replaced or deleted.
