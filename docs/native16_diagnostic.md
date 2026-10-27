# native-16bit-diagnostic-1: device capability check

| | |
|---|---|
| version | `native-16bit-diagnostic-1` |
| stage | 1 of 4: diagnostic |
| depends on | nothing |
| changes production code? | **no**. It only adds files; nothing is replaced or deleted. |
| device testing required before stage 2? | **yes** |

This package finds out what Apple's Accelerate / BNNS can do **on your
iPhone**. Stage 2 (a native 16-bit matrix-vector experiment) is designed
from these results. The background and the reasoning are in
`docs/native16_feasibility.md`.

## Files

| action | path | purpose |
|---|---|---|
| add | `install_native_16bit.py` | the installer, kept in tiny_gpt so later stages need only a ZIP |
| add | `rollback_native_16bit.py` | undo script |
| add | `tools/native16_diagnostics.py` | the diagnostic |
| add | `tests/test_native16_diagnostics.py` | host tests (they skip in Pythonista, where there is no C compiler) |
| add | `tests/native16_fake_bnns.c` | fake BNNS used only by those host tests |
| add | `docs/native16_diagnostic.md` | this file |
| add | `docs/native16_feasibility.md` | feasibility report |
| replace | — | none |
| delete | — | none |

## Install (Pythonista)

You never delete or overwrite anything by hand. The installer does the
replacing, and backs everything up first.

**First time** (no `install_native_16bit.py` in tiny_gpt yet):

1. Save `native_16bit_diagnostic.zip` to the Files app.
2. In Pythonista's file browser, open the `tiny_gpt` folder, tap **+**,
   choose **Import…**, then **Files**, and pick the ZIP. Any folder under
   Documents works. If Pythonista names it `native_16bit_diagnostic 2.zip`,
   that's fine.
3. Tap the ZIP to extract it. You get a `native_16bit_diagnostic` folder.
4. Open that folder, open `install_native_16bit.py`, and press **▶︎ Run**.

   (Alternatively, import `install_native_16bit.py` on its own as well and
   run that. It finds the ZIP without extracting.)

**Later stages:** import the new ZIP the same way, then run
`tiny_gpt/install_native_16bit.py`. There's no need to extract it.

What the installer does then:

- It finds tiny_gpt and the package(s), recognising them by contents, not
  file name.
- It prints the plan: here 7 × `ADD`, nothing replaced or deleted.
- It asks **`Install? [Y/n]`**. Press Enter.
- It installs, checks that every Python file compiles, and runs the package
  tests. Most of those skip on iOS; the rest must pass.
- It prints `SUCCESS: native-16bit-diagnostic-1 installed.`
- It offers to **tidy up**: the used ZIP, the extracted folder, duplicates
  and stray installer copies are *moved* (not deleted) into
  `tiny_gpt/.native16/imported/`, so the next import never collides with an
  old name. Press Enter.

If it reports files that "already exist and differ" (for example an older
copy you put there by hand), it explains and asks before replacing them.
Each one is saved to `tiny_gpt/.native16/backups/` first.

## Install (desktop / command line)

```
python install_native_16bit.py --zip native_16bit_diagnostic.zip --repo /path/to/tiny_gpt --dry-run
python install_native_16bit.py --zip native_16bit_diagnostic.zip --repo /path/to/tiny_gpt --run-tests
```

## Test procedure (on the iPhone, in Pythonista)

Close other heavy apps, keep the phone off the charger and not hot, and
don't run a model in the same Pythonista session first.

1. Open `tiny_gpt/tools/native16_diagnostics.py` and press ▶︎ Run.
2. The **symbols** stage runs first. It calls no Accelerate functions, only
   looks them up. It prints a report and writes
   `tiny_gpt/native16_results/diag_symbols_<time>.txt` and `.json`.
3. It then asks `Run the probe stage too (tiny BNNS calls)? [y/N]`. Answer
   **y**. The probe calls documented BNNS / vImage functions on tiny
   matrices (at most 256×512). It takes a few seconds.
4. **If Pythonista crashes or quits during the probe:** reopen it and run
   the script again. It reports `PREVIOUS RUN DIED DURING: <step>`, skips
   that step (and the rest of its data-type group) and carries on.
   Repeat until it finishes. The crash is itself a
   result, so please report it.
5. If a model is already downloaded under `tiny_gpt/hf_models/`, the symbols
   stage also checks the alignment of its first model's tensors. It reads
   headers and addresses only; no weights are loaded. From a shell you can
   pick the model with `--model DIR`.

## What to send back

Send **both** `.txt` files (symbols and probe), the probe `.json`, and any
crash notes. The `.json` holds the details stage 2 needs (allocation sizes,
errors, device flags).

## Expected output

On the build host (Linux, no Accelerate) every symbol shows `-`, and
`probe` reports that Accelerate cannot be loaded. That is the only output
measured so far.

Expected on an iPhone but **not yet observed**:

- `libraries`: `Accelerate loaded`.
- `cblas_sgemv` / `cblas_sgemm` `PRESENT` (tiny_gpt already uses them).
- `structs ok`, and `BNNSNDArrayGetDataSize ok` in the probe. If either says
  `MISMATCH`, the probe stops before any heavy BNNS call. That is the safety
  gate.
- Each `FC_W<type>_x<type>_y<type>_<shape>_<clientptr|copy>` line is one of
  `ok`, `refused`, `apply-failed` or `wrong-result`. Which ones are `ok` is
  exactly what nobody knows yet.

## Results only your device can provide

- whether the BNNS fully connected and `BNNSMatMul` entry points still
  exist on your iOS version (they are deprecated since iOS 18)
- which weight / input / output dtype combinations they accept
  (BF16? FP16? FP32 activations with 16-bit weights?)
- whether a filter keeps reading the caller's weight memory (zero-copy) or
  makes a private copy, and how many bytes it allocates
- numerical error against a float64 reference
- CPU features (`FEAT_FP16`, `FEAT_BF16`), cache sizes, iOS build

## STOP

**Do not install anything else yet.** Send the results back. The
proof-of-concept package (stage 2) will be built from them. Until then no
performance or support claim has been verified on iOS.

## Rollback / uninstall

Pythonista: run `rollback_native_16bit.py` and answer `y`. Command line:

```
python rollback_native_16bit.py --repo /path/to/tiny_gpt --dry-run
python rollback_native_16bit.py --repo /path/to/tiny_gpt
```

This moves the 7 added files (including the two scripts) into `tiny_gpt/.native16/rollbacks/<time>/`
(they are not deleted outright). Results in `native16_results/` are yours,
and are never touched by the installer or the rollback.

## Compatibility

- Pythonista 3.4 (Python 3.10, NumPy 1.22): the target.
- macOS: works the same way; useful as a second opinion, not a substitute.
- Linux / Windows: symbols mode runs and reports everything unavailable.
  The host tests use a fake BNNS built with the system C compiler.
- Standard library + NumPy + ctypes only. Nothing is compiled on the
  device.
