# Native FP16/BF16 matrix-vector on iOS: feasibility report

Status: **stage 1 (diagnostic). No device measurements yet.**

Question: can lowmem decoding multiply directly against the memory-mapped
BF16/FP16 weights through Accelerate, instead of widening every slab to
float32 (`Linear.matvec()` → `to_f32()` → `cblas_sgemv`), and does that make
decoding faster without raising memory?

Each finding below is labelled by where it comes from:

- **[doc]**: Apple SDK headers / documentation.
- **[host]**: inspected or run on the Linux build host (no Accelerate there).
- **[device]**: measured in Pythonista on the iPhone. *None yet.*
- **[hypothesis]**: inferred and unverified.

## 1. Sources

- [doc] Accelerate BNNS headers from the macOS 15.5 SDK (`bnns.h`,
  `bnns_structures.h`, `bnns_constants.h`, `bnns_graph.h`), plus `cblas.h`,
  `cblas_new.h` and vImage `Conversion.h`. macOS 15.5 pairs with iOS 18.5.
  Every declaration carries `ios(...)` availability. The iOS 26 SDK headers
  were not available to the host, so additions or removals in iOS 26 are
  **unknown**. The device diagnostic checks symbols directly.
- [doc] developer.apple.com pages for `BNNSFilterCreateLayerFullyConnected`
  and `BNNSDataType`. They add nothing about which data types each layer
  supports.

## 2. Candidate APIs

| API | iOS availability [doc] | Status [doc] | Relevance |
|---|---|---|---|
| `BNNSFilterCreateLayerFullyConnected` + `BNNSFilterApply` | 14.0 | **deprecated since 18.0** ("Use BNNSGraph* APIs") | y = W·x with a reusable filter object. Main candidate. |
| `BNNSMatMul` + `BNNSMatMulWorkspaceSize` | 16.0 | deprecated since 18.0 | Stateless C = A·B with a caller-supplied workspace. Second candidate. |
| `BNNSDirectApplyBroadcastMatMul` | 14.0 | deprecated since 16.0 | Superseded by `BNNSMatMul`. |
| `BNNSCopy` | 14.0 | current | Dtype conversion between descriptors (a possible BF16→F32 converter). |
| `BNNSNDArrayGetDataSize` | 16.0 | current | Harmless. The diagnostic uses it to check the ctypes struct layout before any heavy call. |
| `BNNSGraphCompileFromFile` / `BNNSGraphContextExecute` | 18.0 | current (the replacement) | Compiles a Core ML model *file*. The 15.5 SDK has no C graph builder. |
| `vImageConvert_Planar16FtoPlanarF` | 5.0 | current | FP16→F32 only. vImage has no BF16. |
| half/bf16 CBLAS (`cblas_hgemv` …) | — | **not declared anywhere** in `cblas.h` / `cblas_new.h` | The diagnostic only looks these names up. It never calls them. |

Data types [doc]: `BNNSDataTypeFloat16 = 0x10010` (iOS 10),
`BNNSDataTypeFloat32 = 0x10020` (iOS 10),
`BNNSDataTypeBFloat16 = 0x18010` (**iOS 15**).

### What the headers do *not* say

- [doc] Which data-type combinations the fully connected layer or
  `BNNSMatMul` accept. The FC header says only "some combinations of the
  parameters may not be supported, in which case the call will fail". So
  FP16/BF16 weights, mixed FP32 activations with 16-bit weights, and the
  accumulation precision are all **undocumented**. Other layers do list
  types (normalization: "FP32, BF16, and FP16"; one layer: "fp16/fp32 input
  and output"), so the omission for FC is not reassuring either way.
- [doc] Whether `BNNSFlagsUseClientPtr` means zero-copy *execution*. The text
  is: "keep the pointers provided by the client at creation time (weights,
  bias), and work directly from this data". That rules out a persistent
  private copy *if honoured*. It says nothing about per-call conversion into
  a temporary buffer, or about packing for a layout or dtype the kernel
  prefers.

### Weight layout matches the checkpoint [doc]

The FC layer stores `Weight(o,i)` at `weights[i + o*in_size]`, described as
`BNNSDataLayoutRowMajorMatrix` with `size = {in, out}`. That is exactly
safetensors' row-major `(out, in)`. A BNNS filter could point straight at
the mapped tensor, and at a row range of it, as the slabs do today.

## 3. Reachability from Pythonista

- [doc] Every candidate is a plain C function taking plain C structs and
  pointers. No blocks, no Objective-C, no Swift. `ctypes` can call them
  exactly as `backend.py` already calls `cblas_sgemv`, so nothing needs
  compiling.
- [host] The ctypes definitions of `BNNSNDArrayDescriptor` (176 bytes),
  `BNNSActivation` (48), `BNNSLayerParametersFullyConnected` (752),
  `BNNSFilterParameters` (32) and `vImage_Buffer` (32) match the C layout
  that `clang -target arm64-apple-ios16.0` computes from the header
  definitions (static asserts on sizes and offsets).
- [doc] `BNNSFilterParameters.alloc_memory` / `free_memory` let the caller
  supply the allocator. The diagnostic uses that to count the bytes BNNS
  allocates at filter creation and per apply. That count is the
  evidence for or against a hidden FP32 weight copy.
- [hypothesis] Deprecated symbols are still exported on iOS 18/26.
  Deprecation is a compile-time annotation. The device symbol check settles
  this.
- BNNSGraph [doc]: the C API compiles a `.mlmodelc`/`.mlpackage` file. Using
  it would mean building a Core ML program offline (coremltools on a
  desktop), presumably with the weight matrix as a runtime *argument* so it
  can stay in the safetensors mapping. [hypothesis] Possible, but it is a
  separate project with its own unknowns. It is not pursued unless BNNS FC
  and MatMul both fail.

## 4. Hardware: does "two FP16 per 32-bit lane" matter?

- [doc/general] Apple CPUs since A11 implement ARMv8.2 FP16 arithmetic.
  A 128-bit NEON register holds 8 FP16 values against 4 FP32, so one
  instruction does twice the arithmetic. BF16 dot-product instructions
  (FEAT_BF16: `BFDOT`/`BFMMLA`) only exist on newer cores. The diagnostic
  reports `hw.optional.arm.FEAT_FP16` / `FEAT_BF16` for the actual device.
  Accelerate also has the undocumented AMX matrix unit. [hypothesis]
  AMX FP16 and BF16 support per chip generation comes from community
  reverse engineering, not Apple, and is unverified here.
- [general] **Single-token decoding is bandwidth bound.** Each weight is
  used exactly once per token: 2 FLOPs per weight, against 2 bytes (BF16) or
  4 bytes (FP32) loaded. Arithmetic width does not set the speed; bytes
  moved per weight do. Per weight, per token:

  | path | memory traffic |
  |---|---|
  | float32 mode (`cblas_sgemv` on resident FP32) | read 4 B |
  | lowmem today (widen into 8 MB scratch, then `sgemv`) | read 2 B + write 4 B + read 4 B (scratch partly cache-resident) |
  | ideal native 16-bit kernel | read 2 B |

  The FP16 packing benefit only matters if a kernel streams the 16-bit data
  itself. A kernel that widens to FP32 internally, per call, is the current
  path moved into Apple's code.

## 5. What it could be worth [hypothesis, projection only]

From the README's iPhone numbers for SmolLM2-135M: float32 12.8 ms/token
(~70% matvec, so ≈ 9 ms matvec and ≈ 3.8 ms everything else), lowmem
19.3 ms/token.

- If a native 16-bit kernel streams 2 B/weight at the same effective
  bandwidth as `sgemv` streams 4 B/weight, matvec becomes ≈ 4.5–9 ms. That
  gives ≈ 8–13 ms/token, i.e. **≈ 75–120 tok/s at lowmem's ~120 MB
  footprint**, versus 52 today.
- If it widens internally per call, expect roughly today's 52 tok/s.
- If it builds a private FP32 copy at creation, it is float32 mode with
  extra steps, and the memory goal fails.

These are projections, not results. The proof-of-concept stage exists to
replace them with device numbers.

## 6. Host-side facts about the weights

- [host] SmolLM2-135M-Instruct `model.safetensors`: all 272 tensors are BF16.
  The data section starts at byte 30536, and **every tensor's absolute file
  offset is ≡ 8 (mod 64)**. Mapped weights are 8-byte aligned but *not*
  16-byte aligned. A kernel that needs 16-byte alignment would fail or need
  a copy. The probe uses NumPy buffers, so stage 2 must also test unaligned
  pointers. `--model DIR` in the diagnostic reports this on the device.
- [host] BF16→FP16 is not lossless in general: FP16's range is ±65504, and
  values under 6e-5 become subnormal or flush to zero. If only FP16 weights
  are accepted, a one-time offline conversion to an FP16 safetensors file
  (still memory-mapped) is the alternative. It would need a range and
  underflow report per tensor before anyone trusts it.

## 7. Fallbacks if BNNS cannot do it (stage 6 candidates)

Cheap, and testable without new native APIs:

1. **Slab size.** The scratch is 8 MB (2M floats). Smaller slabs (e.g.
   256 KB–1 MB) keep the freshly widened rows in L1/L2 for the `sgemv` that
   immediately reads them. [hypothesis] This can remove most of the "write
   4 B + read 4 B" DRAM traffic.
2. **Faster widening.** `np.left_shift` is single-threaded. `BNNSCopy`
   (BF16→F32) or `vImageConvert_Planar16FtoPlanarF` (FP16) may be faster,
   and the probe checks that they are exact.
3. Reusing the per-slab temporary in `Linear.matmul()` (prefill), which
   allocates a fresh `tmp` every slab.

## 8. Decision rules for stage 2 (from the probe output)

| probe result | meaning | next |
|---|---|---|
| `FC_WBF16_xF32_yF32_*_clientptr` **ok**, `reads-client-W=True`, `create-alloc` ≪ `weight_bytes`, `apply-alloc` ≈ 0 | BNNS reads mapped BF16 directly, with no private copy | PoC: BF16 FC filters on real shapes, timing and footprint |
| only `FC_WF16_*` ok | FP16 only | PoC with FP16, plus an offline BF16→FP16 converter and range report |
| ok but `reads-client-W=False`, or `create-alloc` ≥ `weight_bytes` | private copy, possibly FP32 | memory goal at risk. Measure footprint, likely reject |
| `apply-alloc` ≈ `out*in*4` per apply | per-call widening | unlikely to beat today. Measure once, likely reject |
| `MatMul_*` ok with `workspace_bytes` small | stateless alternative | PoC compares it with FC |
| everything refused / symbols absent | not available | stage 6 fallbacks (slab size, BNNSCopy/vImage widening) |
