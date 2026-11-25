# V100 benchmark record

All throughput values are tokens/second. Hardware: Tesla V100 (SM70, 32,494 MiB reported by CUDA). Model: `Swift-1.5-Qwen3.8-27B-Uncensored-Dynamic-MTP-UD-Q3_K_XL.gguf` (llama.cpp reports Q3_K, 27.32B, 12.23 GiB). Build: llama.cpp `90ce70c26` / source base `90ce70c269d9037dcac6d885b724e9d82651b7f1`. Flash attention was enabled. These are model-level end-to-end `llama-bench` results, not isolated attention-kernel timings.

## Full prompt prefill

All paths below use `batch 4096`, `ubatch 2048`, Flash Attention enabled, and the same Q3_K_XL model. Throughput is end-to-end `llama-bench` prompt tokens/second. “Custom forced” selects the 32-value Q8 loader even where the default dispatcher now avoids it.

| Prompt tokens | FP16 K / FP16 V | Original FP16 K / Q8_0 V | Custom forced | Default dispatcher |
|---:|---:|---:|---:|---:|
| 32,768 | 675.328 | 662.216 | **722.947** | **722.947** |
| 65,536 | not measured | 583.707 mean (563.186, 604.227) | 541.491 | 584.425, standard mixed-cache path |
| 131,072 | 427.353 | 427.500 | 396.861 | standard mixed-cache path selected; not rerun |

Each cell is one timed sample, except the 65,536 mixed-cache mean, which averages two single-sample runs with a 7% spread. At 32K, the custom kernel is 9.2% faster than the original mixed-cache path and 7.1% faster than FP16. At 64K, forcing the custom loader is slower than the mixed-cache reference; the automatic dispatcher uses the standard path instead. The 128K forced custom result also trails both references, so the automatic dispatcher uses the standard path from 64K upward for the validated Qwen3.8 D=256/GQA=6 prefill shape. The threshold is conservative: 32K was tested as the custom win and 64K as the first loss. Small-query decode remains on the custom path.

A same-build 64K custom tile screen compared the default 16-query × 2-head tile (541.491 tok/s) with a 32-query × 1-head tile (519.027 tok/s). Both used the validated 32-value loader and 32-row KV step; the default tile remained faster. The 32-query tile result is exploratory and does not affect dispatch.

## 2,048-token decode after a large prefill

The depth column is llama-bench's pre-generation context depth. “Total context” adds the 2,048 generated tokens. Each configuration had two depth/generation repetitions.

| Prefill depth | Total context after generation | FP16 K / FP16 V | Original FP16 K / Q8_0 V | Custom FP16 K / Q8_0 V |
|---:|---:|---:|---:|---:|
| 32,768 | 34,816 | 24.53 ± 1.01 | 21.30 ± 0.29 | 25.45 ± 0.02 |
| 65,536 | 67,584 | 18.25 ± 0.34 | 16.89 ± 0.45 | 22.91 ± 0.42 |
| 129,024 | 131,072 | 13.55 ± 0.19 | 12.19 ± 0.24 | 19.66 ± 0.12 |

At 128K total context, the custom result is 45% faster than the FP16/FP16 reference and 61% faster than the original mixed-cache path (based on the reported means). The 2K decode demonstrates sustained long-context generation, not only a short first-token/steady-state sample.

A separate 128-token run at 130,944 prefill depth measured 13.71 ± 0.58 tok/s for FP16/FP16, 11.40 ± 0.82 for original FP16/Q8_0, and 20.59 ± 0.21 for the custom path. It is separate from the 2K decode table and should not be directly compared as the same benchmark workload.

The updated loader split was also checked with 2,048 generated tokens after a 65,536-token prefill, using `batch 4096` and `ubatch 2048`: custom FP16-K/Q8_0-V measured **23.110 tok/s** and the custom-path-disabled mixed-cache reference measured **16.691 tok/s**. This is one sample per path. The small-query decode tile keeps the original 8-value Q8 loader; the wider loader is compiled for prefill tiles.

## Local-window prefill status

A matched-ubatch 20K-token prefill appended at a depth of about 54K measured 443.49 ± 9.24 tok/s for FP16/FP16 and 438.72 ± 6.60 for the original FP16/Q8_0 path at ubatch 2048. The custom-path run was interrupted before a result was recorded. No custom local-window number is claimed.

## Reproduction caveats

- These numbers came from the Q3_K_XL model named above, not the later Q5_K_S model used for the running minion servers.
- The reference build and custom build are the same source build label, with dispatch/type settings selecting paths; the original Q8 path and new path share cache dtype but use different attention kernels.
- The earlier prefill table used mismatched ubatches and has been replaced by matched `ubatch 2048` results above.
- Benchmarks ran on a single V100. No other GPU or engine integration has been benchmarked here.
- Correctness was checked in the llama.cpp backend tests and the kernel’s numerical tests during development; the results here record performance measurements, not independent verification of another engine's implementation.
