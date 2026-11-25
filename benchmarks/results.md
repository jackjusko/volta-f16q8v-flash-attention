# V100 benchmark record

All throughput values are tokens/second. Hardware: Tesla V100 (SM70, 32,494 MiB reported by CUDA). Model: `Swift-1.5-Qwen3.8-27B-Uncensored-Dynamic-MTP-UD-Q3_K_XL.gguf` (llama.cpp reports Q3_K, 27.32B, 12.23 GiB). Build: llama.cpp `90ce70c26` / source base `90ce70c269d9037dcac6d885b724e9d82651b7f1`. Flash attention was enabled. These are model-level end-to-end `llama-bench` results, not isolated attention-kernel timings.

## Full prompt prefill

| Prompt tokens | FP16 K / FP16 V, ubatch 512 | Original FP16 K / Q8_0 V, ubatch 512 | Custom FP16 K / Q8_0 V, ubatch 2048 |
|---:|---:|---:|---:|
| 32,768 | 585.90 ± 0.96 | 543.31 (1 run) | 738.30 ± 4.39 |
| 65,536 | 461.72 ± 10.98 | 451.20 (1 run) | 574.17 ± 15.43 |
| 131,072 | 359.70 ± 0.21 | 357.09 (1 run) | 395.19 ± 1.77 |

The FP16 reference used two runs; the original Q8 reference log contains one run, so no standard deviation is available. The custom path used two runs. The original-reference and custom-path `ubatch` values differ, which is a confound: these results show the tested configurations, not an apples-to-apples kernel-only prefill comparison. The custom path's 2048 query-row ceiling is exercised at the configured upper bound; smaller ubatches (including 512) remain within the dispatch guard.

## 2,048-token decode after a large prefill

The depth column is llama-bench's pre-generation context depth. “Total context” adds the 2,048 generated tokens. Each configuration had two depth/generation repetitions.

| Prefill depth | Total context after generation | FP16 K / FP16 V | Original FP16 K / Q8_0 V | Custom FP16 K / Q8_0 V |
|---:|---:|---:|---:|---:|
| 32,768 | 34,816 | 24.53 ± 1.01 | 21.30 ± 0.29 | 25.45 ± 0.02 |
| 65,536 | 67,584 | 18.25 ± 0.34 | 16.89 ± 0.45 | 22.91 ± 0.42 |
| 129,024 | 131,072 | 13.55 ± 0.19 | 12.19 ± 0.24 | 19.66 ± 0.12 |

At 128K total context, the custom result is 45% faster than the FP16/FP16 reference and 61% faster than the original mixed-cache path (based on the reported means). The 2K decode demonstrates sustained long-context generation, not only a short first-token/steady-state sample.

A separate 128-token run at 130,944 prefill depth measured 13.71 ± 0.58 tok/s for FP16/FP16, 11.40 ± 0.82 for original FP16/Q8_0, and 20.59 ± 0.21 for the custom path. It is separate from the 2K decode table and should not be directly compared as the same benchmark workload.

## Local-window prefill status

A matched-ubatch 20K-token prefill appended at a depth of about 54K measured 443.49 ± 9.24 tok/s for FP16/FP16 and 438.72 ± 6.60 for the original FP16/Q8_0 path at ubatch 2048. The custom-path run was interrupted before a result was recorded. No custom local-window number is claimed.

## Reproduction caveats

- These numbers came from the Q3_K_XL model named above, not the later Q5_K_S model used for the running minion servers.
- The reference build and custom build are the same source build label, with dispatch/type settings selecting paths; the original Q8 path and new path share cache dtype but use different attention kernels.
- Full-prompt prefill reference vs custom uses ubatch 512 vs 2048. Interpret that table as configuration-level comparison, not solely a kernel effect.
- Benchmarks ran on a single V100. No other GPU or engine integration has been benchmarked here.
- Correctness was checked in the llama.cpp backend tests and the kernel’s numerical tests during development; the results here record performance measurements, not independent verification of another engine's implementation.
