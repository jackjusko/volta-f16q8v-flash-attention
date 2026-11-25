# Faster long-context attention on NVIDIA V100

This project shares a CUDA attention kernel for a mixed-precision KV cache: **FP16 keys and Q8_0 values**. It was developed and measured on a Tesla V100 (Volta, SM70), with Qwen3.8 as the main target.

## The standout result

At a 128K-token context, generating 2,048 tokens reached **19.66 tokens/second** with this kernel. In the same benchmark, FP16 keys and values reached 13.55 tokens/second, and the original FP16-key/Q8_0-value path reached 12.19 tokens/second.

That is about **45% faster than the FP16 path** and **61% faster than the original mixed-cache path** in this test. The advantage grew with context length:

| Total context | FP16 K/V | Original FP16 K + Q8_0 V | This kernel |
|---:|---:|---:|---:|
| 34,816 | 24.53 tok/s | 21.30 tok/s | **25.45 tok/s** |
| 67,584 | 18.25 tok/s | 16.89 tok/s | **22.91 tok/s** |
| 131,072 | 13.55 tok/s | 12.19 tok/s | **19.66 tok/s** |

Each run generated 2,048 tokens after a long prefill. The full results and measurement notes are in [benchmarks/results.md](benchmarks/results.md).

## Why it helps

During generation, the model repeatedly looks back over the conversation so far. At long context, that means reading a large cache of past keys and values for every new token. This kernel keeps keys in FP16, but stores values in the smaller Q8_0 format. It expands only the small value blocks needed for the current calculation, instead of first converting the whole cache to FP16.

The attention work is also done in tiles. The kernel processes a manageable slice of the past, keeps a running summary of which earlier tokens matter most, then moves to the next slice. It does not create a huge table of every query-to-token score. That keeps temporary memory small as context grows.

On V100, the kernel uses the GPU's tensor-core matrix operations for the main attention calculations. Together, the smaller value reads and tiled processing are the main reasons the long-context decode results improve.

## What happens, step by step

1. **Read the current query and a slice of past keys.** A query represents what the model is looking for. Keys help determine which past tokens are relevant. The reference implementation converts each query tile to FP16 for the tensor-core calculation and reads keys stored in FP16.
2. **Score that slice.** The kernel compares the query with the keys, applies the causal mask so a token cannot see the future, and updates its running record of the strongest matches.
3. **Read the matching value slice.** Values carry the information that gets brought forward. Each Q8_0 block contains 32 signed 8-bit values and one FP16 scale. The kernel applies that scale and expands only the current block into a shared-memory tile.
4. **Add this slice to the running answer.** The kernel combines the slice's weighted values with the work from earlier slices, adjusting for the updated match strengths. Once all visible past tokens have been processed, it writes the final attention result.

A useful way to picture the cache: FP16 keys are the index cards used to find relevant pages; Q8_0 values are compact copies of the page contents. The kernel searches with the keys and opens only the value pages needed for each small step.

## Default dispatch policy

Use hybrid dispatch by default on supported SM70 layouts. The measured Qwen3.8 case (D=256, GQA 6) favors the custom path for decode and 32K prefill, while the standard mixed-cache path is faster for prefill at 64K and above. The llama.cpp selector keeps the custom path for supported decode and shorter prefills, then falls back to standard mixed-cache attention when this shape has more than 8 query rows and at least 65,536 KV rows.

Treat 64K as a conservative measured guard, not a universal crossover: only 32K and 64K bracketed the prefill comparison. When porting to another engine or model shape, benchmark matched prefill and decode workloads and set an engine-specific cutoff. After correctness and performance checks pass, make the hybrid selector the default; retain a runtime opt-out and the engine's native fallback.

## Prefill results

Prefill uses `batch 4096` and `ubatch 2048`. The latest two-run measurements show the new path winning at 32K; at 64K and above, the standard mixed-cache path is faster, so the default dispatcher switches back to it.

| Prompt length | FP16 K/V | Original FP16 K + Q8_0 V | New path forced | Hybrid default dispatch |
|---:|---:|---:|---:|---:|
| 32,768 | 675.33 tok/s | 662.22 tok/s | **738.30 ± 4.39 tok/s** | Same new-path sample; default route not separately retimed |
| 65,536 | — | 583.71 tok/s mean (563.19–604.23) | 574.17 ± 15.43 tok/s | 584.43 tok/s, standard path |
| 131,072 | 427.35 tok/s | 427.50 tok/s | 395.19 ± 1.77 tok/s | standard path selected; not retimed |

The 32K reference values are single samples; the new-path result averages two runs. The 65K mixed-cache mean combines two single-sample runs with a 7% spread; other reference cells are single samples. At 32K the new path beats the original mixed-cache result by 11.5%, and the default route selects that path, though we did not retime after changing dispatch. Forced use loses at 64K and 128K, so the dispatcher selects llama.cpp's standard mixed-cache path at 64K and above. The 128K default was not retimed after the dispatch change. See [the full benchmark record](benchmarks/results.md) for setup and caveats.

## What is included

- `kernel/llama.cpp-volta-f16k-q8v.patch` — the CUDA source changes, against the recorded llama.cpp base revision.
- `prompts/` — one integration task prompt each for vLLM, SGLang, TensorRT-LLM, Hugging Face TGI, LMDeploy, Ollama, Aphrodite Engine, and ExLlamaV2.
- `benchmarks/results.md` — complete measured prefill and decode tables with setup details.

The patch is source code, not a compiled llama.cpp build. It adds Q8_0 value loading to llama.cpp's existing CUDA attention implementation, so it depends on that codebase's types and launch machinery. The other engine prompts ask an agent to inspect the engine's actual cache format and backend before porting; an engine's generic “INT8 KV” mode is not automatically compatible with Q8_0.

## Supported path and limits

The reference dispatch targets NVIDIA SM70, head dimensions 64, 128, or 256, grouped-query ratios 2–8, and query batches up to 2048 when the cache layout, alignment, mask, and other guards match. Unsupported shapes should use the engine's normal attention path. The llama.cpp path can be disabled with `GGML_CUDA_VOLTA_F16Q8_FATTN_MMA_DISABLE`.

This is a FlashAttention-style tiled kernel, not a universal drop-in implementation for every FlashAttention-2 backend. Page tables, split-KV scheduling, graph capture, other KV encodings, and newer GPU architectures need their own integration and validation.

## Source and license

The patch targets llama.cpp revision `90ce70c269d9037dcac6d885b724e9d82651b7f1` (`90ce70c26`). It is derived from the MIT-licensed [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp); see [LICENSE](LICENSE).
