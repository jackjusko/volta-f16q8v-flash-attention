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

## Prefill results

The best measured full-prompt prefill rate was **738.30 tok/s at 32K** and **395.19 tok/s at 128K** with this kernel. The table below includes the two reference paths:

| Prompt length | FP16 K/V, ubatch 512 | Original FP16 K + Q8_0 V, ubatch 512 | This kernel, ubatch 2048 |
|---:|---:|---:|---:|
| 32,768 | 585.90 tok/s | 543.31 tok/s | **738.30 tok/s** |
| 65,536 | 461.72 tok/s | 451.20 tok/s | **574.17 tok/s** |
| 131,072 | 359.70 tok/s | 357.09 tok/s | **395.19 tok/s** |

The custom run used ubatch 2048; both reference runs used ubatch 512. Since ubatch affects prefill speed, these numbers compare the tested configurations and do **not** isolate the kernel's contribution by themselves. Decode used the same 2,048 generated tokens for each path and is the cleaner long-context comparison. The original Q8 prefill reference had one run; other table values use two runs. See [the full benchmark record](benchmarks/results.md) for deviations, local-window status, and caveats.

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
