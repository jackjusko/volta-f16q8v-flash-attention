# Volta FlashAttention-style kernel: FP16 K + Q8_0 V

A CUDA attention-kernel contribution and integration brief for serving engines that want an optimized mixed-precision KV-cache path on NVIDIA Volta (SM70, tested on Tesla V100).

This repository publishes the kernel source changes as a patch against llama.cpp, the benchmark record, and one integration prompt per serving engine. It does **not** ship a llama.cpp build, compiled binary, model, or turnkey plugin. The implementation is integrated into llama.cpp's existing attention template; its patch is the source of truth, and the prompts guide an outside coding agent through adapting the algorithm to another engine's own layouts and runtime.

## What the kernel computes

For each query head, causal attention is

\[
S = \frac{QK^T}{\sqrt{d}} + M,\qquad P=\operatorname{softmax}(S),\qquad O=PV,
\]

where `M` is the causal/additive mask. The reference llama.cpp op supplies Q in FP32 and converts each query tile to FP16 for tensor-core MMA; K is stored as FP16; V is stored as Q8_0. The benchmarked model used a 256-wide head and 24 query heads grouped over 4 KV heads (GQA ratio 6). Dispatch in the reference adapter also admits head dimensions 64 and 128, GQA ratios 2–8, and query batches up to 2048 when the other layout/mask/alignment guards match.

“Q8_0” is a specific block format, not a generic synonym for INT8 KV: each 32-value block has a 16-bit scale and 32 signed 8-bit values. Its values are reconstructed as `v[i] = scale * q[i]`. This kernel reads each Q8_0 V block and widens small chunks to FP16 while staging an attention tile in shared memory. The tensor-core weighted-value product then consumes that tile. Keys remain FP16, so the score product does not pay a K dequantization step.

## Step by step: one attention tile

1. **Choose a work tile.** The kernel assigns work across query rows, grouped query heads, and a block of KV positions. The reference path uses tiled MMA/WMMA operations and shared-memory staging. Larger query batches use a query tile suited to prefill; decode-sized batches select smaller tiles. This is why the dispatch is shape-aware rather than a single launch configuration.
2. **Load Q and K.** Query rows and the current key block are staged in the layout expected by Volta tensor cores. The MMA computes a tile of dot products. Scaling and the causal/additive mask are applied before normalization. Invalid tail rows are guarded by the adapter's bounds checks.
3. **Update softmax online.** The kernel does not build the full `sequence_length × sequence_length` score matrix. As each K block arrives, it tracks a running maximum `m`, normalization sum `l`, and weighted-value accumulator `a`. For a new score tile with maximum `m'`, it rescales the old state by `alpha = exp(m - max(m,m'))`, folds in the new tile's exponentials, and continues. At the end, `O = a/l` (with the implementation's existing numerical bookkeeping). This is the key long-context property: memory for intermediate attention scores is bounded by tile size rather than growing quadratically with context.
4. **Load and expand Q8_0 values.** For the V tile only, each worker decodes Q8_0 scale/value chunks and writes FP16 values into shared memory using the swizzled tile layout. It expands only the tile currently needed; it does not convert the whole cache to FP16. The analogy is unpacking just the current page of a compressed book onto a desk, rather than photocopying the whole library.
5. **Accumulate weighted V.** MMA multiplies the normalized probability tile by the staged V tile. A partial weighted sum is combined with prior KV tiles using the same online-softmax rescaling, then the final result is stored.
6. **Dispatch only when safe.** The llama.cpp adapter checks SM70, data types, head dimensions, GQA ratio, query count (1–2048), contiguity/alignment, mask shape/type, and unsupported features such as nonzero softcap or extra inputs. Unsupported shapes use the engine's existing attention fallback. The environment variable `GGML_CUDA_VOLTA_F16Q8_FATTN_MMA_DISABLE` disables this route.

A useful analogy: ordinary attention lays every question-to-token score card on a huge table, sorts and normalizes all of them, then revisits the values. Tiled online attention processes a stack of cards a handful at a time, keeps only the running normalization and weighted answer, and discards each score tile. Q8_0 is a compact shelf for V; the kernel expands only the cards' current shelf section when it needs to combine them.

## Why this mixed cache can help

For equal K and V widths, FP16 K plus Q8_0 V is about 3.06 bytes per KV element pair on average (2 bytes for K and 34/32 bytes for Q8_0 V), versus 4 bytes for FP16/FP16. That is about 23% less KV payload than FP16/FP16. It is not half-sized because keys stay FP16, and the Q8_0 scales add overhead. At long context, fewer V bytes moved can help bandwidth-bound decode; prefill can instead be limited by projection/GEMM work and tile scheduling. The measured gains below are specific to the stated V100/model/build/ubatch setup and should not be generalized without a same-shape benchmark.

## Scope and compatibility

- Hardware tested: NVIDIA Tesla V100, compute capability 7.0 (SM70), 32 GiB device.
- Cache layout: FP16 K and llama.cpp-compatible Q8_0 V blocks. Other engines' INT8/FP8/FP8-E4M3/scaled formats need an explicit format adapter and numerical validation.
- Reference dispatch shapes: D=64/128/256; GQA 2–8; query rows up to 2048; additional mask, layout, and alignment constraints apply.
- Not a universal CUDA FlashAttention-2 drop-in. It is a FlashAttention-style tiled online-softmax path using the llama.cpp MMA attention machinery, with Q8_0 V tile decoding added. Several memory-layout and runtime details are host/backend-specific.
- Long-context support does not imply every sequence length or scheduler shape is legal. In particular, padding, page tables, split-KV reduction, graph capture, sliding-window masks, sinks, softcaps, and ragged batches must be validated per serving engine.

## Source and application

`kernel/llama.cpp-volta-f16k-q8v.patch` is the source patch against llama.cpp base revision `90ce70c269d9037dcac6d885b724e9d82651b7f1` (build label `90ce70c26`). It changes only the CUDA attention source files needed for the mixed path; it does not include local tests or build products. Apply to that exact base to inspect the source in context. This is a reference integration, not a promise that the patch applies to later llama.cpp revisions without conflict.

The patch deliberately reuses the existing generic FP16 MMA kernel and adds a Q8_0 V tile loader. This is useful reference code for ports, but a serving engine with a different cache representation or kernel DSL should adapt the algorithm rather than copy the ggml wrapper verbatim.

## Benchmarks

See [benchmarks/results.md](benchmarks/results.md) for test details and caveats. Short version: on a Tesla V100, the custom path reached 395.19 tok/s on a 131,072-token prompt versus 359.70 FP16/FP16 and 357.09 original FP16-K/Q8_0-V. For a 2,048-token decode at a 131,072-token total context, it reached 19.66 tok/s versus 13.55 and 12.19 tok/s respectively. The prefill comparison uses different `ubatch` settings for the original references (512) and custom kernel (2048), so the full-prompt speedup cannot be attributed to the kernel alone. Decode runs used the same 2,048 generated tokens and two repetitions per path.

## Engine integration prompts

The selected targets cover widely used self-hosted runtimes and CUDA-oriented serving stacks. Their interfaces and hardware matrices change; each prompt begins by asking the agent to verify the current checkout.

| Prompt | Project |
|---|---|
| [vLLM](prompts/vllm.md) | [vllm-project/vllm](https://github.com/vllm-project/vllm) |
| [SGLang](prompts/sglang.md) | [sgl-project/sglang](https://github.com/sgl-project/sglang) |
| [TensorRT-LLM](prompts/tensorrt-llm.md) | [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) |
| [Text Generation Inference](prompts/tgi.md) | [huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference) |
| [LMDeploy](prompts/lmdeploy.md) | [InternLM/lmdeploy](https://github.com/InternLM/lmdeploy) |
| [Ollama](prompts/ollama.md) | [ollama/ollama](https://github.com/ollama/ollama) |
| [Aphrodite Engine](prompts/aphrodite.md) | [aphrodite-engine/aphrodite-engine](https://github.com/aphrodite-engine/aphrodite-engine) |
| [ExLlamaV2 / successor](prompts/exllamav2.md) | [turboderp-org/exllamav2](https://github.com/turboderp-org/exllamav2) |

Each file in `prompts/` is a standalone task prompt for an outside coding agent. It asks the agent to inspect the current engine source, identify its actual cache and attention APIs, implement a guarded backend path, add correctness coverage, and measure performance against the engine's native references. The prompts are plans for ports; they do not claim that those integrations already exist or that every target supports SM70 today.

## Attribution

The reference patch is derived from ggml-org/llama.cpp, licensed under MIT. See [LICENSE](LICENSE). Benchmark values were collected locally with llama.cpp build `90ce70c26`; the exact source logs are available in the development environment, not vendored here.
