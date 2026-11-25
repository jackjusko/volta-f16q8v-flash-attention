# Integration task: ExLlamaV2 / successor status check

Investigate a port of this repository's Volta FP16-K/Q8_0-V tiled attention algorithm to ExLlamaV2 or its maintained successor. Read the root README and benchmark results first. The reference patch is tied to llama.cpp and cannot be applied directly.

1. Check the current repository/maintenance state and identify whether ExLlamaV2 or its successor is the right target. Inspect its CUDA extensions, attention kernels, cache allocator, cache quantization, GQA layout, and supported SM targets. Confirm that SM70 is buildable and supported.
2. Determine whether its paged/unpaged cache uses compatible Q8_0 blocks. ExLlama's cache formats may differ; define an exact encoding adapter if needed. Preserve its generator/batching and cache-reuse semantics.
3. Integrate the algorithm at the CUDA extension layer: FP16 K dot products, per-tile Q8 V decode to FP16 shared-memory values, and online softmax accumulation. Keep all native attention routes available for unsupported layouts and devices.
4. Gate to SM70, head dimensions 64/128/256, GQA 2–8, query rows 1–2048, valid alignment and supported masks. Add compile-time arch guards and a runtime opt-out.
5. Test against the project's reference path for single-token generation, batched prompts, partial KV blocks, long context, GQA, causal masks, cache quantization error, and repeated/prefix-cached prompts.
6. If V100 is supported, benchmark matched 32K/64K/128K prefill/decode against FP16/FP16 and native mixed KV cache. Record model format, quantized weights, CUDA/PyTorch/compiler versions, generation batch, throughput, latency, and memory. Separate kernel microbenchmarks from model-level throughput.
7. Deliver a small patch, tests, and support statement. If ExLlamaV2 is archived or SM70 is unsupported, target the maintained successor only if its APIs are stable; document the migration and do not claim compatibility with the archived engine.
