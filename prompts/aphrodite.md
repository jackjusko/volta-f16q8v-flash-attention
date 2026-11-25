# Integration task: Aphrodite Engine

Port this repository's Volta FP16-K/Q8_0-V tiled attention path into the current Aphrodite Engine source. Read the root README and benchmark record; the llama.cpp patch is an algorithm reference only.

1. Inspect the current Aphrodite/vLLM-derived architecture, supported GPU targets, attention backend selection, paged KV cache, quantized-cache metadata, scheduler, and CUDA graph behavior. Verify SM70 support in the actual selected backend and current dependency versions.
2. Identify the backend hook best suited for a CUDA extension. Check whether the engine's current paged layout and quantization formats can represent Q8_0 exactly. Add an explicit cache format if needed; never reinterpret FP8/INT8 data silently.
3. Implement the kernel path with FP16 K score tiles, Q8_0 V tile-local dequantization into shared memory, and online-softmax accumulation. Preserve paging, batch mapping, GQA head mapping, masking, and the native fallback.
4. Restrict dispatch to SM70, D=64/128/256, GQA 2–8, query chunks up to 2048, aligned supported layouts, and validated features. Keep it opt-in initially with a disable/fallback route.
5. Add correctness tests versus current attention for decode, prefill, variable-length batches, cache pages and tails, GQA, causal/additive masks, and long contexts. Check CUDA graph capture and sanitizers.
6. Benchmark a V100 against FP16/FP16 and the existing mixed cache at 32K/64K/128K, controlling scheduler, batch, cache block size, and ubatch. Report prefill/decode throughput, p50/p99 latency, memory, kernel selection, and software versions.
7. Send back a minimal patch, reproducible command lines, and compatibility matrix. Do not upstream-enable until tests pass; if current Aphrodite no longer has an independent backend suitable for SM70, explain the preferred upstream target.
