# Integration task: Hugging Face Text Generation Inference (TGI)

Port the Volta FP16-K/Q8_0-V attention algorithm from this repository into the current TGI implementation if its present architecture and maintenance status make that a viable target. Read the root README and benchmark record first; the llama.cpp patch is only a low-level reference.

1. Check the current TGI repository status, supported deployment paths, CUDA architecture targets, and its current attention backend/dependencies. Verify that the selected backend supports SM70 today. If TGI has delegated attention to another package, identify the correct upstream extension point.
2. Inspect paged KV-cache representation, page tables, quantization scales, batch scheduler, continuous batching, CUDA graph use, and prefix reuse. Keep Q8_0 distinct from generic INT8/FP8 cache formats unless the byte layout and scales are identical.
3. Propose and implement the smallest supported backend route for FP16 K and Q8_0 V: tiled QK, dequantized V tile in shared memory, online softmax accumulation, and correct page/table addressing. Preserve the native fallback and API behavior.
4. Dispatch only on confirmed SM70 and valid shape/layout: D=64/128/256, GQA 2–8, query chunk 1–2048, aligned K/V blocks, supported masks. Add feature gates and a runtime disable switch.
5. Add reference comparisons for decode, prefill, causal masks, variable lengths, page boundaries/tails, GQA, and long contexts. Verify concurrent/continuous batches and request cancellation do not corrupt cache state.
6. Run V100 benchmarks only if supported by current TGI dependencies. Compare matched FP16/FP16, existing FP16-K/Q8-V, and custom implementations at 32K, 64K, and 128K; report exact model, server flags, batch/ubatch, latency, throughput, and memory.
7. Deliver a patch and an honest support note. If TGI is in maintenance mode or its supported backend excludes SM70, report that and point to the maintained dependency that owns the attention kernel; do not fork an obsolete path without justification.
