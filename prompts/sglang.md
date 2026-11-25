# Integration task: SGLang

Port the Volta FP16-K/Q8_0-V attention algorithm in this repository into the current SGLang source. Start with the root README, the benchmark record, and the llama.cpp reference patch. Do not treat the patch as directly applicable.

1. Inspect the current SGLang revision and CUDA backend selection, attention kernel (including FlashInfer/Triton/CUDA paths), KV pool allocation, radix/prefix cache, page size, and graph-capture flow. Verify whether SM70 is supported by the actual build and selected attention backend.
2. Identify how SGLang represents quantized KV formats and per-block scales. The target format is llama-compatible Q8_0: 32 signed int8 values plus one FP16 scale per block. If SGLang's allocator/page layout cannot represent it, propose a minimal explicit cache format rather than reinterpreting an existing dtype.
3. Integrate a guarded kernel path that computes QK with FP16 K tiles, dequantizes Q8_0 V only while loading V tiles, and uses FlashAttention-style online softmax to avoid materializing the full score matrix. Preserve request batching, causal masks, prefix reuse, and page indexing.
4. Gate to SM70 and proven supported dimensions (64/128/256), GQA 2–8, compatible page alignment, and query chunks through 2048. Add an opt-out and safe fallback. Explicitly check graph capture, stream semantics, and cross-attention/softcap/sinks feature conflicts.
5. Add numerical tests against SGLang's existing attention path for decode and prefill; test long contexts, page boundaries, non-multiple tails, batch/ragged sequences, GQA, and cache reuse. Include a test for scales at block boundaries.
6. Benchmark on V100 versus FP16/FP16 and SGLang's pre-existing FP16-K/Q8-V implementation, if present. Hold scheduling and page settings constant. Report prefill throughput and decode throughput at 32K/64K/128K, plus latency, memory, and the exact kernel selected.
7. Provide patch, compatibility matrix, tests, and benchmark commands. If the project's supported backend cannot run on SM70, state that clearly and keep the port as an experimental backend rather than silently enabling a slower/unsupported route.
