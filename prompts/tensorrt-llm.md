# Integration task: NVIDIA TensorRT-LLM

Integrate the repository's Volta FP16-K/Q8_0-V attention path into the current TensorRT-LLM source/runtime. Read the root README, `benchmarks/results.md`, and the llama.cpp patch as algorithm/reference only.

1. Inspect the current TensorRT-LLM release and supported GPU/compute capability matrix first. Confirm whether a V100/SM70 can execute the target runtime and custom plugin path; current-version support may differ from historical support.
2. Map the kernel to TensorRT-LLM's attention plugin/backend and KV-cache manager. Inspect its paged KV layout, cache quantization metadata, scales, context FMHA, in-flight batching, CUDA graph, and split-attention pathways. Do not label FP8/INT8 formats Q8_0 unless block layout and scale semantics match exactly.
3. Choose a custom plugin or backend hook that preserves the engine's workspace, stream, graph, and scheduler contracts. Port FP16 K scoring plus tile-local Q8_0 V dequantization and online softmax. Preserve a reference fallback.
4. Add strict dispatch guards for SM70, head width 64/128/256, GQA 2–8, query chunk <=2048, alignment, mask/features, and cache block layout. Use plugin capability checks and reject/fallback cleanly on unsupported hardware or features.
5. Validate numerical correctness against TensorRT-LLM's FP16 reference for single-token decode, multi-token prefill, causal masking, partial blocks, GQA, long contexts, and any cache reuse modes. Verify graph replay and concurrent requests.
6. Benchmark on V100 if the current release supports it; otherwise provide a compile-time/unit-level test and explicitly state the hardware blocker. Compare matched FP16/FP16, native mixed cache, and custom path at 32K/64K/128K with fixed batch/scheduler settings. Record throughput, latency, peak workspace/cache memory, and build compatibility.
7. Submit a minimal plugin/backend patch, reproducible tests, and a support table. Do not claim general TensorRT-LLM compatibility based solely on compiling the standalone CUDA code.
