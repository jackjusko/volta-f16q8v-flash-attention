# Integration task: LMDeploy / TurboMind

Port the Volta FP16-K/Q8_0-V tiled attention algorithm in this repository to LMDeploy's relevant runtime. Read the root README, benchmark caveats, and reference patch before modifying code.

1. Inspect the current LMDeploy revision and distinguish TurboMind from the PyTorch engine. Identify which path supports SM70, which owns attention kernels, and how the paged KV cache and block allocation are implemented.
2. Examine existing KV quantization formats and scale packing. The target is Q8_0's 32 int8 values plus FP16 scale per block. Decide whether TurboMind can consume that exact representation without conversion; otherwise add an explicit cache type and allocator path.
3. Add the kernel at the lowest stable backend interface that retains block-table addressing, tensor-parallel/GQA behavior, stream and graph semantics, and continuous batching. Compute scores from FP16 K, decode V blocks into shared-memory tiles, and carry online softmax state across KV tiles.
4. Enable only for verified SM70, supported head widths 64/128/256, GQA 2–8, query batches <=2048, supported masks/features, and layout alignment. Keep FP16/native fallback and an opt-out; do not change quantization defaults globally.
5. Test exact numerical output against existing LMDeploy attention, including page boundary/tail cases, partial tiles, long contexts, GQA, batched prefill/decode, and cache reuse. Add checks for graph capture and concurrency if those are used on this path.
6. Benchmark V100 at 32K/64K/128K against FP16/FP16 and existing mixed KV cache with matched scheduler, batch, and ubatch settings. Report prefill/decode tok/s, request latency, VRAM use, and which LMDeploy engine/backend ran.
7. Return a minimal patch, tests, and version/hardware support table. If TurboMind or its toolchain has dropped SM70, state the blocker and leave the feature opt-in/unsupported.
