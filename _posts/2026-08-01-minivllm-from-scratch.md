---
layout: post
title: "Building a minimal vLLM implementation"
categories: [GPU, LLM, Inference]
topic: technical
---
![DSpark thumbnail](/assets/image/posts/mvllm/mvllm_logo.png)

- vLLM: efficient inference + serving multiple users. 
    - core innovations: paged attention, continuous batching, OpenAI-compatible API. 
    - Supports quantized models. 
    - Optimized GEMM/MoE and attention kernels. 
    - Automatic kernel generation and graph level transformations. 
- The original codebase is massive, so I'll read through a minimal implementation called nano-vllm. 
- I'll come up with my own implementation in the process. Maybe add a few more features (support for more models, speculative decoding, etc). 
### nano-vLLM: 
#### config.py: 
- A config class has:
    - `model`: model name. 
    - `max_num_batched_tokens`: maximum number of tokens that can be processed in a single batch. 
    - `max_num_seqs`: how many requests the engine is allowed to keep active at the same time. 
    - `max_model_len`: maximum number of tokens a single sequence is allowed to have. 
    - `gpu_memory_utilization`: fraction of GPU memory to use. 
    - `tensor_parallel_size`: how many GPUs should jointly execute one model, by default it's set to one. 
