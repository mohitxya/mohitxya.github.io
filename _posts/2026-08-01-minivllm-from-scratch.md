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
    - `kvcache_block_size`: belongs to PagedAttention. Fixed size token blocks.  
- Then the `__post_init__` methods runs a few checks. 
#### sequence.py
- This class is the central datastructure of the inference engine. 
- Everything operates on `Sequence` objects. 
- We have `WAITING`, `RUNNING` & `FINISHED` enums in sequence status class. 
- In the `__init__` method of our sequence class: 
    - We have parameters like the sequence ID, the status (enums), the token IDs of the tokens in that sequence, last token & number of tokens.
    - `num_cached_tokens`: how many tokens are already cached. 
    - `num_scheduled_tokens`: 
    - `block_table`: stores mapping between logical blocks and its actual physical location. 
- We keep track of several metrics too: 
    - if the sequece status is finished or not. 
    - prompt_token_ids, completion_token_ids, num_blocks, last_block_num_tokens. 
    - method to get a block using its ID. 
    - method to append a new token to our sequence. 
```python
def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

def __setstate__(self, state):
    self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
    if isinstance(last_state, list):
        self.token_ids = last_state
        self.last_token = self.token_ids[-1]
    else:
        self.token_ids = []
        self.last_token = last_state
```
