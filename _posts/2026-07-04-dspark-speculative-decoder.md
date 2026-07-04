---
layout: post
title: "DSpark: how does it work?"
categories: [Speculative decoder, LLM, Inference]
topic: technical
---
In line with my recent fascination with inference, I've been catching up on some of the latest work in the area. 
Deepseek's DSpark was released earlier this week and caught my eye since it achieves really remarkable acceleration in LLM inference (boosts single-user generation speeds by 60% to 85%! crazy). Building on traditional approaches to speculative 
decoding, it has introduced some very interesting changes which I want to talk about after I lay down some initial ground-work on the theoretical background. 

### why bother? 
Inference in LLMs usually means two stages: prefill and decode. In prefill you build up the KV cache.Caching the Keys and Values for a given prompt instead of recomputing them from scratch did solve a major bottleneck in computation. But, we still end up loading all the model weights during forward pass for every single token. From GPU VRAM/HBM to GPU compute cores. The GPU obviously can't keep all model weights inside fast on-chip memory (It's expensive so to make it work economically, it's kept relatively small).

Speculative decoding helps by making one expensive forward pass do work for multiple tokens instead of just a single token. We train a small draft model to predict several positions at once. Our main model simply verifies them over it's own distribution and drops them if they seem improbable. 

### speculative decoding: brief survey
Early draft models were autoregressive which meant they would condition each position on previously sampled tokens. The belief was their autoregressive nature would allow them to generate more meaningful draft tokens. Their drafting latency grows linearly with block size (number of draft tokens generated) and this forced researchers to use short blocks and shallow architectures. 

Parallel drafters on the other hand predict each position independently, so they can't really model inter-token dependencies within a block. This makes them susceptible to a thing called "rapid acceptance decay": 

> say, we predict 4 tokens (a, b, c and d). Token d could be a good guess for that position, but not necessariliy for the specific future created by tokens a, b and c. 

They benchmarked Qwen3 (4B, 8B and 14B) and Gemma4-12B mainly in three domains: 
1. Mathematical Reasoning 
2. Code Generation
3. Daily Chat 
(The specifics on the data sets used can be found here: )

- Specifically across Qwen3-4B, 8B, and 14B models, DSpark improves the 