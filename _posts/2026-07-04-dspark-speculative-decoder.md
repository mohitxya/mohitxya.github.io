---
layout: post
title: "Speeding up Generation: From Speculative Decoding to DSpark"
categories: [Speculative decoder, LLM, Inference]
topic: technical
---
![DSpark thumbnail](/assets/image/posts/dspark/dspark_thumb.png)

In line with my recent fascination with inference, I've been catching up on some of the latest work in the area. 
Deepseek's DSpark was released earlier this week and caught my eye since it achieves really remarkable acceleration in LLM inference (boosts single-user generation speeds by 60% to 85%! crazy). Building on traditional approaches to speculative 
decoding, it has introduced some very interesting changes which I want to talk about after I lay down some initial ground-work on the theoretical background. 

### the inference bottleneck: why bother? 
Inference in LLMs usually means two stages: prefill and decode. In prefill you build up the KV cache.Caching the Keys and Values for a given prompt instead of recomputing them from scratch did solve a major bottleneck in computation. But, we still end up loading all the model weights during forward pass for every single token. From GPU VRAM/HBM to GPU compute cores. The GPU obviously can't keep all model weights inside fast on-chip memory (It's expensive so to make it work economically, it's kept relatively small).

Speculative decoding helps by making one expensive forward pass do work for multiple tokens instead of just a single token. We train a smaller model called draft model to predict several positions at once. Our main model simply verifies them over it's own distribution and drops them if they seem improbable. 

Put more formally, Speculative decoding accelerates the inference of a target model $M_{T}$ using a lightweight model $M_{d}$. At each decoding cycle, the draft model proposes $\lambda$ candidate tokens. The target model verifies all candidates in a single forward pass, accepting the longest prefix consistent with its own distribution. 

$acceptance probability = min(1, p_t^k(x_k) / p_d^k(x_k))$

> For the draft token $x_k$ at position k, how willing is the target model to accept it compared to how likely the drafter thought it was?

### autoregressive vs parallel drafters

Early draft models were autoregressive which meant they would condition each position on previously sampled tokens. The belief was that their autoregressive nature would allow them to generate more meaningful draft tokens. Their drafting latency grows linearly with block size (number of draft tokens generated) and this forced researchers to use short blocks and shallow architectures. 

> Logically one would conclude that autoregressive models (like Eagle3) create higher quality draft tokens but what ends up happening in practice is this: their sequential nature forces them to be shallow (mapping connections is expensive!). Parallel or semi-autoregressive drafters can afford more architectural capacity, and that extra capacity can outweigh the lack of full autoregressive dependency. Bummer. 

Parallel drafters on the other hand predict each position independently, so they can't really model inter-token dependencies within a block. This makes them susceptible to a thing called "rapid acceptance decay": 

> say, we predict 4 tokens (a, b, c and d). Token d could be a good guess for that position, but not necessariliy for the specific future created by tokens a, b and c. 

- mention the formula
- talk about semi autoregressive generation
#### DFlash
- DSpark builds on it. important. depth. 
- intuition
- math: break down formulae
- diagrams


### what DSpark brings to the table? 
- Minor modification to DFlash
- talk about sequential temperature scaling
#### Sequential stage
#### Confidence-scheduled verificaion
#### Hardawre aware prefix scheduler

### Results 

### Where are we headed? 


