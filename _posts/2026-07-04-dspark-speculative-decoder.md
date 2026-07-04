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
Inference in LLMs usually means two stages: prefill and decode. In prefill you build up the KV cache. 

> A quick recap on attention: $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$, To figure out how much context a token needs from the rest of the sentence, the formula takes the query for that token:
- calculates dot product against the key of every token in the sequence, including itself. 
- Scales it. 
- Calculates its Softmax. 
- Multiplies each word's value vector with this softmax percentage

KV Cache just allows you to skip this tedius calculation and skip re-computation for every token in your sequence. 

Caching the Keys and Values for a given prompt did solve a major bottleneck in computation. But, we still end up loading all the model weights during forward pass for every single token. From GPU VRAM/HBM to GPU compute cores. The GPU obviously can't keep all model weights inside fast on-chip memory (It's expensive so to make it work economically, it's kept relatively small).

Speculative decoding helps by making one expensive forward pass do work for multiple tokens instead of just a single token. We train a smaller model called draft model to predict several positions at once. Our main model simply verifies them over it's own distribution and drops them if they seem improbable. 

Put more formally, Speculative decoding accelerates the inference of a target model $M_{T}$ using a lightweight model $M_{d}$. At each decoding cycle, the draft model proposes $\lambda$ candidate tokens. The target model verifies all candidates in a single forward pass, accepting the longest prefix consistent with its own distribution. 

$\text{acceptance probability} = min(1, p_t^k(x_k) / p_d^k(x_k))$

> Intuition: For the draft token $x_k$ at position k, How likely did the drafter think this exact token was ($p_d^k(x_k)$)? How likely does the target model think this exact token is ($p_t^k(x_k)$)? 

 - Let 𝜏 denote the number of accepted tokens per-cycle
 - let $𝑇_{draft}$ and $𝑇_{verify}$ be the wall-clock times of the drafting and verification passes, respectively. 
 - The average latency per generated token is:

$L=(T_{draft} + T_{verify})/\tau$

Improving speedup therefore reduces to three levers: lowering $𝑇_{draft}$ (draft faster), raising 𝜏 (draft better), or reducing the effective $𝑇_{verify}$ (verify smarter).

### autoregressive vs parallel drafters

Early draft models were autoregressive which meant they would condition each position on previously sampled tokens. The belief was that their autoregressive nature would allow them to generate more meaningful draft tokens. Their drafting latency grows linearly with block size (number of draft tokens generated) and this forced researchers to use short blocks and shallow architectures. 

> Logically one would conclude that autoregressive models (like Eagle3) create higher quality draft tokens but what ends up happening in practice is this: their sequential nature forces them to be shallow (mapping connections is expensive!). Parallel or semi-autoregressive drafters can afford more architectural capacity, and that extra capacity can outweigh the lack of full autoregressive dependency. Bummer. 

Parallel drafters on the other hand predict each position independently, so they can't really model inter-token dependencies within a block. They produce all y tokens in a single forward pass making the drafting time nearly independent of the block size.

This does make them susceptible to a thing called "rapid acceptance decay": 

> say, we predict 4 tokens (a, b, c and d). Token d could be a good guess for that position, but not necessariliy for the specific future created by tokens a, b and c. So, our predictions at later stages are more likely to be rejected. 

 DSpark adds a light-weight sequential block on top of a particular type of parallel drafter architecture (DFlash) making it a "Semi-autoregressive" drafter. 

### DFlash

It was released earlier this year and is a parallel drafter. For the parallel generation to work, it uses the big target model's hidden states as guidance. When the target model processes the prompt, its internal hidden representations contain rich information about the context and even hints about possible future tokens. DFlash extracts hidden features from several layers of the target model, fuses them, and feeds them into the draft model.

Instead of only adding target features to the input embeddings, DFlash injects target features into the Key and Value cache of every draft layer.

- Input fusion: Gives the draft model target info once at the beginning. 
- KV injection: Keeps the target info available inside every draft layer. 

**How does training look like?**

```
prompt: p1 p2 p3 p4
response: r1 r2 r3 r4 r5 r6 ...
```
- You pick an anchor token from response. 
- Mask the next few tokens

```
r1 [MASK] [MASK] [MASK]
```
- The model learns to predict the masked tokens in parallel. 

The DFlash implementation: [github](https://github.com/z-lab/dflash)

We can see how the concepts are implemented (quick example):

![KV](/assets/image/posts/dspark/kv_injection.png)

- here we run the `target_hidden` through the same K/V projection matrices used for draft's own tokens, concatenating the results into the KV sequence that every query attends over. 
- `target_hidden` in case you didn't notice is the hidden representation we talked about.  

### what DSpark brings to the table? 
- Minor modification to DFlash
- talk about sequential temperature scaling
#### Sequential stage
#### Confidence-scheduled verificaion
#### Hardawre aware prefix scheduler

### Results 

### Where are we headed? 

I'm super new to this world of inference optimization, but I do wonder if speedups like DSpark, DFlash, MTP, etc might get us to a point where model spillover to the RAM (CPU) is more tolerable. It may make offloading more practical by giving the system extra time to prefetch/compress/prepare future KV-cache data.

When KV cache is the bottle-neck, this paper ([QuantSpec](https://arxiv.org/abs/2502.10424?utm_source=chatgpt.com)) introduces a pretty cool idea: the draft model shares the architecture of the target model but employs a hierarchical 4-bit quantized KV cache and 4-bit quantized weights for acceleration (haven't read the paper properly yet). 

