---
layout: post
title: "RAG is only good as it's retriever"
categories: [RAG, LLM, Information Retrieval]
topic: technical
---
> This blog is work in progress and is being continually updated as I learn more about the topic. Follow my progress here: [GitHub](https://github.com/mohitxya/domain-rag-retriever)

![Cute Golden Retriever](/assets/image/posts/dog_rag.png)

- Before neural networks, search engines usually used **lexical retrieval**.

> lexical retrieval matches documents based on keywords. 

- Earlier RAGs: Sparse word vectors (Huge dimensions, mostly zero).
	- Bag of Words, BM25, TF-IDF, etc. 
- Modern RAGs: Dense Neural Embedding (Mostly non-zero floats, smaller dimensions).
	- Word2Vec, BERT, etc.
- Search loop has remained same: 
	- query vector
	- compare against document vectors
	- rank by similarity
	- return top-k
- Sentence-BERT made this neural retrieval style practical by producing sentence embeddings that can be compared with cosine similarity. [S-BERT](https://arxiv.org/abs/1908.10084)
- FAISS gives us fast similarity search over large collections of dense vectors. 
- **Bag of Words model**: 
	- `similarity = (A · B) / (|A| × |B|)`

- **TF-IDF retrieval**:
	- Term frequency and Inverse document frequency. 
	- `TF("cat", doc) = count of "cat" in doc / total words in doc` (How often does the word appear in this document?)
	- `IDF("cat") = log(total docs / docs containing "cat")` (How rare is the word across all documents?)
	- `TF-IDF(word, doc) = TF × IDF`

- **BM-25 retrieval**: 
	- Best Matching.
	- Builds upon TF-IDF and fixes some of it's weaknesses. 
	- Problem 1: TF grows unboundedly
		- `TF_saturated = TF × (k1 + 1) / (TF + k1)`
		- k1 is typically 1.2-2.0.
	- Problem 2: Long documents get unfairly rewarded
		- `TF_normalized = TF / (1 - b + b × (doc_length / avg_doc_length))`
		- b is typically 0.75
	- Formula: `BM25(word, doc) = IDF × [TF × (k1 + 1)] / [TF + k1 × (1 - b + b × (L / L_avg))]`
	- `k1`: saturation control, `b`: length penalty strength, `L`: document length, `L_avg`: average doc length across corpus.

- **Dense retrieval**: 
	- Fixed size vectors. 
	- Texts with similar meaning would have nearby vectors. 
	- FAISS (Facebook AI Similarity Search) is a library built for one problem: Given a query vector, find the k most similar vectors from a giant collection - fast.
	- `IndexFlatIP`: Highly optimized CUDA/C++ version. 
	- To get cosine behavior we would have to pre-normalize our vectors since it only does dot product. 
	- Other options: `IndexIVFFlat`, `IndexHNSW`.
- Embedding model is a neural network trained to compress text into a vector that preserves useful relationships. 

#### Evaluation
- A benchmark has query, expected relevant document and corpus of candidate documents. 
- **Major benchmarks**: BIER (paper emphasizes that BM-25 remains a robust baseline and dense retrievers can underperform out of domain.)
- Metrics:
	- `Recall@k`: Is the correct document somewhere in top k?
	- `MRR@k`: Mean reciprocal rank, it rewards putting the answer early. (2nd place: 1/2, 3rd place: 1/3, etc)
- General embedding models are trained on broad internet/text data. 
- But your domain might have: medical terms, legal clauses, company-specific abbreviations, etc. 
- fine-tuning teaches domain-specific mappings. 

#### MNRL
- **Multiple negatives ranking loss** trains on pairs: `(query, positive_document)`.
- Correct score must be high, rest should be low. 
- Cached MNRL lets us use a larger effective batch size without needing a huge GPU. 

> Modify benchmarks with 5 harder paraphrased queries: 
> 1. Which queries failed? 
> 2. What was retrieved instead? 
> 3. Was the failure lexical, semantic, or domain-specific? 

- Failed query "What objective makes similar examples close in embedding space?".
- D2 was retrieved instead. 
- I feel the error was lexical. Since, D2 contained more similar words even though it wasn't the right answer. 

> Reading: Sentence-BERT, FAISS, BEIR. 

#### Data Processing for retrieval
- Retrieval is often bottlenecked by chunk quality, not the embedding model. 
- Retrieval systems retrieve units of texts, could call them chunks. 
- Each chunk should be large enough to contain useful meaning, small enough to be specific, self-contained enough to mean something. 
- Historical Context
	- Classic search engines worked with web pages. 
	- A page was the retrieval unit. 
	- Modern RAG systems often work with chunks because LLM context windows and embedding models have limits.
- Daft is a high performance data engine for AI and multimodal workloads. Raw text rows into clean training ready datasets.
- Data leakage: If chunks 1–8 are in train and chunks 9–10 are in test, your test set is contaminated. The model has seen almost the same document during training.

>Read Daft docs, README, chunking/ RAG reading, BEIR paper again. 

#### Cached MNRL
- For each query, every other positive document in the batch acts as negative. 
- If batch size is 128, each query gets 127 in-batch negatives. 
- large batch also means large GPU memory usage. 
- Gradient accumulation over multiple batches doesn't work since a particular query may not get exposure to other negatives then. 
- We want: large logical contrastive batch but small physical encoder batch.
- It trades time for memory. 
- A hard negative is a negative that looks relevant. 