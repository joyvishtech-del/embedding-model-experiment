# Embedding Model Evaluation Report

## Introduction

Embedding models convert text into high‑dimensional vectors that capture
semantic meaning. These vectors enable modern AI systems such as search
engines, recommendation systems, and retrieval‑augmented generation
pipelines.

This report evaluates several widely used sentence embedding models to
understand how effectively they capture semantic relationships.

## Models Evaluated

* all-MiniLM-L6-v2
* BAAI/bge-small-en-v1.5
* intfloat/e5-base-v2
* mixedbread-ai/mxbai-embed-large-v1

## Experimental Setup

### Dataset

30 sentences grouped into three semantic categories:

* Healthcare
* Food
* Sports

Each category contains 10 sentences.

### Queries

Three queries were designed to represent retrieval tasks:

* AI in healthcare
* Italian food dishes
* popular sports competitions

### Methodology

1. Encode all sentences using each embedding model.
2. Encode queries into embeddings.
3. Compute cosine similarity between queries and sentences.
4. Identify the highest scoring sentence.
5. Compute category‑level similarity averages.
6. Measure separation between relevant and non‑relevant categories.

## Metrics

### Max Similarity

Measures the strongest semantic match retrieved by the model.

### Category Average

Evaluates how strongly a query aligns with sentences in each category.

### Separation Score

Separation = Relevant Avg Similarity − Non‑Relevant Avg Similarity

Higher values indicate better semantic discrimination.

## Visualization

t‑SNE was used to project embeddings into two dimensions to visualize
clustering behavior of sentence groups.



## Observations

* All evaluated models demonstrate the ability to capture semantic
  similarity.
* Larger embedding models tend to produce stronger category
  separation.

Retrieval‑optimized models perform better on query matching tasks.



## Final Model Ranking (Based on this experiment)



| Rank | Model      | Summary                               |

| ---- | ---------- | ------------------------------------- |

| 🥇 1 | \*\*MiniLM\*\* | Best semantic separation              |

| 🥈 2 | \*\*MXBAI\*\*  | Strong overall performance            |

| 🥉 3 | \*\*BGE\*\*    | Moderate performance                  |

| 4    | \*\*E5\*\*     | High similarity but weaker separation |



## Conclusion

Embedding quality varies depending on model architecture and training
objective. Strong category separation suggests a model is better suited
for retrieval tasks such as semantic search and RAG pipelines.

In this experiment, MiniLM produced the strongest semantic separation 
between relevant and unrelated content, making it the most effective model 
for retrieval-oriented tasks among the evaluated embedding models.

