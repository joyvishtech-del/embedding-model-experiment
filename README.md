# Embedding Model Benchmark

A research-style experiment comparing modern sentence embedding models
on semantic similarity, retrieval behavior, and embedding space
structure.

## Objective

Embeddings power many modern AI systems including:

-   Semantic Search
-   Retrieval Augmented Generation (RAG)
-   Document Clustering
-   Recommendation Systems
-   Vector Databases

This project evaluates how different embedding models represent semantic
meaning and retrieve relevant text.

## Models Compared

| Model | HuggingFace Model | Embedding Dimension |
|------|------------------|--------------------|
| MiniLM | all-MiniLM-L6-v2 | 384 |
| BGE | BAAI/bge-small-en-v1.5 | 384 |
| E5 | intfloat/e5-base-v2 | 768 |
| MXBAI | mixedbread-ai/mxbai-embed-large-v1 | 1024 |

## Dataset

30 sentences grouped into 3 semantic categories:

-   Healthcare (10)
-   Food (10)
-   Sports (10)

## Queries

-   AI in healthcare
-   Italian food dishes
-   popular sports competitions

## Evaluation Metrics

### Best Similarity Score

Maximum cosine similarity retrieved for each query.

### Category Average Similarity

Average similarity between the query and each semantic category.

### Category Separation Score

Difference between similarity to relevant vs non-relevant categories.

Higher separation indicates stronger semantic discrimination.

## Visualizations

-   Best similarity score comparison
-   Category separation score comparison
-   t-SNE visualization of embedding clusters

## Run the Experiment

Install dependencies:

pip install -r requirements.txt

Run the script:

python experiment.py

Results are saved in:

results/
