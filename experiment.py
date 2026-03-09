from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer

# -----------------------------
# Dataset
# -----------------------------
healthcare_sentences = [
    "AI helps doctors diagnose disease",
    "Machine learning improves medical imaging",
    "Deep learning detects cancer in CT scans",
    "Radiology uses MRI and CT scans",
    "Hospitals treat patients with advanced technology",
    "AI assists doctors in analyzing X-rays",
    "Medical imaging helps detect tumors early",
    "Healthcare systems use data to improve patient care",
    "AI tools assist radiologists in diagnosis",
    "Modern hospitals rely on digital medical records",
]

food_sentences = [
    "Pizza is a popular Italian food",
    "Pasta and pizza are famous dishes",
    "Italian cuisine includes pasta and cheese",
    "Restaurants serve a variety of Italian meals",
    "Cheese is commonly used in many Italian dishes",
    "Burgers and fries are popular fast foods",
    "Many people enjoy eating pasta with tomato sauce",
    "Street food is popular in many countries",
    "Desserts like cake and ice cream are sweet treats",
    "Food festivals celebrate different cuisines",
]

sports_sentences = [
    "Football is the most popular sport",
    "Soccer players score goals in matches",
    "Basketball players shoot the ball into the hoop",
    "Tennis players compete in international tournaments",
    "Cricket is widely played in many countries",
    "Olympic athletes train for years",
    "Sports teams compete for championships",
    "Fans cheer loudly during football matches",
    "Professional athletes maintain strict training routines",
    "Stadiums host thousands of sports fans",
]

sentences = healthcare_sentences + food_sentences + sports_sentences

queries = {
    "AI in healthcare": "healthcare",
    "Italian food dishes": "food",
    "popular sports competitions": "sports",
}

models = {
    "MiniLM": "all-MiniLM-L6-v2",
    "BGE": "BAAI/bge-small-en-v1.5",
    "E5": "intfloat/e5-base-v2",
    "MXBAI": "mixedbread-ai/mxbai-embed-large-v1",
}

category_slices = {
    "healthcare": slice(0, 10),
    "food": slice(10, 20),
    "sports": slice(20, 30),
}

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)


def encode_for_model(model_name: str, model: SentenceTransformer, texts: list[str], is_query: bool) -> np.ndarray:
    """Handle model-specific formatting where useful."""
    if model_name == "E5":
        if is_query:
            texts = [f"query: {t}" for t in texts]
        else:
            texts = [f"passage: {t}" for t in texts]
    return model.encode(texts, normalize_embeddings=True)


max_similarity_records: list[dict] = []
category_average_records: list[dict] = []
separation_records: list[dict] = []

for short_name, model_path in models.items():
    print(f"\nLoading model: {short_name} -> {model_path}")
    start = time.perf_counter()
    model = SentenceTransformer(model_path)
    load_time = time.perf_counter() - start

    sentence_embeddings = encode_for_model(short_name, model, sentences, is_query=False)

    for query, relevant_category in queries.items():
        query_embedding = encode_for_model(short_name, model, [query], is_query=True)
        sim = cosine_similarity(query_embedding, sentence_embeddings)[0]

        best_idx = int(np.argmax(sim))
        best_sentence = sentences[best_idx]
        best_score = float(sim[best_idx])

        rel_slice = category_slices[relevant_category]
        relevant_avg = float(np.mean(sim[rel_slice]))

        non_relevant_scores = []
        for category_name, cat_slice in category_slices.items():
            avg_score = float(np.mean(sim[cat_slice]))
            category_average_records.append(
                {
                    "model": short_name,
                    "query": query,
                    "category": category_name,
                    "average_similarity": round(avg_score, 6),
                }
            )
            if category_name != relevant_category:
                non_relevant_scores.extend(sim[cat_slice])

        non_relevant_avg = float(np.mean(non_relevant_scores))
        separation = relevant_avg - non_relevant_avg

        max_similarity_records.append(
            {
                "model": short_name,
                "query": query,
                "best_match": best_sentence,
                "best_similarity": round(best_score, 6),
                "model_path": model_path,
                "model_load_seconds": round(load_time, 3),
            }
        )

        separation_records.append(
            {
                "model": short_name,
                "query": query,
                "relevant_category": relevant_category,
                "relevant_avg_similarity": round(relevant_avg, 6),
                "non_relevant_avg_similarity": round(non_relevant_avg, 6),
                "separation_score": round(separation, 6),
            }
        )

# Save CSVs
max_df = pd.DataFrame(max_similarity_records)
cat_df = pd.DataFrame(category_average_records)
sep_df = pd.DataFrame(separation_records)

max_df.to_csv(results_dir / "max_similarity_scores.csv", index=False)
cat_df.to_csv(results_dir / "category_average_scores.csv", index=False)
sep_df.to_csv(results_dir / "separation_scores.csv", index=False)

# -----------------------------
# Plot 1: max similarity scores
# -----------------------------
max_plot_df = max_df.pivot(index="query", columns="model", values="best_similarity")
plt.figure(figsize=(10, 6))
max_plot_df.plot(kind="bar")
plt.title("Best Similarity Score by Query and Model")
plt.ylabel("Cosine Similarity")
plt.xlabel("Query")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(results_dir / "max_similarity_scores.png", dpi=200)
plt.close()

# -----------------------------
# Plot 2: separation scores
# -----------------------------
sep_plot_df = sep_df.pivot(index="query", columns="model", values="separation_score")
plt.figure(figsize=(10, 6))
sep_plot_df.plot(kind="bar")
plt.title("Category Separation Score by Query and Model")
plt.ylabel("Relevant Avg - Non-Relevant Avg")
plt.xlabel("Query")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(results_dir / "separation_scores.png", dpi=200)
plt.close()

# -----------------------------
# Plot 3: t-SNE visualization using MiniLM
# -----------------------------
viz_model = SentenceTransformer(models["MiniLM"])
viz_embeddings = encode_for_model("MiniLM", viz_model, sentences, is_query=False)

# Keep perplexity below sample size and suitable for 30 points
reduced = TSNE(n_components=2, perplexity=5, random_state=42).fit_transform(viz_embeddings)

colors = (["tab:blue"] * 10) + (["tab:orange"] * 10) + (["tab:green"] * 10)
labels = (["Healthcare"] * 10) + (["Food"] * 10) + (["Sports"] * 10)

plt.figure(figsize=(12, 9))
seen_labels = set()
for i, sentence in enumerate(sentences):
    label = labels[i]
    scatter_label = label if label not in seen_labels else None
    plt.scatter(reduced[i, 0], reduced[i, 1], label=scatter_label)
    plt.text(reduced[i, 0] + 0.15, reduced[i, 1] + 0.15, sentence[:28], fontsize=8)
    seen_labels.add(label)

plt.title("t-SNE Visualization of Sentence Embeddings (MiniLM)")
plt.legend()
plt.tight_layout()
plt.savefig(results_dir / "tsne_minilm.png", dpi=200)
plt.close()

print("\nDone. Files saved in ./results")
print(max_df)
print(sep_df)
