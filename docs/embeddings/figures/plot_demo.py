import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

with open("embed_demo_result.json", "r", encoding="utf-8") as f:
    result = json.load(f)

snippets = result["snippets"]
sim = np.array(result["similarity_matrix"])
vectors = np.load("embed_demo_vectors.npy")

genre_colors = {
    "Hadith": "#2a78d6",
    "Tafsir": "#1baf7a",
    "Fiqh": "#eb6834",
    "Aqidah": "#4a3aa7",
    "Philology": "#898781",
    "Quran": "#eda100",
}

labels = [f"{s['genre']}\n{s['book']}" for s in snippets]
short_labels = [s["id"] for s in snippets]
genres = [s["genre"] for s in snippets]
colors = [genre_colors[g] for g in genres]

# --- Figure 1: similarity heatmap ---
fig, ax = plt.subplots(figsize=(9, 7.5), dpi=150)
im = ax.imshow(sim, cmap="Blues", vmin=0.5, vmax=1.0)
ax.set_xticks(range(len(short_labels)))
ax.set_yticks(range(len(short_labels)))
ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(short_labels, fontsize=8)
for i in range(len(short_labels)):
    for j in range(len(short_labels)):
        ax.text(
            j,
            i,
            f"{sim[i, j]:.2f}",
            ha="center",
            va="center",
            color="white" if sim[i, j] > 0.82 else "black",
            fontsize=7.5,
        )
plt.colorbar(im, ax=ax, label="cosine similarity")
ax.set_title(
    "Real cosine similarity matrix\nintfloat/multilingual-e5-small on 8 real Shamela passages",
    fontsize=11,
)
plt.tight_layout()
plt.savefig("similarity_heatmap.png", facecolor="white")
plt.close()

# --- Figure 2: PCA 2D projection ---
pca = PCA(n_components=2)
coords = pca.fit_transform(vectors)
explained = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
seen_genres = []
for i, s in enumerate(snippets):
    g = s["genre"]
    lbl = g if g not in seen_genres else None
    if lbl:
        seen_genres.append(g)
    ax.scatter(
        coords[i, 0],
        coords[i, 1],
        s=180,
        color=genre_colors[g],
        edgecolor="white",
        linewidth=1.5,
        label=lbl,
        zorder=3,
    )
    ax.annotate(
        s["id"],
        (coords[i, 0], coords[i, 1]),
        textcoords="offset points",
        xytext=(8, 6),
        fontsize=8,
    )

ax.set_title(
    f"Real PCA projection of 8 real embeddings\n(explained variance: {explained[0] * 100:.1f}% + {explained[1] * 100:.1f}%)",
    fontsize=11,
)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("pca_projection.png", facecolor="white")
plt.close()

print("wrote similarity_heatmap.png and pca_projection.png")
print("PCA explained variance ratio:", explained)
