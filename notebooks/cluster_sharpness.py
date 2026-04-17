# %%
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np


# Calculate laplacian variance as a measure of sharpness
def sharpness_score(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return laplacian.var()


def sharpness_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Laplacian (jouw huidige)
    lap_var = cv2.Laplacian(img, cv2.CV_64F).var()

    # Sobel (edge strength)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    sobel = np.sqrt(sobelx**2 + sobely**2).mean()

    # Tenengrad (sterke focus metric)
    tenengrad = (sobelx**2 + sobely**2).mean()

    return lap_var, sobel, tenengrad


# %%
# Load the images and score
folder = "../images/sharpness"
scores = []

for file in os.listdir(folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(folder, file)
        # score = sharpness_score(path)
        lap, sobel, tenengrad = sharpness_features(path)
        # score = compute_score(lap, sobel, tenengrad)
        scores.append((file, lap, sobel, tenengrad))

# %%
# Normalize the scores with mean and standard deviation
# convert to array for easier manipulation
files = [s[0] for s in scores]
features = np.array([[s[1], s[2], s[3]] for s in scores])
mean = features.mean(axis=0)
std = features.std(axis=0)
# prevent zero division
std[std == 0] = 1
# normalize each feature column
features_norm = (features - mean) / std
features_norm

# %%
# Combine the normalized features into a single sharpness score (simple weighted sum)
weights = np.array([0.7, 0.1, 0.2]) # laplacian, sobel, tenengrad
scores_combined = features_norm @ weights
scores_combined

# %%
# Combine the file names with the scores
normalized_scores = []

for i, file in enumerate(files):
    lap, sobel, ten = features_norm[i]
    score = scores_combined[i]

    normalized_scores.append((file, score, lap, sobel, ten))


normalized_scores.sort(key=lambda x: x[1], reverse=True)
for file, score, lap, sobel, ten in normalized_scores:
    # print(f"{file}: {score:.2f}")
    print(f"{file}: {score:.2f} (Laplacian: {lap:.2f}, Sobel: {sobel:.2f}, Tenengrad: {ten:.2f})")



# %%
# Histogram of the combined scores
plt.hist([s for _, s, _, _, _ in normalized_scores], bins=15)
plt.show()

# %%
# Scores sorted by combined score
df_scores = pd.DataFrame(normalized_scores, columns=["File", "Combined Score", "Laplacian", "Sobel", "Tenengrad"])
df_scores

# %%
# Scatter plot of the normalized features
score = [s[1] for s in normalized_scores]
laplacian = [s[2] for s in normalized_scores]
sobel = [s[3] for s in normalized_scores]
tenengrad = [s[4] for s in normalized_scores]
plt.scatter(laplacian, score, label="Laplacian", alpha=0.7)
plt.scatter(sobel, score, label="Sobel", alpha=0.7)
plt.scatter(tenengrad, score, label="Tenengrad", alpha=0.7)
plt.xlabel("Feature Value")
plt.ylabel("Combined Sharpness Score")
plt.legend()
plt.show()

# %%
# Simple thresholding to classify images as sharp or blurry
threshold = 0  # tune yourself based on the distribution of scores

sharp = [f for f, s, _, _, _ in normalized_scores if s >= threshold]
blurry = [f for f, s, _, _, _ in normalized_scores if s < threshold]

# Add as a column to the DataFrame
df_scores["isSharp"] = df_scores["Combined Score"].apply(lambda x: True if x >= threshold else False)
df_scores


# %%
# Show the images with matplotlib
def show_images(image_files, title):
    if not image_files:
        print(f"No images to show for {title}.")
        return

    max_cols = 3
    cols = min(max_cols, len(image_files))
    rows = int(np.ceil(len(image_files) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, file in zip(axes, image_files):
        img = cv2.imread(os.path.join(folder, file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(file)
        ax.axis("off")

    # Hide any unused subplot axes in the last row.
    for ax in axes[len(image_files):]:
        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

show_images(sharp, "Sharp Images")
show_images(blurry, "Blurry Images")

# %%
# Cluster with K-means
# use the lap, sobel, ten to cluster instead of the combined score
values = np.array([[lap, sobel, ten] for file, score, lap, sobel, ten in normalized_scores])

kmeans = KMeans(n_clusters=3, random_state=0).fit(values)
labels = kmeans.labels_

cluster1 = []
cluster2 = []
cluster3 = []

for (file, score, _, _, _), label in zip(normalized_scores, labels):
    if label == 0:
        cluster1.append(file)
    elif label == 1:
        cluster2.append(file)
    else:
        cluster3.append(file)

means = [values[labels == i].mean() for i in range(3)]
sharp_cluster = int(np.argmax(means))

# %%
# Convert the clusters to "sharp", "average", "blurry" based on the mean combined score of each cluster
cluster_scores = [np.mean([score for (file, score, _, _, _), label in zip(normalized_scores, labels) if label == i]) for i in range(3)]
cluster_labels = ["sharp", "average", "blurry"]
cluster_labels = [cluster_labels[i] for i in np.argsort(cluster_scores)[::-1]]
cluster_mapping = {i: cluster_labels[i] for i in range(3)}
labels_named = [cluster_mapping[label] for label in labels]
labels_named


# %%
# Add the cluster labels to the DataFrame
df_scores["Cluster"] = labels_named
df_scores


# %%
# Plot the clusters
plt.scatter(values[:, 0], values[:, 1], c=labels, cmap="viridis")
plt.xlabel("Laplacian")
plt.ylabel("Sobel")
plt.title("K-means Clustering of Sharpness Features")
plt.colorbar(label="Cluster Label")
plt.show()

# %%
sharp_cluster

# %%



