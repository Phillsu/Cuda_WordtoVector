import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import numpy as np

# 讀取並解析詞向量
word_vectors = {}
with open("../output/vector.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.split(" weights: ")
        word = parts[0].strip()
        vector = np.array(eval(parts[1].strip()))  # Convert string list to numpy array
        word_vectors[word] = vector

# 提取向量
vectors = np.array(list(word_vectors.values()))

# 使用 PCA 將 300 維向量降至 3 維
pca = PCA(n_components=3)
reduced_vectors = pca.fit_transform(vectors)

# 3D 繪圖
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], reduced_vectors[:, 2], alpha=0.6, s=10)

# 設定標籤
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
ax.set_title("Word Embeddings Visualization")

# 儲存圖檔
plt.savefig("hp100.png")  # 儲存為 PNG 檔案
