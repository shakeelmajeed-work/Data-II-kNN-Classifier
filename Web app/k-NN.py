# ==========================================================
# CNN Embedding + PCA + kNN  (Single File Pipeline)
# ==========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import time

# ------------------------------
# 1. Load Data
# ------------------------------
train_df = pd.read_csv("product_images.csv")
test_df  = pd.read_csv("product_images_for_prediction.csv")

X_train = train_df.drop("label", axis=1).values.astype(np.float32)
y_train = train_df["label"].values
X_test  = test_df.values.astype(np.float32)

# Normalize to [0,1]
X_train /= 255.0
X_test  /= 255.0

# Reshape to (N,1,28,28)
X_train = X_train.reshape(-1, 1, 28, 28)
X_test  = X_test.reshape(-1, 1, 28, 28)

# Torch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_tensor = TensorDataset(
    torch.tensor(X_train),
    torch.tensor(y_train)
)

train_loader = DataLoader(train_tensor, batch_size=128, shuffle=True)

# ------------------------------
# 2. Small CNN Definition
# ------------------------------
class SmallCNN(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, embedding_dim)
        self.out = nn.Linear(embedding_dim, 10)

    def forward(self, x, return_embedding=False):
        x = self.pool(F.relu(self.conv1(x)))   # 28→14
        x = self.pool(F.relu(self.conv2(x)))   # 14→7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        embedding = F.relu(self.fc2(x))
        logits = self.out(embedding)

        if return_embedding:
            return embedding
        return logits


# ------------------------------
# 3. Train CNN
# ------------------------------
model = SmallCNN(embedding_dim=256).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("Training CNN...")
start = time.time()

for epoch in range(12):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/12  |  Loss: {total_loss:.4f}")

print(f"Training time: {time.time() - start:.1f}s")


# ------------------------------
# 4. Extract Embeddings
# ------------------------------
def extract_embeddings(model, X):
    model.eval()
    embeddings = []

    loader = DataLoader(
        torch.tensor(X),
        batch_size=256,
        shuffle=False
    )

    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            emb = model(xb, return_embedding=True)
            embeddings.append(emb.cpu())

    return torch.cat(embeddings).numpy()


print("Extracting embeddings...")
X_train_emb = extract_embeddings(model, X_train)
X_test_emb  = extract_embeddings(model, X_test)

print("Embedding shape:", X_train_emb.shape)


# ------------------------------
# 5. StandardScaler
# ------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_emb)
X_test_scaled  = scaler.transform(X_test_emb)


# ------------------------------
# 6. PCA (200 dims)
# ------------------------------
pca = PCA(n_components=200, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

print("After PCA:", X_train_pca.shape)


# ------------------------------
# 7. kNN (LOOCV - exact & efficient)
# ------------------------------
print("Running LOOCV for kNN...")

best_k = None
best_score = 0

for k in [5, 7, 9]:
    # Use k+1 because each point is its own nearest neighbor
    knn = KNeighborsClassifier(n_neighbors=k+1, metric="euclidean")
    knn.fit(X_train_pca, y_train)

    # Get neighbors for every training point
    distances, indices = knn.kneighbors(X_train_pca)

    # Remove self neighbor (first column)
    neighbor_labels = y_train[indices[:, 1:k+1]]

    # Majority vote
    preds = np.array([
        np.bincount(row).argmax()
        for row in neighbor_labels
    ])

    acc = (preds == y_train).mean()

    print(f"k={k}  LOOCV accuracy: {acc:.4f}")

    if acc > best_score:
        best_score = acc
        best_k = k

print(f"Best k from LOOCV: {best_k}")

# ------------------------------
# 8. Final Train + Predict
# ------------------------------
knn_final = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
knn_final.fit(X_train_pca, y_train)

test_predictions = knn_final.predict(X_test_pca)

# Save submission
submission = pd.DataFrame({
    "label": test_predictions
})
submission.to_csv("knn_predictions.csv", index=False)

print("Done. Predictions saved to knn_predictions.csv")