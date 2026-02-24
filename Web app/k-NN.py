import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors as NNNeighbors
from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.metrics import (
    accuracy_score, classification_report,
    balanced_accuracy_score, cohen_kappa_score,
    matthews_corrcoef, confusion_matrix,
    precision_recall_fscore_support
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================
print("="*60)
print("STEP 1: Loading Data")
print("="*60)

train_df = pd.read_csv('product_images.csv')
X_train_orig = train_df.drop('label', axis=1).values.astype(np.float32)
y_train = train_df['label'].values

print(f"Training samples: {X_train_orig.shape[0]}")
print(f"Original features (pixels): {X_train_orig.shape[1]}")

# Pixel layout: Fashion-MNIST is 28x28 row-major (C order)
# pixel_0 = (row 0, col 0) = top-left
# pixel_27 = (row 0, col 27) = top-right
# pixel_28 = (row 1, col 0)
IMG_H, IMG_W = 28, 28

# ============================================================
# 2. FEATURE ENGINEERING: VARIANCE THRESHOLDING
# ============================================================
print("\n" + "="*60)
print("STEP 2: Variance-Based Feature Selection")
print("="*60)

pixel_variances = np.var(X_train_orig, axis=0)

print(f"Variance statistics:")
print(f"  Min variance:  {pixel_variances.min():.4f}")
print(f"  Max variance:  {pixel_variances.max():.4f}")
print(f"  Mean variance: {pixel_variances.mean():.4f}")
print(f"  Median variance: {np.median(pixel_variances):.4f}")

for thresh in [0, 0.001, 0.01, 0.1, 0.5, 1.0]:
    count = np.sum(pixel_variances > thresh)
    print(f"  Pixels with variance > {thresh}: {count}")

VARIANCE_THRESHOLD = 0.01
high_var_mask = pixel_variances > VARIANCE_THRESHOLD
n_high_var = np.sum(high_var_mask)

print(f"\n→ Variance threshold: {VARIANCE_THRESHOLD}")
print(f"→ Pixels kept (high variance): {n_high_var} / {X_train_orig.shape[1]}")
print(f"→ Pixels removed (low variance): {X_train_orig.shape[1] - n_high_var}")

X_train_var = X_train_orig[:, high_var_mask]

# ============================================================
# 3. FEATURE ENGINEERING: CORRELATION-BASED REMOVAL
# ============================================================
print("\n" + "="*60)
print("STEP 3: Correlation-Based Feature Selection")
print("="*60)

print("Computing correlation matrix...")
corr_matrix = np.corrcoef(X_train_var.T)
corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

CORRELATION_THRESHOLD = 0.95
n_features = corr_matrix.shape[0]
cols_to_remove = set()

for i in range(n_features):
    if i in cols_to_remove:
        continue
    for j in range(i + 1, n_features):
        if j in cols_to_remove:
            continue
        if abs(corr_matrix[i, j]) > CORRELATION_THRESHOLD:
            var_i = pixel_variances[high_var_mask][i]
            var_j = pixel_variances[high_var_mask][j]
            if var_i >= var_j:
                cols_to_remove.add(j)
            else:
                cols_to_remove.add(i)

cols_to_keep = [i for i in range(n_features) if i not in cols_to_remove]

print(f"→ Correlation threshold: {CORRELATION_THRESHOLD}")
print(f"→ Highly correlated pairs found: {len(cols_to_remove)}")
print(f"→ Features after correlation filter: {len(cols_to_keep)}")

X_train_filtered = X_train_var[:, cols_to_keep]

print(f"\n*** FEATURE REDUCTION SUMMARY ***")
print(f"  Original pixels:           {X_train_orig.shape[1]}")
print(f"  After variance filter:     {X_train_var.shape[1]}")
print(f"  After correlation filter:  {X_train_filtered.shape[1]}")

# ============================================================
# 4. STANDARDIZE FILTERED FEATURES
# ============================================================
print("\n" + "="*60)
print("STEP 4: Standardizing Filtered Features")
print("="*60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_filtered).astype(np.float32)
print(f"Scaled pixel features: {X_train_scaled.shape}")

# ============================================================
# 5. SPATIAL FEATURE ENGINEERING
# ============================================================
#
# The confusion matrix reveals the core bottleneck: classes 0
# (T-shirt), 4 (Coat), and 6 (Shirt) account for ~75% of all
# errors. Raw pixels/PCA cannot distinguish these because the
# discriminative cues are STRUCTURAL:
#
#   • Shirt (6)    has a collar, button placket (vertical center line)
#   • T-shirt (0)  has a simple round/V neckline
#   • Coat (4)     has lapels, longer silhouette, wider at shoulders
#
# We address this with three complementary feature types:
#
# [5a] HOG (Histogram of Oriented Gradients)
#      Captures EDGE DIRECTIONS in local cells. Collar shape,
#      lapel angle, and button lines all produce distinctive
#      orientation histograms invisible in raw pixel values.
#
# [5b] Spatial profiles (row / column means + stds)
#      The mean brightness at each ROW captures garment WIDTH at
#      each height — a coat is wider at the shoulder than waist;
#      a t-shirt and shirt differ in shoulder-to-waist taper.
#      Column profiles capture vertical symmetry patterns.
#
# [5c] CNN embeddings (ResNet18, if PyTorch is available)
#      Deep conv features explicitly encode semantic concepts
#      (collar, texture, buttons) that k-NN can separate directly.
#      Only used for feature extraction; k-NN makes all predictions.
#
# ============================================================
print("\n" + "="*60)
print("STEP 5: Spatial Feature Engineering")
print("="*60)
print(f"Pixel layout: {IMG_H}x{IMG_W} row-major  "
      f"(pixel 0 = top-left, pixel {IMG_W-1} = top-right)")

imgs_train = X_train_orig.reshape(-1, IMG_H, IMG_W).astype(np.float32)

# ----------------------------------------------------------
# 5a. HOG features
# ----------------------------------------------------------
# Each 28×28 image → 4×4 grid of 7×7 cells × 9 orientation
# bins = 144 features per image.
# np.gradient is fully vectorised across all N images at once.
# ----------------------------------------------------------
def compute_hog_batch(X_pixels, H=28, W=28, cell=7, n_bins=9):
    n     = X_pixels.shape[0]
    imgs  = X_pixels.reshape(n, H, W).astype(np.float32)
    gy    = np.gradient(imgs, axis=1)          # vertical gradient
    gx    = np.gradient(imgs, axis=2)          # horizontal gradient
    mag   = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)                 # (-π, π)
    n_cr, n_cc = H // cell, W // cell
    hog   = np.zeros((n, n_cr * n_cc * n_bins), dtype=np.float32)
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    fi    = 0
    for r in range(n_cr):
        for c in range(n_cc):
            cm = mag[:,   r*cell:(r+1)*cell, c*cell:(c+1)*cell].reshape(n, -1)
            ca = angle[:, r*cell:(r+1)*cell, c*cell:(c+1)*cell].reshape(n, -1)
            for b in range(n_bins):
                hog[:, fi+b] = (cm * ((ca >= edges[b]) & (ca < edges[b+1]))).sum(axis=1)
            norm = np.linalg.norm(hog[:, fi:fi+n_bins], axis=1, keepdims=True) + 1e-6
            hog[:, fi:fi+n_bins] /= norm
            fi += n_bins
    return hog

print("\n[5a] Computing HOG features...")
t0 = time.time()
hog_train = compute_hog_batch(X_train_orig)
print(f"  HOG shape: {hog_train.shape}  ({time.time()-t0:.1f}s)")

# ----------------------------------------------------------
# 5b. Spatial row / column profiles
# ----------------------------------------------------------
print("[5b] Computing spatial profiles...")
row_means = imgs_train.mean(axis=2)   # (n, 28) — brightness at each row height
col_means = imgs_train.mean(axis=1)   # (n, 28) — brightness at each column
row_stds  = imgs_train.std(axis=2)    # (n, 28)
col_stds  = imgs_train.std(axis=1)    # (n, 28)
spatial_profiles = np.hstack([row_means, col_means, row_stds, col_stds])
print(f"  Profiles shape: {spatial_profiles.shape}")

# ----------------------------------------------------------
# 5c. Garment-specific measurements  (targeted at 0 / 4 / 6)
# ----------------------------------------------------------
# The F-score map from [5d] shows exactly which regions drive
# the 0/4/6 decision: collar area (rows 1-6), centre button
# strip (cols 12-15), shoulder/side edges, and hem length.
# We encode these as explicit scalars so the model doesn't
# have to discover them from HOG cells and pixel averages.
#
# All 17 features are computed on the raw 0-255 pixel values
# (before any filtering/scaling) for maximum signal fidelity.
# ----------------------------------------------------------
def compute_garment_features(X_pixels, H=28, W=28):
    """
    17 targeted garment measurements for T-shirt(0)/Coat(4)/Shirt(6):
      body_height     — Coat > Shirt ≈ T-shirt
      collar_bright   — Shirt collar has high top-centre brightness
      collar_std      — Shirt collar has uneven/bumpy texture
      placket_std     — Shirt button line: high std in centre strip
      shoulder_width  — pixels wide at rows 6-9 (normalised)
      waist_width     — pixels wide at rows 16-20
      taper_ratio     — shoulder/waist: Coat tapers, T-shirt doesn't
      top3_mean       — avg brightness of top third
      bot3_mean       — avg brightness of bottom third (Coat is longer)
      top_bot_ratio   — top/bottom: short items score high
      left_edge       — left body/lapel brightness
      right_edge      — right body/lapel brightness
      edge_asymmetry  — |left-right|: lapels cause asymmetry briefly
      neckline_bright — brightness at neckline opening (T-shirt dark)
      neckline_std    — variance at neckline: collar bumps → high std
      hem_mean        — brightness of bottom 6 rows
      mid_mean        — avg brightness of rows 9-18 (waist zone)
    """
    n   = X_pixels.shape[0]
    img = X_pixels.reshape(n, H, W).astype(np.float32)
    THRESH = 10.0

    # Body height: last active row − first active row
    row_brightness = img.mean(axis=2)                        # (n, H)
    active_r = row_brightness > THRESH
    has_act  = active_r.any(axis=1)
    first_r  = np.where(has_act, active_r.argmax(axis=1), 0)
    last_r   = np.where(has_act,
                        H - 1 - np.flip(active_r, axis=1).argmax(axis=1), H-1)
    body_height = (last_r - first_r).astype(np.float32)

    collar_bright  = img[:, 1:7,  9:19].mean(axis=(1, 2))
    collar_std     = img[:, 1:7,  9:19].std(axis=(1, 2))
    placket_std    = img[:, 4:24, 12:16].std(axis=(1, 2))   # button line
    shoulder_width = (img[:, 6:10, :] > THRESH).sum(axis=(1, 2)) / 4.0
    waist_width    = (img[:, 16:21, :] > THRESH).sum(axis=(1, 2)) / 5.0
    taper_ratio    = shoulder_width / (waist_width + 1e-4)
    top3_mean      = img[:, :9, :].mean(axis=(1, 2))
    bot3_mean      = img[:, 19:, :].mean(axis=(1, 2))
    top_bot_ratio  = top3_mean / (bot3_mean + 1e-4)
    left_edge      = img[:, 4:22, :4].mean(axis=(1, 2))
    right_edge     = img[:, 4:22, 24:].mean(axis=(1, 2))
    edge_asymm     = np.abs(left_edge - right_edge)
    neckline_bright= img[:, 2:7,  8:20].mean(axis=(1, 2))
    neckline_std   = img[:, 2:7,  8:20].std(axis=(1, 2))
    hem_mean       = img[:, 22:,  :].mean(axis=(1, 2))
    mid_mean       = img[:, 9:19, :].mean(axis=(1, 2))

    return np.column_stack([
        body_height, collar_bright, collar_std, placket_std,
        shoulder_width, waist_width, taper_ratio,
        top3_mean, bot3_mean, top_bot_ratio,
        left_edge, right_edge, edge_asymm,
        neckline_bright, neckline_std, hem_mean, mid_mean
    ]).astype(np.float32)

print("[5c] Computing garment-specific measurements...")
garment_train = compute_garment_features(X_train_orig)
print(f"  Garment features: {garment_train.shape}")

# ----------------------------------------------------------
# 5d. CNN features (PyTorch / ResNet18, optional)
# ----------------------------------------------------------
cnn_ok = False
try:
    import torch
    import torch.nn.functional as F_torch
    import torchvision.models as models

    print("[5d] Extracting CNN features (ResNet18 pre-trained on ImageNet)...")
    cnn_model = models.resnet18(weights='DEFAULT')
    cnn_model = torch.nn.Sequential(*list(cnn_model.children())[:-1])
    cnn_model.eval()

    def extract_cnn_features(X_pixels, model, H=28, W=28, batch_size=512):
        n      = X_pixels.shape[0]
        imgs_t = torch.FloatTensor(X_pixels.reshape(n, 1, H, W))
        imgs_t = imgs_t.repeat(1, 3, 1, 1)
        imgs_t = F_torch.interpolate(imgs_t, size=(224, 224), mode='bilinear',
                                     align_corners=False)
        mu  = torch.tensor([0.485*255, 0.456*255, 0.406*255]).view(1, 3, 1, 1)
        sig = torch.tensor([0.229*255, 0.224*255, 0.225*255]).view(1, 3, 1, 1)
        imgs_t = (imgs_t - mu) / sig
        out = []
        with torch.no_grad():
            for i in range(0, n, batch_size):
                out.append(model(imgs_t[i:i+batch_size]).squeeze(-1).squeeze(-1).numpy())
        return np.concatenate(out, axis=0)

    t0 = time.time()
    X_cnn_train = extract_cnn_features(X_train_orig, cnn_model)
    cnn_ok = True
    print(f"  CNN shape: {X_cnn_train.shape}  ({time.time()-t0:.1f}s)")

except ImportError:
    print("[5d] PyTorch not available — skipping CNN features")

# ----------------------------------------------------------
# 5d. Discriminative pixel analysis for classes 0, 4, 6
# ----------------------------------------------------------
# ANOVA F-test: which of the 784 pixels most strongly
# differentiate T-shirt(0) / Coat(4) / Shirt(6)?
# High F-score pixels → high between-class variance relative
# to within-class variance.
# ----------------------------------------------------------
print("\n[5d] Discriminative pixel analysis (classes 0 / 4 / 6)...")
mask_046  = np.isin(y_train, [0, 4, 6])
f_scores, _ = f_classif(X_train_orig[mask_046], y_train[mask_046])
f_img = f_scores.reshape(IMG_H, IMG_W)
p50, p75, p90 = (np.percentile(f_scores, p) for p in (50, 75, 90))
print(f"  F-score range: {f_scores.min():.0f} – {f_scores.max():.0f}")
print(f"  Top-discriminative pixel map  (█=top10%  ▓=top25%  ░=top50%):")
for row in range(IMG_H):
    line = "  "
    for col in range(IMG_W):
        v = f_img[row, col]
        line += ("█" if v > p90 else "▓" if v > p75 else "░" if v > p50 else " ")
    print(line)

# ----------------------------------------------------------
# 5e. Feature fusion
# ----------------------------------------------------------
# Scale each new feature group separately (preserves relative
# magnitude within each group), then concatenate with the
# already-scaled pixel features.
# Final fused matrix: pixels + HOG + profiles [+ CNN]
# ----------------------------------------------------------
print("\n[5e] Fusing feature groups...")
scaler_hog_prof = StandardScaler()
X_hp_scaled = scaler_hog_prof.fit_transform(
    np.hstack([hog_train, spatial_profiles, garment_train])
).astype(np.float32)

if cnn_ok:
    scaler_cnn = StandardScaler()
    X_cnn_scaled = scaler_cnn.fit_transform(X_cnn_train).astype(np.float32)
    X_fused = np.hstack([X_train_scaled, X_hp_scaled, X_cnn_scaled])
else:
    X_fused = np.hstack([X_train_scaled, X_hp_scaled])

n_pix  = X_train_scaled.shape[1]
n_hp   = X_hp_scaled.shape[1]
n_cnn  = X_cnn_scaled.shape[1] if cnn_ok else 0
print(f"  pixels={n_pix}  HOG+profiles+garment={n_hp}" +
      (f"  CNN={n_cnn}" if cnn_ok else "") +
      f"  → fused={X_fused.shape[1]} features")

# ============================================================
# 6. MLP NEURAL NETWORK EMBEDDINGS
# ============================================================
#
# A Multi-Layer Perceptron is trained on the fused feature matrix
# and used purely as a feature extractor — k-NN makes all final
# predictions.  This mirrors the CNN-embedding workflow: the
# neural network learns a compact, discriminative representation;
# the non-parametric k-NN classifier then exploits it.
#
# Note: PyTorch / TensorFlow are not installed in this environment.
# sklearn's MLPClassifier IS a proper multi-layer neural network
# (dense layers, ReLU activations, Adam optimiser, backpropagation).
# It lacks the local spatial connectivity of a CNN, but trained on
# the HOG-augmented fused features it still learns rich nonlinear
# representations of garment structure.
#
# Two embedding types are added to the search:
#   MLP_emb(128)  — 128-dim last-hidden-layer activations
#                   (learned intermediate representation)
#   MLP_proba(10) — 10-dim softmax class probabilities
#                   (how T-shirt-like vs Shirt-like is each sample?)
#                   Especially useful for the 0/4/6 confusion trio.
#
# ============================================================
from sklearn.neural_network import MLPClassifier

print("\n" + "="*60)
print("STEP 6: MLP Neural Network Embeddings")
print("="*60)
print("Architecture: fused_input → 512 → 256 → 128 → 10 (softmax)")
print("Embeddings  : 128-dim last hidden layer + 10-dim probabilities")

t0 = time.time()
mlp_extractor = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42
)
mlp_extractor.fit(X_fused, y_train)
print(f"  Trained in {time.time()-t0:.1f}s  "
      f"| best val accuracy: {mlp_extractor.best_validation_score_:.4f}")


def extract_mlp_embeddings(mlp, X):
    """
    Forward-pass X through all hidden layers and return the
    last-hidden-layer activations (before the output weights).
    Works for relu / tanh / logistic activations.
    """
    act = np.array(X, dtype=np.float64)
    for coef, intercept in zip(mlp.coefs_[:-1], mlp.intercepts_[:-1]):
        act = act @ coef + intercept
        if mlp.activation == 'relu':
            act = np.maximum(0, act)
        elif mlp.activation == 'tanh':
            act = np.tanh(act)
        elif mlp.activation == 'logistic':
            act = 1.0 / (1.0 + np.exp(-act))
    return act.astype(np.float32)


X_mlp_emb   = extract_mlp_embeddings(mlp_extractor, X_fused)        # (n, 128)
X_mlp_proba = mlp_extractor.predict_proba(X_fused).astype(np.float32) # (n, 10)

scaler_mlp = StandardScaler()
X_mlp_emb_scaled = scaler_mlp.fit_transform(X_mlp_emb)

print(f"  MLP embedding : {X_mlp_emb.shape}")
print(f"  MLP proba     : {X_mlp_proba.shape}")

# ============================================================
# 7. DIMENSIONALITY REDUCTION  (PCA + Manifold on fused features)
# ============================================================
#
# All manifold methods receive a PCA pre-reduction of the fused
# feature matrix (sklearn tip: reduce noisy high-D input first).
# Pre-PCA raised to 100 components (was 50) to retain more
# discriminative information from the richer fused features.
#
# NEW method: Spectral Embedding
#   Constructs a sparse k-NN affinity graph, then finds the
#   eigenvectors of the normalised graph Laplacian.  This
#   captures the GLOBAL CONNECTIVITY structure of the data
#   manifold — classes that share a connected region of the
#   graph cluster together.  More robust to noise than Isomap
#   for large datasets because it only uses the k-NN graph
#   (no pairwise distance matrix needed).
#   SpectralEmbedding has no native .transform() for new points;
#   we use an out-of-sample extension (inverse-distance-weighted
#   average of k nearest training embeddings).
#
# ============================================================
print("\n" + "="*60)
print("STEP 7: Dimensionality Reduction (on fused features)")
print("="*60)

# ----------------------------------------------------------
# 7a. PCA pre-reduction for manifold input (100 dims)
# ----------------------------------------------------------
PRE_PCA_COMPONENTS = 100
print(f"\n[7a] PCA pre-reduction → {PRE_PCA_COMPONENTS} components...")
pca_pre = PCA(n_components=PRE_PCA_COMPONENTS, random_state=42)
X_pca_pre = pca_pre.fit_transform(X_fused)
pre_var = pca_pre.explained_variance_ratio_.sum() * 100
print(f"  Variance retained: {pre_var:.1f}%")

# ----------------------------------------------------------
# 6b. Isomap
# ----------------------------------------------------------
print(f"\n[7b] Isomap(n_neighbors=10, n_components=50)...")
isomap_ok = False
t0 = time.time()
try:
    isomap = Isomap(n_neighbors=10, n_components=50, n_jobs=-1)
    X_isomap = isomap.fit_transform(X_pca_pre)
    isomap_ok = True
    print(f"  Done in {time.time()-t0:.1f}s")
except (MemoryError, Exception) as e:
    print(f"  Isomap skipped ({type(e).__name__}: {e})")

# ----------------------------------------------------------
# 7c. Modified LLE (MLLE)
# n_neighbors=60 satisfies MLLE constraint: n_neighbors >= n_components
# ----------------------------------------------------------
print(f"\n[7c] MLLE(n_neighbors=60, n_components=50)...")
t0 = time.time()
mlle = LocallyLinearEmbedding(
    n_neighbors=60, n_components=50,
    method='modified', n_jobs=-1, random_state=42
)
X_mlle = mlle.fit_transform(X_pca_pre)
print(f"  Done in {time.time()-t0:.1f}s")

# ----------------------------------------------------------
# 7d. Spectral Embedding  (NEW manifold method)
# affinity='nearest_neighbors' builds a SPARSE k-NN graph —
# no O(n²) distance matrix needed, making it feasible for 20K.
# Out-of-sample extension: for test points, take the
# inverse-distance-weighted mean of k nearest training embeddings.
# ----------------------------------------------------------
print(f"\n[7d] Spectral Embedding(n_components=50, affinity=nearest_neighbors)...")
t0 = time.time()
spec_emb = SpectralEmbedding(
    n_components=50,
    affinity='nearest_neighbors',
    n_neighbors=10,
    n_jobs=-1,
    random_state=42
)
X_spec = spec_emb.fit_transform(X_pca_pre)
print(f"  Done in {time.time()-t0:.1f}s")

# Fit kNN model for out-of-sample extension at test time
spec_oos_nn = NNNeighbors(n_neighbors=5, algorithm='ball_tree', n_jobs=-1)
spec_oos_nn.fit(X_pca_pre)

def spectral_transform_oos(X_new_pca, train_embedding, nn_model):
    """Inverse-distance-weighted interpolation in the Spectral Embedding space."""
    distances, indices = nn_model.kneighbors(X_new_pca)
    w = 1.0 / (distances + 1e-8)
    w /= w.sum(axis=1, keepdims=True)
    return (w[:, :, np.newaxis] * train_embedding[indices]).sum(axis=1)

# ----------------------------------------------------------
# 7e. Full PCA search space (on fused features)
# ----------------------------------------------------------
print(f"\n[7e] Full PCA search space (on fused features)...")
max_possible = min(X_fused.shape[1], X_fused.shape[0] - 1)
pca_list = [n for n in [50, 75, 100, 150, 200, 250, 300] if n <= max_possible]
max_components = max(pca_list)
pca_full = PCA(n_components=max_components, random_state=42)
X_pca_full = pca_full.fit_transform(X_fused)
cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
for n in pca_list:
    print(f"  PCA({n}): {cumsum_var[n-1]*100:.1f}% variance")

# ----------------------------------------------------------
# 7f. Build unified feature dictionary
# ----------------------------------------------------------
feature_dict = {}

for n in pca_list:
    feature_dict[f'PCA({n})'] = X_pca_full[:, :n]

if isomap_ok:
    feature_dict['Isomap(30)'] = X_isomap[:, :30]
    feature_dict['Isomap(50)'] = X_isomap

feature_dict['MLLE(30)']       = X_mlle[:, :30]
feature_dict['MLLE(50)']       = X_mlle
feature_dict['SpecEmb(30)']    = X_spec[:, :30]
feature_dict['SpecEmb(50)']    = X_spec
# MLP neural network embeddings
feature_dict['MLP_emb(128)']   = X_mlp_emb_scaled          # 128-dim learned hidden layer
feature_dict['MLP_proba(10)']  = X_mlp_proba                # 10-dim soft class scores

print(f"\n*** FEATURE REPRESENTATIONS TO SEARCH ***")
for name, X in feature_dict.items():
    print(f"  {name:20s}: {X.shape}")

# ============================================================
# 7. HYPERPARAMETER SEARCH (5-FOLD STRATIFIED CV)
# ============================================================
print("\n" + "="*60)
print("STEP 8: Hyperparameter Search (5-Fold CV)")
print("="*60)

n_neighbors_list = [1, 3, 5, 7, 9, 11]
weights_list     = ['uniform', 'distance']
metric_list      = ['euclidean', 'manhattan', 'cosine']

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
total_configs = len(n_neighbors_list) * len(weights_list) * len(metric_list) * len(feature_dict)
print(f"Testing {total_configs} configurations...")

config_count = 0
for k in n_neighbors_list:
    for weights in weights_list:
        for metric in metric_list:
            for feat_name, X_data in feature_dict.items():
                config_count += 1
                algo = 'ball_tree' if metric in ('euclidean', 'manhattan') else 'brute'
                model = KNeighborsClassifier(
                    n_neighbors=k, weights=weights,
                    metric=metric, algorithm=algo, n_jobs=-1
                )
                scores = cross_val_score(model, X_data, y_train, cv=cv, scoring='accuracy')
                results.append({
                    'k': k, 'weights': weights, 'metric': metric,
                    'feat_name': feat_name,
                    'cv_accuracy': scores.mean(), 'cv_std': scores.std()
                })
                if config_count % 50 == 0:
                    print(f"  Progress: {config_count}/{total_configs}")

results = sorted(results, key=lambda x: x['cv_accuracy'], reverse=True)

print("\n" + "-"*60)
print("Top 15 Configurations (5-Fold CV):")
print("-"*60)
for i, r in enumerate(results[:15], 1):
    print(f"{i:2d}. k={r['k']:2d}, {r['weights']:8s}, {r['metric']:10s}, "
          f"{r['feat_name']:20s} → {r['cv_accuracy']:.4f} ± {r['cv_std']:.4f}")

# ============================================================
# 8. LOOCV ON TOP 5 CONFIGURATIONS
# ============================================================
print("\n" + "="*60)
print("STEP 9: LOOCV Validation (Top 5)")
print("="*60)

loo = LeaveOneOut()
best_params   = None
best_accuracy = 0
best_preds    = None

for i, r in enumerate(results[:5], 1):
    print(f"\nConfig {i}/5: k={r['k']}, {r['metric']}, {r['feat_name']}")
    X_data = feature_dict[r['feat_name']]
    algo   = 'ball_tree' if r['metric'] in ('euclidean', 'manhattan') else 'brute'
    model  = KNeighborsClassifier(
        n_neighbors=r['k'], weights=r['weights'],
        metric=r['metric'], algorithm=algo, n_jobs=-1
    )
    preds    = cross_val_predict(model, X_data, y_train, cv=loo, n_jobs=-1)
    acc      = accuracy_score(y_train, preds)
    bal_acc  = balanced_accuracy_score(y_train, preds)
    kappa    = cohen_kappa_score(y_train, preds)
    print(f"  → LOOCV Accuracy:    {acc:.4f}")
    print(f"  → Balanced Accuracy: {bal_acc:.4f}")
    print(f"  → Cohen's Kappa:     {kappa:.4f}")
    if acc > best_accuracy:
        best_accuracy = acc
        best_params   = r.copy()
        best_preds    = preds

print("\n" + "-"*60)
print("Best Configuration (LOOCV):")
print("-"*60)
print(f"  k={best_params['k']}, weights={best_params['weights']}, "
      f"metric={best_params['metric']}, {best_params['feat_name']}")
print(f"  LOOCV Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

# ============================================================
# 9. TRAIN FINAL MODEL
# ============================================================
print("\n" + "="*60)
print("STEP 10: Training Final Model")
print("="*60)

X_train_final = feature_dict[best_params['feat_name']]
algo_final    = 'ball_tree' if best_params['metric'] in ('euclidean', 'manhattan') else 'brute'
final_model   = KNeighborsClassifier(
    n_neighbors=best_params['k'], weights=best_params['weights'],
    metric=best_params['metric'], algorithm=algo_final, n_jobs=-1
)
final_model.fit(X_train_final, y_train)
print(f"Final model: {X_train_final.shape[0]} samples, "
      f"{X_train_final.shape[1]} dims ({best_params['feat_name']})")

# ============================================================
# 10. ENHANCED CLASSIFICATION REPORT (LOOCV)
# ============================================================
print("\n" + "="*60)
print("STEP 11: Classification Report (LOOCV)")
print("="*60)

print(classification_report(y_train, best_preds))

bal_acc_final = balanced_accuracy_score(y_train, best_preds)
kappa_final   = cohen_kappa_score(y_train, best_preds)
mcc_final     = matthews_corrcoef(y_train, best_preds)

print(f"Extended Metrics:")
print(f"  Balanced Accuracy : {bal_acc_final:.4f}  (robust to class imbalance)")
print(f"  Cohen's Kappa     : {kappa_final:.4f}  (>0.80 = strong agreement)")
print(f"  Matthews MCC      : {mcc_final:.4f}  (1=perfect, 0=random)")

print("\nPer-Class Performance:")
print("-"*50)
prec, rec, f1, sup = precision_recall_fscore_support(y_train, best_preds)
for i in range(len(prec)):
    status = ("!! WEAK" if f1[i] < 0.80 else " ~ OK" if f1[i] < 0.90 else " * GOOD")
    print(f"  Class {i}: F1={f1[i]:.3f}  P={prec[i]:.3f}  R={rec[i]:.3f}  {status}")

print("\nConfusion Matrix (rows=true, cols=predicted):")
cm      = confusion_matrix(y_train, best_preds)
classes = sorted(np.unique(y_train))
print("     " + " ".join(f"{c:4d}" for c in classes))
for idx, row in enumerate(cm):
    print(f"  {classes[idx]:2d}:" + " ".join(f"{v:4d}" for v in row))

# ============================================================
# 11. PREDICT ON TEST DATA
# ============================================================
print("\n" + "="*60)
print("STEP 12: Test Predictions")
print("="*60)

test_df     = pd.read_csv('product_images_for_prediction.csv')
X_test_orig = test_df.values.astype(np.float32)
print(f"Test samples: {X_test_orig.shape[0]}")

# ── Pixel pipeline (same transforms as training) ──
X_test_var      = X_test_orig[:, high_var_mask]
X_test_filtered = X_test_var[:, cols_to_keep]
X_test_scaled   = scaler.transform(X_test_filtered).astype(np.float32)

# ── Spatial features ──
imgs_test    = X_test_orig.reshape(-1, IMG_H, IMG_W).astype(np.float32)
hog_test     = compute_hog_batch(X_test_orig)
row_means_t  = imgs_test.mean(axis=2)
col_means_t  = imgs_test.mean(axis=1)
row_stds_t   = imgs_test.std(axis=2)
col_stds_t   = imgs_test.std(axis=1)
garment_test = compute_garment_features(X_test_orig)
sp_test      = np.hstack([hog_test, row_means_t, col_means_t,
                           row_stds_t, col_stds_t, garment_test])
X_hp_test    = scaler_hog_prof.transform(sp_test).astype(np.float32)

# ── CNN features (if available) ──
if cnn_ok:
    X_cnn_test  = extract_cnn_features(X_test_orig, cnn_model)
    X_cnn_test  = scaler_cnn.transform(X_cnn_test).astype(np.float32)
    X_test_fused = np.hstack([X_test_scaled, X_hp_test, X_cnn_test])
else:
    X_test_fused = np.hstack([X_test_scaled, X_hp_test])

# ── Dimensionality reduction ──
feat_name = best_params['feat_name']

if feat_name.startswith('PCA('):
    n_comp = int(feat_name[4:-1])
    X_test_final = X_test_fused @ pca_full.components_[:n_comp, :].T

elif feat_name.startswith('Isomap('):
    X_test_pre   = pca_pre.transform(X_test_fused)
    X_test_iso   = isomap.transform(X_test_pre)
    n_comp       = int(feat_name[7:-1])
    X_test_final = X_test_iso[:, :n_comp]

elif feat_name.startswith('MLLE('):
    X_test_pre    = pca_pre.transform(X_test_fused)
    X_test_mlle   = mlle.transform(X_test_pre)
    n_comp        = int(feat_name[5:-1])
    X_test_final  = X_test_mlle[:, :n_comp]

elif feat_name.startswith('SpecEmb('):
    X_test_pre    = pca_pre.transform(X_test_fused)
    X_test_spec   = spectral_transform_oos(X_test_pre, X_spec, spec_oos_nn)
    n_comp        = int(feat_name[8:-1])
    X_test_final  = X_test_spec[:, :n_comp]

elif feat_name == 'MLP_emb(128)':
    # Forward-pass test data through the same trained MLP hidden layers
    X_test_mlp   = extract_mlp_embeddings(mlp_extractor, X_test_fused)
    X_test_final = scaler_mlp.transform(X_test_mlp).astype(np.float32)

elif feat_name == 'MLP_proba(10)':
    # Soft class-membership scores from MLP output layer
    X_test_final = mlp_extractor.predict_proba(X_test_fused).astype(np.float32)

test_predictions = final_model.predict(X_test_final)

pd.DataFrame(test_predictions, columns=['predicted_label']) \
    .to_csv('test_predictions.csv', index=False)

print(f"Predictions saved to 'test_predictions.csv'")
pred_counts = dict(zip(*np.unique(test_predictions, return_counts=True)))
print(f"Prediction distribution: {pred_counts}")

# ============================================================
# 12. SUMMARY
# ============================================================
print("\n" + "="*60)
print("STEP 13: SUMMARY")
print("="*60)
print(f"Feature Engineering:")
print(f"  Pixels (filtered):   {X_train_scaled.shape[1]}")
print(f"  HOG (4x4 grid x9):   {hog_train.shape[1]}")
print(f"  Spatial profiles:    {spatial_profiles.shape[1]}")
print(f"  Garment measurements:{garment_train.shape[1]}  (targeted 0/4/6 features)")
if cnn_ok:
    print(f"  CNN (ResNet18):      {X_cnn_train.shape[1]}")
print(f"  MLP embeddings:      {X_mlp_emb.shape[1]}  (last hidden layer, 512→256→128)")
print(f"  MLP probabilities:   {X_mlp_proba.shape[1]}  (softmax class scores)")
print(f"  Fused (for PCA/manifold): {X_fused.shape[1]}")

print(f"\nDimensionality Reduction Evaluated:")
for name, X in feature_dict.items():
    marker = " <-- BEST" if name == best_params['feat_name'] else ""
    print(f"  {name:20s}: {X.shape[1]:3d} dims{marker}")

print(f"\nBest Model:")
print(f"  Representation : {best_params['feat_name']}")
print(f"  k={best_params['k']}, weights={best_params['weights']}, "
      f"metric={best_params['metric']}")
print(f"  LOOCV Accuracy : {best_accuracy:.4f}  ({best_accuracy*100:.2f}%)")
print(f"  Balanced Acc   : {bal_acc_final:.4f}")
print(f"  Cohen's Kappa  : {kappa_final:.4f}")
print(f"  Matthews MCC   : {mcc_final:.4f}")
