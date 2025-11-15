# ===============================================
# features.py — handcrafted feature extraction
# ===============================================

import numpy as np
from scipy import ndimage

# ---------- PCA (from scratch) ----------
def pca_fit_transform(X, variance_ratio=0.95):
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    cov = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    total = np.sum(eigvals)
    var_sum = np.cumsum(eigvals) / total
    n_comp = np.searchsorted(var_sum, variance_ratio) + 1
    components = eigvecs[:, :n_comp]
    X_reduced = X_centered @ components
    return X_reduced, components, X_mean


def pca_transform(X, components, mean):
    return (X - mean) @ components


# ---------- HOG ----------
def compute_hog(img, cell_size=7, num_bins=8):
    img = img.reshape(28, 28).astype(np.float32)
    gx = ndimage.sobel(img, axis=1)
    gy = ndimage.sobel(img, axis=0)
    mag = np.hypot(gx, gy)
    ori = np.arctan2(gy, gx)
    ori = (ori + np.pi) * (num_bins / (2 * np.pi))
    ori = np.clip(ori, 0, num_bins - 1)
    hog_vec = []
    for cy in range(4):
        for cx in range(4):
            cell_mag = mag[cy*cell_size:(cy+1)*cell_size,
                           cx*cell_size:(cx+1)*cell_size]
            cell_ori = ori[cy*cell_size:(cy+1)*cell_size,
                           cx*cell_size:(cx+1)*cell_size]
            hist = np.zeros(num_bins)
            for i in range(cell_mag.shape[0]):
                for j in range(cell_mag.shape[1]):
                    hist[int(cell_ori[i, j])] += cell_mag[i, j]
            hog_vec.extend(hist)
    return np.array(hog_vec)


def extract_hog_features(X):
    return np.array([compute_hog(x) for x in X])


# ---------- Directional gradient energies ----------
def directional_energy(img):
    gx = ndimage.sobel(img, axis=1)
    gy = ndimage.sobel(img, axis=0)
    g45 = ndimage.sobel(img, axis=1) + ndimage.sobel(img, axis=0)
    g135 = ndimage.sobel(img, axis=1) - ndimage.sobel(img, axis=0)
    return np.array([
        np.sum(np.abs(gx)),
        np.sum(np.abs(gy)),
        np.sum(np.abs(g45)),
        np.sum(np.abs(g135))
    ])


def extract_directional_features(X):
    return np.array([directional_energy(x.reshape(28, 28)) for x in X])


# ---------- Zonal features ----------
def zonal_features(X, grid=(4, 4)):
    N = X.shape[0]
    gx, gy = grid
    feats = []
    for i in range(N):
        img = X[i].reshape(28, 28)
        sx = 28 // gx
        sy = 28 // gy
        blocks = [img[x*sx:(x+1)*sx, y*sy:(y+1)*sy].sum()
                  for x in range(gx) for y in range(gy)]
        row_sums = img.sum(axis=1)
        col_sums = img.sum(axis=0)
        feats.append(blocks + [row_sums.mean(), row_sums.std(), col_sums.mean(), col_sums.std()])
    return np.array(feats)
