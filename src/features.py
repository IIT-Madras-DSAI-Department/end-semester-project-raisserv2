# ===============================================
# features.py — Optimized handcrafted feature set
# ===============================================
import numpy as np
from scipy import ndimage

# ---------- StandardScaler ----------
class StandardScalerCustom:
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0) + 1e-12
        return self
    def transform(self, X):
        return (X - self.mean_) / self.std_
    def fit_transform(self, X):
        return self.fit(X).transform(X)

# ---------- PCA (retain variance) ----------
def pca_fit_transform(X, variance_ratio=0.95):
    X_mean = np.mean(X, axis=0)
    Xc = X - X_mean
    cov = np.cov(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    total = np.sum(eigvals)
    n_comp = np.searchsorted(np.cumsum(eigvals) / total, variance_ratio) + 1
    comps = eigvecs[:, :n_comp]
    return Xc @ comps, comps, X_mean

def pca_transform(X, comps, mean):
    return (X - mean) @ comps

# ---------- HOG (7x7 cells, 6 bins) ----------
def compute_hog(img, cell_size=4, num_bins=6):
    img = img.reshape(28, 28).astype(np.float32)
    gx = ndimage.sobel(img, axis=1)
    gy = ndimage.sobel(img, axis=0)
    mag = np.hypot(gx, gy)
    ori = (np.arctan2(gy, gx) + np.pi) * (num_bins / (2*np.pi))
    ori = np.clip(ori, 0, num_bins - 1)
    cells = 28 // cell_size
    hog = []
    for cy in range(cells):
        for cx in range(cells):
            m = mag[cy*cell_size:(cy+1)*cell_size,
                    cx*cell_size:(cx+1)*cell_size]
            o = ori[cy*cell_size:(cy+1)*cell_size,
                    cx*cell_size:(cx+1)*cell_size]
            hist = np.zeros(num_bins)
            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    hist[int(o[i, j])] += m[i, j]
            hog.extend(hist)
    return np.array(hog)

def extract_hog_features(X):
    print("Extracting fine-grained HOG (7×7 cells, 6 bins)...")
    return np.array([compute_hog(x) for x in X])

# ---------- Directional energies (4 dirs + 2 ratios) ----------
def directional_energy(img):
    gx = ndimage.sobel(img, axis=1)
    gy = ndimage.sobel(img, axis=0)
    g45 = gx + gy
    g135 = gx - gy
    E = np.array([np.sum(np.abs(gx)),
                  np.sum(np.abs(gy)),
                  np.sum(np.abs(g45)),
                  np.sum(np.abs(g135))])
    ratios = np.array([E[0]/(E[1]+1e-6), E[2]/(E[3]+1e-6)])
    return np.concatenate([E, ratios])

def extract_directional_features(X):
    print("Extracting directional features...")
    return np.array([directional_energy(x.reshape(28, 28)) for x in X])

# ---------- Zonal (4×4 blocks + row/col stats) ----------
def zonal_features(X, grid=(4, 4)):
    N = X.shape[0]; gx, gy = grid
    feats = []
    for i in range(N):
        img = X[i].reshape(28, 28)
        sx, sy = 28//gx, 28//gy
        blocks = [img[x*sx:(x+1)*sx, y*sy:(y+1)*sy].sum()
                  for x in range(gx) for y in range(gy)]
        row_sums = img.sum(axis=1)
        col_sums = img.sum(axis=0)
        feats.append(blocks + [row_sums.mean(), row_sums.std(),
                               col_sums.mean(), col_sums.std()])
    return np.array(feats)
