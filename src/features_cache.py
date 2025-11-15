# features_cache.py
import json, hashlib, numpy as np
from scipy import ndimage

# ---------- utils ----------
def sha1_file(path, block=1<<20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b: break
            h.update(b)
    return h.hexdigest()

class StandardScalerCustom:
    def fit(self, X):
        self.mean_ = X.mean(axis=0).astype(np.float32)
        self.std_  = (X.std(axis=0) + 1e-12).astype(np.float32)
        return self
    def transform(self, X):  return ((X - self.mean_) / self.std_).astype(np.float32)
    def fit_transform(self, X): self.fit(X); return self.transform(X)

def pca_fit_transform(X, var_ratio=0.95):
    X = X.astype(np.float32)
    mean = X.mean(axis=0).astype(np.float32)
    Xc = (X - mean).astype(np.float32)
    cov = np.cov(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    n_comp = int(np.searchsorted(np.cumsum(eigvals)/eigvals.sum(), var_ratio) + 1)
    comps = eigvecs[:, :n_comp].astype(np.float32)
    Xp = (Xc @ comps).astype(np.float32)
    return Xp, comps, mean

def pca_transform(X, comps, mean):
    return ((X - mean) @ comps).astype(np.float32)

# ---------- vectorized HOG (cells = 28//cell_size) ----------
import numpy as np
from scipy.ndimage import sobel

def extract_hog_features(X, cell_size=7, num_bins=8):
    """
    Extracts HOG features from a batch of images.
    
    X: (n_samples, n_features) array (e.g., n_features=784)
    cell_size: The size of each square cell in pixels (e.g., 7)
    num_bins: The number of orientation bins (e.g., 8)
    """
    # Assume images are 28x28
    images = X.reshape(-1, 28, 28)
    n_samples = images.shape[0]
    
    # Dimensions for HOG
    cells_per_dim = 28 // cell_size
    n_cells = cells_per_dim * cells_per_dim
    total_features = n_cells * num_bins
    
    # Final HOG features for all samples
    all_hogs = np.zeros((n_samples, total_features))
    
    # Calculate features for each sample
    for k in range(n_samples):
        img = images[k]
        
        # 1. Calculate gradients (magnitude and orientation)
        # Note: axis 0 is y (rows), axis 1 is x (columns)
        gy = sobel(img, axis=0, mode='constant')
        gx = sobel(img, axis=1, mode='constant')
        
        m = np.sqrt(gx**2 + gy**2)                 # Magnitude
        angles_rad = np.arctan2(gy, gx)           # Angle in radians
        angles_deg = np.rad2deg(angles_rad) % 180 # Unsigned angle 0-180
        
        # 2. Calculate orientation bins
        # --- FIX IS HERE ---
        # We use np.floor and then clip to ensure indices are [0, num_bins-1]
        # This prevents an angle of 180.0 from becoming bin 8
        bin_width = 180.0 / num_bins
        o = np.floor(angles_deg / bin_width).astype(int)
        o = np.clip(o, 0, num_bins - 1) 
        
        # 3. Accumulate histograms over cells
        hist_for_this_sample = np.zeros(total_features)
        
        cell_idx = 0
        for r in range(cells_per_dim):
            for c in range(cells_per_dim):
                # Get cell boundaries
                r_start, r_end = r * cell_size, (r + 1) * cell_size
                c_start, c_end = c * cell_size, (c + 1) * cell_size
                
                # Extract magnitudes and orientations for this cell
                m_cell = m[r_start:r_end, c_start:c_end].ravel()
                o_cell = o[r_start:r_end, c_start:c_end].ravel()
                
                # 4. Calculate histogram for this cell
                # This is the line that caused the original error
                # Now, o_cell will never contain '8', so the bincount works
                hist_cell = np.bincount(o_cell, weights=m_cell, minlength=num_bins)
                
                # Assign to the full feature vector for this sample
                hist_for_this_sample[cell_idx*num_bins : (cell_idx+1)*num_bins] = hist_cell
                cell_idx += 1
                
        all_hogs[k] = hist_for_this_sample
            
    return all_hogs

def extract_directional_features(X):
    N = X.shape[0]
    imgs = X.reshape(N, 28, 28).astype(np.float32)
    gx = ndimage.sobel(imgs, axis=2)
    gy = ndimage.sobel(imgs, axis=1)
    g45  = gx + gy
    g135 = gx - gy
    E0 = np.sum(np.abs(gx),   axis=(1,2))
    E1 = np.sum(np.abs(gy),   axis=(1,2))
    E2 = np.sum(np.abs(g45),  axis=(1,2))
    E3 = np.sum(np.abs(g135), axis=(1,2))
    r0 = E0 / (E1 + 1e-6)
    r1 = E2 / (E3 + 1e-6)
    return np.stack([E0,E1,E2,E3,r0,r1], axis=1).astype(np.float32)

def zonal_features(X, grid=(4,4)):
    N = X.shape[0]; gx, gy = grid
    imgs = X.reshape(N,28,28).astype(np.float32)
    sx, sy = 28//gx, 28//gy
    blocks = []
    for ix in range(gx):
        for iy in range(gy):
            blk = imgs[:, ix*sx:(ix+1)*sx, iy*sy:(iy+1)*sy]
            blocks.append(blk.sum(axis=(1,2)))
    blocks = np.stack(blocks, axis=1)  # (N, gx*gy)
    row_stats = np.stack([imgs.sum(2).mean(1), imgs.sum(2).std(1)], axis=1)
    col_stats = np.stack([imgs.sum(1).mean(1), imgs.sum(1).std(1)], axis=1)
    out = np.concatenate([blocks, row_stats, col_stats], axis=1)
    return out.astype(np.float32)

def save_json(path, obj):
    with open(path, "w") as f: json.dump(obj, f, indent=2)
