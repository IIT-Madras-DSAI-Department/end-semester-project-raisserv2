# build_cache.py
import os, json, time, numpy as np
from features_cache import (
    sha1_file, StandardScalerCustom,
    pca_fit_transform, pca_transform,
    extract_hog_features, extract_directional_features, zonal_features,
    save_json
)

TRAIN = "MNIST_train.csv"
VAL   = "MNIST_validation.csv"
CACHE_DIR = "cache"
NPZ_PATH  = os.path.join(CACHE_DIR, "features_final_v1.npz")
META_PATH = os.path.join(CACHE_DIR, "meta.json")

def load_csvs(train_csv, val_csv):
    tr = np.loadtxt(train_csv, delimiter=',', skiprows=1)
    va = np.loadtxt(val_csv, delimiter=',', skiprows=1)
    if tr.shape[1] > 785: tr = tr[:, :785]
    if va.shape[1] > 785: va = va[:, :785]
    y_tr, X_tr = tr[:,0].astype(int), tr[:,1:].astype(np.float32)
    y_va, X_va = va[:,0].astype(int), va[:,1:].astype(np.float32)
    return X_tr, y_tr, X_va, y_va

if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)
    t0 = time.time()
    print("Loading CSVs...")
    X_tr, y_tr, X_va, y_va = load_csvs(TRAIN, VAL)

    # 1) scale pixels + PCA
    print("Scaling pixels + PCA(0.95)...")
    sc_pix = StandardScalerCustom()
    X_tr_s = sc_pix.fit_transform(X_tr)
    X_va_s = sc_pix.transform(X_va)
    X_tr_pca, comps, mean = pca_fit_transform(X_tr_s, 0.95)
    X_va_pca = pca_transform(X_va_s, comps, mean)
    print("PCA comps:", X_tr_pca.shape[1])

    # 2) feature families
    print("Extracting HOG/Directional/Zonal...")
    hog_tr = extract_hog_features(X_tr, cell_size=7, num_bins=8)
    hog_va = extract_hog_features(X_va, cell_size=7, num_bins=8)
    dir_tr = extract_directional_features(X_tr)  # 6
    dir_va = extract_directional_features(X_va)
    zon_tr = zonal_features(X_tr)               # 20
    zon_va = zonal_features(X_va)

    # 3) scale non-PCA groups independently
    print("Scaling HOG/Directional/Zonal...")
    sc_hog = StandardScalerCustom().fit(hog_tr)
    hog_tr = sc_hog.transform(hog_tr); hog_va = sc_hog.transform(hog_va)
    sc_dir = StandardScalerCustom().fit(dir_tr)
    dir_tr = sc_dir.transform(dir_tr); dir_va = sc_dir.transform(dir_va)
    sc_zon = StandardScalerCustom().fit(zon_tr)
    zon_tr = sc_zon.transform(zon_tr); zon_va = sc_zon.transform(zon_va)

    # 4) concatenate + final scaler
    print("Concatenating & final scaling...")
    X_train_final = np.hstack([X_tr_pca, hog_tr, dir_tr, zon_tr]).astype(np.float32)
    X_val_final   = np.hstack([X_va_pca, hog_va, dir_va, zon_va]).astype(np.float32)
    sc_final = StandardScalerCustom().fit(X_train_final)
    X_train_final = sc_final.transform(X_train_final)
    X_val_final   = sc_final.transform(X_val_final)

    # 5) precompute normalized copies for KNN cosine (pure preprocessing)
    def row_norm(x):
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return (x / n).astype(np.float32)
    X_train_final_norm = row_norm(X_train_final)
    X_val_final_norm   = row_norm(X_val_final)

    # 6) write cache
    np.savez(
        NPZ_PATH,
        X_train_final=X_train_final,
        X_val_final=X_val_final,
        X_train_final_norm=X_train_final_norm,
        X_val_final_norm=X_val_final_norm,
        y_train=y_tr,
        y_val=y_va
    )
    meta = {
        "train_sha1": sha1_file(TRAIN),
        "val_sha1":   sha1_file(VAL),
        "pca_var": 0.95,
        "hog": {"cell_size": 7, "bins": 8},
        "directional": {"dims": 6},
        "zonal": {"grid": [4,4]},
        "dtype": "float32",
        "created_s": round(time.time() - t0, 2)
    }
    save_json(META_PATH, meta)
    print(f"Done. Cache at {NPZ_PATH}. Built in {meta['created_s']} s.")
