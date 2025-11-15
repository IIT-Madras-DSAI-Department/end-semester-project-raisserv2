# main_phase2_cached.py — FINAL OPTIMIZED TRAINING (no preprocessing)
import os, json, time, numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ==== import your custom learners ====
from algorithms import (
    LogisticRegressionCustom, KNNCustom,
    RandomForestFast, GradientBoostingFast,
    SigmoidCalibrator, StackingCV
)

# ==== paths ====
CACHE_DIR = "cache"
NPZ_PATH  = os.path.join(CACHE_DIR, "features_final_v1.npz")
META_PATH = os.path.join(CACHE_DIR, "meta.json")

def load_cached_arrays():
    assert os.path.exists(NPZ_PATH), "Cache not found. Run: python build_cache.py first."
    t0 = time.time()
    print("Loading precomputed cache (memory-mapped)...")

    # --- load uncompressed .npz or memory-map .npy ---
    data = np.load(NPZ_PATH, allow_pickle=False)
    meta = json.load(open(META_PATH))
    print("Cache meta:", meta.get("pca_var", "N/A"), "variance retained")

    # --- make all arrays contiguous in memory ---
    arrs = {name: np.ascontiguousarray(data[name]) for name in data.files}
    load_time = time.time() - t0
    print(f"Cache loaded in {load_time:.2f}s")
    return arrs

# ==== main ====
if __name__ == "__main__":
    start = time.time()
    data = load_cached_arrays()

    X_train = data["X_train_final"]
    X_val   = data["X_val_final"]
    X_train_norm = data["X_train_final_norm"]
    X_val_norm   = data["X_val_final_norm"]
    y_train = data["y_train"]
    y_val   = data["y_val"]

    print(f"Shapes: train={X_train.shape}, val={X_val.shape}")

    # ---- instantiate base learners ----
    rf = RandomForestFast(
        n_estimators=35, max_depth=10, sample_ratio=0.35,
        n_thresholds=20, random_state=42
    )
    knn = KNNCustom(k=5)   # cosine or Euclidean
    lr  = LogisticRegressionCustom(lr=0.35, n_iter=400, reg=1e-4, random_state=42)

    # ---- fit learners ----
    print("\nTraining base learners...")
    rf.fit(X_train, y_train)
    knn.fit(X_train_norm, y_train)  # uses cached normalized data
    lr.fit(X_train, y_train)

    # ---- meta learner ----
    meta = GradientBoostingFast(
        n_estimators=25, lr=0.25, max_depth=5,
        subsample=0.5, n_thresholds=10, random_state=42
    )

    # ---- stacking (no verbose prints inside CV) ----
    print("\nTraining stacking ensemble...")
    stack = StackingCV(
        base_learners=[("rf", rf), ("knn", knn), ("lr", lr)],
        meta_learner=meta,
        cv=3, passthrough=True, random_state=42
    )
    stack.fit(X_train, y_train)

    # ---- calibrate ONCE (not per fold) ----
    print("\nCalibrating meta-probabilities (Platt scaling)...")
    P_train = stack.predict_proba(X_train)
    cal = SigmoidCalibrator(max_iter=200, lr=0.05).fit(P_train, y_train, stack.classes_)
    P_val = cal.transform(stack.predict_proba(X_val))

    # ---- soft temperature adjustment ----
    P_val = (P_val ** (1/1.15)).astype(np.float32)
    P_val /= P_val.sum(axis=1, keepdims=True)

    preds = stack.classes_[np.argmax(P_val, axis=1)]

    # ---- evaluation ----
    print("\n--- FINAL EVALUATION ---")
    print("Accuracy:", accuracy_score(y_val, preds))
    print("Weighted F1:", f1_score(y_val, preds, average='weighted'))
    print(classification_report(y_val, preds, digits=4))

    runtime = time.time() - start
    print(f"\nTotal training time (no preprocessing): {runtime:.2f}s")
