#directional hog + added specialists

import time
import numpy as np
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import matplotlib.pyplot as plt

try:
    from xgboost import XGBClassifier
except ImportError:
    print("XGBoost not found. Run: pip install xgboost")
    raise


# ============================================================
#  DATA LOADING
# ============================================================
def load_data(train_csv, val_csv):
    print(f"Loading data from {train_csv} and {val_csv} ...")
    try:
        train_data = np.loadtxt(train_csv, delimiter=',', skiprows=1)
        val_data = np.loadtxt(val_csv, delimiter=',', skiprows=1)
    except Exception as e:
        print("Error:", e)
        return None

    if train_data.shape[1] > 785:
        train_data = train_data[:, :785]
    if val_data.shape[1] > 785:
        val_data = val_data[:, :785]

    y_train = train_data[:, 0].astype(int)
    X_train = train_data[:, 1:]

    y_val = val_data[:, 0].astype(int)
    X_val = val_data[:, 1:]

    print("Loaded shapes:", X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    return X_train, y_train, X_val, y_val


# ============================================================
#  SIMPLE HOG IMPLEMENTATION
# ============================================================
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
                    bin_idx = int(cell_ori[i, j])
                    hist[bin_idx] += cell_mag[i, j]
            hog_vec.extend(hist)
    return np.array(hog_vec)


def extract_hog_features(X):
    print("Extracting HOG features...")
    return np.array([compute_hog(x) for x in X])


# ============================================================
#  DIRECTIONAL ENERGY FEATURES (4-bin coarse gradient summary)
# ============================================================
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
    print("Extracting directional HOG features...")
    return np.array([directional_energy(x.reshape(28, 28)) for x in X])


# ============================================================
#  MAIN PIPELINE
# ============================================================
if __name__ == "__main__":
    TRAIN_FILE = "MNIST_train.csv"
    VAL_FILE = "MNIST_validation.csv"

    start_total = time.time()
    data = load_data(TRAIN_FILE, VAL_FILE)
    if data is None:
        exit()
    X_train, y_train, X_val, y_val = data

    # Step 1 — Extract HOG + Directional
    hog_train = extract_hog_features(X_train)
    hog_val = extract_hog_features(X_val)
    dir_train = extract_directional_features(X_train)
    dir_val = extract_directional_features(X_val)

    # Step 2 — PCA (95%)
    print("\n--- Scaling + PCA(95%) on Raw Pixels ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    # Step 3 — Combine PCA + HOG + Directional
    X_train_final = np.hstack([X_train_pca, hog_train, dir_train])
    X_val_final = np.hstack([X_val_pca, hog_val, dir_val])
    print("Final feature shape:", X_train_final.shape)

    # Step 4 — Base XGB
    base_xgb = XGBClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=42,
        objective="multi:softprob",
        num_class=10,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )
    base_xgb.fit(X_train_final, y_train)

    # Step 5 — Stacking Ensemble
    estimators = [
        ("rf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        ("logreg", LogisticRegression(
            multi_class="multinomial", solver="lbfgs", max_iter=200, random_state=42))
    ]
    meta_xgb = XGBClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=42,
        objective="multi:softprob",
        num_class=10,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_xgb,
        cv=5,
        n_jobs=-1,
        passthrough=True,
        stack_method="predict_proba"
    )

    # ---------- BEGIN PATCH: Zonal + Calibration + Specialists ----------
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.base import clone

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

    print("Extracting zonal features...")
    ztrain = zonal_features(X_train)
    zval = zonal_features(X_val)
    X_train_boost = np.hstack([X_train_final, ztrain])
    X_val_boost = np.hstack([X_val_final, zval])
    print("Boosted feature shapes:", X_train_boost.shape, X_val_boost.shape)

    stacking_boost = clone(stacking)
    print("Training stacking_boost...")
    stacking_boost.fit(X_train_boost, y_train)

    calibrated = CalibratedClassifierCV(
        estimator=stacking_boost,
        method='sigmoid',
        cv=3
    )
    print("Calibrating probabilities...")
    calibrated.fit(X_train_boost, y_train)

    # Specialists: 3-vs-rest, 7-vs-rest, 9-vs-rest, 3vs5, 7vs9
    spec_clf = LogisticRegression(solver='lbfgs', max_iter=300)
    def train_binary_spec(X, y, pos_classes, pairwise=False):
        """
        If pairwise=True, trains only on samples from the two classes.
        Otherwise trains a one-vs-rest specialist (binary labels 0/1).
        """
        if pairwise:
            mask = np.isin(y, pos_classes)
            X_sub, y_sub = X[mask], y[mask]
            y_sub = (y_sub == pos_classes[0]).astype(int)
        else:
            X_sub = X
            y_sub = np.isin(y, pos_classes).astype(int)
        clf = clone(spec_clf)
        clf.fit(X_sub, y_sub)
        return clf

    spec3  = train_binary_spec(X_train_boost, y_train, [3], pairwise=False)
    spec7  = train_binary_spec(X_train_boost, y_train, [7], pairwise=False)
    spec9  = train_binary_spec(X_train_boost, y_train, [9], pairwise=False)
    spec35 = train_binary_spec(X_train_boost, y_train, [3,5], pairwise=True)
    spec79 = train_binary_spec(X_train_boost, y_train, [7,9], pairwise=True)

    cal_proba_val = calibrated.predict_proba(X_val_boost)
    base_preds_val = np.argmax(cal_proba_val, axis=1)

    def apply_specialists(base_preds, base_proba, Xv):
        preds = base_preds.copy()
        p3, p7, p9 = [s.predict_proba(Xv)[:, 1]
                      for s in (spec3, spec7, spec9)]
        p35 = spec35.predict_proba(Xv)[:, 1]
        p79 = spec79.predict_proba(Xv)[:, 1]
        for i in range(len(preds)):
            p = base_proba[i]
            pred = preds[i]
            if pred in (3, 5) and p[pred] < 0.7:
                preds[i] = 3 if p35[i] >= 0.5 else 5
                continue
            if pred in (7, 9) and p[pred] < 0.7:
                preds[i] = 7 if p79[i] >= 0.5 else 9
                continue
            if pred == 3 and p[3] < 0.7 and p3[i] >= 0.5:
                preds[i] = 3
            if pred == 7 and p[7] < 0.7 and p7[i] >= 0.5:
                preds[i] = 7
            if pred == 9 and p[9] < 0.7 and p9[i] >= 0.5:
                preds[i] = 9
        return preds

    final_preds = apply_specialists(base_preds_val, cal_proba_val, X_val_boost)

    print("\n--- FINAL EVALUATION ---")
    print("Accuracy:", accuracy_score(y_val, final_preds))
    print("Weighted F1:", f1_score(y_val, final_preds, average='weighted'))
    print(classification_report(y_val, final_preds, digits=4))

    


    # Show wrong predictions
    mis_idx = np.where(y_val != final_preds)[0]
    print(f"\nTotal misclassified samples: {len(mis_idx)} / {len(y_val)}")
    for i in mis_idx[:20]:
        print(f"Index {i:4d}: True={y_val[i]}, Pred={final_preds[i]}")

    def show_misclassified(X, y_true, y_pred, indices, n=25):
        plt.figure(figsize=(12, 12))
        for i, idx in enumerate(indices[:n]):
            img = X[idx].reshape(28, 28)
            plt.subplot(5, 5, i+1)
            plt.imshow(img, cmap='gray')
            plt.title(f"T:{y_true[idx]}  P:{y_pred[idx]}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    show_misclassified(X_val, y_val, final_preds, mis_idx)
    # ---------- END PATCH ----------

    end_total = time.time()
    print(f"\nTotal execution time: {end_total - start_total:.2f} seconds")
