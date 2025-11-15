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

    # Trim extra columns (some MNIST CSVs have trailing commas)
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
#  SIMPLE HOG IMPLEMENTATION (project-safe)
# ============================================================
def compute_hog(img, cell_size=7, num_bins=8):
    """
    Custom HOG implementation using numpy only.
    Produces 16 cells × 8 bins = 128-dim feature vector.
    """
    img = img.reshape(28, 28).astype(np.float32)

    # Compute gradients
    gx = ndimage.sobel(img, axis=1)
    gy = ndimage.sobel(img, axis=0)

    # Magnitude and orientation
    mag = np.hypot(gx, gy)
    ori = np.arctan2(gy, gx)  # range [-pi, pi]
    ori = (ori + np.pi) * (num_bins / (2 * np.pi))  # scale to bin index
    ori = np.clip(ori, 0, num_bins - 1)

    # Divide into 4×4 = 16 cells (each 7×7 pixels)
    hog_vec = []
    idx = 0
    for cy in range(4):
        for cx in range(4):
            cell_mag = mag[cy*cell_size:(cy+1)*cell_size,
                           cx*cell_size:(cx+1)*cell_size]
            cell_ori = ori[cy*cell_size:(cy+1)*cell_size,
                           cx*cell_size:(cx+1)*cell_size]

            hist = np.zeros(num_bins)
            # Soft binning not needed; just integer bins
            for i in range(cell_mag.shape[0]):
                for j in range(cell_mag.shape[1]):
                    bin_idx = int(cell_ori[i, j])
                    hist[bin_idx] += cell_mag[i, j]

            hog_vec.extend(hist)

    return np.array(hog_vec)


def extract_hog_features(X):
    print("Extracting HOG features...")
    output = []
    for img_flat in X:
        output.append(compute_hog(img_flat))
    return np.array(output)


# ============================================================
#  MAIN PIPELINE (PCA + HOG + STACKING + SPECIALIST)
# ============================================================
if __name__ == "__main__":
    TRAIN_FILE = "MNIST_train.csv"
    VAL_FILE   = "MNIST_validation.csv"

    start_total = time.time()
    loaded = load_data(TRAIN_FILE, VAL_FILE)
    if loaded is None:
        exit()
    X_train, y_train, X_val, y_val = loaded

    # -------------------------------------------------------------
    # STEP 1 — Extract HOG features
    # -------------------------------------------------------------
    hog_train = extract_hog_features(X_train)   # shape: (N, 128)
    hog_val   = extract_hog_features(X_val)

    # -------------------------------------------------------------
    # STEP 2 — Scaling + PCA(95%) on raw pixel intensities
    # -------------------------------------------------------------
    print("\n--- Scaling + PCA(95%) on Raw Pixels ---")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca   = pca.transform(X_val_scaled)

    print("PCA output shape:", X_train_pca.shape)

    # -------------------------------------------------------------
    # STEP 3 — CONCATENATE PCA + HOG for final feature vector
    # -------------------------------------------------------------
    print("\n--- Concatenating PCA + HOG Features ---")
    X_train_final = np.hstack([X_train_pca, hog_train])
    X_val_final   = np.hstack([X_val_pca,   hog_val])
    print("Final feature shape:", X_train_final.shape)


    # -------------------------------------------------------------
    # STEP 4 — Train BASE XGBoost
    # -------------------------------------------------------------
    print("\n--- Training Base XGB ---")
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

    print("Base accuracy:", accuracy_score(y_val, base_xgb.predict(X_val_final)))

    # -------------------------------------------------------------
    # STEP 5 — STACKING (RF + KNN + LR → XGB)
    # -------------------------------------------------------------
    print("\n--- Training Stacking Ensemble ---")

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
##################################################################################################################
    # ---------- begin patch: specialists + calibration + zonal features ----------
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import GridSearchCV
    from sklearn.base import clone
    import itertools

    # ---------- 0) small engineered features: zonal sums (4x4), row/col sums stats ----------
    def zonal_features(X, grid=(4,4)):
        # X: (N, 784)
        N = X.shape[0]
        gx, gy = grid
        out = []
        for i in range(N):
            img = X[i].reshape(28,28)
            blocks = []
            sx = 28 // gx
            sy = 28 // gy
            for bx in range(gx):
                for by in range(gy):
                    sub = img[bx*sx:(bx+1)*sx, by*sy:(by+1)*sy]
                    blocks.append(sub.sum())
            # row/col sums mean & std
            row_sums = img.sum(axis=1)
            col_sums = img.sum(axis=0)
            feats = blocks + [row_sums.mean(), row_sums.std(), col_sums.mean(), col_sums.std()]
            out.append(feats)
        return np.array(out)

    print("Extracting zonal features...")
    ztrain = zonal_features(X_train)   # (N_train, 4*4 + 4) = 20 features
    zval   = zonal_features(X_val)

    # Concatenate to final features
    X_train_boost = np.hstack([X_train_final, ztrain])
    X_val_boost   = np.hstack([X_val_final,   zval])

    print("Boosted feature shapes:", X_train_boost.shape, X_val_boost.shape)

    # ---------- 1) (Re)Train stacking on boosted features (or re-fit calibrated later) ----------
    # If you already have a stacking model trained on X_train_final, re-train a fresh one on X_train_boost:
    stacking_boost = clone(stacking)
    print("Training stacking_boost on PCA+HOG+Zonal features...")
    stacking_boost.fit(X_train_boost, y_train)

    # ---------- 2) Probability calibration for stacking (improves thresholds) ----------
    # Calibrate with sigmoid (Platt scaling) using 3-fold; this is relatively cheap because meta-learner is light.
    calibrated = CalibratedClassifierCV(estimator=stacking_boost, method='sigmoid', cv=3, n_jobs=-1)
    print("Calibrating stacking probabilities (CalibratedClassifierCV)...")
    calibrated.fit(X_train_boost, y_train)   # fits internally with CV

    # ---------- 3) Train specialists: 3-vs-rest, 7-vs-rest, 9-vs-rest, and pairwise 3vs5 ----------
    spec_clf = LogisticRegression(solver='lbfgs', max_iter=300)

    # Train 3-vs-rest
    y3 = (y_train == 3).astype(int); spec3 = clone(spec_clf); spec3.fit(X_train_boost, y3)
    # Train 7-vs-rest
    y7 = (y_train == 7).astype(int); spec7 = clone(spec_clf); spec7.fit(X_train_boost, y7)
    # Train 9-vs-rest
    y9 = (y_train == 9).astype(int); spec9 = clone(spec_clf); spec9.fit(X_train_boost, y9)
    # Pairwise specialist 3 vs 5
    mask35 = np.isin(y_train, [3,5])
    X35 = X_train_boost[mask35]; y35 = y_train[mask35]
    # encode 3->1, 5->0
    y35b = (y35 == 3).astype(int)
    spec35 = clone(spec_clf); spec35.fit(X35, y35b)

    # ---------- 4) Automatic threshold tuning (grid search on validation) ----------
    # We'll search thresholds for each specialist independently using validation set predictions
    cal_proba_val = calibrated.predict_proba(X_val_boost)   # calibrated stacking probabilities
    base_preds_val = np.argmax(cal_proba_val, axis=1)

    # function to tune threshold for one specialist
    def tune_threshold(base_probas, base_preds, specialist, class_id, Xv, yv, thresholds=np.linspace(0.5, 0.95, 10)):
        best_t = 0.5
        best_f1 = -1
        # specialist_prob: probability that sample is class_id
        spec_proba = specialist.predict_proba(Xv)[:,1]
        for t in thresholds:
            preds = base_preds.copy()
            for i in range(len(preds)):
                if preds[i] == class_id and base_probas[i, class_id] < t:
                    # ask specialist
                    p = spec_proba[i]
                    if p >= 0.5:
                        preds[i] = class_id
                    else:
                        # keep base prediction (or could use argmax)
                        preds[i] = preds[i]
            f1 = f1_score(yv, preds, average='weighted')
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        return best_t, best_f1

    print("Tuning thresholds on validation set...")
    t3, f3 = tune_threshold(cal_proba_val, base_preds_val, spec3, 3, X_val_boost, y_val)
    t7, f7 = tune_threshold(cal_proba_val, base_preds_val, spec7, 7, X_val_boost, y_val)
    t9, f9 = tune_threshold(cal_proba_val, base_preds_val, spec9, 9, X_val_boost, y_val)
    # For pairwise 3v5 - tune separately for cases where base predicts 3 or 5
    # We'll use a simple approach: when base predicts 3 or 5 and prob of predicted class < t35, call spec35
    def tune_pairwise_threshold(base_probas, base_preds, pair_clf, classes, Xv, yv, thresholds=np.linspace(0.5, 0.95, 10)):
        best_t, best_f = 0.5, -1
        pair_mask = np.isin(base_preds, classes)
        pair_idx = np.where(pair_mask)[0]
        if len(pair_idx)==0:
            return 0.5, -1
        spec_proba = pair_clf.predict_proba(Xv)[:,1]  # prob of class '3' in our encoding
        for t in thresholds:
            preds = base_preds.copy()
            for i in pair_idx:
                if base_probas[i, preds[i]] < t:
                    # ask pairwise specialist
                    p = spec_proba[i]
                    if p >= 0.5:
                        preds[i] = classes[0]  # 3
                    else:
                        preds[i] = classes[1]  # 5
            f1 = f1_score(yv, preds, average='weighted')
            if f1 > best_f:
                best_f = f1
                best_t = t
        return best_t, best_f

    t35, f35 = tune_pairwise_threshold(cal_proba_val, base_preds_val, spec35, (3,5), X_val_boost, y_val)

    print("Chosen thresholds: t3=%.3f (f1=%.4f), t7=%.3f (%.4f), t9=%.3f (%.4f), t35=%.3f (%.4f)" %
        (t3, f3, t7, f7, t9, f9, t35, f35))

    # ---------- 5) Apply calibrated stacking + specialists with tuned thresholds ----------
    final_preds = base_preds_val.copy()
    spec3_proba_val = spec3.predict_proba(X_val_boost)[:,1]
    spec7_proba_val = spec7.predict_proba(X_val_boost)[:,1]
    spec9_proba_val = spec9.predict_proba(X_val_boost)[:,1]
    spec35_proba_val = spec35.predict_proba(X_val_boost)[:,1]

    for i in range(len(final_preds)):
        p = cal_proba_val[i]
        pred = final_preds[i]
        # pairwise 3 vs 5 first (if base predicted 3 or 5)
        if pred in (3,5) and p[pred] < t35:
            # ask pairwise specialist
            if spec35_proba_val[i] >= 0.5:
                final_preds[i] = 3
            else:
                final_preds[i] = 5
            continue
        # specialist flow
        if pred == 3 and p[3] < t3:
            final_preds[i] = 3 if spec3_proba_val[i]>=0.5 else final_preds[i]
        if pred == 7 and p[7] < t7:
            final_preds[i] = 7 if spec7_proba_val[i]>=0.5 else final_preds[i]
        if pred == 9 and p[9] < t9:
            final_preds[i] = 9 if spec9_proba_val[i]>=0.5 else final_preds[i]

    # Evaluate
    from sklearn.metrics import classification_report
    print("After calibration + specialists: Accuracy:", accuracy_score(y_val, final_preds))
    print("Weighted F1:", f1_score(y_val, final_preds, average='weighted'))
    print(classification_report(y_val, final_preds, digits=4))


    # Find misclassified samples
    mis_idx = np.where(y_val != final_preds)[0]
    print(f"\nTotal misclassified samples: {len(mis_idx)} / {len(y_val)}")

    # Print first 20 for inspection
    print("\nFirst 20 misclassifications (index, true → predicted):")
    for i in mis_idx[:20]:
        print(f"Index {i:4d}: True={y_val[i]}, Pred={final_preds[i]}")

    # Optional: visualize them
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

    # Show a few images
    show_misclassified(X_val, y_val, final_preds, mis_idx)


##############################################################################################################
    end_total = time.time()
    print(f"\nTotal execution time: {end_total - start_total:.2f} seconds")
