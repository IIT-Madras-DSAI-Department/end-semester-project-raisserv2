

import time
import numpy as np
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    print("XGBoost not found. Run: pip install xgboost")
    raise


# ============================================================
#  LOAD DATA
# ============================================================
def load_data(train_csv, val_csv):
    """Loads MNIST CSV where col0 = label, cols1-784 = pixels."""
    print(f"Loading data from {train_csv} and {val_csv} ...")

    try:
        train_data = np.loadtxt(train_csv, delimiter=',', skiprows=1)
        val_data = np.loadtxt(val_csv, delimiter=',', skiprows=1)
    except Exception as e:
        print("Failed loading:", e)
        return None

    # Fix if there are extra empty columns (common in MNIST CSV)
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
#  DESKEW (MAJOR BOOST FOR DIGIT 3)
# ============================================================
def deskew(img):
    """Deskew using central moments. img is 28×28 float array."""
    H, W = img.shape
    c, r = np.meshgrid(np.arange(W), np.arange(H))
    total = img.sum()

    if total == 0:
        return img

    cy = (r * img).sum() / total
    cx = (c * img).sum() / total

    mu11 = ((r - cy) * (c - cx) * img).sum() / total
    mu20 = ((r - cy)**2 * img).sum() / total
    mu02 = ((c - cx)**2 * img).sum() / total

    skew = mu11 / np.sqrt((mu20 * mu02) + 1e-8)

    # Affine transform
    M = np.float32([[1, skew, -0.5 * skew * H],
                    [0,    1,               0]])

    out = ndimage.affine_transform(
        img,
        M,
        offset=0,
        order=1,
        mode='nearest'
    )
    return out


# ============================================================
#  SMOOTHING (3×3 MEAN, ALLOWED)
# ============================================================
def smooth_image(img):
    kernel = np.ones((3, 3)) / 9.0
    return ndimage.convolve(img, kernel, mode='nearest')


# ============================================================
#  PREPROCESS FUNCTION
# ============================================================
def preprocess_images(X):
    out = []
    for flat in X:
        img = flat.reshape(28, 28)
        img = deskew(img)
        img = smooth_image(img)
        out.append(img.flatten())
    return np.array(out)


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    TRAIN_FILE = "MNIST_train.csv"
    VAL_FILE   = "MNIST_validation.csv"

    start_total = time.time()

    loaded = load_data(TRAIN_FILE, VAL_FILE)
    if loaded is None:
        exit()

    X_train, y_train, X_val, y_val = loaded

    # ----------------------------------------------------------
    # STEP 1: Deskew + smooth
    # ----------------------------------------------------------
    print("\n--- Deskewing + Smoothing ---")
    X_train = preprocess_images(X_train)
    X_val   = preprocess_images(X_val)

    # ----------------------------------------------------------
    # STEP 2: Scaling + PCA (98% variance)
    # ----------------------------------------------------------
    print("\n--- Scaling + PCA (98%) ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    pca = PCA(n_components=0.98, svd_solver='full')
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca   = pca.transform(X_val_scaled)

    print("PCA components:", pca.n_components_)
    print("Train PCA shape:", X_train_pca.shape)

    # ----------------------------------------------------------
    # STEP 3: Baseline XGB
    # ----------------------------------------------------------
    print("\n--- Training Base XGB ---")

    base_xgb = XGBClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=42,
        objective="multi:softprob",
        num_class=10,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    base_xgb.fit(X_train_pca, y_train)

    y_pred_base = base_xgb.predict(X_val_pca)
    print("Base XGB Accuracy:", accuracy_score(y_val, y_pred_base))

    # ----------------------------------------------------------
    # STEP 4: Stacking Ensemble
    # ----------------------------------------------------------
    print("\n--- Training Stacking Ensemble ---")

    estimators = [
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ("knn", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        ("logreg", LogisticRegression(
            multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=42))
    ]

    meta_xgb = XGBClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=42,
        objective="multi:softprob",
        num_class=10,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_xgb,
        cv=5,
        n_jobs=-1,
        passthrough=True,
        stack_method="predict_proba",
    )

    stacking.fit(X_train_pca, y_train)

    y_pred_stack = stacking.predict(X_val_pca)
    print("Stacking Accuracy:", accuracy_score(y_val, y_pred_stack))
    print("Stacking F1:", f1_score(y_val, y_pred_stack, average="weighted"))

    # ----------------------------------------------------------
    # STEP 5: Specialist 3-vs-Rest
    # ----------------------------------------------------------
    print("\n--- Training Specialist (3-vs-Rest) ---")

    y_train_3 = (y_train == 3).astype(int)
    specialist3 = LogisticRegression(solver="lbfgs", max_iter=200)
    specialist3.fit(X_train_pca, y_train_3)

    # ----------------------------------------------------------
    # STEP 6: Apply specialist routing
    # ----------------------------------------------------------
    print("\n--- Applying Specialist Routing ---")

    proba_stack = stacking.predict_proba(X_val_pca)
    final_preds = []
    THRESHOLD = 0.70

    for i in range(len(y_val)):
        p = proba_stack[i]
        pred = np.argmax(p)

        if pred == 3 and p[3] < THRESHOLD:
            # Ask specialist
            if specialist3.predict(X_val_pca[i].reshape(1, -1))[0] == 1:
                final_preds.append(3)
            else:
                final_preds.append(np.argmax(p))
        else:
            final_preds.append(pred)

    final_preds = np.array(final_preds)

    # ----------------------------------------------------------
    # FINAL REPORT
    # ----------------------------------------------------------
    print("\n--- FINAL REPORT (with Specialist) ---")
    print("Final Accuracy:", accuracy_score(y_val, final_preds))
    print("Final Weighted F1:", f1_score(y_val, final_preds, average="weighted"))
    print("\nClassification Report:")
    print(classification_report(y_val, final_preds, digits=4))

    print("\nTotal runtime: %.2fs" % (time.time() - start_total))
