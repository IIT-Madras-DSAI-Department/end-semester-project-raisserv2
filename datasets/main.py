# ===============================================
# main.py — Phase-2 optimized training pipeline
# ===============================================
import time, numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

from features import (
    StandardScalerCustom, pca_fit_transform, pca_transform,
    extract_hog_features, extract_directional_features, zonal_features
)
from algorithms import (
    LogisticRegressionCustom, KNNCustom,
    RandomForestFast, GradientBoostingFast,
    SigmoidCalibrator, StackingCV
)

# ------------------ Load data ------------------
def load_data(train_csv, val_csv):
    print(f"Loading data from {train_csv} and {val_csv} ...")
    train = np.loadtxt(train_csv, delimiter=',', skiprows=1)
    val = np.loadtxt(val_csv, delimiter=',', skiprows=1)
    if train.shape[1] > 785: train = train[:, :785]
    if val.shape[1] > 785: val = val[:, :785]
    y_train, X_train = train[:, 0].astype(int), train[:, 1:]
    y_val, X_val = val[:, 0].astype(int), val[:, 1:]
    print("Shapes:", X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    return X_train, y_train, X_val, y_val

# ------------------ Specialists ------------------
def train_binary_spec(X, y, pos_classes, pairwise=False):
    clf = LogisticRegressionCustom(lr=0.4, n_iter=500, reg=1e-4)
    if pairwise:
        mask = np.isin(y, pos_classes)
        Xs, ys = X[mask], y[mask]
        ys = (ys == pos_classes[0]).astype(int)
        clf.fit(Xs, ys)
    else:
        ys = np.isin(y, pos_classes).astype(int)
        clf.fit(X, ys)
    return clf

# ------------------ MAIN ------------------
if __name__ == "__main__":
    TRAIN_FILE = "MNIST_train.csv"
    VAL_FILE   = "MNIST_validation.csv"
    start = time.time()

    X_train, y_train, X_val, y_val = load_data(TRAIN_FILE, VAL_FILE)

    # ---------- PCA ----------
    print("\nScaling + PCA(95%)...")
    scaler_pix = StandardScalerCustom()
    X_train_s = scaler_pix.fit_transform(X_train)
    X_val_s   = scaler_pix.transform(X_val)
    X_train_pca, comps, mean = pca_fit_transform(X_train_s, 0.95)
    X_val_pca = pca_transform(X_val_s, comps, mean)

    # ---------- Features ----------
    print("\nExtracting features...")
    hog_tr, hog_va = extract_hog_features(X_train), extract_hog_features(X_val)
    dir_tr, dir_va = extract_directional_features(X_train), extract_directional_features(X_val)
    zon_tr, zon_va = zonal_features(X_train), zonal_features(X_val)

    # ---scale non-PCA features separately ---
    sc_hog = StandardScalerCustom().fit(hog_tr)
    hog_tr = sc_hog.transform(hog_tr)
    hog_va = sc_hog.transform(hog_va)

    sc_dir = StandardScalerCustom().fit(dir_tr)
    dir_tr = sc_dir.transform(dir_tr)
    dir_va = sc_dir.transform(dir_va)

    sc_zon = StandardScalerCustom().fit(zon_tr)
    zon_tr = sc_zon.transform(zon_tr)
    zon_va = sc_zon.transform(zon_va)

    # --- Combine everything ---
    X_train_f = np.hstack([X_train_pca, hog_tr, dir_tr, zon_tr])
    X_val_f   = np.hstack([X_val_pca,   hog_va, dir_va, zon_va])

    # --- Final global normalization ---
    scaler_f = StandardScalerCustom()
    X_train_f = scaler_f.fit_transform(X_train_f)
    X_val_f   = scaler_f.transform(X_val_f)

    print("Final feature shape:", X_train_f.shape)


    # ---------- Base + Meta ----------
    rf  = RandomForestFast(n_estimators=35, max_depth=10,
                           sample_ratio=0.35, n_thresholds=20, random_state=42)
    knn = KNNCustom(k=5)
    lr  = LogisticRegressionCustom(lr=0.4, n_iter=500, reg=1e-4)

    meta = GradientBoostingFast(n_estimators=30, lr=0.20,
                                max_depth=5, subsample=0.5,
                                n_thresholds=10, random_state=42)

    stack = StackingCV([("rf", rf), ("knn", knn), ("lr", lr)],
                       meta, cv=3, passthrough=True, random_state=42)
    print("\nTraining stacking ensemble...")
    stack.fit(X_train_f, y_train)

    # ---------- Calibrate ----------
    print("Calibrating probabilities...")
    P_train = stack.predict_proba(X_train_f)
    calibrator = SigmoidCalibrator(max_iter=200, lr=0.05)
    calibrator.fit(P_train, y_train, stack.classes_)
    P_val = calibrator.transform(stack.predict_proba(X_val_f))

    # Temperature scaling (stability)
    # P_val = P_val ** (1/1.2)
    temps = np.clip(1.0 + 0.5*(np.std(P_val,axis=0)), 1.0, 1.4)
    for k in range(P_val.shape[1]):
        P_val[:,k] = P_val[:,k] ** (1/temps[k])
    P_val /= np.sum(P_val, axis=1, keepdims=True)
    P_val /= np.sum(P_val, axis=1, keepdims=True)
    preds = stack.classes_[np.argmax(P_val, axis=1)]

    # ---------- Specialists ----------
    print("Training specialists...")
    s3  = train_binary_spec(X_train_f, y_train, [3])
    s7  = train_binary_spec(X_train_f, y_train, [7])
    s9  = train_binary_spec(X_train_f, y_train, [9])
    s35 = train_binary_spec(X_train_f, y_train, [3,5], pairwise=True)
    s79 = train_binary_spec(X_train_f, y_train, [7,9], pairwise=True)

    # thresholds = {"t3":0.7,"t7":0.7,"t9":0.7,"t35":0.75,"t79":0.75}
    thresholds = {"t3":0.65,"t7":0.65,"t9":0.65,"t35":0.70,"t79":0.70}
    cls_idx = {c:i for i,c in enumerate(stack.classes_)}
    final = preds.copy()
    for i,p in enumerate(preds):
        prob = P_val[i, cls_idx[p]]
        if p in (3,5) and prob < thresholds["t35"]:
            final[i] = 3 if s35.predict(X_val_f[i:i+1])[0]==1 else 5
        elif p in (7,9) and prob < thresholds["t79"]:
            final[i] = 7 if s79.predict(X_val_f[i:i+1])[0]==1 else 9
        elif p==3 and prob<thresholds["t3"] and s3.predict(X_val_f[i:i+1])[0]==1:
            final[i]=3
        elif p==7 and prob<thresholds["t7"] and s7.predict(X_val_f[i:i+1])[0]==1:
            final[i]=7
        elif p==9 and prob<thresholds["t9"] and s9.predict(X_val_f[i:i+1])[0]==1:
            final[i]=9

    # ---------- Evaluation ----------
    print("\n--- FINAL EVALUATION ---")
    print("Accuracy:", accuracy_score(y_val, final))
    print("Weighted F1:", f1_score(y_val, final, average='weighted'))
    print(classification_report(y_val, final, digits=4))
    print("Runtime: %.2fs" % (time.time()-start))
