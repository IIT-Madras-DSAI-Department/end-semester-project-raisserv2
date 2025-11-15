# ===============================================
# main.py — training pipeline
# ===============================================

import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from algorithms import LogisticRegressionCustom, KNNCustom, RandomForestCustom, GradientBoostingCustom
from features import pca_fit_transform, pca_transform, extract_hog_features, extract_directional_features, zonal_features


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


if __name__ == "__main__":
    TRAIN_FILE = "MNIST_train.csv"
    VAL_FILE = "MNIST_validation.csv"
    start_total = time.time()

    X_train, y_train, X_val, y_val = load_data(TRAIN_FILE, VAL_FILE)

    # ---- Feature extraction ----
    X_train_pca, components, mean = pca_fit_transform(X_train, 0.95)
    X_val_pca = pca_transform(X_val, components, mean)

    hog_train = extract_hog_features(X_train)
    hog_val = extract_hog_features(X_val)
    dir_train = extract_directional_features(X_train)
    dir_val = extract_directional_features(X_val)
    ztrain = zonal_features(X_train)
    zval = zonal_features(X_val)

    X_train_final = np.hstack([X_train_pca, hog_train, dir_train, ztrain])
    X_val_final = np.hstack([X_val_pca, hog_val, dir_val, zval])

    # ---- Train base learners ----
    logreg = LogisticRegressionCustom(lr=0.1, n_iter=150)
    knn = KNNCustom(k=5)
    rf = RandomForestCustom(n_estimators=30, max_depth=8)
    gb = GradientBoostingCustom(n_estimators=50, lr=0.1)

    print("Training logistic regression...")
    logreg.fit(X_train_final, y_train)
    print("Training KNN...")
    knn.fit(X_train_final, y_train)
    print("Training RandomForest...")
    rf.fit(X_train_final, y_train)

    print("Training GradientBoost (meta)...")
    gb.fit(X_train_final, y_train)

    # ---- Evaluate ----
    preds = gb.predict(X_val_final)
    print("\nAccuracy:", accuracy_score(y_val, preds))
    print("Weighted F1:", f1_score(y_val, preds, average='weighted'))
    print(classification_report(y_val, preds, digits=4))

    print(f"\nTotal runtime: {time.time()-start_total:.2f}s")
