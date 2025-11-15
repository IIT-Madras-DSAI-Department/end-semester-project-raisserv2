import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

try:
    from xgboost import XGBClassifier
    XGB_INSTALLED = True
except ImportError:
    XGB_INSTALLED = False
    print("XGBoost not found. To run this, you must: pip install xgboost")
    raise

# -----------------------
# Helper: Load CSV data
# -----------------------
def load_data(train_csv, val_csv):
    """Loads CSVs where first column = label, remaining 784 columns = pixels."""
    print(f"Loading data from {train_csv} and {val_csv} ...")
    try:
        train_data = np.loadtxt(train_csv, delimiter=',', skiprows=1)
        X_train = train_data[:, 1:]
        y_train = train_data[:, 0].astype(int)

        val_data = np.loadtxt(val_csv, delimiter=',', skiprows=1)
        X_val = val_data[:, 1:]
        y_val = val_data[:, 0].astype(int)
    except Exception as e:
        print("Failed to load data:", e)
        return None
    print("Data loaded.")
    return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    TRAIN_FILE = "MNIST_train.csv"
    VAL_FILE = "MNIST_validation.csv"

    start_total = time.time()
    data = load_data(TRAIN_FILE, VAL_FILE)
    if data is None:
        raise SystemExit("Data load failed — check file paths/format.")

    X_train, y_train, X_val, y_val = data
    print(f"Shapes: X_train={X_train.shape}, y_train={y_train.shape}; X_val={X_val.shape}, y_val={y_val.shape}")

    # -----------------------
    # Preprocessing
    # -----------------------
    print("\n--- Preprocessing: StandardScaler + PCA (retain 95% variance) ---")
    t0 = time.time()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    pca = PCA(n_components=0.99999999999, svd_solver='full')
    X_train_proc = pca.fit_transform(X_train_scaled)
    X_val_proc = pca.transform(X_val_scaled)
    t1 = time.time()
    print(f"PCA finished. n_components = {pca.n_components_}. Time: {t1-t0:.2f}s")
    print(f"Processed shapes: X_train_proc={X_train_proc.shape}, X_val_proc={X_val_proc.shape}")

    # -----------------------
    # Baseline single XGBoost on PCA features
    # -----------------------
    print("\n--- Training baseline XGBoost (on PCA features) ---")
    t0 = time.time()
    base_xgb = XGBClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=42,
        objective='multi:softprob',  # probability output
        num_class=10,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    base_xgb.fit(X_train_proc, y_train, verbose=False)
    t1 = time.time()
    print(f"Base XGB trained in {t1-t0:.2f}s")

    y_pred_base = base_xgb.predict(X_val_proc)
    acc_base = accuracy_score(y_val, y_pred_base)
    f1_base = f1_score(y_val, y_pred_base, average='weighted')
    print(f"Base XGB -> Accuracy: {acc_base:.6f}, F1(weighted): {f1_base:.6f}")

    # -----------------------
    # Define stacking ensemble (all trained on PCA features)
    # -----------------------
    print("\n--- Defining Stacking Ensemble (all on PCA features) ---")
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ('knn', KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        ('logreg', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200, random_state=42))
    ]

    meta_xgb = XGBClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=42,
        objective='multi:softprob',  # ensure probabilities are used
        num_class=10,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )

    # stack_method='predict_proba' ensures meta-learner receives class probabilities (recommended)
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_xgb,
        cv=5,
        n_jobs=-1,
        passthrough=True,
        stack_method='predict_proba'  # explicitly request probabilities
    )

    # -----------------------
    # Train stacking (ONLY on PCA features)
    # -----------------------
    print("\n--- Training Stacking Ensemble (on PCA features) ---")
    t0 = time.time()
    stacking.fit(X_train_proc, y_train)
    t1 = time.time()
    print(f"Stacking trained in {t1-t0:.2f}s")

    # -----------------------
    # Evaluate stacking
    # -----------------------
    print("\n--- Evaluating Stacking Ensemble (on PCA features) ---")
    y_pred_stack = stacking.predict(X_val_proc)
    acc_stack = accuracy_score(y_val, y_pred_stack)
    f1_stack = f1_score(y_val, y_pred_stack, average='weighted')
    print(f"Stacking -> Accuracy: {acc_stack:.6f}, F1(weighted): {f1_stack:.6f}")

    print("\nClassification report (stacking):")
    print(classification_report(y_val, y_pred_stack, digits=4))

    # Optional: confusion matrix if you want to inspect common confusions
    # print("Confusion matrix:")
    # print(confusion_matrix(y_val, y_pred_stack))

    end_total = time.time()
    print(f"\nTotal runtime: {end_total - start_total:.2f}s")
