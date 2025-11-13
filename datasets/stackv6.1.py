import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans # REMOVED - This was noise
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import LinearSVC # REMOVED - per your request
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

try:
    from xgboost import XGBClassifier
    XGB_INSTALLED = True
except ImportError:
    XGB_INSTALLED = False
    print("XGBoost not found. To run this, you must: pip install xgboost")
    exit()

# --- 1. Helper Functions ---

def load_data(train_csv, val_csv):
    """Loads the MNIST training and validation datasets."""
    print(f"Loading data from {train_csv} and {val_csv}...")
    try:
        train_data = np.loadtxt(train_csv, delimiter=',', skiprows=1)
        X_train = train_data[:, 1:]
        y_train = train_data[:, 0].astype(int)
        
        val_data = np.loadtxt(val_csv, delimiter=',', skiprows=1)
        X_val = val_data[:, 1:]
        y_val = val_data[:, 0].astype(int)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure CSV files are in the same directory.")
        return None
    
    print("Data loaded successfully.")
    return X_train, y_train, X_val, y_val

# --- 2. Main Execution ---

if __name__ == "__main__":
    
    # --- Configuration ---
    TRAIN_FILE = 'MNIST_train.csv'
    VAL_FILE = 'MNIST_validation.csv'
    LOW_THRESHOLD = 0  # Your new low threshold
    HIGH_THRESHOLD = 255 # Your new high threshold
    
    # --- STAGE 1: Load Data ---
    start_total_time = time.time()
    
    data = load_data(TRAIN_FILE, VAL_FILE)
    if data is None:
        exit()
        
    X_train, y_train, X_val, y_val = data
    
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Val shapes:   X={X_val.shape}, y={y_val.shape}")
    
    
    # --- STAGE 1.5: Clip Data (Your New Experiment) ---
    print(f"\n--- Clipping data: < {LOW_THRESHOLD} -> 0, > {HIGH_THRESHOLD} -> 255 ---")
    
    # Make a copy to work on
    X_train_clipped = np.copy(X_train)
    X_val_clipped = np.copy(X_val)

    # Apply lower threshold
    X_train_clipped[X_train_clipped < LOW_THRESHOLD] = 0.0
    X_val_clipped[X_val_clipped < LOW_THRESHOLD] = 0.0
    
    # Apply upper threshold
    X_train_clipped[X_train_clipped > HIGH_THRESHOLD] = 255.0
    X_val_clipped[X_val_clipped > HIGH_THRESHOLD] = 255.0

    print("Clipping complete. Values are 0, 255, or 50-205.")


    # --- STAGE 2: Preprocessing (StandardScaler + PCA) ---
    # We will now run the pipeline on the new CLIPPED data.
    print("\n--- Preprocessing CLIPPED Data (StandardScaler + PCA) ---")
    
    scaler = StandardScaler()
    # Fit the scaler on the new clipped training data
    X_train_scaled = scaler.fit_transform(X_train_clipped)
    # Transform the clipped validation data
    X_val_scaled = scaler.transform(X_val_clipped)

    print("Scaling complete. Starting PCA...")
    pca = PCA(n_components=0.96)
    X_train_processed = pca.fit_transform(X_train_scaled)
    X_val_processed = pca.transform(X_val_scaled)
    
    n_pca_features = X_train_processed.shape[1]
    print(f"PCA complete. Selected {n_pca_features} components.")
    print(f"Data shape after PCA: {X_train_processed.shape}")
    
    
    # --- STAGE 3: KMeans Feature Engineering ---
    # REMOVED THIS ENTIRE STAGE. It was hurting performance.
    

    # --- STAGE 4: Train Base Model (XGBoost Only, on PCA data) ---
    # This is our baseline "Best Single Model" *on PCA data*.
    print(f"\n--- Training Base Model (XGBoost Only, on PCA data) ---")
    start_xgb_time = time.time()
    
    base_model = XGBClassifier(
        n_estimators=100, 
        n_jobs=-1,
        random_state=42, 
        objective='multi:softmax', 
        num_class=10,
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    # Fit on the PROCESSED (PCA-only) data
    base_model.fit(X_train_processed, y_train)
    
    end_xgb_time = time.time()
    print(f"Base XGBoost trained. Time: {end_xgb_time:.2f}s")
    
    # Predict on the PROCESSED (PCA-only) data
    y_pred_base = base_model.predict(X_val_processed)
    f1_base = f1_score(y_val, y_pred_base, average='weighted')
    acc_base = accuracy_score(y_val, y_pred_base)


    # --- STAGE 5: Define the Stacking Ensemble ---
    print("\n--- Defining the Stacking Ensemble (RF+KNN+LogReg -> XGB) ---")

    # Level 0: The "Estimators" (Base Models)
    # This is the stack you requested.
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ('knn', KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        ('logreg', LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42)),
        # ('svm', LinearSVC(max_iter=2000, random_state=42, dual=True)) # REMOVED
    ]

    # Level 1: The "Final Estimator" (Meta-Model)
    # We use XGBoost as the meta-model to learn from the base models
    final_estimator = XGBClassifier(
        n_estimators=100, 
        n_jobs=-1,
        random_state=42, 
        objective='multi:softmax', 
        num_class=10,
        eval_metric='mlogloss',
        use_label_encoder=False
    )

    # Build the Stacking Classifier
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=3,  # Use 3 folds for speed.
        n_jobs=-1,
        passthrough=True # The meta-model gets preds + the PCA data
    )

    # --- STAGE 6: Train the Stacking Ensemble ---
    print(f"\n--- Training Stacking Ensemble (on PCA data) ---")
    start_stack_time = time.time()
    
    # Fit on the PROCESSED (PCA-only) data
    stacking_model.fit(X_train_processed, y_train)
    
    end_stack_time = time.time()
    print(f"Stacking Ensemble trained. Time: {end_stack_time - start_stack_time:.2f}s")

    
    # --- STAGE 7: Evaluate Ensemble vs. Base ---
    print("\n--- Evaluating Final Ensemble ---")
    
    # Predict with the Stacking Model on PROCESSED (PCA-only) data
    y_pred_stack = stacking_model.predict(X_val_processed)
    
    # Stacking Ensemble
    f1_stack = f1_score(y_val, y_pred_stack, average='weighted')
    acc_stack = accuracy_score(y_val, y_pred_stack)
    
    print("\n" + "="*70)
    print(f" --- REALISTIC PINNACLE BENCHMARK (Clipped <{LOW_THRESHOLD}, >{HIGH_THRESHOLD} + PCA) ---")
    print("="*70)
    print(f" Base Model (XGBoost Only, on PCA data):")
    print(f"   F1 Score: {f1_base:.6f} | Accuracy: {acc_base:.6f}")
    print(f"\n Stacking Ensemble (RF+KNN+LogReg -> XGB, on PCA data):")
    print(f"   F1 Score: {f1_stack:.6f} | Accuracy: {acc_stack:.6f}")
    print("="*70)
    
    end_total_time = time.time()
    print(f"\n--- Total Benchmark Runtime: {end_total_time - start_total_time:.2f}s ---")