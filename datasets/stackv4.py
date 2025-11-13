#xgb as a base was bad for obvious reaosns 

# lets try some kmeans juice

import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans # Import KMeans
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
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
    
    # --- STAGE 1: Load Data ---
    start_total_time = time.time()
    
    data = load_data(TRAIN_FILE, VAL_FILE)
    if data is None:
        exit()
        
    X_train, y_train, X_val, y_val = data
    
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Val shapes:   X={X_val.shape}, y={y_val.shape}")
    
    
    # --- STAGE 2: Preprocessing (StandardScaler + PCA) ---
    # This is the *realistic* pipeline for your final project.
    # We run PCA ONCE at the start to meet the 5-min time limit.
    print("\n--- Preprocessing Data (StandardScaler + PCA) ---")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("Scaling complete. Starting PCA...")
    pca = PCA(n_components=0.95)
    X_train_processed = pca.fit_transform(X_train_scaled)
    X_val_processed = pca.transform(X_val_scaled)
    
    n_pca_features = X_train_processed.shape[1]
    print(f"PCA complete. Selected {n_pca_features} components.")
    print(f"Data shape after PCA: {X_train_processed.shape}")
    
    
    # --- STAGE 3: KMeans Feature Engineering ---
    print("\n--- Engineering new features with KMeans ---")
    n_clusters = 10 # One cluster per digit
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    print(f"Fitting KMeans(n_clusters={n_clusters}) on PCA data...")
    kmeans.fit(X_train_processed)
    
    # Use .transform() to get distances to each cluster centroid
    print("Creating cluster-distance features...")
    X_train_kmeans_feats = kmeans.transform(X_train_processed)
    X_val_kmeans_feats = kmeans.transform(X_val_processed)
    
    # Combine PCA features and KMeans features
    X_train_enhanced = np.hstack([X_train_processed, X_train_kmeans_feats])
    X_val_enhanced = np.hstack([X_val_processed, X_val_kmeans_feats])
    
    n_total_features = X_train_enhanced.shape[1]
    print(f"Feature engineering complete. New shape: {X_train_enhanced.shape}")
    print(f"   ({n_pca_features} PCA + {n_clusters} KMeans = {n_total_features} total features)")
    

    # --- STAGE 4: Train Base Model (XGBoost Only, on ENHANCED data) ---
    # This is our baseline "Best Single Model" *on enhanced data*.
    print(f"\n--- Training Base Model (XGBoost Only, on ENHANCED data) ---")
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
    # Fit on the ENHANCED (PCA + KMeans) data
    base_model.fit(X_train_enhanced, y_train)
    
    end_xgb_time = time.time()
    print(f"Base XGBoost trained. Time: {end_xgb_time - start_xgb_time:.2f}s")
    
    # Predict on the ENHANCED (PCA + KMeans) data
    y_pred_base = base_model.predict(X_val_enhanced)
    f1_base = f1_score(y_val, y_pred_base, average='weighted')
    acc_base = accuracy_score(y_val, y_pred_base)


    # --- STAGE 5: Define the Stacking Ensemble ---
    print("\n--- Defining the Stacking Ensemble (RF+KNN+LogReg -> XGB) ---")

    # Level 0: The "Estimators" (Base Models)
    # This is the RF+KNN+LogReg stack you wanted, which is very diverse.
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ('knn', KNeighborsClassifier(n_neighbors=5, n_jobs=-1)), # Added KNN
        ('logreg', LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42))
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
        passthrough=True # The meta-model gets preds + the ENHANCED data
    )

    # --- STAGE 6: Train the Stacking Ensemble ---
    print(f"\n--- Training Stacking Ensemble (on ENHANCED data) ---")
    start_stack_time = time.time()
    
    # Fit on the ENHANCED (PCA + KMeans) data
    stacking_model.fit(X_train_enhanced, y_train)
    
    end_stack_time = time.time()
    print(f"Stacking Ensemble trained. Time: {end_stack_time - start_stack_time:.2f}s")

    
    # --- STAGE 7: Evaluate Ensemble vs. Base ---
    print("\n--- Evaluating Final Ensemble ---")
    
    # Predict with the Stacking Model on ENHANCED (PCA + KMeans) data
    y_pred_stack = stacking_model.predict(X_val_enhanced)
    
    # Fit on the PROCESSED (PCA) data
    stacking_model.fit(X_train_processed, y_train)
    
    end_stack_time = time.time()
    print(f"Stacking Ensemble trained. Time: {end_stack_time - start_stack_time:.2f}s")

    
    # --- STAGE 6: Evaluate Ensemble vs. Base ---
    print("\n--- Evaluating Final Ensemble ---")
    
    # Predict with the Stacking Model on PROCESSED (PCA) data
    y_pred_stack = stacking_model.predict(X_val_processed)
    
    # Stacking Ensemble
    f1_stack = f1_score(y_val, y_pred_stack, average='weighted')
    acc_stack = accuracy_score(y_val, y_pred_stack)
    
    print("\n" + "="*60)
    print(" --- REALISTIC PINNACLE BENCHMARK (PCA + KMeans Features) ---")
    print("="*60)
    print(f" Base Model (XGBoost Only, on ENHANCED data):")
    print(f"   F1 Score: {f1_base:.6f} | Accuracy: {acc_base:.6f}")
    print(f"\n Stacking Ensemble (RF+KNN+LogReg -> XGB, on ENHANCED data):")
    print(f"   F1 Score: {f1_stack:.6f} | Accuracy: {acc_stack:.6f}")
    print("="*60)
    
    end_total_time = time.time()
    print(f"\n--- Total Benchmark Runtime: {end_total_time - start_total_time:.2f}s ---")