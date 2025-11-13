import numpy as np
import time
from sklearn.preprocessing import StandardScaler
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
    
    
    # --- STAGE 2: Preprocessing (NO PCA) ---
    # We are finding the "pinnacle" score, so we use all 784 features.
    print("\n--- Preprocessing Data (StandardScaler, NO PCA) ---")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print(f"Data shape after scaling: {X_train_scaled.shape}")
    

    # --- STAGE 3: Train Base Model (XGBoost Only) ---
    # This is our baseline "Best Single Model" from the previous run.
    print(f"\n--- Training Base Model (XGBoost Only) for Comparison ---")
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
    base_model.fit(X_train_scaled, y_train)
    
    end_xgb_time = time.time()
    print(f"Base XGBoost trained. Time: {end_xgb_time - start_xgb_time:.2f}s")
    
    y_pred_base = base_model.predict(X_val_scaled)
    f1_base = f1_score(y_val, y_pred_base, average='weighted')
    acc_base = accuracy_score(y_val, y_pred_base)


    # --- STAGE 4: Define the Stacking Ensemble ---
    # This is the "pinnacle" architecture.
    print("\n--- Defining the Stacking Ensemble ---")

    # Level 0: The "Estimators" (Base Models)
    # We pick a diverse set of strong models from our benchmark.
    # Note: 'passthrough=True' means the meta-model *also* gets the original data.
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ('knn', KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        ('logreg', LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42))
    ]

    # Level 1: The "Final Estimator" (Meta-Model)
    # We use XGBoost as the meta-model to learn the complex patterns
    # from the Level 0 models' predictions.
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
    # cv=3 means it will use 3-fold cross-validation to generate
    # the "out-of-fold" predictions for the meta-model, preventing leaks.
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=3,  # Use 3 folds for speed. Can increase to 5 for more robustness.
        n_jobs=-1,
        passthrough=True # The meta-model gets both predictions AND original data
    )

    # --- STAGE 5: Train the Stacking Ensemble ---
    print(f"\n--- Training Stacking Ensemble (This will take a few minutes...) ---")
    start_stack_time = time.time()
    
    stacking_model.fit(X_train_scaled, y_train)
    
    end_stack_time = time.time()
    print(f"Stacking Ensemble trained. Time: {end_stack_time - start_stack_time:.2f}s")

    
    # --- STAGE 6: Evaluate Ensemble vs. Base ---
    print("\n--- Evaluating Final Ensemble ---")
    
    # Predict with the Stacking Model
    y_pred_stack = stacking_model.predict(X_val_scaled)
    
    # Specialist Ensemble
    f1_stack = f1_score(y_val, y_pred_stack, average='weighted')
    acc_stack = accuracy_score(y_val, y_pred_stack)
    
    print("\n" + "="*45)
    print(" --- PINNACLE BENCHMARK RESULTS ---")
    print("="*45)
    print(f" Base Model (XGBoost Only):")
    print(f"   F1 Score: {f1_base:.6f} | Accuracy: {acc_base:.6f}")
    print(f"\n Stacking Ensemble (RF+KNN+LogReg -> XGB):")
    print(f"   F1 Score: {f1_stack:.6f} | Accuracy: {acc_stack:.6f}")
    print("="*45)
    
    end_total_time = time.time()
    print(f"\n--- Total Benchmark Runtime: {end_total_time - start_total_time:.2f}s ---")