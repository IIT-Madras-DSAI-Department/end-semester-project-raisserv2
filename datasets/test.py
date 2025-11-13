import numpy as np
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score, classification_report

# --- Model Imports ---
# We'll import the sklearn equivalents of the models you've built
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Note:
# - Ridge/Lasso are for regression. LogisticRegression is the classification equivalent.
# - XGBoost is a library, so we won't use it here (per project rules).
# - K-Means/DBSCAN are clustering (unsupervised) and are not direct classifiers.

def load_data(train_csv, val_csv):
    """
    Loads the MNIST training and validation datasets.
    """
    print(f"Loading data from {train_csv} and {val_csv}...")
    
    # Load training data
    try:
        train_data = np.loadtxt(train_csv, delimiter=',', skiprows=1)
    except FileNotFoundError:
        print(f"Error: Training file not found at {train_csv}")
        print("Please make sure 'MNIST_train.csv' is in the same directory.")
        return None, None, None, None
        
    X_train = train_data[:, 1:]
    y_train = train_data[:, 0].astype(int)
    
    # Load validation data
    try:
        val_data = np.loadtxt(val_csv, delimiter=',', skiprows=1)
    except FileNotFoundError:
        print(f"Error: Validation file not found at {val_csv}")
        print("Please make sure 'MNIST_validation.csv' is in the same directory.")
        return None, None, None, None

    X_val = val_data[:, 1:]
    y_val = val_data[:, 0].astype(int)
    
    print("Data loaded successfully.")
    return X_train, y_train, X_val, y_val

def run_benchmarks():
    """
    Loads data, defines models, and runs the benchmarking pipeline.
    """
    
    # --- 1. Load Data ---
    X_train, y_train, X_val, y_val = load_data('MNIST_train.csv', 'MNIST_validation.csv')
    if X_train is None:
        return # Stop execution if files weren't found

    print(f"\nData Shapes:\n  X_train: {X_train.shape}\n  y_train: {y_train.shape}\n  X_val:   {X_val.shape}\n  y_val:   {y_val.shape}")

    # --- 2. Define Models ---
    # We create a dictionary of all the models we want to test.
    # We use 'random_state=42' for reproducibility.
    # We set 'n_jobs=-1' on Random Forest to use all CPU cores.
    models_to_test = {
        "Logistic Regression (OvR)": LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42),
        "K-Nearest Neighbors (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Linear SVM (LinearSVC)": LinearSVC(max_iter=2000, random_state=42, dual=True), # dual=True is default, but good to be explicit
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest (100 trees)": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    
    # --- 3. Run Pipeline for Each Model ---
    print("\n--- Starting Model Benchmarks (with Scaling + PCA) ---")
    
    results = {}

    for model_name, model_obj in models_to_test.items():
        print(f"\nTesting Model: {model_name}")
        
        # Create the pipeline:
        # 1. StandardScaler: Scales data (mean 0, std 1). Better for PCA/Linear models.
        # 2. PCA: Reduces dimensions. n_components=0.95 means "keep components
        #    that explain 95% of the variance". This is key for speed.
        # 3. model: The classifier itself.
        pipeline = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)), 
            ('model', model_obj)
        ])
        
        # --- Train ---
        start_train = time.time()
        pipeline.fit(X_train, y_train)
        end_train = time.time()
        train_time = end_train - start_train
        
        # --- Predict ---
        start_pred = time.time()
        y_pred = pipeline.predict(X_val)
        end_pred = time.time()
        pred_time = end_pred - start_pred
        
        # --- Evaluate ---
        # The project requires maximizing the F1 score [cite: 27]
        f1 = f1_score(y_val, y_pred, average='weighted')
        acc = accuracy_score(y_val, y_pred)
        
        # Store and print results
        results[model_name] = {'f1': f1, 'acc': acc, 'train_time': train_time}
        
        print(f"  Training Time: {train_time:.3f}s")
        print(f"  Prediction Time: {pred_time:.3f}s")
        print(f"  Validation F1 Score (Weighted): {f1:.4f}")
        print(f"  Validation Accuracy: {acc:.4f}")
        
        # Optional: Uncomment to see per-class details
        # print("\nClassification Report:")
        # print(classification_report(y_val, y_pred))

    # --- 4. Final Summary ---
    print("\n\n--- Benchmark Summary (Sorted by F1 Score) ---")
    
    # Sort results by F1 score, descending
    sorted_results = sorted(results.items(), key=lambda item: item[1]['f1'], reverse=True)
    
    print(f"{'Model':<30} | {'F1 Score':<10} | {'Accuracy':<10} | {'Train Time (s)':<15}")
    print("-" * 70)
    
    for model_name, metrics in sorted_results:
        print(f"{model_name:<30} | {metrics['f1']:<10.4f} | {metrics['acc']:<10.4f} | {metrics['train_time']:<15.3f}")

if __name__ == "__main__":
    run_benchmarks()