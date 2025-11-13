#included xgb maybe?

import numpy as np
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score

# --- Model Imports ---
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Try to import XGBoost, but don't fail if it's not installed
try:
    from xgboost import XGBClassifier
    XGB_INSTALLED = True
except ImportError:
    XGB_INSTALLED = False
    print("XGBoost not found. To benchmark it, run: pip install xgboost")
    print("-" * 30)

def load_data(train_csv, val_csv):
    """
    Loads the MNIST training and validation datasets.
    """
    print(f"Loading data from {train_csv} and {val_csv}...")
    
    try:
        train_data = np.loadtxt(train_csv, delimiter=',', skiprows=1)
    except FileNotFoundError:
        print(f"Error: Training file not found at {train_csv}")
        return None, None, None, None
        
    X_train = train_data[:, 1:]
    y_train = train_data[:, 0].astype(int)
    
    try:
        val_data = np.loadtxt(val_csv, delimiter=',', skiprows=1)
    except FileNotFoundError:
        print(f"Error: Validation file not found at {val_csv}")
        return None, None, None, None

    X_val = val_data[:, 1:]
    y_val = val_data[:, 0].astype(int)
    
    print("Data loaded successfully.")
    return X_train, y_train, X_val, y_val

def run_pipeline(models_to_test, X_train, y_train, X_val, y_val, use_pca=True):
    """
    Helper function to run the benchmark pipeline.
    """
    results = {}
    
    for model_name, model_obj in models_to_test.items():
        print(f"  Testing Model: {model_name}")
        
        # Define the pipeline steps
        steps = [('scaler', StandardScaler())]
        if use_pca:
            # PCA(0.95) keeps ~87 components on this dataset
            steps.append(('pca', PCA(n_components=0.95))) 
        
        steps.append(('model', model_obj))
        
        pipeline = Pipeline(steps=steps)
        
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
        f1 = f1_score(y_val, y_pred, average='weighted')
        acc = accuracy_score(y_val, y_pred)
        
        results[model_name] = {'f1': f1, 'acc': acc, 'train_time': train_time}
        
        print(f"    Train Time: {train_time:.3f}s | Pred Time: {pred_time:.3f}s | F1: {f1:.4f} | Acc: {acc:.4f}")

    return results

def print_summary(title, results):
    """
    Prints a formatted summary table.
    """
    print(f"\n\n--- {title} (Sorted by F1 Score) ---")
    
    sorted_results = sorted(results.items(), key=lambda item: item[1]['f1'], reverse=True)
    
    print(f"{'Model':<30} | {'F1 Score':<10} | {'Accuracy':<10} | {'Train Time (s)':<15}")
    print("-" * 70)
    
    for model_name, metrics in sorted_results:
        print(f"{model_name:<30} | {metrics['f1']:<10.4f} | {metrics['acc']:<10.4f} | {metrics['train_time']:<15.3f}")

def run_benchmarks_v2():
    """
    Loads data, defines models, and runs benchmarks WITH and WITHOUT PCA.
    """
    
    # --- 1. Load Data ---
    X_train, y_train, X_val, y_val = load_data('MNIST_train.csv', 'MNIST_validation.csv')
    if X_train is None:
        return

    #--- 2. Define Models ---
    models_to_test = {
        # "Logistic Regression (OvR)": LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42),
        "K-Nearest Neighbors (k=25)": KNeighborsClassifier(n_neighbors=1, n_jobs=-1) # Use all cores
        # "Naive Bayes (Gaussian)": GaussianNB(),
        # "Linear SVM (LinearSVC)": LinearSVC(max_iter=2000, random_state=42, dual=True), 
        # "Decision Tree": DecisionTreeClassifier(random_state=42),
        # "Random Forest (100 trees)": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    # models_to_test = {}
    # for i in range(1, 26):
    #     models_to_test[f"K-Nearest Neighbors (k={i+1}"] = KNeighborsClassifier(n_neighbors=i+1, n_jobs=-1)
    
    # if XGB_INSTALLED:
    #     models_to_test["XGBoost (100 trees)"] = XGBClassifier(
    #         n_estimators=100, m_jobs=-1, random_state=42, 
    #         objective='multi:softmax', num_class=10,
    #         eval_metric='mlogloss', use_label_encoder=False
    #     )

    # --- 3. Run Pipeline WITH PCA ---
    print("\n--- Starting Benchmarks WITH PCA (n_components=0.95) ---")
    pca_results = run_pipeline(models_to_test, X_train, y_train, X_val, y_val, use_pca=True)
    
    # --- 4. Run Pipeline WITHOUT PCA ---
    print("\n\n--- Starting Benchmarks WITHOUT PCA (Full 784 features) ---")
    # We remove models that are too slow or bad on high-dimensional data
    models_to_test_no_pca = models_to_test.copy()
    
    # Note: KNN is VERY slow on high-dimensional data (curse of dimensionality)
    # We can keep it for comparison, but expect it to be much slower.
    
    # GaussianNB assumes features are independent, which isn't true, 
    # but it's fast, so we keep it.
    
    no_pca_results = run_pipeline(models_to_test_no_pca, X_train, y_train, X_val, y_val, use_pca=False)
    
    # --- 5. Final Summaries ---
    print_summary("Benchmark Summary WITH PCA (0.95)", pca_results)
    print_summary("Benchmark Summary WITHOUT PCA (Full 784 Features)", no_pca_results)


if __name__ == "__main__":
    run_benchmarks_v2()