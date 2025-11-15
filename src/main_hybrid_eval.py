"""
Phase 3 – Hybrid Verification
Author: Khaja Mohammed

Uses custom feature pipeline (PCA + HOG + Directional + Zonal)
but sklearn implementations of models for speed parity check.
"""

import time, numpy as np, matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from xgboost import XGBClassifier

# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
def load_data(train_csv, val_csv):
    print(f"Loading data from {train_csv} and {val_csv} ...")
    tr = np.loadtxt(train_csv, delimiter=',', skiprows=1)
    va = np.loadtxt(val_csv, delimiter=',', skiprows=1)
    if tr.shape[1] > 785: tr = tr[:, :785]
    if va.shape[1] > 785: va = va[:, :785]
    y_tr, X_tr = tr[:, 0].astype(int), tr[:, 1:]
    y_va, X_va = va[:, 0].astype(int), va[:, 1:]
    print("Loaded shapes:", X_tr.shape, y_tr.shape, X_va.shape, y_va.shape)
    return X_tr, y_tr, X_va, y_va

# -------------------------------------------------------------------
# Feature extractors (same as your project)
# -------------------------------------------------------------------
def compute_hog(img, cell_size=7, num_bins=8):
    img = img.reshape(28,28).astype(np.float32)
    gx, gy = ndimage.sobel(img,1), ndimage.sobel(img,0)
    mag, ori = np.hypot(gx,gy), np.arctan2(gy,gx)
    ori = (ori + np.pi)*(num_bins/(2*np.pi)); ori = np.clip(ori,0,num_bins-1)
    out=[]
    for cy in range(4):
        for cx in range(4):
            m = mag[cy*cell_size:(cy+1)*cell_size, cx*cell_size:(cx+1)*cell_size]
            o = ori[cy*cell_size:(cy+1)*cell_size, cx*cell_size:(cx+1)*cell_size]
            hist=np.zeros(num_bins)
            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    hist[int(o[i,j])] += m[i,j]
            out.extend(hist)
    return np.array(out)

def extract_hog_features(X): return np.array([compute_hog(x) for x in X])

def directional_energy(img):
    gx, gy = ndimage.sobel(img,1), ndimage.sobel(img,0)
    g45, g135 = gx+gy, gx-gy
    return np.array([np.sum(np.abs(gx)), np.sum(np.abs(gy)),
                     np.sum(np.abs(g45)), np.sum(np.abs(g135))])
def extract_directional_features(X):
    return np.array([directional_energy(x.reshape(28,28)) for x in X])

def zonal_features(X, grid=(4,4)):
    gx, gy = grid; out=[]
    for img in X:
        img = img.reshape(28,28)
        sx, sy = 28//gx, 28//gy
        blocks=[img[i*sx:(i+1)*sx,j*sy:(j+1)*sy].sum()
                for i in range(gx) for j in range(gy)]
        rs, cs = img.sum(1), img.sum(0)
        out.append(blocks+[rs.mean(),rs.std(),cs.mean(),cs.std()])
    return np.array(out)

# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------
if __name__ == "__main__":
    TRAIN = os.path.join("data", "MNIST_train.csv")
    VAL = os.path.join("data", "MNIST_validation.csv")

    t0_total = time.time()
    X_train, y_train, X_val, y_val = load_data(TRAIN, VAL)

    # --- Features ---------------------------------------------------
    print("\nExtracting features...")
    hog_tr, hog_va = extract_hog_features(X_train), extract_hog_features(X_val)
    dir_tr, dir_va = extract_directional_features(X_train), extract_directional_features(X_val)
    zon_tr, zon_va = zonal_features(X_train), zonal_features(X_val)

    # Scale non-PCA groups individually
    sc_hog, sc_dir, sc_zon = StandardScaler(), StandardScaler(), StandardScaler()
    hog_tr, hog_va = sc_hog.fit_transform(hog_tr), sc_hog.transform(hog_va)
    dir_tr, dir_va = sc_dir.fit_transform(dir_tr), sc_dir.transform(dir_va)
    zon_tr, zon_va = sc_zon.fit_transform(zon_tr), sc_zon.transform(zon_va)

    # PCA on pixels (retain 95 % variance)
    print("\nPCA on raw pixels...")
    sc_pix = StandardScaler()
    Xtr_s, Xva_s = sc_pix.fit_transform(X_train), sc_pix.transform(X_val)
    pca = PCA(n_components=0.95, svd_solver="full")
    Xtr_pca, Xva_pca = pca.fit_transform(Xtr_s), pca.transform(Xva_s)
    print("PCA components:", pca.n_components_)

    # Merge all feature families
    Xtr_f = np.hstack([Xtr_pca, hog_tr, dir_tr, zon_tr])
    Xva_f = np.hstack([Xva_pca, hog_va, dir_va, zon_va])
    Xtr_f = StandardScaler().fit_transform(Xtr_f)
    Xva_f = StandardScaler().fit_transform(Xva_f)
    print("Final feature shape:", Xtr_f.shape)

    # --- Base learners ---------------------------------------------
    print("\nTraining base learners...")
    rf = RandomForestClassifier(
        n_estimators=250, max_depth=None, max_features='sqrt',
        min_samples_leaf=1, random_state=42, n_jobs=-1)
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    lr  = LogisticRegression(multi_class='multinomial',
                             solver='lbfgs', max_iter=250, random_state=42)
    rf.fit(Xtr_f, y_train); knn.fit(Xtr_f, y_train); lr.fit(Xtr_f, y_train)

    # --- Meta learner (XGB) ----------------------------------------
    meta_xgb = XGBClassifier(
        n_estimators=250, learning_rate=0.05, max_depth=4,
        subsample=0.9, colsample_bytree=0.8,
        objective='multi:softprob', num_class=10,
        eval_metric='mlogloss', random_state=42, n_jobs=-1)

    stack = StackingClassifier(
        estimators=[('rf',rf),('knn',knn),('lr',lr)],
        final_estimator=meta_xgb,
        cv=5, n_jobs=-1, passthrough=True, stack_method='predict_proba')

    print("Training stacking ensemble...")
    stack.fit(Xtr_f, y_train)

    # --- Calibration -----------------------------------------------
    print("Calibrating probabilities...")
    cal = CalibratedClassifierCV(stack, method='isotonic', cv=3)
    cal.fit(Xtr_f, y_train)

    # --- Specialists ------------------------------------------------
    print("Training specialists...")
    spec_clf = LogisticRegression(solver='lbfgs', max_iter=300)
    def train_spec(X, y, pos, pair=False):
        if pair:
            mask=np.isin(y,pos); Xs, ys=X[mask],y[mask]; ys=(ys==pos[0]).astype(int)
        else:
            Xs, ys=X, np.isin(y,pos).astype(int)
        c=clone(spec_clf); c.fit(Xs,ys); return c
    s3=train_spec(Xtr_f,y_train,[3]); s7=train_spec(Xtr_f,y_train,[7])
    s9=train_spec(Xtr_f,y_train,[9]); s35=train_spec(Xtr_f,y_train,[3,5],True)
    s79=train_spec(Xtr_f,y_train,[7,9],True)

    # --- Evaluation -------------------------------------------------
    print("\nEvaluating...")
    proba = cal.predict_proba(Xva_f)
    preds = np.argmax(proba,1)
    def apply_specs(preds, proba, Xv):
        p3,p7,p9=[s.predict_proba(Xv)[:,1] for s in (s3,s7,s9)]
        p35,s79p=[s35.predict_proba(Xv)[:,1], s79.predict_proba(Xv)[:,1]]
        new=preds.copy()
        for i,pred in enumerate(preds):
            p=proba[i]
            if pred in (3,5) and p[pred]<0.7:
                new[i]=3 if p35[i]>=0.5 else 5; continue
            if pred in (7,9) and p[pred]<0.7:
                new[i]=7 if s79p[i]>=0.5 else 9; continue
            if pred==3 and p[3]<0.7 and p3[i]>=0.5: new[i]=3
            if pred==7 and p[7]<0.7 and p7[i]>=0.5: new[i]=7
            if pred==9 and p[9]<0.7 and p9[i]>=0.5: new[i]=9
        return new
    final_preds=apply_specs(preds,proba,Xva_f)

    # --- Results ----------------------------------------------------
    print("\n--- FINAL EVALUATION ---")
    print("Accuracy:", accuracy_score(y_val, final_preds))
    print("Weighted F1:", f1_score(y_val, final_preds, average='weighted'))
    print(classification_report(y_val, final_preds, digits=4))

    t1_total = time.time()
    print(f"Total runtime: {t1_total - t0_total:.2f}s")
