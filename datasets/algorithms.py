# ===============================================
# algorithms.py — FAST classical models (from scratch, single-core)
# ===============================================
import numpy as np
from collections import Counter

# ----------------------------
# Utils
# ----------------------------
def _rng(seed):
    return np.random.RandomState(None if seed is None else int(seed))

def _onehot(y, classes):
    y_one = np.zeros((len(y), len(classes)), dtype=float)
    for i, c in enumerate(classes):
        y_one[:, i] = (y == c).astype(float)
    return y_one

def _gini_from_counts(counts):
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts / n
    return 1.0 - np.sum(p * p)

# ----------------------------
# Logistic Regression (multiclass softmax)
# ----------------------------
class LogisticRegressionCustom:
    def __init__(self, lr=0.3, n_iter=400, reg=1e-4, random_state=None):
        self.lr = lr
        self.n_iter = n_iter
        self.reg = reg
        self.random_state = random_state

    @staticmethod
    def _softmax(z):
        z = z - np.max(z, axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / (np.sum(ez, axis=1, keepdims=True) + 1e-12)

    def fit(self, X, y):
        n, d = X.shape
        self.classes_ = np.unique(y)
        C = len(self.classes_)
        y_one = _onehot(y, self.classes_)
        rng = _rng(self.random_state)
        self.W = 0.01 * rng.randn(d, C)
        self.b = np.zeros((1, C))

        for _ in range(self.n_iter):
            S = X @ self.W + self.b
            P = self._softmax(S)
            gradW = (X.T @ (P - y_one)) / n + self.reg * self.W
            gradb = np.mean(P - y_one, axis=0, keepdims=True)
            self.W -= self.lr * gradW
            self.b -= self.lr * gradb
        return self

    def predict_proba(self, X):
        S = X @ self.W + self.b
        return self._softmax(S)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

# ----------------------------
# KNN (cosine, fast single-core)
# ----------------------------
class KNNCustom:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        self.y = y
        return self

    def predict(self, X):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        sims = Xn @ self.X.T
        idx = np.argpartition(-sims, self.k-1, axis=1)[:, :self.k]
        preds = []
        for row in idx:
            labels = self.y[row]
            preds.append(Counter(labels).most_common(1)[0][0])
        return np.array(preds)

# ----------------------------
# Decision Tree (fast)
# ----------------------------
class DecisionTreeFast:
    def __init__(self,
                 max_depth=8,
                 min_samples_split=20,
                 min_samples_leaf=5,
                 max_features='sqrt',
                 n_thresholds=16,
                 random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.random_state = random_state
        self._rng = _rng(random_state)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.K_ = len(self.classes_)
        self.D_ = X.shape[1]
        self.root_ = self._grow(X, y, 0)
        return self

    def _num_features(self):
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(self.D_)))
        if isinstance(self.max_features, int):
            return max(1, min(self.D_, self.max_features))
        if isinstance(self.max_features, float):
            return max(1, int(self.D_ * self.max_features))
        return self.D_

    def _best_split(self, X, y):
        n, d = X.shape
        if n < 2 * self.min_samples_leaf:
            return None
        parent_counts = np.bincount(np.searchsorted(self.classes_, y), minlength=self.K_)
        g_parent = _gini_from_counts(parent_counts)
        best = (None, None, -1.0)

        feat_idxs = np.random.choice(X.shape[1],
                                    int(np.sqrt(X.shape[1])),
                                    replace=False)
        for f in feat_idxs:
            col = X[:, f]
            if self.n_thresholds > 0:
                qs = np.linspace(0, 100, self.n_thresholds + 2)[1:-1]
                ths = np.unique(np.percentile(col, qs))
            else:
                ths = np.unique(col)
            if ths.size == 0:
                continue
            for thr in ths:
                left = col <= thr
                nl = left.sum()
                if nl < self.min_samples_leaf or (n - nl) < self.min_samples_leaf:
                    continue
                counts_l = np.bincount(np.searchsorted(self.classes_, y[left]), minlength=self.K_)
                counts_r = parent_counts - counts_l
                g_l = _gini_from_counts(counts_l)
                g_r = _gini_from_counts(counts_r)
                gain = g_parent - (nl / n) * g_l - ((n - nl) / n) * g_r
                if gain > best[2]:
                    best = (f, thr, gain)
        if best[2] <= 1e-12:
            return None
        return best

    def _grow(self, X, y, depth):
        node = {}
        if (depth >= self.max_depth or
            X.shape[0] < self.min_samples_split or
            np.unique(y).size == 1):
            node['leaf'] = True
            node['pred'] = Counter(y).most_common(1)[0][0]
            return node
        split = self._best_split(X, y)
        if split is None:
            node['leaf'] = True
            node['pred'] = Counter(y).most_common(1)[0][0]
            return node
        f, thr, _ = split
        left = X[:, f] <= thr
        node['leaf'] = False
        node['feat'] = f
        node['thr'] = thr
        node['left'] = self._grow(X[left], y[left], depth + 1)
        node['right'] = self._grow(X[~left], y[~left], depth + 1)
        return node

    def _pred_one(self, x, node):
        while not node.get('leaf', True):
            node = node['left'] if x[node['feat']] <= node['thr'] else node['right']
        return node['pred']

    def predict(self, X):
        return np.array([self._pred_one(x, self.root_) for x in X])

# ----------------------------
# Random Forest (fast)
# ----------------------------
class RandomForestFast:
    def __init__(self,
                 n_estimators=60,
                 max_depth=8,
                 sample_ratio=0.30,
                 max_features='sqrt',
                 min_samples_split=20,
                 min_samples_leaf=5,
                 n_thresholds=16,
                 bootstrap=True,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.sample_ratio = sample_ratio
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_thresholds = n_thresholds
        self.bootstrap = bootstrap
        self.random_state = random_state
        self._rng = _rng(random_state)

    def fit(self, X, y):
        self.trees_ = []
        n = X.shape[0]
        for _ in range(self.n_estimators):
            m = max(1000, int(self.sample_ratio * n))
            idx = self._rng.choice(n, m, replace=self.bootstrap)
            Xt, yt = X[idx], y[idx]
            tree = DecisionTreeFast(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                n_thresholds=self.n_thresholds,
                random_state=self._rng.randint(0, 1_000_000)
            )
            tree.fit(Xt, yt)
            self.trees_.append(tree)
        return self

    def predict(self, X):
        all_preds = np.vstack([t.predict(X) for t in self.trees_])
        return np.array([Counter(all_preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])])

# ----------------------------
# Gradient Boosting (fast multi-class meta)
# ----------------------------
class GradientBoostingFast:
    def __init__(self,
                 n_estimators=35,
                 lr=0.15,
                 max_depth=4,
                 min_samples_split=10,
                 min_samples_leaf=5,
                 n_thresholds=8,
                 subsample=0.40,
                 early_stop=True,
                 random_state=None):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_thresholds = n_thresholds
        self.subsample = subsample
        self.early_stop = early_stop
        self.random_state = random_state
        self._rng = _rng(random_state)

    def fit(self, X, y):
        n = X.shape[0]
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        Y = _onehot(y, self.classes_)
        self.trees_ = [[] for _ in range(K)]
        P = np.full((n, K), 1.0 / K)

        for m in range(self.n_estimators):
            G = Y - P
            if self.early_stop and np.mean(np.abs(G)) < 0.05:
                break
            msize = max(2000, int(self.subsample * n))
            # idx = self._rng.choice(n, msize, replace=False)
            idx = np.random.choice(len(y), int(0.6*len(y)), replace=False)
            Xsub, Gsub = X[idx], G[idx]
            for k in range(K):
                ybin = (Gsub[:, k] > 0).astype(int)
                if ybin.min() == ybin.max():
                    continue
                tree = DecisionTreeFast(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features='sqrt',
                    n_thresholds=self.n_thresholds,
                    random_state=self._rng.randint(0, 1_000_000),
                )
                tree.fit(Xsub, ybin)
                self.trees_[k].append(tree)
            P = self._predict_proba_internal(X)
        return self

    def _predict_proba_internal(self, X):
        n = X.shape[0]
        K = len(self.classes_)
        scores = np.zeros((n, K))
        for k in range(K):
            for t in self.trees_[k]:
                scores[:, k] += self.lr * t.predict(X)  # (0/1) outputs
        scores -= np.max(scores, axis=1, keepdims=True)
        es = np.exp(scores)
        return es / (np.sum(es, axis=1, keepdims=True) + 1e-12)

    def predict_proba(self, X):
        return self._predict_proba_internal(X)

    def predict(self, X):
        return self.classes_[np.argmax(self._predict_proba_internal(X), axis=1)]

# ----------------------------
# Platt scaling (per-class sigmoid calibration)
# ----------------------------
class SigmoidCalibrator:
    """
    Calibrates multi-class probabilities with per-class Platt scaling:
    p' = sigmoid(a_k * logit(p_k) + b_k)
    """
    def __init__(self, max_iter=200, lr=0.05):
        self.max_iter = max_iter
        self.lr = lr

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return np.log(p / (1 - p))

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def fit(self, P, y, classes):
        # P: (n, K) base probabilities; y: labels
        self.classes_ = classes
        K = len(classes)
        self.A_ = np.zeros(K)
        self.B_ = np.zeros(K)
        for k, c in enumerate(classes):
            t = (y == c).astype(float)
            z = self._logit(P[:, k])
            a, b = 1.0, 0.0
            for _ in range(self.max_iter):
                s = self._sigmoid(a * z + b)
                grad_a = np.mean((s - t) * z)
                grad_b = np.mean(s - t)
                a -= self.lr * grad_a
                b -= self.lr * grad_b
            self.A_[k], self.B_[k] = a, b
        return self

    def transform(self, P):
        P = np.clip(P, 1e-12, 1 - 1e-12)
        logits = self._logit(P)
        out = np.zeros_like(P)
        for k in range(P.shape[1]):
            out[:, k] = self._sigmoid(self.A_[k] * logits[:, k] + self.B_[k])
        out = out / (np.sum(out, axis=1, keepdims=True) + 1e-12)
        return out

# ----------------------------
# Stacking with OOF (like sklearn StackingClassifier)
# ----------------------------
class StackingCV:
    def __init__(self, base_learners, meta_learner, cv=3, passthrough=True, random_state=None):
        """
        base_learners: list of (name, estimator) with fit/predict or predict_proba
        meta_learner: estimator with fit/predict or predict_proba
        """
        self.base_learners = base_learners
        self.meta_learner = meta_learner
        self.cv = cv
        self.passthrough = passthrough
        self.random_state = random_state
        self._rng = _rng(random_state)

    def _kfold(self, n):
        idx = np.arange(n)
        self._rng.shuffle(idx)
        folds = np.array_split(idx, self.cv)
        return folds

    def fit(self, X, y):
        n = X.shape[0]
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        folds = self._kfold(n)

        # out-of-fold meta features
        oof_parts = []
        for _, est in self.base_learners:
            if hasattr(est, "predict_proba"):
                oof = np.zeros((n, K))
            else:
                oof = np.zeros((n, 1))
            oof_parts.append(oof)

        for fold_id in range(self.cv):
            val_idx = folds[fold_id]
            tr_idx = np.setdiff1d(np.arange(n), val_idx)

            Xtr, ytr = X[tr_idx], y[tr_idx]
            Xva = X[val_idx]

            for k, (name, est) in enumerate(self.base_learners):
                est_fold = self._clone(est)
                est_fold.fit(Xtr, ytr)
                if hasattr(est_fold, "predict_proba"):
                    oof_parts[k][val_idx] = est_fold.predict_proba(Xva)
                else:
                    oof_parts[k][val_idx, 0] = est_fold.predict(Xva)

        meta_X = np.hstack(oof_parts)
        if self.passthrough:
            meta_X = np.hstack([meta_X, X])

        self.meta_ = self._clone(self.meta_learner).fit(meta_X, y)

        # fit full base models for test-time
        self.fitted_bases_ = []
        for name, est in self.base_learners:
            e = self._clone(est).fit(X, y)
            self.fitted_bases_.append((name, e))
        return self

    def predict_proba(self, X):
        K = len(self.classes_)
        parts = []
        for name, est in self.fitted_bases_:
            if hasattr(est, "predict_proba"):
                parts.append(est.predict_proba(X))
            else:
                parts.append(est.predict(X).reshape(-1, 1))
        meta_X = np.hstack(parts)
        if self.passthrough:
            meta_X = np.hstack([meta_X, X])

        if hasattr(self.meta_, "predict_proba"):
            proba = self.meta_.predict_proba(meta_X)
            # ensure columns map to classes_ order if meta returns ints
            if proba.shape[1] != K:
                # fallback to one-vs-rest style if needed
                out = np.zeros((X.shape[0], K))
                for i, c in enumerate(self.classes_):
                    out[:, i] = proba[:, i] if i < proba.shape[1] else 1.0 / K
                proba = out
            return proba
        else:
            preds = self.meta_.predict(meta_X)
            out = np.zeros((X.shape[0], K))
            for i, c in enumerate(self.classes_):
                out[:, i] = (preds == c).astype(float)
            out = out / (np.sum(out, axis=1, keepdims=True) + 1e-12)
            return out

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    @staticmethod
    def _clone(est):
        import copy
        return copy.deepcopy(est)
