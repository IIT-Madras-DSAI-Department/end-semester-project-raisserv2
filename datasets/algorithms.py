# ===============================================
# algorithms.py — Classical models implemented from scratch
# ===============================================

import numpy as np
from collections import Counter


# ---------- Logistic Regression (multiclass softmax) ----------
class LogisticRegressionCustom:
    def __init__(self, lr=0.1, n_iter=200, reg=1e-4):
        self.lr = lr
        self.n_iter = n_iter
        self.reg = reg

    def _softmax(self, z):
        z -= np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        classes = np.unique(y)
        self.classes_ = classes
        n_classes = len(classes)
        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros((1, n_classes))

        y_onehot = np.zeros((n_samples, n_classes))
        for i, c in enumerate(classes):
            y_onehot[:, i] = (y == c).astype(float)

        for _ in range(self.n_iter):
            scores = X @ self.W + self.b
            probs = self._softmax(scores)
            grad_W = (X.T @ (probs - y_onehot)) / n_samples + self.reg * self.W
            grad_b = np.mean(probs - y_onehot, axis=0, keepdims=True)
            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b

    def predict(self, X):
        scores = X @ self.W + self.b
        probs = self._softmax(scores)
        return self.classes_[np.argmax(probs, axis=1)]


# ---------- K-Nearest Neighbors ----------
class KNNCustom:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        preds = []
        for x in X:
            dists = np.sum((self.X_train - x)**2, axis=1)
            idx = np.argsort(dists)[:self.k]
            labels = self.y_train[idx]
            preds.append(Counter(labels).most_common(1)[0][0])
        return np.array(preds)


# ---------- Decision Tree (very simple, for RandomForest) ----------
class DecisionTreeCustom:
    def __init__(self, max_depth=8, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y, depth=0):
        self.depth = depth
        self.n_classes = len(np.unique(y))
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:
            self.label = Counter(y).most_common(1)[0][0]
            self.left = self.right = None
            self.split = None
            return
        n_features = X.shape[1]
        best_feat, best_thresh, best_gain = None, None, 0
        parent_entropy = self._entropy(y)
        for feat in range(n_features):
            vals = np.unique(X[:, feat])
            for t in vals:
                left_mask = X[:, feat] <= t
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                gain = parent_entropy - (
                    np.sum(left_mask) / len(y) * self._entropy(y[left_mask]) +
                    np.sum(right_mask) / len(y) * self._entropy(y[right_mask])
                )
                if gain > best_gain:
                    best_feat, best_thresh, best_gain = feat, t, gain
        if best_feat is None:
            self.label = Counter(y).most_common(1)[0][0]
            self.left = self.right = None
            return
        self.split = (best_feat, best_thresh)
        left_mask = X[:, best_feat] <= best_thresh
        self.left = DecisionTreeCustom(self.max_depth, self.min_samples_split)
        self.left.fit(X[left_mask], y[left_mask], depth+1)
        self.right = DecisionTreeCustom(self.max_depth, self.min_samples_split)
        self.right.fit(X[~left_mask], y[~left_mask], depth+1)

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return -np.sum(p * np.log2(p + 1e-9))

    def predict_one(self, x):
        if self.split is None:
            return self.label
        feat, thresh = self.split
        branch = self.left if x[feat] <= thresh else self.right
        if branch is None:
            return self.label
        return branch.predict_one(x)

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])


# ---------- Random Forest ----------
class RandomForestCustom:
    def __init__(self, n_estimators=30, max_depth=6, sample_ratio=0.15, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.sample_ratio = sample_ratio
        self.max_features = max_features  # None defaults to sqrt(n_features)

    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape
        max_feats = self.max_features or int(np.sqrt(n_features))

        for _ in range(self.n_estimators):
            idx = np.random.choice(n_samples, int(self.sample_ratio * n_samples), replace=True)
            X_sub, y_sub = X[idx], y[idx]
            features = np.random.choice(n_features, max_feats, replace=False)
            X_feat = X_sub[:, features]
            tree = DecisionTreeCustom(max_depth=self.max_depth)
            tree.features_used = features
            tree.fit(X_feat, y_sub)
            self.trees.append(tree)

    def predict(self, X):
        all_preds = []
        for tree in self.trees:
            preds = tree.predict(X[:, tree.features_used])
            all_preds.append(preds)
        all_preds = np.array(all_preds)
        final_preds = [Counter(all_preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]
        return np.array(final_preds)



# ---------- Gradient Boosting (simplified XGB) ----------
class GradientBoostingCustom:
    def __init__(self, n_estimators=20, lr=0.2, max_depth=3, subsample=0.3, early_stop=True):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.subsample = subsample
        self.early_stop = early_stop

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        n_classes = len(classes)
        n_samples = X.shape[0]
        y_enc = np.zeros((n_samples, n_classes))
        for i, c in enumerate(classes):
            y_enc[:, i] = (y == c).astype(float)

        prob = np.full_like(y_enc, 1 / n_classes)
        self.trees = [[] for _ in range(n_classes)]

        for round in range(self.n_estimators):
            grad = y_enc - prob

            # early stop if residuals are small
            if self.early_stop and np.mean(np.abs(grad)) < 0.05:
                print(f"Early stopping at round {round}")
                break

            # train trees on a random subset
            idx = np.random.choice(n_samples, int(self.subsample * n_samples), replace=False)
            X_sub, grad_sub = X[idx], grad[idx]

            for k in range(n_classes):
                tree = DecisionTreeCustom(max_depth=self.max_depth)
                y_bin = (grad_sub[:, k] > 0).astype(int)
                if len(np.unique(y_bin)) < 2:
                    continue
                tree.fit(X_sub, y_bin)
                self.trees[k].append(tree)

            # update probabilities
            prob = self._predict_proba(X)

    def _predict_proba(self, X):
        scores = np.zeros((X.shape[0], len(self.classes_)))
        for k in range(len(self.classes_)):
            for t in self.trees[k]:
                scores[:, k] += self.lr * np.array(t.predict(X))
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def predict(self, X):
        prob = self._predict_proba(X)
        return self.classes_[np.argmax(prob, axis=1)]

