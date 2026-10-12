"""
Decision Tree Comparison: ID3 vs C4.5 vs CART
Dataset : Titanic (survived = 1 / 0)

Run:  pip install -r requirements.txt
      python main.py
Output: graphs + summary in the  results/  folder
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

os.makedirs("results", exist_ok=True)
SEED = 42
BINS = 10          # number of groups for age and fare
COLORS = {"ID3": "#1f3a5f", "C4.5": "#2e86c1", "CART": "#e67e22"}


# ----------------------------------------------------------------------
# 1. DATA  (put a local titanic.csv / Kaggle train.csv here to use it)
# ----------------------------------------------------------------------
LOCAL = "titanic.csv"
URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(LOCAL if os.path.exists(LOCAL) else URL)
df.columns = df.columns.str.lower()

df = df[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]].copy()
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# ID3 / C4.5 (here) need categories, so numbers are cut into BINS groups.
# The SAME data goes to all three algorithms, so the comparison is fair.
df["age"] = pd.qcut(df["age"], BINS, duplicates="drop").astype(str)
df["fare"] = pd.qcut(df["fare"], BINS, duplicates="drop").astype(str)
df = df.astype(str).assign(survived=df["survived"].astype(int))

X = df.drop(columns="survived").values
y = df["survived"].values
print("Rows:", len(df), "| Features:", X.shape[1])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


# ----------------------------------------------------------------------
# 2. ID3 and C4.5 written from scratch
# ----------------------------------------------------------------------
def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return -(p * np.log2(p)).sum()


def split_scores(x, y):
    """Information Gain and Gain Ratio when splitting on column x."""
    n, remainder, split_info = len(y), 0.0, 0.0
    for v in np.unique(x):
        mask = x == v
        w = mask.sum() / n
        remainder += w * entropy(y[mask])
        split_info -= w * np.log2(w)
    gain = entropy(y) - remainder
    ratio = gain / split_info if split_info > 0 else 0
    return gain, ratio


class DecisionTree:
    def __init__(self, method="id3", max_depth=None):
        self.method, self.max_depth = method, max_depth

    def fit(self, X, y):
        self.root = self._build(X, y, list(range(X.shape[1])), 0)
        return self

    def _build(self, X, y, attrs, depth):
        labels, counts = np.unique(y, return_counts=True)
        majority = labels[counts.argmax()]
        if len(labels) == 1 or not attrs or depth == self.max_depth:
            return {"leaf": majority}

        scores = [split_scores(X[:, a], y) for a in attrs]
        gains = np.array([s[0] for s in scores])
        ratios = np.array([s[1] for s in scores])
        if self.method == "id3":                      # ID3  -> highest Information Gain
            best = gains.argmax()
        else:                                         # C4.5 -> highest Gain Ratio
            ratios = np.where(gains >= gains.mean(), ratios, -1)   # (among above-average gains)
            best = ratios.argmax()
        if gains[best] <= 1e-9:
            return {"leaf": majority}

        a = attrs[best]
        rest = [x for x in attrs if x != a]
        kids = {}
        for v in np.unique(X[:, a]):
            m = X[:, a] == v
            kids[v] = self._build(X[m], y[m], rest, depth + 1)
        return {"attr": a, "kids": kids, "default": majority}

    def predict(self, X):
        return np.array([self._one(self.root, row) for row in X])

    def _one(self, node, row):
        while "leaf" not in node:
            node = node["kids"].get(row[node["attr"]], {"leaf": node["default"]})
        return node["leaf"]

    def depth(self, node=None):
        node = node or self.root
        return 0 if "leaf" in node else 1 + max(self.depth(k) for k in node["kids"].values())

    def n_nodes(self, node=None):
        node = node or self.root
        return 1 if "leaf" in node else 1 + sum(self.n_nodes(k) for k in node["kids"].values())


# ----------------------------------------------------------------------
# 3. CART (scikit-learn, Gini index, binary splits)
# ----------------------------------------------------------------------
class CART:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def _encode(self, X):
        return pd.get_dummies(pd.DataFrame(X)).astype(int)

    def fit(self, X, y):
        enc = self._encode(X)
        self.cols = enc.columns
        self.model = DecisionTreeClassifier(criterion="gini", max_depth=self.max_depth,
                                            random_state=SEED).fit(enc, y)
        return self

    def predict(self, X):
        return self.model.predict(self._encode(X).reindex(columns=self.cols, fill_value=0))

    def depth(self):
        return self.model.get_depth()

    def n_nodes(self):
        return self.model.tree_.node_count


def make(name, max_depth=None):
    if name == "CART":
        return CART(max_depth)
    return DecisionTree("id3" if name == "ID3" else "c45", max_depth)


# ----------------------------------------------------------------------
# 4. TRAIN + EVALUATE
# ----------------------------------------------------------------------
names = ["ID3", "C4.5", "CART"]
rows, preds = {}, {}
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
cv_scores = {}

for n in names:
    t0 = time.perf_counter()
    model = make(n).fit(X_tr, y_tr)
    train_time = time.perf_counter() - t0
    p = model.predict(X_te)
    preds[n] = p
    rows[n] = {
        "Accuracy": accuracy_score(y_te, p),
        "Precision": precision_score(y_te, p),
        "Recall": recall_score(y_te, p),
        "F1": f1_score(y_te, p),
        "Train Acc": accuracy_score(y_tr, model.predict(X_tr)),
        "Nodes": model.n_nodes(),
        "Depth": model.depth(),
        "Time (s)": train_time,
    }
    cv_scores[n] = [accuracy_score(y[te], make(n).fit(X[tr], y[tr]).predict(X[te]))
                    for tr, te in skf.split(X, y)]
    rows[n]["CV Acc"] = np.mean(cv_scores[n])
    rows[n]["CV Std"] = np.std(cv_scores[n])

res = pd.DataFrame(rows).T
res["Overfit Gap"] = res["Train Acc"] - res["Accuracy"]      # train acc - test acc
res.round(4).to_csv("results/results.csv")
print("\n", res.round(3).to_string())

# accuracy at different tree depths
depths = range(1, 11)
dep_test = {n: [] for n in names}
dep_train = {n: [] for n in names}
for d in depths:
    for n in names:
        m = make(n, d).fit(X_tr, y_tr)
        dep_test[n].append(accuracy_score(y_te, m.predict(X_te)))
        dep_train[n].append(accuracy_score(y_tr, m.predict(X_tr)))


# ----------------------------------------------------------------------
# 5. GRAPHS
# ----------------------------------------------------------------------
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 11})
col = [COLORS[n] for n in names]

# 5a. metrics
metrics = ["Accuracy", "Precision", "Recall", "F1"]
fig, ax = plt.subplots(figsize=(10, 5.5))
w = 0.26
for i, n in enumerate(names):
    vals = [res.loc[n, m] for m in metrics]
    bars = ax.bar(np.arange(4) + (i - 1) * w, vals, w, label=n, color=COLORS[n])
    ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=2)
ax.set_xticks(range(4)); ax.set_xticklabels(metrics)
ax.set_ylim(0.5, 1.0); ax.set_ylabel("Score"); ax.legend()
ax.set_title("Test-set performance")
plt.tight_layout(); plt.savefig("results/1_metrics.png", dpi=200); plt.close()

# 5b. cross-validation
fig, ax = plt.subplots(figsize=(7, 5))
means = [res.loc[n, "CV Acc"] for n in names]
stds = [res.loc[n, "CV Std"] for n in names]
bars = ax.bar(names, means, yerr=stds, capsize=6, color=col)
ax.bar_label(bars, fmt="%.3f", padding=14)
for i, n in enumerate(names):
    ax.scatter([i] * 5, cv_scores[n], color="black", s=14, zorder=3)
ax.set_ylim(0.6, 0.9); ax.set_ylabel("Accuracy")
ax.set_title("5-fold cross-validation accuracy (dots = each fold)")
plt.tight_layout(); plt.savefig("results/2_cross_validation.png", dpi=200); plt.close()

# 5c. accuracy vs depth
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, data, title in [(axes[0], dep_train, "Training accuracy"), (axes[1], dep_test, "Test accuracy")]:
    for n in names:
        ax.plot(list(depths), data[n], marker="o", label=n, color=COLORS[n])
    ax.set_title(title); ax.set_xlabel("Max tree depth"); ax.legend()
axes[0].set_ylabel("Accuracy")
plt.suptitle("Accuracy as the tree grows deeper")
plt.tight_layout(); plt.savefig("results/3_accuracy_vs_depth.png", dpi=200); plt.close()

# 5d. confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, n in zip(axes, names):
    cm = confusion_matrix(y_te, preds[n])
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", fontsize=14,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Died", "Survived"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Died", "Survived"])
    ax.set_title(n); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout(); plt.savefig("results/4_confusion_matrices.png", dpi=200); plt.close()

# 5e. size, overfitting, speed
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, c, t in zip(axes, ["Nodes", "Overfit Gap", "Time (s)"],
                    ["Tree size (nodes)", "Overfitting gap (train acc - test acc)", "Training time (s)"]):
    bars = ax.bar(names, res[c], color=col)
    ax.bar_label(bars, fmt="%.3g", padding=2)
    ax.set_title(t); ax.margins(y=0.15)
plt.tight_layout(); plt.savefig("results/5_complexity.png", dpi=200); plt.close()


# ----------------------------------------------------------------------
# 6. SUMMARY - which one is best and why
# ----------------------------------------------------------------------
gap = res["CV Acc"].max() - res["CV Acc"].min()
noise = res["CV Std"].max()
if gap < noise:
    verdict = (f"NOTE: CV accuracy differs by only {gap:.3f} (fold-to-fold std is {noise:.3f}), so the three "
               "algorithms perform about the same on this data. The clear difference is the tree shape "
               "(ID3/C4.5 split many ways -> shallow tree, CART splits in two -> deep tree).")
else:
    verdict = f"NOTE: CV accuracy differs by {gap:.3f}, larger than the fold-to-fold std ({noise:.3f}), so the difference is real."
best = res.sort_values(["CV Acc", "Nodes"], ascending=[False, True]).index[0]
lines = [
    "RESULT SUMMARY",
    "-" * 55,
    f"Best overall (highest CV accuracy): {best}  ({res.loc[best, 'CV Acc']:.3f})",
    f"Best test F1-score               : {res['F1'].idxmax()}  ({res['F1'].max():.3f})",
    f"Smallest / simplest tree         : {res['Nodes'].idxmin()}  ({int(res['Nodes'].min())} nodes)",
    f"Most overfitted (biggest gap)    : {res['Overfit Gap'].idxmax()}  ({res['Overfit Gap'].max():.3f})",
    f"Least overfitted (smallest gap)  : {res['Overfit Gap'].idxmin()}  ({res['Overfit Gap'].min():.3f})",
    f"Fastest to train                 : {res['Time (s)'].idxmin()}",
    f"Tree depth                       : " + ", ".join(f"{n}={int(res.loc[n, 'Depth'])}" for n in names),
    "",
    verdict,
    "",
    res.round(3).to_string(),
]
open("results/summary.txt", "w").write("\n".join(lines))
print("\n" + "\n".join(lines[:8]))
print("\nGraphs saved in the results/ folder")
