Lab 02: Decision Trees (ID3, CART, C4.5)
Objective
Implement ID3, CART and C4.5 from scratch, compare them on a real-life dataset, and understand why each algorithm follows the previous one.
Dataset
Titanic passengers (891 records). Target: `survived`.
Attributes: `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked` (numeric + categorical, with missing values in `age` and `embarked`).
Algorithms
Algorithm	Split measure	Split type	Key idea
ID3	Information Gain (entropy)	Multiway	Categorical data only, no pruning, biased to many-valued attributes
CART	Gini Index	Binary	Numeric thresholds, pruning, also works for regression
C4.5	Gain Ratio	Multiway / Binary	Fixes ID3 bias, handles numeric + missing values, pruning
Manual Calculation (root node, attribute `sex`)
Computed on the training set (70% split, 623 passengers). The notebook prints the same numbers.
Formulas
Entropy: H(S) = - Σ pᵢ log₂ pᵢ
Information Gain: IG = H(S) - Σ (|Sᵥ|/|S|) · H(Sᵥ)
Gini: G(S) = 1 - Σ pᵢ², Gini Decrease = G(S) - Σ (|Sᵥ|/|S|) · G(Sᵥ)
Split Info: SI = - Σ (|Sᵥ|/|S|) log₂ (|Sᵥ|/|S|), Gain Ratio = IG / SI
Step 1: Root node
623 passengers: 239 survived (p = 0.3836), 384 died (q = 0.6164)
H(root) = -(0.3836 · log₂ 0.3836 + 0.6164 · log₂ 0.6164) = 0.9606
G(root) = 1 - (0.3836² + 0.6164²) = 0.4729
Step 2: Split on `sex`
Branch	n	Survived	Died	Weight	Entropy	Gini
female	217	163	54	217/623 = 0.3483	0.8094	0.3738
male	406	76	330	406/623 = 0.6517	0.6956	0.3043
Step 3: Weighted impurity after the split
Entropy: 0.3483 · 0.8094 + 0.6517 · 0.6956 = 0.2819 + 0.4533 = 0.7352
Gini: 0.3483 · 0.3738 + 0.6517 · 0.3043 = 0.1302 + 0.1983 = 0.3285
Step 4: Score of the split
Algorithm	Score	Calculation
ID3	Information Gain = 0.2253	0.9606 - 0.7352
CART	Gini Decrease = 0.1444	0.4729 - 0.3285
C4.5	Gain Ratio = 0.2416	SI = 0.9326, so 0.2253 / 0.9326
Step 5: Choose the best attribute
Each algorithm repeats Step 2 to Step 4 for every attribute and picks the best score. `sex` wins at the root in all three algorithms.
Attribute	Information Gain
sex	0.2253
pclass	0.0814
fare (binned)	0.0660
embarked	0.0277
sibsp	0.0259
parch	0.0258
age (binned)	0.0254
Why C4.5 uses Gain Ratio (manual check with a unique ID attribute)
A useless `row_id` (unique per passenger) makes every branch pure, so its gain equals the full root entropy.
Attribute	Information Gain	Split Info	Gain Ratio
row_id	0.9606	log₂ 623 = 9.2831	0.9606 / 9.2831 = 0.1035
sex	0.2253	0.9326	0.2416
ID3 ranks `row_id` first (highest gain)
C4.5 ranks `sex` first (highest gain ratio)
Results (mean test accuracy over 10 random 70/30 splits)
Model	Mean Test Acc	Std
ID3 (full)	0.8190	0.0169
CART (full)	0.7892	0.0213
CART (depth <= 4)	0.8187	0.0193
C4.5 (full)	0.7974	0.0140
C4.5 (pruned)	0.8272	0.0134
Observations
Pruning / depth limiting improves test accuracy for both CART and C4.5
The three algorithms perform similarly on this dataset (differences are within about one standard deviation)
ID3 picks a useless unique-ID attribute as root; C4.5's gain ratio does not
Scratch CART matches scikit-learn exactly (27 nodes, test accuracy 0.7948 at depth 4)
Files
`decision_trees_id3_cart_c45.ipynb`: implementation, manual calculation, comparison charts
`data/titanic.csv`: dataset
