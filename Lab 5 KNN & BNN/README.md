# Lab: K-Nearest Neighbors (KNN) and Bayesian Belief Network (BBN)

Both algorithms are written **from scratch** (NumPy and Pandas only). No scikit-learn and no BBN library is used. Matplotlib is used only to draw graphs.

**Contents:** Files and how to run, Dataset, **Part 1 KNN** (idea, steps, results, *mathematical background*, *worked example by hand*), **Part 2 BBN** (idea, steps, results, *mathematical background*, *worked example by hand*), KNN vs BBN.

## Files in this folder

| File | What it does |
|---|---|
| `generate_dataset.py` | Creates the dataset (`patient_data.csv`, 1000 rows) |
| `patient_data.csv` | The dataset: patient details and whether the patient has diabetes |
| `knn_from_scratch.py` | Full KNN implementation and evaluation |
| `KNN_from_scratch.ipynb` | The same KNN as a Jupyter notebook (step by step, with outputs) |
| `bbn_from_scratch.py` | Full Bayesian Belief Network implementation and inference |
| `BBN_from_scratch.ipynb` | The same BBN as a Jupyter notebook (step by step, with outputs) |
| `requirements.txt` | Libraries needed |
| `*.png` | Output graphs (K vs accuracy, confusion matrices, network structure) |

## How to run

```bash
pip install -r requirements.txt
python generate_dataset.py      # optional: the CSV already exists
python knn_from_scratch.py
python bbn_from_scratch.py
```

Run the scripts from the folder that contains `patient_data.csv`. For the notebooks, put `patient_data.csv` next to the `.ipynb` file and click "Run All".

---

## Dataset

**Problem:** use a patient's details to predict whether the patient has **diabetes**.

| Column | Type | Meaning |
|---|---|---|
| `Age` | number | Age in years (20-80) |
| `BMI` | number | Body Mass Index |
| `BloodPressure` | number | Blood pressure |
| `Glucose` | number | Blood glucose level |
| `FamilyHistory` | 0/1 | Does diabetes run in the family? |
| `Smoker` | 0/1 | Does the patient smoke? |
| `Outcome` | 0/1 | **Target**: 1 = Diabetes, 0 = No diabetes |

- 1000 rows, about 30% positive (303 positive, 697 negative).
- The data is **synthetic**. It follows real medical ideas (higher glucose and BMI mean a higher chance of diabetes). `generate_dataset.py` creates it with `seed=42`, so you get the same data every time. It is synthetic on purpose: we know which variable affects which, so it is easy to understand and check the BBN structure.
- Train/test split: 80% / 20%, stratified (both parts keep the same share of positive cases).
- To use a real dataset (for example Pima Indians Diabetes), change `FEATURES` in the KNN code and `STRUCTURE` in the BBN code.

---

# Part 1: KNN (K-Nearest Neighbors)

## Idea

> "Tell me who your friends are and I will tell you who you are."

KNN is a **lazy learner**. It does not learn anything during training. It only remembers all the training data. When a new patient arrives, it finds the **K most similar** past patients and decides by their **majority**.

## How it works (step by step)

1. **Scaling (z-score):** `z = (x - mean) / std`. This is needed because KNN works on distance. Without scaling, `BloodPressure` (about 100) would overpower `Smoker` (0 or 1). The mean and std come from the **train data only**, so the model never sees the test data (this avoids data leakage).
2. **Find distances:** between the new point `x` and every training point `t`:
   - Euclidean: `d = sqrt( sum( (x_i - t_i)^2 ) )`
   - Manhattan: `d = sum( |x_i - t_i| )`
3. **Pick the K nearest:** sort the distances and take the K smallest.
4. **Vote:** the class that appears most among those K neighbors is the prediction.
   - *Weighted* option: closer neighbors get a bigger vote (`weight = 1 / distance`).
5. **Find the best K:** 5-fold cross-validation tries K = 1, 3, 5, ..., 31. If K is too small, the model memorizes noise (overfitting). If K is too large, the model becomes too smooth and misses details (underfitting). K is kept odd so votes cannot tie.

## Where things are in the code

| Item | Location |
|---|---|
| Scaling | `StandardScaler` |
| KNN model | `KNNClassifier` (`fit`, `_distances`, `predict`) |
| Best K | `cross_validate_k` |
| Metrics | `confusion_matrix`, `classification_report` |

## Results (seed = 42)

- **Best K = 9** (CV accuracy about 0.829)
- Test set (200 rows), Euclidean distance, majority vote:

| Accuracy | Precision | Recall | F1 |
|---|---|---|---|
| 0.835 | 0.818 | 0.590 | 0.686 |

Confusion matrix:

|  | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 131 (TN) | 8 (FP) |
| **Actual 1** | 25 (FN) | 36 (TP) |

Graphs: `knn_k_vs_accuracy.png`, `knn_confusion_matrix.png`

**Observation:** accuracy is good but recall is low (0.59). The model misses many diabetic patients (25 false negatives). In a medical problem a false negative is costly, so do not trust accuracy alone.

## Mathematical background (KNN)

**Notation**

```
Training set : (x_1, y_1), ..., (x_n, y_n)     x_i has d features, y_i is 0 or 1
New point    : x
K            : number of neighbors
```

### 1. Distance between two points

The general (Minkowski) distance between points a and b with d features:

```
d_p(a, b) = ( SUM_j |a_j - b_j|^p ) ^ (1/p)

p = 2  ->  Euclidean : d(a, b) = sqrt( SUM_j (a_j - b_j)^2 )
p = 1  ->  Manhattan : d(a, b) = SUM_j |a_j - b_j|
```

A distance is never negative, is 0 only when a = b, is the same in both directions, and obeys the triangle inequality. Euclidean squares each difference, so one big difference counts more. Manhattan counts every difference equally.

### 2. Scaling (z-score)

```
mean   : mu_j    = (1/n) * SUM_i x_ij
std    : sigma_j = sqrt( (1/n) * SUM_i (x_ij - mu_j)^2 )     (population std, same as NumPy)
z-score: z_ij    = (x_ij - mu_j) / sigma_j
```

After scaling, every feature has mean 0 and std 1, so every feature has the same weight in the distance. The test point is scaled with the **train** mu and sigma (never its own).

### 3. The KNN decision rule

```
N_K(x)   = the K training points with the smallest d(x, x_i)

n_c(x)   = SUM over i in N_K(x) of  1[ y_i = c ]          (votes for class c)

Majority vote : y_hat = argmax_c n_c(x)
Probability   : P(y = c | x) is estimated as n_c(x) / K

Weighted vote : w_i = 1 / d(x, x_i)
                score_c = SUM over i in N_K(x) with y_i = c of w_i
                P(y = c | x) is estimated as score_c / (score_0 + score_1)
```

For 2 classes and an odd K (plain vote), a tie is impossible.

### 4. Why it works

KNN estimates P(y | x) by averaging the labels of the nearby points. When n grows, K grows, and K/n goes to 0, the KNN error approaches the **Bayes error** (the smallest error any classifier can reach). For K = 1 the error is at most about twice the Bayes error (Cover and Hart, 1967).

### 5. Choosing K (bias and variance)

| K | Bias | Variance | Problem |
|---|---|---|---|
| Very small (1) | low | high | overfitting (follows noise) |
| Very large (K = n) | high | low | underfitting (always predicts the majority class) |

### 6. Cross-validation

Split the train data into F folds (F = 5). For each K, train on F-1 folds, test on the left-out fold, and repeat so every fold is left out once:

```
CV(K) = (1/F) * SUM over folds f of  Accuracy_f(K)
best K = argmax_K CV(K)
```

The scaler is learned again inside each fold, so the validation fold never leaks into training.

### 7. Evaluation metrics

```
                 Predicted 0    Predicted 1
Actual 0            TN             FP
Actual 1            FN             TP

Accuracy    = (TP + TN) / (TP + TN + FP + FN)
Precision   = TP / (TP + FP)
Recall      = TP / (TP + FN)           (also called sensitivity)
Specificity = TN / (TN + FP)
F1          = 2 * Precision * Recall / (Precision + Recall)  =  2TP / (2TP + FP + FN)
```

### 8. Cost

Training costs nothing (the data is only stored). One prediction needs n distances (n x d operations) plus sorting (about n log n). Memory is n x d. In many dimensions all points become almost equally far apart, so KNN gets weaker (the "curse of dimensionality").

## Worked example by hand (KNN)

To keep the arithmetic short we use **8 real patients** from the dataset and only **2 features** (Glucose and BMI), and one new patient. K = 3.

| ID | Glucose | BMI | Diabetes |
|---|---|---|---|
| 1 | 70 | 20.7 | 0 |
| 2 | 95 | 28.2 | 0 |
| 3 | 80 | 24.4 | 0 |
| 4 | 104 | 21.5 | 1 |
| 5 | 111 | 23.9 | 1 |
| 6 | 112 | 29.9 | 1 |
| 7 | 100 | 29.8 | 0 |
| 8 | 102 | 29.2 | 1 |
| **New** | **98** | **27.0** | **?** |

**Step 1: means**

```
mean Glucose = (70+95+80+104+111+112+100+102) / 8 = 774 / 8     = 96.75
mean BMI     = (20.7+28.2+24.4+21.5+23.9+29.9+29.8+29.2) / 8 = 207.6 / 8 = 25.95
```

**Step 2: standard deviations**

```
Glucose deviations (x - 96.75): -26.75, -1.75, -16.75, 7.25, 14.25, 15.25, 3.25, 5.25
squares: 715.5625, 3.0625, 280.5625, 52.5625, 203.0625, 232.5625, 10.5625, 27.5625
sum = 1525.5      variance = 1525.5 / 8 = 190.6875      std = sqrt(190.6875) = 13.809

BMI deviations (x - 25.95): -5.25, 2.25, -1.55, -4.45, -2.05, 3.95, 3.85, 3.25
squares: 27.5625, 5.0625, 2.4025, 19.8025, 4.2025, 15.6025, 14.8225, 10.5625
sum = 100.02      variance = 100.02 / 8 = 12.5025       std = sqrt(12.5025) = 3.536
```

**Step 3: z-scores** `z = (x - mean) / std`

Example for patient 1: `z_Glucose = (70 - 96.75) / 13.809 = -1.937`, `z_BMI = (20.7 - 25.95) / 3.536 = -1.485`.
New patient: `z_Glucose = (98 - 96.75) / 13.809 = 0.091`, `z_BMI = (27.0 - 25.95) / 3.536 = 0.297`.

**Step 4: distance from the new patient to every patient**

Example for patient 8 (z = 0.380, 0.919):

```
Euclidean = sqrt( (0.380 - 0.091)^2 + (0.919 - 0.297)^2 )
          = sqrt( 0.0835 + 0.3869 ) = sqrt(0.4704) = 0.686
Manhattan = |0.380 - 0.091| + |0.919 - 0.297| = 0.289 + 0.622 = 0.911   (0.912 from unrounded values)
```

All patients:

| ID | z Glucose | z BMI | Label | Euclidean | Manhattan | Rank (Euclid.) |
|---|---|---|---|---|---|---|
| 1 | -1.937 | -1.485 | 0 | 2.699 | 3.809 | 8 |
| 2 | -0.127 | 0.636 | 0 | **0.403** | 0.557 | **1** |
| 3 | -1.213 | -0.438 | 0 | 1.497 | 2.039 | 6 |
| 4 | 0.525 | -1.259 | 1 | 1.615 | 1.990 | 7 |
| 5 | 1.032 | -0.580 | 1 | 1.286 | 1.818 | 4 |
| 6 | 1.104 | 1.117 | 1 | 1.304 | 1.834 | 5 |
| 7 | 0.235 | 1.089 | 0 | **0.805** | 0.937 | **3** |
| 8 | 0.380 | 0.919 | 1 | **0.686** | 0.912 | **2** |

**Step 5: take the K = 3 nearest and vote**

```
Nearest 3: patient 2 (label 0), patient 8 (label 1), patient 7 (label 0)
Votes: class 0 = 2, class 1 = 1
P(diabetes) = 1/3 = 0.33   ->   prediction: NO diabetes
```

**Step 6: what changes with other settings**

```
K = 1 : nearest is patient 2 (label 0)                          -> No diabetes
K = 5 : add patient 5 (1) and patient 6 (1): votes 0 -> 2, 1 -> 3
        P(diabetes) = 3/5 = 0.60                                -> DIABETES
K = 5, distance-weighted (w = 1/d):
        patient 2: 1/0.403 = 2.482 (class 0)
        patient 8: 1/0.686 = 1.457 (class 1)
        patient 7: 1/0.805 = 1.242 (class 0)
        patient 5: 1/1.286 = 0.777 (class 1)
        patient 6: 1/1.304 = 0.767 (class 1)
        score_0 = 2.482 + 1.242 = 3.724      score_1 = 1.457 + 0.777 + 0.767 = 3.001
        P(diabetes) = 3.001 / 6.725 = 0.446                     -> No diabetes
Manhattan, K = 3 : same 3 nearest (2, 8, 7)                     -> No diabetes
```

The answer changes with K and with the voting rule. This is exactly why we choose K with cross-validation.

**Step 7: why scaling matters**

Without scaling, the raw distance from the new patient to patient 8 is `sqrt(4^2 + 2.2^2) = 4.57`. Glucose gives 16 of the 20.84 under the root (77%), so BMI is almost ignored. The raw ranking puts patient 7 ahead of patient 8 (order 2, 7, 8), while the scaled ranking is 2, 8, 7.

**Step 8: metrics by hand** (from the lab's test set, 200 patients)

KNN: TN = 131, FP = 8, FN = 25, TP = 36

```
Accuracy    = (36 + 131) / 200 = 167 / 200 = 0.835
Precision   = 36 / (36 + 8)    = 36 / 44   = 0.818
Recall      = 36 / (36 + 25)   = 36 / 61   = 0.590
Specificity = 131 / (131 + 8)  = 131 / 139 = 0.942
F1          = 2*36 / (2*36 + 8 + 25) = 72 / 105 = 0.686
```

Every number above was checked against the code in `knn_from_scratch.py`.

---

# Part 2: Bayesian Belief Network (BBN)

## Idea

A BBN is a **graph** where
- each **node** is a variable (Age, BMI, Glucose, ...)
- each **arrow** means "this variable directly affects that variable"
- each node has a **CPT** (Conditional Probability Table): the probability of that node for every combination of its parents' values.

It is called a "belief" network because when we get new evidence we **update our belief (probability)**.

## Network structure (`bbn_structure.png`)

The full graph is in `bbn_structure.png` (9 arrows). The parents of each node:

| Node | Parents |
|---|---|
| Age, FamilyHistory, Smoker | none |
| BMI | Age |
| BloodPressure | Age, BMI, Smoker |
| Glucose | Age, BMI, FamilyHistory |
| Outcome | Glucose, BMI, FamilyHistory |

We chose this structure ourselves using domain knowledge (the way a doctor would think). Learning the structure automatically from data is not part of this lab.

## How it works (step by step)

### Step 1: Discretization
A BBN works with **discrete** values, so `Age`, `BMI`, `BloodPressure` and `Glucose` are split into **Low / Medium / High**. The cut points are the 33% and 67% quantiles of the train data (`QuantileDiscretizer`), so each group gets about the same amount of data.

### Step 2: Splitting the joint probability
The probability of the whole network is the product of all the CPTs:

```
P(X1, X2, ..., Xn) = PRODUCT over i of  P( Xi | Parents(Xi) )
```

For this network:

```
P(Age, Fam, Smk, BMI, BP, Glu, Out) =
    P(Age) * P(Fam) * P(Smk)
  * P(BMI | Age)
  * P(BP  | Age, BMI, Smk)
  * P(Glu | Age, BMI, Fam)
  * P(Out | Glu, BMI, Fam)
```

This works because each variable is **conditionally independent** of the others once its parents are known. It keeps the tables small. A full joint table would have 3x3x3x3x2x2x2 = 648 entries to learn.

### Step 3: Parameter learning (CPTs by counting)

```
P(X = x | parents = pa) = ( count(x, pa) + 1 ) / ( count(pa) + |X| )
```

This is **MLE + Laplace smoothing (alpha = 1)**. The "+1" is there so that a combination never seen in the training data does not get probability 0 (which would make the whole product 0).

Example: the CPT entry `P(Outcome=1 | Glucose=High, BMI=High, FamilyHistory=1) = 0.807` means about 81% of such patients had diabetes.

### Step 4: Inference (exact, by enumeration)
Question: `P(Outcome | evidence)`.

1. Variables that are neither the query nor in the evidence are called **hidden**.
2. For each value of Outcome, **add up** the joint probability over **all combinations** of the hidden variables.
3. **Normalize** so the total is 1.

```
P(Out=o | e) = SUM over hidden h of P(Out=o, e, h)  /  (normalizing constant)
```

This is in `BayesianNetwork.query()`. It is exact and simple for a small network. Very large networks use Variable Elimination or sampling instead.

### Step 5: Prediction
Use all 6 features of a test patient as evidence and compute `P(Outcome=1 | ...)`. If it is above 0.5, predict "Diabetes".

## Where things are in the code

| Item | Location |
|---|---|
| Discretization | `QuantileDiscretizer` |
| Structure | `STRUCTURE` dictionary |
| CPT learning | `BayesianNetwork.fit` |
| Joint probability | `BayesianNetwork.joint_probability` |
| Inference | `BayesianNetwork.query` |
| Prediction | `BayesianNetwork.predict` |

## Results (seed = 42)

Test set (200 rows):

| Accuracy | Precision | Recall | F1 |
|---|---|---|---|
| 0.855 | 0.742 | 0.803 | 0.772 |

|  | Pred 0 | Pred 1 |
|---|---|---|
| **Actual 0** | 122 (TN) | 17 (FP) |
| **Actual 1** | 12 (FN) | 49 (TP) |

Inference examples (`P(Outcome=1 | evidence)`):

| Evidence | Probability |
|---|---|
| None (prior) | 0.310 |
| Glucose = High | 0.645 |
| Glucose = High, BMI = High | 0.739 |
| Glucose = High, BMI = High, FamilyHistory = 1 | 0.807 |
| Glucose = Low, BMI = Low, FamilyHistory = 0 | 0.039 |

See how the belief is updated as evidence is added: 0.31, then 0.65, then 0.74, then 0.81.

Graphs: `bbn_structure.png`, `bbn_confusion_matrix.png`

A special strength of the BBN is **reverse (diagnostic) reasoning**. For example `P(Glucose | Outcome=1)`: among diabetic patients, 74% had High glucose. KNN cannot do this.

## Mathematical background (BBN)

### 1. Probability rules

```
Conditional probability : P(A | B) = P(A, B) / P(B)
Product rule            : P(A, B) = P(A | B) * P(B)
Sum rule (marginal)     : P(A) = SUM over b of P(A, B = b)
Bayes' theorem          : P(A | B) = P(B | A) * P(A) / P(B)
```

Bayes' theorem lets us turn "cause -> effect" knowledge into "effect -> cause" reasoning.

### 2. Chain rule and the network factorization

For any variables, the chain rule is always true:

```
P(X1, ..., Xn) = P(X1) * P(X2 | X1) * P(X3 | X1, X2) * ... * P(Xn | X1, ..., Xn-1)
```

A Bayesian network adds one assumption (the Markov condition): **each variable is conditionally independent of its other non-descendants, given its parents.** Then each factor keeps only the parents:

```
P(X1, ..., Xn) = PRODUCT over i of  P( Xi | Parents(Xi) )
```

The graph must be a DAG (no cycles), so every variable can be placed after its parents.

### 3. How many numbers must we learn?

A node with k states and m parent combinations needs m x (k - 1) free numbers.

```
Full joint table : 3 x 3 x 3 x 3 x 2 x 2 x 2 = 648 entries -> 647 free numbers

Our network:
  Age           : 3-1                        =  2
  FamilyHistory : 2-1                        =  1
  Smoker        : 2-1                        =  1
  BMI | Age                  : 3 x (3-1)     =  6
  BloodPressure | Age,BMI,Smoker : 18 x (3-1) = 36
  Glucose | Age,BMI,FamilyHistory : 18 x (3-1) = 36
  Outcome | Glucose,BMI,FamilyHistory : 18 x (2-1) = 18
  Total = 100 numbers
```

So the factorized form needs 100 numbers instead of 647. This saves data and makes the estimates more reliable.

### 4. Learning the CPTs: maximum likelihood (MLE)

For one node X with parent combination pa, let N(x, pa) be the number of training rows with X = x and parents = pa. The likelihood of the data is

```
L = PRODUCT over pa, x of  theta(x | pa) ^ N(x, pa)          where theta(x | pa) = P(X = x | pa)
log L = SUM N(x, pa) * log theta(x | pa)
```

Maximize `log L` with the rule that each row of the CPT adds to 1 (SUM over x of theta(x | pa) = 1). With a Lagrange multiplier lambda:

```
N(x, pa) / theta(x | pa) = lambda   ->   theta(x | pa) = N(x, pa) / N(pa)
```

So the best CPT value is simply **count divided by count**. Because the likelihood splits into one piece per node and per parent combination, every CPT row can be learned on its own.

### 5. Laplace smoothing

If a combination never appears, MLE gives 0, and a single 0 makes the whole product 0. We add alpha = 1 to every count:

```
theta(x | pa) = ( N(x, pa) + alpha ) / ( N(pa) + alpha * |X| )
```

This is the posterior mean when the prior over each CPT row is a symmetric Dirichlet(alpha) (alpha = 1 means a flat prior). The values are never 0, and they get closer to the MLE as the data grows.

### 6. Discretization with quantiles

```
q1 = 33.3% quantile of the train data      q2 = 66.7% quantile of the train data
value x  ->  Low if x < q1,   Medium if q1 <= x < q2,   High if x >= q2
```

### 7. Exact inference by enumeration

Let Q be the query variable, E the evidence (observed values e), and H the hidden variables (everything else).

```
P(Q = q | e) = SUM over h of P(Q = q, e, h)  /  SUM over q', h of P(Q = q', e, h)
```

Each `P(Q = q, e, h)` is the product of CPT values from section 2. The bottom sum is the normalizing constant (it makes the answers add up to 1).

Number of terms = (states of Q) x PRODUCT over hidden variables of (their states). In our network with only Glucose observed: 2 x (3 x 2 x 2 x 3 x 3) = 216 terms. With all 6 features observed there are no hidden variables, so only 2 terms.

### 8. Prediction rule and metrics

```
y_hat = 1  if  P(Outcome = 1 | all features) >= 0.5,  else 0
```

The metrics (accuracy, precision, recall, F1) are the same formulas as in the KNN section.

## Worked example by hand (BBN)

A full network is too big to solve by hand, so we use a **mini network with 3 variables** and **16 real patients** from the dataset. Glucose is grouped into just two levels: Low (below 100) and High (100 or more).

```
Arrows:  FamilyHistory -> Glucose
         FamilyHistory -> Outcome
         Glucose       -> Outcome
```

Factorization: `P(FH, G, Out) = P(FH) * P(G | FH) * P(Out | FH, G)`

**Step 1: count the training data**

| FamilyHistory | Glucose | Outcome = 0 | Outcome = 1 | Rows |
|---|---|---|---|---|
| 1 | Low | 1 | 2 | 3 |
| 1 | High | 0 | 2 | 2 |
| 0 | Low | 4 | 2 | 6 |
| 0 | High | 2 | 3 | 5 |
| | | 7 | 9 | **16** |

So FamilyHistory = 1 in 5 rows and FamilyHistory = 0 in 11 rows.

**Step 2: build the CPTs** with `(count + 1) / (total + number of states)`

```
P(FH):
  P(FH=1) = (5 + 1) / (16 + 2) = 6/18  = 0.3333
  P(FH=0) = (11 + 1) / (16 + 2) = 12/18 = 0.6667

P(G | FH):
  FH=1: P(High) = (2 + 1) / (5 + 2)  = 3/7  = 0.4286     P(Low) = 4/7  = 0.5714
  FH=0: P(High) = (5 + 1) / (11 + 2) = 6/13 = 0.4615     P(Low) = 7/13 = 0.5385

P(Out=1 | FH, G):
  FH=1, Low : (2 + 1) / (3 + 2) = 3/5 = 0.600     (so P(Out=0) = 0.400)
  FH=1, High: (2 + 1) / (2 + 2) = 3/4 = 0.750     (so P(Out=0) = 0.250)
  FH=0, Low : (2 + 1) / (6 + 2) = 3/8 = 0.375     (so P(Out=0) = 0.625)
  FH=0, High: (3 + 1) / (5 + 2) = 4/7 = 0.571     (so P(Out=0) = 0.429)
```

Why smoothing? For FH=1, High the raw counts are 0 and 2, so MLE would say `P(Out=1) = 2/2 = 1` and `P(Out=0) = 0`. Smoothing gives 0.75 and 0.25 instead, so nothing is called impossible just because 2 patients were seen.

**Step 3: the full joint table** (every row = P(FH) x P(G | FH) x P(Out | FH, G))

| FH | Glucose | Outcome | Calculation | Joint |
|---|---|---|---|---|
| 1 | High | 1 | 1/3 x 3/7 x 3/4 | 0.107143 |
| 1 | High | 0 | 1/3 x 3/7 x 1/4 | 0.035714 |
| 1 | Low | 1 | 1/3 x 4/7 x 3/5 | 0.114286 |
| 1 | Low | 0 | 1/3 x 4/7 x 2/5 | 0.076190 |
| 0 | High | 1 | 2/3 x 6/13 x 4/7 | 0.175824 |
| 0 | High | 0 | 2/3 x 6/13 x 3/7 | 0.131868 |
| 0 | Low | 1 | 2/3 x 7/13 x 3/8 | 0.134615 |
| 0 | Low | 0 | 2/3 x 7/13 x 5/8 | 0.224359 |
| | | | **Sum** | **1.000000** |

The sum is 1, so the CPTs are consistent.

**Step 4: inference (query A): P(Outcome = 1 | Glucose = High)**

FamilyHistory is hidden, so we add over both of its values (sum rule):

```
P(Out=1, G=High) = 0.107143 + 0.175824 = 0.282967
P(Out=0, G=High) = 0.035714 + 0.131868 = 0.167582
Normalizer       = 0.282967 + 0.167582 = 0.450549

P(Out=1 | G=High) = 0.282967 / 0.450549 = 0.628
P(Out=0 | G=High) = 0.167582 / 0.450549 = 0.372
```

The prior is `P(Out=1) = 0.282967 + P(Out=1, G=Low) = 0.282967 + (0.114286 + 0.134615) = 0.532`. Seeing High glucose moves the belief from 0.532 to 0.628.

**Step 5: query B: P(Outcome = 1 | Glucose = High, FamilyHistory = 1)**

No hidden variables are left:

```
P(Out=1, FH=1, G=High) = 0.107143      P(Out=0, FH=1, G=High) = 0.035714
P(Out=1 | FH=1, G=High) = 0.107143 / (0.107143 + 0.035714) = 0.75
```

This is exactly the CPT row `P(Out=1 | FH=1, High) = 3/4`. More evidence moved the belief from 0.628 to 0.75.

**Step 6: query C: reverse reasoning P(Glucose = High | Outcome = 1)**

```
P(G=High, Out=1) = 0.282967
P(G=Low,  Out=1) = 0.114286 + 0.134615 = 0.248901

P(G=High | Out=1) = 0.282967 / (0.282967 + 0.248901) = 0.282967 / 0.531868 = 0.532
```

Check with Bayes' theorem: `P(High | Out=1) = P(Out=1 | High) x P(High) / P(Out=1) = 0.628 x 0.450549 / 0.531868 = 0.532`. The two ways agree.

**Step 7: query D: P(FamilyHistory = 1 | Outcome = 1, Glucose = High)**

```
P(FH=1, G=High, Out=1) = 0.107143
P(FH=0, G=High, Out=1) = 0.175824

P(FH=1 | Out=1, G=High) = 0.107143 / (0.107143 + 0.175824) = 0.379
```

The prior `P(FH=1) = 0.333` rises to 0.379 after we learn the patient has diabetes with high glucose.

**Step 8: prediction and metrics by hand** (from the lab's test set, 200 patients)

For a patient with Glucose = High and FH = 1 the network says P(diabetes) = 0.75 >= 0.5, so it predicts "Diabetes".

BBN: TN = 122, FP = 17, FN = 12, TP = 49

```
Accuracy  = (49 + 122) / 200 = 171 / 200 = 0.855
Precision = 49 / (49 + 17)   = 49 / 66   = 0.742
Recall    = 49 / (49 + 12)   = 49 / 61   = 0.803
F1        = 2*49 / (2*49 + 17 + 12) = 98 / 127 = 0.772
```

All the numbers in this worked example were checked against the code in `bbn_from_scratch.py` (the `BayesianNetwork` class gives the same CPTs and the same answers 0.628, 0.75, 0.532 and 0.379).

**Note:** in this tiny 3-variable example the factorization saves nothing (7 numbers either way), because it is the complete chain rule. The saving appears in the real network: 100 numbers instead of 647.

---

# KNN vs BBN

| Point | KNN | BBN |
|---|---|---|
| Type | Instance-based (lazy) | Probabilistic graphical model |
| Training | None, it just stores the data | Counts to build CPTs (fast) |
| Data type | Numbers (scaling needed) | Discrete values (grouping needed) |
| Prediction speed | Slow (distance to every training point) | Fast (table lookup and sum) |
| Easy to explain? | Only "these were the neighbors" | CPTs and arrows are easy to read |
| Missing values | Hard | Easy (leave the feature out of the evidence, it becomes hidden) |
| Probability / uncertainty | Only the vote share | Real probabilities |
| Domain knowledge | Not needed | Needed for the structure |
| Accuracy in this lab | 0.835 | 0.855 |

**Note:** this accuracy difference is measured on a small test set (200 rows), so it is not solid proof that the BBN is better. The BBN did well here partly because we gave it a structure that matches how the data was made.

## What you can try next
- Lower the KNN threshold below 0.5 to improve recall.
- Learn the BBN structure from data (for example Hill-Climbing with a BIC score).
- Run both models on a real dataset (Pima Indians Diabetes).
- Compare the two models fairly with cross-validation.
