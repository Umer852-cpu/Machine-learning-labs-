# Lab 01: Find-S Algorithm

## Objective
Learn the most specific hypothesis that describes a very good (`vgood`) car using the Find-S algorithm.

## Dataset
Car Evaluation (UCI / Kaggle): 1,728 records, 6 categorical attributes
(`buying`, `maint`, `doors`, `persons`, `lug_boot`, `safety`).

## Method
- Positive class: `vgood`; all other classes are negative
- Find-S uses only positive examples and generalizes mismatching attributes to `?`
- 80/20 train-test split (random_state=42)

## Results
- Final hypothesis: `<?, ?, ?, ?, ?, high>`
- Test accuracy: 0.7341
- Confusion: TP = 15, FN = 0, FP = 92, TN = 239

## Observation
Every `vgood` car has high safety, so it is the only attribute that stays specific.
Find-S ignores negative examples, so the hypothesis is too general and produces false positives.

## Files
- `find_s_car_evaluation.ipynb`: implementation with step-by-step trace
- `data/car_evaluation.csv`: dataset
