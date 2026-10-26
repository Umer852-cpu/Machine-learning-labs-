"""
generate_dataset.py
-------------------
Creates a patient dataset (1000 rows) that has a known cause-and-effect structure:

    Age ──► BMI ──► BloodPressure ◄── Smoker
     │       │
     ▼       ▼
   Glucose ◄─ FamilyHistory
     │       │
     ▼       ▼
        Outcome (Diabetes: 0 = No, 1 = Yes)

The same dataset is used for KNN (on the number values) and for the
Bayesian Belief Network (on the values grouped into Low / Medium / High).

Run:    python generate_dataset.py
Output: patient_data.csv
"""
import numpy as np
import pandas as pd

SEED = 42
N = 1000

rng = np.random.default_rng(SEED)

age = np.clip(rng.normal(45, 14, N), 20, 80)
family = rng.binomial(1, 0.30, N)
smoker = rng.binomial(1, 0.25, N)

bmi = np.clip(20 + 0.07 * age + rng.normal(0, 4, N), 16, 45)
bp = np.clip(90 + 0.35 * age + 0.8 * (bmi - 25) + 3 * smoker + rng.normal(0, 8, N), 70, 190)
glucose = np.clip(
    70 + 1.2 * (bmi - 20) + 0.3 * age + 10 * family + rng.normal(0, 14, N), 60, 250
)

# Chance of diabetes: glucose matters most, then BMI, family history and age
logit = (
    0.12 * (glucose - glucose.mean())
    + 0.12 * (bmi - bmi.mean())
    + 0.9 * family
    + 0.02 * (age - 45)
    - 2.0
)
prob = 1 / (1 + np.exp(-logit))
outcome = rng.binomial(1, prob)

df = pd.DataFrame(
    {
        "Age": age.round(0).astype(int),
        "BMI": bmi.round(1),
        "BloodPressure": bp.round(0).astype(int),
        "Glucose": glucose.round(0).astype(int),
        "FamilyHistory": family,
        "Smoker": smoker,
        "Outcome": outcome,
    }
)

df.to_csv("patient_data.csv", index=False)
print(f"patient_data.csv saved: {df.shape[0]} rows, {df.shape[1]} columns")
print("Outcome distribution:")
print(df["Outcome"].value_counts().rename({0: "No Diabetes (0)", 1: "Diabetes (1)"}))
print("\nFirst 5 rows:")
print(df.head())
