# Lab 03: Linear Regression

## Objective
Predict a car's fuel efficiency (miles per gallon) with linear regression built from scratch: simple regression, gradient descent and multiple regression.

## Dataset
Auto MPG (398 cars, 392 after removing 6 rows with missing horsepower). Target: `mpg`.
Attributes: `cylinders`, `displacement`, `horsepower`, `weight`, `acceleration`, `model_year`, `origin`.

## Methods
| Step | Method |
|---|---|
| A | Simple linear regression (`weight`), closed-form least squares |
| B | Gradient descent on the same problem (learning-rate experiment) |
| C | Multiple linear regression with the normal equation β = (XᵀX)⁻¹Xᵀy |
| D | Residual diagnostics, quadratic-weight extension, scikit-learn cross-check |

## Manual Calculation (first 8 cars, `weight` in 1000 lbs)
**Formulas**
- b₁ = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²
- b₀ = ȳ - b₁ x̄
- R² = 1 - SSE / SST

**Step 1: Means**
- x̄ = 3.81525, ȳ = 15.875

**Step 2: Deviations**
| Car | x | y | x - x̄ | y - ȳ | (x - x̄)(y - ȳ) | (x - x̄)² |
|---|---|---|---|---|---|---|
| chevrolet chevelle malibu | 3.504 | 18 | -0.3112 | 2.125 | -0.6614 | 0.0969 |
| buick skylark 320 | 3.693 | 15 | -0.1222 | -0.875 | 0.1070 | 0.0149 |
| plymouth satellite | 3.436 | 18 | -0.3792 | 2.125 | -0.8059 | 0.1438 |
| amc rebel sst | 3.433 | 16 | -0.3822 | 0.125 | -0.0478 | 0.1461 |
| ford torino | 3.449 | 17 | -0.3662 | 1.125 | -0.4120 | 0.1341 |
| ford galaxie 500 | 4.341 | 15 | 0.5258 | -0.875 | -0.4600 | 0.2764 |
| chevrolet impala | 4.354 | 14 | 0.5388 | -1.875 | -1.0102 | 0.2903 |
| plymouth fury iii | 4.312 | 14 | 0.4968 | -1.875 | -0.9314 | 0.2468 |
| **Sum** | | | | | **Sxy = -4.2218** | **Sxx = 1.3493** |

**Step 3: Slope and intercept**
- b₁ = -4.2218 / 1.3493 = **-3.1288**
- b₀ = 15.875 - (-3.1288)(3.81525) = **27.8120**
- Equation: mpg = 27.812 - 3.129 · weight

**Step 4: Check the fit**
| Car | y | ŷ | Residual | Residual² | (y - ȳ)² |
|---|---|---|---|---|---|
| chevrolet chevelle malibu | 18 | 16.849 | 1.151 | 1.325 | 4.516 |
| buick skylark 320 | 15 | 16.257 | -1.257 | 1.581 | 0.766 |
| plymouth satellite | 18 | 17.062 | 0.938 | 0.881 | 4.516 |
| amc rebel sst | 16 | 17.071 | -1.071 | 1.147 | 0.016 |
| ford torino | 17 | 17.021 | -0.021 | 0.000 | 1.266 |
| ford galaxie 500 | 15 | 14.230 | 0.770 | 0.593 | 0.766 |
| chevrolet impala | 14 | 14.189 | -0.189 | 0.036 | 3.516 |
| plymouth fury iii | 14 | 14.321 | -0.321 | 0.103 | 3.516 |
| **Sum** | | | | **SSE = 5.666** | **SST = 18.875** |

- R² = 1 - 5.666 / 18.875 = **0.6998**
- RMSE = √(5.666 / 8) = 0.8416, MAE = 0.7149
- Prediction for a 3000 lb car: 27.812 - 3.129 · 3.0 = **18.43 mpg**

**Note:** this 8-car sample (all large 1970 American cars, narrow weight range) gives a slope of -3.13, while the full dataset gives about -7.90. Small, narrow samples give unstable slopes.

## Results (80/20 split, 79 test cars)
| Model | Test R² | Test RMSE (mpg) | Test MAE (mpg) |
|---|---|---|---|
| Baseline (predict the mean) | -0.0114 | 7.1847 | 5.8810 |
| Simple (weight) | 0.6533 | 4.2064 | 3.4641 |
| Simple + weight² | 0.6730 | 4.0852 | 3.1383 |
| Multiple (8 features) | 0.7923 | 3.2561 | 2.4620 |
| Multiple + weight² | 0.8238 | 2.9991 | 2.2518 |

Full-data simple model: mpg = 47.20 - 7.90 · weight (weight in 1000 lbs); gradient descent reaches the same coefficients.

## Observations
- Weight alone explains about 65% of the variation in mpg; adding the other specifications raises this to about 79%
- Gradient descent matches the closed-form solution; a learning rate that is too large makes the cost diverge
- Scratch implementations match scikit-learn exactly
- Displacement, cylinders, weight and horsepower are strongly correlated (multicollinearity), so individual coefficients should be interpreted with care
- Residual plots show curvature and uneven spread; adding weight² helps
- Results come from a single split, so small differences between models should not be over-interpreted

## Files
- `linear_regression_auto_mpg.ipynb`: implementation, manual calculation, diagnostics and comparison
- `data/auto_mpg.csv`: dataset
