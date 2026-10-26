# QRT: Hierarchical Modeling and Ensemble Experiments

## Research questions

Predict return direction from historical returns, volume, and cross-sectional information. The central questions are how much structure should be shared across instruments and whether local corrections complement a shared model. The training set contains 527073 rows, 278 allocations, and 2522 timestamps (TS); 50.7184% of targets are nonnegative.

## Modeling ideas and economic assumptions

The global model tests for shared price–volume relationships. Group and cluster experts allow conditional relationships to differ across subsets, while allocation residuals capture instrument-specific differences. CatBoost learns from statistical features and nonlinear interactions; GRU learns historical sequence patterns. The ensemble tests whether these representations complement each other.

These are modeling assumptions. The economic meaning of GROUP and ALLOCATION has not been verified, so they are not treated as industry or asset-class labels. Clusters reflect statistical feature similarity. Directional accuracy is not a trading return; profitability after costs, turnover, and risk adjustment has not been established.

## Core experimental results

| path | control | OOF accuracy changes | change percentage points |
| --- | --- | --- | ---: |
| CatBoost | Global → add group residuals | 52.0524% → 51.8382% | -0.2142 |
| CatBoost | Cluster → add allocation residuals | 52.1719% → 52.0791% | -0.0928 |
| GRU final base model | Global head → add group residuals | 52.4404% → 52.5743% | +0.1339 |
| GRU outer layer correction | Global + group → add cluster residuals | 52.5743% → 52.6701% | +0.0958 |
| Fusion | GRU normalized output → allocation fusion | 52.6426% → 52.8356% | +0.1930 |

The global head is a branch of the jointly trained GRU, including identifier embeddings; it is not a separately trained global-only model. Historical configurations may also differ in features, splits, or training parameters. The [presentation notebook](analysis.ipynb) distinguishes paired branch comparisons from historical architecture comparisons and marks missing records.

The final archived configuration combines **cluster CatBoost + allocation residual** with **joint global/group GRU + outer cluster residual**, using allocation-specific linear blending. Its 52.8356% accuracy is measured on the same OOF predictions used to select the weights. It is not an independent test score or a claim of independently verified superiority over every historical experiment.

## What does component ablation explain?

With final allocation weights fixed, removing the CatBoost allocation residual reduces accuracy by 0.4109 percentage points, and removing the GRU outer cluster residual reduces it by 0.2119 points. Removing the entire CatBoost or GRU branch reduces accuracy by 0.1930 and 0.8052 points, respectively.

CatBoost residuals have no gain when the model is used alone, but they contribute in the current fusion, indicating that the combined performance should also be examined when retaining components. This conclusion applies to branch ablation of already trained systems; there is no component removal followed by re-optimization of the remaining model.

## Display and result boundaries

[analysis.ipynb](analysis.ipynb) presents data, features, model structures, comparisons within both model families, blending, component ablations, and training diagnostics. Historical aggregate tables and figures are embedded, so viewing them requires no intermediate models or row-level prediction files. Only selected checkpoints were saved; a complete epoch-by-epoch loss curve is unavailable.

The current verification is not equivalent to strict forward rolling prediction. Some preprocessing uses complete training features, and the same OOF is used for fusion weight selection and score reporting. The structure and branching results support the interpretation on the current sample, and generalization and economic benefits in independent time periods still need to be evaluated separately. See [README.md](README.md) for the reproduction command and product location.
