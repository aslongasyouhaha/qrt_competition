# QRT earnings direction forecast

The final training framework: cluster CatBoost and allocation residuals, combined with global and group GRU and cluster residuals, and then linearly fused according to allocation.

For experimental demonstration, see [analysis.ipynb](analysis.ipynb), and for research ideas and conclusions, see [summary.md](summary.md). Notebook has built-in summary tables and pictures, and no intermediate model or historical prediction files are required for reading and running.

## Directory

```text
.
├── README.md
├── summary.md
├── analysis.ipynb
├── requirements.txt
├── data/ # Original input data
├── qrt/ # Features, training and fusion code of the final solution
└── scripts/
    └── run_pipeline.py # Unified training entrance
```

## Data

| File | content |
| --- | --- |
| `data/X_train.csv` | ROW_ID, TS, ALLOCATION, GROUP, 20 lags of RET and SIGNED_VOLUME, MEDIAN_DAILY_TURNOVER |
| `data/y_train.csv` | ROW_ID and continuous revenue target target |
| `data/X_test.csv` | Test input features |
| `data/sample_submission.csv` | Submission format reference |

Inputs and targets are aligned by ROW_ID. During evaluation, target greater than or equal to zero is regarded as a positive class, and the accuracy of the income direction is measured.

`X_train.csv` is about 272 MB, configured as a Git LFS file via `.gitattributes`. Git LFS needs to be installed and enabled on the publishing machine; the corresponding LFS data should also be obtained when cloning. The remaining three CSVs are normal data files.

## Environment

Using Python 3.10 or newer, execute in the root directory of the repository:

```powershell
python -m pip install -r requirements.txt
```

VS Code or Jupyter can open `analysis.ipynb`. Notebook has saved display output and can be read directly on GitHub without installation environment.

## Execute training

```powershell
python scripts/run_pipeline.py --dry-run
python scripts/run_pipeline.py
```

By default, CatBoost is trained, GRU is trained, fusion weights are searched, and submissions are generated. The input directory is `data/` and the output directory is `artifacts/`. Custom paths and devices:

```powershell
python scripts/run_pipeline.py --data-dir data --output-dir artifacts --device cuda
```

It can also be executed in stages:

```powershell
python scripts/run_pipeline.py --stage catboost
python scripts/run_pipeline.py --stage gru
python scripts/run_pipeline.py --stage blend
```

The fusion stage requires that the previous two stages have generated OOF and test scores in the same output directory. Multi-seed, multi-fold, and local expert training may be time-consuming; repeated executions will write results with the same name. Please specify a new output directory when retaining different experiments.

## Output

| path | content |
| --- | --- |
| `artifacts/catboost/` | CatBoost weights, OOF, test scores and metrics |
| `artifacts/gru/` | GRU weights, OOF, test scores and indicators |
| `artifacts/blend/` | Fusion weights, search details and predictions |
| `artifacts/blend/preds_allocation_aware_linear_blend.csv` | Final submission, fields ROW_ID, prediction |

Training products and local historical experiments are excluded by `.gitignore` and are not released with the repository. Notebook retains a small number of summary results and does not include model weights or row-by-row OOF. The dependent version has not been locked; the training entry has passed the syntax check and command preview, and the complete retraining score has not yet been verified.
