# QRT 收益方向预测

最终训练框架：cluster CatBoost 与 allocation 残差，结合 global 与 group 联合 GRU 及 cluster 残差，再按 allocation 线性融合。

实验展示见 [analysis.ipynb](analysis.ipynb)，研究思路与结论见 [summary.md](summary.md)。Notebook 内嵌汇总表与图片，阅读和运行不需要中间模型及历史预测文件。

## 目录

```text
.
├── README.md
├── summary.md
├── analysis.ipynb
├── requirements.txt
├── data/                       # 原始输入数据
├── qrt/                        # 最终方案的特征、训练与融合代码
└── scripts/
    └── run_pipeline.py          # 统一训练入口
```

## 数据

| 文件 | 内容 |
| --- | --- |
| `data/X_train.csv` | ROW_ID、TS、ALLOCATION、GROUP、20 个 lag 的 RET 和 SIGNED_VOLUME、MEDIAN_DAILY_TURNOVER |
| `data/y_train.csv` | ROW_ID 与连续收益目标 target |
| `data/X_test.csv` | 测试输入特征 |
| `data/sample_submission.csv` | 提交格式参考 |

输入和目标按 ROW_ID 对齐。评价时将 target 大于等于零视为正类，衡量收益方向准确率。

`X_train.csv` 约 272 MB，通过 `.gitattributes` 配置为 Git LFS 文件。发布机器需要安装并启用 Git LFS；克隆时也应取得对应 LFS 数据。其余三个 CSV 是普通数据文件。

## 环境

使用 Python 3.10 或更新版本，在仓库根目录执行：

```powershell
python -m pip install -r requirements.txt
```

VS Code 或 Jupyter 可打开 `analysis.ipynb`。Notebook 已保存展示输出，直接在 GitHub 阅读无需安装环境。

## 执行训练

```powershell
python scripts/run_pipeline.py --dry-run
python scripts/run_pipeline.py
```

默认依次训练 CatBoost、训练 GRU、搜索融合权重并生成提交。输入目录为 `data/`，输出目录为 `artifacts/`。自定义路径与设备：

```powershell
python scripts/run_pipeline.py --data-dir data --output-dir artifacts --device cuda
```

也可分阶段执行：

```powershell
python scripts/run_pipeline.py --stage catboost
python scripts/run_pipeline.py --stage gru
python scripts/run_pipeline.py --stage blend
```

融合阶段要求前两个阶段已在同一输出目录生成 OOF 和测试分数。多 seed、多折及局部专家训练可能耗时；重复执行会写入同名结果，保留不同实验时请指定新的输出目录。

## 输出

| 路径 | 内容 |
| --- | --- |
| `artifacts/catboost/` | CatBoost 权重、OOF、测试分数及指标 |
| `artifacts/gru/` | GRU 权重、OOF、测试分数及指标 |
| `artifacts/blend/` | 融合权重、搜索明细和预测 |
| `artifacts/blend/preds_allocation_aware_linear_blend.csv` | 最终提交，字段 ROW_ID、prediction |

训练产物和本地历史实验被 `.gitignore` 排除，不随仓库发布。Notebook 保留少量汇总结果，不包含模型权重或逐行 OOF。依赖版本尚未锁定；训练入口已通过语法检查和命令预演，完整重训分数尚未核验。
