# EXP004-A：多帧 KLT 存活率与标定几何一致性

## 核心目标
检验已有 S5/S10 特征存活率是否代表可靠的几何对应关系。它们是**离线回顾性标签**；本实验暂时不训练新模型，也不评价真正的 VIO 更新成功率。

使用 EXP002 相同的 Shi–Tomasi、8×6 网格轮询、固定起始点 ID、逐相邻帧前后向 KLT、5/10 帧存活定义。另存逐点坐标和失效事件，利用 EuRoC `cam0/sensor.yaml` 的针孔模型内参与 radtan/equidistant 畸变参数，将 KLT 对应点归一化，然后使用 Essential Matrix + RANSAC 进行几何检查。匹配点不足、尾部删失、低视差等单独标记，**不填成错误的“零内点率”**。

几何输出包含内点数量、内点率（相对于存活点和起始点两个不同分母）、Sampson 残差近似像素值、正深度比例、利用恢复旋转补偿后的视差角、几何内点空间覆盖率。RANSAC 和 recoverPose 都使用同一批匹配点，**不是独立真实几何验证**；低视差阈值 0.25° 只是启发式标记，后续 EXP004-B 才用 GT 相对位姿验证。

## 安装与运行

仓库根目录：

```bash
pip install -e ".[dev]"
pytest -q

vio-iqa-exp004 extract --sequence /datasets/MH_01_easy --max-frames 300 --stride 5 --output-dir results/exp004/MH_01_easy
vio-iqa-exp004 extract --sequence /datasets/MH_05_difficult --max-frames 300 --stride 5 --output-dir results/exp004/MH_05_difficult

vio-iqa-exp004 report --input-dir results/exp004/MH_01_easy
vio-iqa-exp004 report --input-dir results/exp004/MH_05_difficult
```

先运行 300 帧检查数据与标定无误，再设置 `--max-frames 0` 运行整条序列。`--stride 5` 只对起始帧抽样，跟踪时仍逐帧递推；如需与 EXP002 每帧标签对照，必须使用相同 `budget=240`、`candidates=3000`、`fb_threshold=1`、ROI 与完整序列设置，然后以 `--stride 1` 运行。

```bash
vio-iqa-exp004 extract --sequence /datasets/MH_01_easy --max-frames 0 --stride 1 --output-dir results/exp004/MH_01_full
vio-iqa-exp004 report --input-dir results/exp004/MH_01_full --exp002-labels results/exp002/MH_01_easy/survival_labels.csv
```

需要包含 `mav0/cam0/sensor.yaml`；也可以用 `--calibration /path/to/sensor.yaml` 明确指定。标定分辨率不符和不支持的模型会报错，不能无标定地回退到原始像素极线计算。

## 输出

| 文件 | 内容 |
|---|---|
| `frame_geometry.csv` | 每个起始帧 × 5/10 帧跨度的存活、几何状态、内点、残差、视差 |
| `feature_tracks.csv` | 每个起始帧、特征 ID、逐帧位置；失效当步状态标为 lost |
| `geometry_matches.csv` | 终点仍存活特征的 t 与 t+K 坐标、RANSAC 内外点、几何残差 |
| `manifest.json` | 相机模型、标定哈希、ROI、特征预算、算法阈值 |
| `geometry_correlations.csv` | S5/S10 与各几何指标的探索性 Spearman |
| `status_counts.csv` | 匹配不足/成功估计/低视差/删失计数 |
| `high_survival_low_inlier_examples.csv` | 高存活率但低几何内点率的候选反例 |
| `exp002_consistency.csv` | 可选，比较与原 EXP002 一致的存活标签 |

注意：逐特征 CSV 较大，上传结果时先压缩。状态为 `insufficient_matches`、`tail_censored` 或 `essential_failed_or_ambiguous` 的记录不能静默去除后宣称全序列的几何质量很高；必须同时报告覆盖率和状态分布。

## 科研限制

1. RANSAC 自拟合内点并非独立的正确匹配真值。若只有纯旋转、近平面或低视差，Essential Matrix 可能估计退化；只报告成功估计的帧容易产生选择偏差。
2. 存活点与初始点分母不同；始终一起报告 `n_initial`、`n_survive`、`n_geom_inlier` 与各自比率。
3. 需要在后续 EXP004-B 中使用 EuRoC 真值轨迹、外参及严格时间对齐，检验独立几何残差。当前尚没有完整 VIO ATE/RPE、创新量或可靠更新成功标签。
4. 相邻帧高度自相关；MH_01/MH_05 目前用于指标开发，不是没有接触过的最终测试集。
