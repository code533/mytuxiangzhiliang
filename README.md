# VIO-IQA Lab

研究问题：当前帧图像质量指标，能否预测下一帧视觉特征跟踪质量？

本项目是 **实验基线**，不是完整 VIO 系统。目前只使用 EuRoC 的单目灰度图像与前后两帧 KLT forward-backward 检验；不会将测试标签当成预测指标，也不将相关性解释为因果关系。

## 安装

要求 Python >= 3.10。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## 数据

从 [EuRoC MAV](https://projects.asl.ethz.ch/datasets/euroc-mav/) 下载 ASL 格式数据，并解压出如 `MH_01_easy/mav0/cam0/data.csv` 与 `data/*.png` 的目录。**不要将数据集提交到本仓库。**

## 运行

```bash
vio-iqa run --sequence /absolute/path/MH_01_easy --max-frames 500 --output results/MH_01_easy.csv
vio-iqa report --input results/MH_01_easy.csv --output-dir results/MH_01_easy_report
```

设置 `--max-frames 0` 可处理完整序列。输出包括逐帧 CSV、探索性 `spearman.csv` 和 `timeline.png`。

## 单帧指标（只读取 I_t）

- `grad_p90`：Sobel 梯度模长第 90 百分位。
- `usable_gradient_ratio`：未饱和且梯度高于阈值的像素比例。
- `laplacian_variance`：Laplacian 方差。
- `dark_ratio` / `saturated_ratio`：暗区与饱和像素比例。
- `fast_count` / `fast_grid_occupancy`：FAST 角点数与 8×6 网格覆盖率。
- `shi_tomasi_count_capped`：最多 1000 个 Shi–Tomasi 特征。

阈值仅用于基线，不代表跨相机、跨分辨率普适的标定结果。

## 后验标签（使用 I_t 和 I_(t+1)）

独立的 Shi–Tomasi + pyramidal LK 前后向跟踪用于计算：
- `klt_fb_success_ratio`：满足前后向一致性要求的点数 / 初始角点数。零角点时为 NaN。
- `klt_fb_good_count`：通过的点数。
- `klt_good_per_1000px`：每千像素通过的点数。
- `klt_fb_median_error_px`：有效 KLT 轨迹的中位前后向误差。

FB 一致性不是地面真实的匹配正确性。高成功率也可能仅来自少量可跟踪角点。

## 研究原则

- 必须同时报告跟踪成功比例和有效点绝对数量。
- 相邻帧相关，不能用每帧 p-value 当作独立显著性证明；最终按完整序列留出测试。
- 运动、纹理、帧间隔、曝光和相机预处理均可能是混杂因素。
- 目前未验证真实 EuRoC 数据，也没有证明能改善 VIO ATE/RPE。
- 下一阶段添加多帧 survival、几何内点率、IMU 分层与受控退化。

## 开发

`pytest -q` 运行单元及数据读取测试；每个 PR 通过 GitHub Actions 验证。

## Experiment 001 — 受控退化和真实低照度

第一轮正式实验已提供命令行入口 `vio-iqa-exp001`，包含 EuRoC 原始序列、12 种成对合成退化以及 TUM-VI slides（需本地下载）、固定 ROI/16-bit 映射、成对差值和探索性相关性。

运行方法、输出字段及实验局限请参阅 [EXP001 实验协议](experiments/exp001/README.md)。目前仍需要在真实数据集上执行实验；合成单元测试不代表研究结论。

## 研究目标与路线图（VIO-oriented Image Quality Predictor）

本项目的长期目标不仅是计算图像锐度或角点数，而是**在视觉更新之前，利用当前图像及截至当前的 IMU 信息预测视觉测量的可靠性**。

研究路线：EXP001（单帧质量与两帧 KLT 关系）→ EXP001.1（特征检测偏差控制）→ **EXP002（预测未来 5/10 帧特征存活率）** → EXP003（融合 IMU）→ EXP004（几何测量可靠性）→ EXP005（校准的 VIO 质量预测器）→ EXP006（在真实 VIO 中检验决策收益）。

**[查看完整研究路线图、标签定义、实验验收标准及防止数据泄漏的规范](docs/research-roadmap.md)**


## EXP002：预测未来 5/10 帧存活率

已实现基于当前帧图像指标预测 KLT 多帧存活率的第一版提取器与严格跨序列 Ridge 基线，支持零特征、末尾删失、固定特征预算和测试序列隔离。详见 [EXP002 运行说明](experiments/exp002/README.md)。此处只是多帧跟踪代理标签，尚未验证真实 VIO 更新成功概率。

## EXP003：图像质量与历史 IMU 融合预测

新增 `vio-iqa-exp003 augment/evaluate`，利用当前图像及当前时刻之前的 0.2/0.5 秒 IMU 运动统计，在严格跨序列测试条件下对比常数、纯图像、纯 IMU 和融合模型。除了存活率 MAE/RMSE，还比较低存活率事件的 AUROC/AUPRC、Brier 与固定预警预算召回率。详见 [EXP003 实验协议](experiments/exp003/README.md)。

## EXP003.2：精简 IMU 与独立数据集风险校准

新增 `vio-iqa-exp0032`，对照全量/精简 IMU 特征、历史窗口差值和图像+IMU 融合，并使用独立验证序列进行概率校准与模型选择。常数模型并列风险按随机排序期望处理。**MH_01/MH_05 已用于提出假设，下一轮需要全新的独立验证与测试序列。** 见 [EXP003.2 实验说明](experiments/exp0032/README.md)。

## EXP004-A：标定几何一致性与多帧存活关系

新增 `vio-iqa-exp004 extract/report`，沿用 EXP002 固定特征组和逐帧 KLT，同时导出带 ID 的多帧轨迹、5/10 帧几何对应、EuRoC 标定畸变校正后的 Essential RANSAC 内点率、Sampson 残差、正深度比例和旋转补偿视差。此处几何模型由同一批匹配点自估，**不是独立几何真值**。详见 [EXP004-A 实验协议](experiments/exp004/README.md)。

## EXP004-B：独立 EuRoC 参考几何验证

新增 `vio-iqa-exp004b`，使用已保存的 KLT 匹配、EuRoC 真实位姿和相机—IMU 外参计算独立的极线一致性、参考平移基线和视差，不需要重新运行 EXP004-A。低基线和缺失 GT 分别标记，不误判为匹配失败。见 [EXP004-B 运行协议](experiments/exp004b/README.md)。

## EXP004-B.1：退化类型与 GT 阈值敏感性

新增 `vio-iqa-exp004b1`，从保存的 EXP004-B 逐点 GT 极线误差重新评价不同的极线阈值及更严格的平移基线，保留 GT 缺失和低基线等状态，并区分自估 RANSAC 与参考几何不一致、以及参考几何较好但自估位姿失败的情况。说明见 [EXP004-B.1 协议](experiments/exp004b1/README.md)。

## EXP005-A：几何退化预测标签构建

新增 `vio-iqa-exp005a`，把 EXP003 的当前图像+历史 IMU 特征与 EXP004-B 完整序列 GT 结果严格对齐，构建未来低平移基线与低 GT 极线一致性两类后验标签，明确区分 GT 缺失、低基线与删失，不训练新模型。详见 [EXP005-A 实验协议](experiments/exp005a/README.md)。
