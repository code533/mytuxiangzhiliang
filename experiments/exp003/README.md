# EXP003：图像质量 + 历史 IMU 预测跟踪退化

目标：在视觉更新前，仅使用当前帧图像指标及过去 0.2s / 0.5s 的 IMU 统计，预测 EXP002 的 S5/S10 多帧 FB-consistent 特征存活率，并重点比较低存活率事件的提前预警。**这是跟踪代理标签，而不是 VIO 更新可靠性的真值。**

## 运行前准备

需要已经通过 EXP002 获得的 `survival_labels.csv`，以及原始 EuRoC 的 `mav0/imu0/data.csv`。此前只包含 EXP002 评估结果的三个 ZIP 不包含时间对齐所需的原始 IMU 数据。无需重新运行光流。

在仓库根目录：

```bash
pip install -e ".[dev]"
pytest -q
```

## 1. 为两条序列添加严格因果 IMU 特征

```bash
vio-iqa-exp003 augment --labels results/exp002/MH_01_easy/survival_labels.csv --imu-csv /datasets/MH_01_easy/mav0/imu0/data.csv --output-dir results/exp003/MH_01_easy
vio-iqa-exp003 augment --labels results/exp002/MH_05_difficult/survival_labels.csv --imu-csv /datasets/MH_05_difficult/mav0/imu0/data.csv --output-dir results/exp003/MH_05_difficult
```

输出 `survival_imu_features.csv` 和 `manifest.json`。将原始 IMU 按统一的相机时刻对齐，只允许 `imu_timestamp<=camera_timestamp`。过去两个时间窗口各提取角速度范数均值/P90/最大值、粗略角速度积分、加速度范数均值及变化等，共 16 项输入特征。超出允许的时间缺口（默认 30ms）或历史覆盖不足时标记窗口无效，不偷看后续 IMU 进行插值。原始 IMU 未进行 gyro bias 校正或光电时间偏移补偿，换数据集时须先检查同步和单位。

## 2. 评估四组模型（同一批有效测试帧）

```bash
vio-iqa-exp003 evaluate --train results/exp003/MH_01_easy/survival_imu_features.csv --test results/exp003/MH_05_difficult/survival_imu_features.csv --horizon 5 --output-dir results/exp003/mh01_to_mh05_s5
vio-iqa-exp003 evaluate --train results/exp003/MH_01_easy/survival_imu_features.csv --test results/exp003/MH_05_difficult/survival_imu_features.csv --horizon 10 --output-dir results/exp003/mh01_to_mh05_s10
vio-iqa-exp003 evaluate --train results/exp003/MH_05_difficult/survival_imu_features.csv --test results/exp003/MH_01_easy/survival_imu_features.csv --horizon 5 --output-dir results/exp003/mh05_to_mh01_s5
vio-iqa-exp003 evaluate --train results/exp003/MH_05_difficult/survival_imu_features.csv --test results/exp003/MH_01_easy/survival_imu_features.csv --horizon 10 --output-dir results/exp003/mh05_to_mh01_s10
```

四组分别为 **constant / image / imu / image_plus_imu**。存活率由固定参数 Ridge 回归，低存活事件 `S_K < 0.7` 由 Logistic Regression 预测发生概率（常数组使用训练集事件比例）。这不是神经网络，不需要 GPU。数据集以整序列划分，所有标准化只由训练数据拟合。阈值 0.7 和预警预算 10% 是预先固定的探索性协议，并非部署阈值。

结果包含 `ablation_metrics.csv`（MAE、RMSE、Spearman、AUROC、AUPRC、Brier、前 10% 高风险帧精确率/召回率）、`ablation_predictions.csv` 和 `evaluation_manifest.json`。

## 评价方法与限制

- **主要问题：** 在同样的低存活事件判定和同一批完整 IMU 窗口样本上，融合模型是否比纯图像模型提高 AUPRC / 高风险召回率，同时不过度提高误报与概率校准误差？
- 若某条训练序列只有正类或负类，会提示增加训练序列；不能强行训练二分类器。
- 早期 IMU 时间窗不完整或存在缺口的帧被排除；四组模型在同一测试子集对比，和此前 EXP002 全部样本的 MAE 不能直接混比。
- 过去的 IMU 不能预知未来突然发生的运动，因此未必解释全部低存活事件。
- 目前尚无几何内点率、ATE/RPE 或完整 VIO 状态更新结果。仅两条 EuRoC 序列不足以证明域外泛化，下一轮需扩充序列并复核相机—IMU 同步、IMU bias 与模型校准。
