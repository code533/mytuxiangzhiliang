# EXP003.2 — 低冗余 IMU 特征、历史趋势和独立序列校准

## 研究目标

测试精简的过去 IMU 运动摘要是否比原始 16 项高度相关的 IMU 特征更有利于预测 KLT 多帧存活率和低存活率事件；**不把 KLT 存活代理标签误认为 VIO 几何可靠性或更新成功**。

MH_01/MH_05 已多次被用于诊断失败与提出特征假设，因此两序列均属于开发数据，**不可再充当从未见过的最终测试集**。

## 预先固定的实验划分

建议使用以下划分（下载、处理新数据之前就固定，不根据输出随意换序列）：

- Train: MH_01_easy、MH_05_difficult（已获取特征 CSV，可以复用）。
- Validation: V1_03_difficult（调模型选择和概率校准）。
- Test: V2_03_difficult（锁定方法后仅做独立评估）。

所有文件均须由相同 EXP002/EXP003 参数生成。三组中的序列、文件哈希不可重合。样本必须具有有效 S5/S10 标签和完整历史 IMU 窗口；任一划分若仅包含单一事件类别，程序会报错而不是返回虚假 AUROC。

## 固定七组消融

| 名称 | 输入 |
|---|---|
| constant | 常数训练先验 |
| image | 7 项当前图像质量指标 |
| imu_full | EXP003 的全部 16 项历史 IMU 指标 |
| imu_compact | 0.2s 平均角速度 + 0.5s 加速度模长标准差 |
| image_imu_full | image + imu_full |
| image_imu_compact | image + imu_compact |
| image_imu_compact_trend | image + imu_compact + 两项历史窗口差值 |

两项趋势是 0.2s - 0.5s 的平均角速度差、加速度模长标准差差值。它们是过去窗口对照，**不是加速度导数，不使用未来 IMU**。直接从已有 `survival_imu_features.csv` 计算，无需再次运行跟踪。

使用训练集拟合 Ridge(alpha=10) 预测 S5/S10，Logistic(C=1) 预测 `S_K < 0.7`。验证集用于 AUPRC 模型选择及单调 Platt 概率校准（校准斜率不得为负），最终测试只报告预先固定的比较。概率校准主要影响 Brier/LogLoss；不能据此假设 AUROC/AUPRC 提高。

Top 10% 预警召回采取并列排名的随机期望定义：常数模型的预期召回应为 10%，**不得根据视频时间顺序错误地计算为 48.8%**。

## 运行步骤

在仓库根目录安装：`pip install -e ".[dev]"`，先运行 `pytest -q`。

1. 为新增验证/测试序列提取多帧存活率：

```bash
vio-iqa-exp002 extract --sequence /datasets/V1_03_difficult --max-frames 0 --output-dir results/exp002/V1_03_difficult
vio-iqa-exp002 extract --sequence /datasets/V2_03_difficult --max-frames 0 --output-dir results/exp002/V2_03_difficult
```

2. 使用历史 IMU 丰富特征：

```bash
vio-iqa-exp003 augment --labels results/exp002/V1_03_difficult/survival_labels.csv --imu-csv /datasets/V1_03_difficult/mav0/imu0/data.csv --output-dir results/exp003/V1_03_difficult
vio-iqa-exp003 augment --labels results/exp002/V2_03_difficult/survival_labels.csv --imu-csv /datasets/V2_03_difficult/mav0/imu0/data.csv --output-dir results/exp003/V2_03_difficult
```

3. 使用 train/validation/test 分组运行 S5：

```bash
vio-iqa-exp0032 \
  --train results/exp003/MH_01_easy/survival_imu_features.csv results/exp003/MH_05_difficult/survival_imu_features.csv \
  --validation results/exp003/V1_03_difficult/survival_imu_features.csv \
  --test results/exp003/V2_03_difficult/survival_imu_features.csv \
  --horizon 5 \
  --output-dir results/exp0032/independent_s5
```

将 `--horizon 5` 改为 10，输出改到 `independent_s10`，可重复 10 帧实验。

## 输出及科学边界

- `heldout_ablation.csv`：MAE、RMSE、Spearman、原始及校准后的 AUROC/AUPRC、Brier、LogLoss、前 10% 风险召回。
- `heldout_predictions.csv`：测试帧实际 S_K、回归预测、raw/calibrated 风险分数。
- `validation_model_selection.csv`、`selected_from_validation.json`：验证集选模型、校准拟合结果。
- `protocol_manifest.json`：全部输入 SHA256、划分、样本数和特征配置。

本阶段不引入 GPU/神经网络。验证结果需按场景和事件发生率分组报告，不能把单一序列结果解释为跨地下/低照度环境普适有效。要进入 EXP004，仍需在全新序列上检验风险排序与校准，并补充几何内点、视差、更新残差等独立质量标签。
