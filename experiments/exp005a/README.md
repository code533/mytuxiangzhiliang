# EXP005-A｜面向 VIO 的几何退化预测标签构建

本阶段仅做**数据集构建、可用性与统计关联审查，不训练模型**。使用现有 EXP003 `survival_imu_features.csv` 和 EXP004-B 全序列 `gt_frame_geometry.csv`。按 `sequence + frame_idx` 精确关联；纳秒时间戳要求逐项 int64 严格相等，禁止近邻匹配和浮点数修补。导出每个起始帧的 K=5 与 K=10 两条记录（EXP004 起始帧 stride=5）。

## 标签

- `label_low_baseline`：GT 相机平移基线 < **1cm**，仅在具有真实相对位姿的记录可定义。 `low_gt_baseline` 属于有标签的正类；其他 GT 可评价记录为负类。无 GT 或序列尾部删失保持 NA。
- `label_low_epipolar`：在 `gt_status == gt_evaluated`、至少有 12 个成功 KLT 对应点的条件下，以 EXP004-B 已用 **1.5px** Sampson 容限得到的参考内点率 < **0.60** 为正类。低基线、GT 缺失、匹配不足、删失保持 NA（不能标成 0）。
- 以上阈值来自 EXP004 已固定的探索性协议；不是 VIO 更新成功/失败的真值，也不是可迁移的安全标准。事件是未来 K 帧后验标签，**不能直接当在线预测输入**。

## 运行

仓库根目录、PR 合并后：
```bash
git pull origin main
pip install -e ".[dev]"
pytest -q

vio-iqa-exp005a \
  --features \
    results/exp003/MH_01_easy/survival_imu_features.csv \
    results/exp003/MH_05_difficult/survival_imu_features.csv \
  --geometry \
    results/exp004b_full/MH_01_easy/gt_frame_geometry.csv \
    results/exp004b_full/MH_05_difficult/gt_frame_geometry.csv \
  --output-dir results/exp005a
```

## 输出与审计

- `MH_01_easy/geometry_prediction_labels.csv`、`MH_05_difficult/geometry_prediction_labels.csv`：当前帧 7 项图像指标、过去 0.2/0.5 s 16 项 IMU、未来几何目标、GT 状态、严格标签可用性 mask。
- `dataset_summary.csv`：按序列、K、目标的可评价样本、正例数量、事件率、特征可用帧数。
- `gt_status_counts.csv`：覆盖率；包含 GT 缺失、低基线、尾部删失。
- `event_runs.csv`：按 EXP004 stride=5 的连续正事件起始组，不把相邻采样组误当独立实验。
- `feature_event_associations.csv`：每项当前/历史预测指标与两类事件的探索性 Spearman 关联；稀有事件时非常不稳定。
- `dataset_manifest.json`：输入 SHA256、阈值、时间戳与禁用输入字段。

**特征输入白名单只包含 EXP003 的 7 项图像 + 16 项历史 IMU 统计。** GT 基线、GT 极线率、未来 S5/S10 和跟踪点数均为后验审查信息，不进入白名单。运行时检查 EXP002/EXP004 的起始特征数和存活率完全一致。

## 基于开发数据的前期审查

MH_01/MH_05 的完整序列几何结果已完成对齐；低基线标签约 9–14%，但正事件主要来自很少的长时间段；低极线内点事件 K5 只有 **9/1028**，K10 **29/1031**。因此不能仅凭这两条序列训练后宣称泛化准确率。更不能因低 GT 几何事件太少就用缺失 GT、低基线状态填充为负类。

进入 EXP005-B 前，建议新增未用于之前指标/模型开发的完整序列，确保每个目标在验证和测试上都有足够**独立事件片段**，并报告基线率、AUPRC/Brier 及分块不确定性。先验证低基线任务；低极线一致性任务若继续稀疏，暂缓二分类模型而保留连续标签研究。
