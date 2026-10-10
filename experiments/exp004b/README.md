# EXP004-B：利用 EuRoC 参考位姿独立检验多帧几何对应

## 目标

利用 EXP004-A 已保存的逐点 KLT 对应关系，结合 EuRoC `mav0/state_groundtruth_estimate0/data.csv` 和 `mav0/cam0/sensor.yaml` 中的 `T_BS`（相机坐标系 C 到 IMU/机体坐标系 B），在固定真实相机相对运动下评价极线几何一致性、参考旋转补偿视差、可三角化正深度比例。**不需要重新运行 KLT，也不使用 RANSAC 自拟合模型作为独立真值。**

EuRoC 约定：地面真值 `T_WB`（机体到世界）、相机 `T_BC=T_BS`（相机到机体），故 `T_WC=T_WB @ T_BC`，`T_C1_C0=inv(T_WC1) @ T_WC0`。数据中的四元数顺序是 `qw,qx,qy,qz`。位姿插值：平移线性插值、旋转 SLERP；相机时间戳必须从 `cam0/data.csv` 按 **int64 纳秒**精确读取，而不是用 EXP004-A 中可能发生浮点舍入的 `end_t_ns`。

## 运行

更新安装 `pip install -e ".[dev]"` 后：

```bash
vio-iqa-exp004b \
  --sequence /datasets/MH_01_easy \
  --exp004-dir results/exp004/MH_01_easy \
  --output-dir results/exp004b/MH_01_easy

vio-iqa-exp004b \
  --sequence /datasets/MH_05_difficult \
  --exp004-dir results/exp004/MH_05_difficult \
  --output-dir results/exp004b/MH_05_difficult
```

要求 EuRoC 数据目录含相机 `data.csv`、`sensor.yaml`、参考位姿 `state_groundtruth_estimate0/data.csv`；EXP004-A 结果须有 `frame_geometry.csv`、`geometry_matches.csv`、`manifest.json`。**脚本会核对原 EXP004-A 相机标定 SHA256 与新的原始标定文件一致**，且起始帧时间戳与相机索引精确匹配，避免误用错误序列数据。

默认只在参考相机平移基线 ≥ 0.01 m 时计算极线内点。基线不足的样本标记 `low_gt_baseline`，`gt_inlier_ratio_survivors` 留空，不将其当成匹配失败。可用 `--min-baseline-m 0.01` 显式设置阈值，但不得根据当前 MH05 的结果事后选择所谓“最佳”阈值。单点 Sampson 阈值默认为等效 1.5 像素，只是探索性指标。

## 输出

- `gt_frame_geometry.csv`：每个 (起始帧, K=5/10) 的真实平移基线、真实旋转角、像素位移、真实极线内点率、真实视差、正深度比例、原 EXP004-A 状态；
- `gt_match_geometry.csv`：每个特征在独立真实 E 下的极线误差、内外点标记、旋转补偿视差，并保留自拟合 RANSAC 标签；
- `status_crosstab.csv`：EXP004-A 状态 × 参考几何状态，便于解释正深度检验不足是否与低基线相关；
- `gt_correlations.csv`：存活率与真实几何质量指标的探索性相关性；
- `high_survival_geometry_warnings.csv`：高存活、GT 低基线或低独立内点率或 EXP004-A 正深度检验不足反例；
- `manifest.json`：传感器文件 SHA256、时间偏移和阈值。

注意：低平移基线场景中不能用本质矩阵约束可靠判别匹配；真实视差、实际跟踪位移、状态估计信息价值也不是等价指标。GT 参考几何仍然不是完整 VIO 更新质量真值。只有前 300 帧属于试运行，不允许从中推断整条序列泛化能力。

## 之后的分析

优先比较 MH_05 前 300 帧的 `insufficient_positive_depth` 是否对应较小参考基线。分别汇总 K=5 和 K=10、全部完整样本、低基线样本及可估计参考极线样本；对高存活、高自拟合内点却独立参考极线低一致性的样本进行重点审查。下一轮再运行全序列及新保留测试序列。
