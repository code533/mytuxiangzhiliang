# EXP004-B.1：几何退化类型与阈值敏感性

该实验**不重新跟踪 KLT、不需要原始图像/GPU**，直接复用 EXP004-B 的 `gt_frame_geometry.csv`、`gt_match_geometry.csv`、`manifest.json`，分开报告低基线、GT 缺失、GT 极线不一致和自估位姿恢复失败。

## 研究约束

1. 源 EXP004-B 的低基线（<1cm）样本没有保存真实极线误差；因此可以把平移阈值从 1cm 提高到 2/5/10cm，**不能在没有重新读取 GT 的情况下把阈值降到 1cm 以下并假装完成了评价**。
2. 极线误差门限对已保存的 GT Sampson 残差逐点重新判断（0.5/1/1.5/2/3px），不重算位姿，不修改 KLT。
3. 明确保留 `missing_gt`、`gt_low_baseline`、`tail_censored`，避免只在成功估计样本上报告几何质量造成选择偏差。
4. 分类是**描述性失效类型**：高自估 RANSAC 与低 GT 一致不自动等于“匹配错误”；GT 极线一致不自动等于“VIO 更新成功”。
5. 现有 MH_01/MH_05 均只采样前 300 帧，任何阈值对比都不能当作全数据集泛化检验。

## 运行

```bash
git pull origin main
pip install -e ".[dev]"
pytest -q
vio-iqa-exp004b1 \
  --input-dirs results/exp004b/MH_01_easy results/exp004b/MH_05_difficult \
  --output-dir results/exp004b1/sensitivity
```

输出：

- `threshold_sensitivity.csv`：基线/极线阈值网格下的有效样本比例、内点率和自拟合-GT 不一致案例数；
- `frame_threshold_sensitivity.csv`：每条有 GT 的轨迹在阈值网格中的内点率；
- `failure_classification.csv`：全部起始帧按失败类型分类，同时保留缺失 GT；
- `failure_class_counts.csv`：各序列、K=5/10 分类计数；
- `high_survival_warning_cases.csv`、`largest_self_vs_gt_disagreements.csv`：值得进一步检查的片段；
- `manifest.json`：本轮参数和禁止降低原始基线阈值的范围约束。

## 现有 300 帧数据的先验观察（仅用于机制分析）

在 MH_05 的 5 帧跨度，基线≥1cm 时 35 组具有 GT 可评估对应；要求≥10cm 时只有 17 组。MH_05 第 250 帧、10 帧跨度，真实极线内点率在 1.5px 下为约 38.2%，3px 放宽仍只有约 56.5%，而自估计 RANSAC 内点率约 87.0%。该现象值得针对原图及时间同步复查；不能直接推出物理因果。

后续建议对 MH_01/MH_05 完整序列运行 EXP004-A/B，并加入之前未参与开发的独立序列。同时优先检查 MH_05 第 95、155、240、250 帧附近的原始图像和对应关系，明确低基线、误匹配以及自估本质矩阵退化三者的区别。
