# EXP002 — 预测未来 5/10 帧特征存活率

本阶段使用当前图像的质量指标预测后验多帧 KLT 存活率，不使用未来帧作为预测输入。详细研究要求见 [研究路线图](../../docs/research-roadmap.md)。

## 标签

从每个起始帧的 Shi–Tomasi 候选点（默认最多 3000）按 8×6 网格轮询选择最多 240 个点。特征不补充、不复活；逐帧计算前后向 KLT 一致性，默认阈值 1 像素。`s_5` 和 `s_10` 分别为连续经过 5/10 次相邻帧转移后仍存活的比例。没有起始点时 `no_features=1` 且比例为 NaN；序列末尾不足 K 帧时标记 `censored_K=1` 且比例为 NaN，不视为失败。

## 运行

安装依赖：`pip install -e '.[dev]'`，测试：`pytest -q`。

```bash
vio-iqa-exp002 extract --sequence /datasets/MH_01_easy --max-frames 0 --output-dir results/exp002/MH_01_easy
vio-iqa-exp002 extract --sequence /datasets/MH_05_difficult --max-frames 0 --output-dir results/exp002/MH_05_difficult
vio-iqa-exp002 evaluate --train results/exp002/MH_01_easy/survival_labels.csv --test results/exp002/MH_05_difficult/survival_labels.csv --horizon 5 --output-dir results/exp002/eval_s5
vio-iqa-exp002 evaluate --train results/exp002/MH_01_easy/survival_labels.csv --test results/exp002/MH_05_difficult/survival_labels.csv --horizon 10 --output-dir results/exp002/eval_s10
```

`--max-frames 0` 表示整条序列。`--stride N` 只减少起始帧数，不会跳过单条轨迹的中间帧。TUM-VI 的 16-bit 图像需显式传入固定 `--uint16-white`，鱼眼图像可使用 `--mask` 或近似 `--circle-radius`。

提取结果为 `survival_labels.csv` 和 `manifest.json`。评价结果为 `heldout_metrics.csv`（常数基线、各单指标 Ridge、多指标 Ridge 的 MAE、RMSE、Spearman）、`heldout_predictions.csv` 和 `split.json`。

## 科学局限

这只是 FB-consistent 的短时/多帧存活代理标签，非真实匹配准确率、几何约束质量或 VIO ATE/RPE。单帧检测本身受到 Shi–Tomasi 相对阈值影响；未来需补固定 clean 候选对照和独立几何 RANSAC。所有标准化与插补仅在训练序列拟合。仅 MH_01 和 MH_05 两序列，尚不足以支持严格泛化结论；后续需要更多完整序列和单独验证集。
