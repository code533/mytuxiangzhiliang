# EXP001 — 成对退化与真实低照度评估

研究目标：检验单帧图像质量指标是否与下一帧视觉跟踪退化有关，不能直接解释为 VIO 轨迹精度变化。

## 安装

在仓库根目录执行：pip install -e ".[dev]" ，然后运行 pytest -q。

## 实验 A：EuRoC 基线和受控退化

从 https://projects.asl.ethz.ch/datasets/euroc-mav/ 获取 EuRoC ASL 目录，确保含有 mav0/cam0/data.csv。

    vio-iqa-exp001 run --sequence /datasets/MH_01_easy --max-frames 300 --output-dir results/exp001/MH_01_easy
    vio-iqa-exp001 report --input-dir results/exp001/MH_01_easy

再使用 MH_05_difficult 重复实验。可以传入 --max-frames 0 处理完整序列，或使用 --stride N 均匀抽取相邻帧对。初次先运行少量数据检查格式。

## 实验 B：TUM-VI slides 真实暗光

下载 https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset 的 slides 序列，选择 Euroc/DSO 512×512 导出格式（不是 ROS bag），保证解压目录具有 mav0/cam0/data.csv 和图像文件。

    vio-iqa-exp001 run --sequence /datasets/dataset-slides1_512_16 --uint16-white 65535 --circle-radius 0.45 --max-frames 300 --output-dir results/exp001/tumvi_slides1
    vio-iqa-exp001 report --input-dir results/exp001/tumvi_slides1

如果源图是 uint16，必须显式指定 --uint16-white，且应根据实际源图量程选择全序列固定值。上面的 65535 仅为 16-bit 满量程示例，不代表已针对 TUM-VI 标定；不要逐帧归一化。若为 uint8 则无需此参数。

鱼眼黑边不能计入暗像素：建议 --mask valid_fov.png 使用经过验证的静态有效视场掩膜；或者以 --circle-radius R 指定临时中心圆形 ROI。圆形 ROI 不等于官方相机标定。掩膜与圆形选项互斥。跨相机/位深/映射设置的梯度量不宜直接比较。

## 科学设计

对同一个相邻帧对 (I_t, I_(t+1))，施加 clean 和 12 个退化条件。两帧受同类别和程度的退化；噪声对每帧使用确定性种子。条件包括 Gaussian Blur、方向性运动模糊、低光照、低对比度、高斯噪声和过曝。低光照仅做亮度缩放，不是实际传感器噪声/增益物理模型。

预测指标严格只用 I_t 计算，标签使用两帧 Shi–Tomasi + KLT 前后向一致性。分析同一帧对的退化与 clean 差值，并按序列和退化种类分层。跟踪成功率与有效跟踪点绝对数需同时报告，以避免改变角点分母造成误导。

## 产出

- frame_pairs.csv：逐帧对 × 条件的指标及跟踪代理标签。
- manifest.json：完整运行条件、随机种子、位深映射和 ROI。
- natural_correlations.csv：clean 图像的自然变化相关性。
- paired_deltas.csv：相对于相同帧对 clean 的指标和跟踪变化。
- condition_summary.csv：按序列 × 条件统计 mean/median delta。
- delta_correlations.csv：按序列 × 退化类型统计探索性 delta Spearman。
- paired_count_delta.png：有效跟踪点的平均配对变化。

## 限制与完成标准

Forward-backward consistency 是代理标签，不是几何真值，也不是 VIO ATE/RPE；同一序列临近帧相关，本阶段不报告 frame-level p-values。强模糊可改变角点检测的分母，需同时看比例和点数。ROI、bit-depth mapping、帧间隔、运动和特征检测阈值是重要混杂因素。后续需按完整序列留出测试并做 IMU 运动分层。

本 PR 只提供可运行程序和合成测试；未下载或运行完整真实 EuRoC、TUM-VI 数据。真实数据需本地运行且不提交到 GitHub 仓库。
