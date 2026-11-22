# 标签 ROI 分类训练流程

本阶段提供可审计的 MobileNetV3-Small 随机初始化基线工具。不下载预训练权重、不训练 `unknown_challenge`，也不把静态分类器解释为完整工艺放行器。

## 数据清单

从 ROI 采集与人工标注数据制作 CSV，模板见 `examples/samples_manifest_template.csv`。必需列：

| 字段 | 含义 |
|---|---|
| `sample_id` | 唯一样本编号 |
| `path` | 标签 ROI 图片路径；相对路径按 CSV 所在目录解析 |
| `label` | `complete`、`crossed_out`、`covered`、`absent` 或 `unknown_challenge` |
| `physical_tube_id` | 匿名物理料管编号 |
| `batch_id` | 生产批次编号 |
| `video_group_id` | 视频/采集组 ID；静态图也要填采集周期或采集组 ID |

建议保留模板里的扩展元数据列：周期、来源视频、raw 原图路径、帧 ID、采集时间、相机型号/分辨率/FPS/曝光/焦点/白平衡、管色、操作事件、匿名操作员编号和 ROI 配置版本。不要把可识别员工身份或产品敏感序列号写入清单。

所有三个分组 ID 都必填。工具用物理管、批次、视频组和文件 SHA256 构建连通分组；共享任一 ID 的样本会整体进入同一个 train/val/test 集。若因批次关联导致独立分组少于三个，工具停止，不会把相关帧拆开凑集合。

精确重复图片按 SHA256 检测。相同内容、相同类别的重复行保留在输出 manifest 中供追溯，但只训练一次；相同内容标成不同类别时中止。近似重复由物理管/批次/视频组隔离策略控制，不做未经验证的感知哈希自动删除。

## 环境和执行

Python 环境中安装 `requirements-training.txt`。先运行只读数据审计和分组 manifest 生成：

```powershell
python tools\train_classifier.py path\to\samples.csv --audit-only --output-dir runs\label_classifier
```

检查生成的 `dataset_manifest.csv`、`dataset_audit.json` 和重复记录后，再开始训练：

```powershell
python tools\train_classifier.py path\to\samples.csv --epochs 20 --batch-size 32 --input-size 320 --seed 2026 --output-dir runs\label_classifier
```

训练期间模型从 `weights=None` 随机初始化。默认按组约 70/15/15 切分；可通过 `--train-ratio`、`--val-ratio`、`--test-ratio` 调整。固定随机种子便于复现分组和训练顺序。缺少任一类的唯一训练样本、缺少验证/测试样本、挑战集与已知类共享管/批次/视频/hash、文件损坏或标签冲突均会中止。

## 每次运行的输出

- `dataset_manifest.csv`：逐样本原路径、标签、组分配、SHA256、去重关系和训练用途。
- `dataset_audit.json`：类别/集合统计、重复数、独立组数量及输入 CSV 哈希。
- `label_classifier.pth`、`label_classifier.onnx`、`model_metadata.json`、`MODEL_CARD.md`。
- `metrics.json`、`classification_report_test.json`、`confusion_matrix_test.csv`、`training_history.json`。
- `SHA256SUMS.txt`：本次输出文件校验和。

测试报告逐类列出 precision/recall/F1，并重点给出真实 `complete` 被误分为 `crossed_out`、`covered`、`absent` 的数量。ONNX 导出后会在最多 16 张独立测试图上比较 PyTorch 与 ONNX Runtime logits；默认 `atol=1e-4`、`rtol=1e-3`，不一致时将以失败退出。

`unknown_challenge` 要求与所有已知类在物理管、批次、视频组及 SHA256 上完全隔离，只用于计算拒识覆盖情况。阈值由验证集正确样本分数的探索性分位数生成，并在模型卡标明 **不可直接用于生产**；最终阈值需由独立挑战集和质量部门风险标准确定。

## 重要边界

本仓库当前没有用户现场标注数据，因此不能生成真实训练权重、实际数据集清单、性能指标或校验和。第一次运行必须提供有权使用的真实 ROI 图片和完整元数据。只用当前两张带检测框叠加的参考图训练不符合数据要求。静态图分类结果不代表全周期 FAR，也不允许模型单独输出生产 PASS。

## 许可证和来源

模型使用本项目训练图片和随机初始化参数；工具不下载或引入预训练权重。框架/运行时许可须按目标机实际安装的 wheel 版本及其随包 notices 复核，见 `THIRD_PARTY_NOTICES.md` 与 `MODEL_LICENSES.md`。图片和标注的授权状态须由数据所有者确认。
