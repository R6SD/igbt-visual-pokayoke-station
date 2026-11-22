# 手套候选检测与 view_valid

本阶段仅输出检测框、图像质量、视角有效性和 `hand_contact_candidate` 视觉候选；没有 FSM、C 计时或 PASS 输出。

## 类别与训练数据

`hand_glove` 与 `marker` 使用独立类别 ID。COCO ID 固定为 1=`hand_glove`、2=`marker`；工具会拒绝 `hand_or_pen`。模型是从随机初始化训练的 torchvision Faster R-CNN MobileNetV3-Large FPN，不请求预训练权重。

准备 `examples/hand_glove_coco_template.json`：

- 框标注手套手部区域，手掌/手指的包含范围须在标注规范里统一；记号笔单独标 `marker`。
- 每张图记录 `physical_tube_id`、`batch_id`、`video_group_id`、站位、匿名操作员、文件名和尺寸。
- 有 `hand_glove` 正框的现场图片必须经人工确认是本工位真实白色针织/ESD 手套，并记录手套类型、复核人匿名编号和复核日期。
- 工具按物理管、批次、视频组及图像 SHA256 连通分组；相同来源组不会跨 train/val/test。
- 工具要求 train、val、test 都含人工复核的真实现场白手套正样本。合成手、裸手公开数据或文字填写不能替代现场图像复核。

运行：

```powershell
python -m pip install -r requirements-vision.txt
python tools\train_hand_glove_detector.py path\to\hand_glove_coco.json --epochs 15 --device auto
```

每次训练生成 `hand_glove_detector.pth`、数据集分组 manifest、IoU 0.5 下的每类 precision/recall/F1、模型卡和 SHA256。检测 adapter 在 `src/igbt_pokayoke/vision/hand_glove_detector.py`；把 `src` 加到 `PYTHONPATH` 后，以 BGR 帧调用 `predict()`。没有白手套现场数据时不可能生成可信权重，训练入口会停止。

## 中央候选事件

`hand_contact.py` 只接受 `hand_glove` 检测框。候选框须达到置信度门槛、覆盖中央 `label_contact_roi`，且未明显进入任一左右握持区。两端 grip zone 都未标定时，函数 fail closed，不发候选。`marker`、未知类别和旧的混合类不触发。`ContactCandidateTracker` 只在候选从区外进入中央 ROI 时发一个事件边沿。

字段名为 `hand_contact_candidate`，证据类型固定为 `2d_visual_proximity_proxy`，`proves_physical_contact=false`。2D RGB 目标框不能证明皮肤/手套与标签表面真实接触；业务若要求物理接触证明，需额外传感器或经验证的强约束。

## view_valid 门控

使用 `examples/view_valid_config_template.json` 作为结构示例，实际 ROI 和画质阈值必须现场标定。模板默认 `quality_thresholds_approved=false`、`orientation_detector_ready=false`，因此结果保持无效，直到真实校准和方向检测验证完成。

判为有效必须同时满足：

- 当前设备 ID 与校准设备一致、分辨率与 ROI 配置一致、相机健康；
- 料管存在且仍在标定视野，标签窗口在位；ROI 框有效且 grip/contact 区已标定；
- 标签 ROI 无手套/笔等遮挡；亮度、过曝、欠曝、Laplace 清晰度通过现场批准的阈值；
- 标签中心没有超出校准窗口，采集帧 ID 连续、帧间隔合规；在稳定帧数和稳定时长内位置变化受控；
- 有可机器验证的方向参考，并确认标签面朝向相机；口头确认或“标签没检测到”都不算方向证据。

料管翻面/背面、标签窗口移出或偏移、遮挡、失焦、过曝/欠曝、丢帧、姿态不稳、方向证据缺失或任何输入未知都会返回 `view_valid=false` 和原因码。这里没有“无标签 => absent”的规则。

## 当前验证边界

代码和配置模板不等于现场模型验收。当前项目没有真实白手套标注数据、方向参考检测器或质量阈值标定报告；因此不能声明手套召回率、方向识别能力或工位 `view_valid=true` 已验证。`view_valid` 中 tube presence、标签窗口、方向参考等语义信号由后续工位定位/方向组件提供，缺省未知会 fail closed。
