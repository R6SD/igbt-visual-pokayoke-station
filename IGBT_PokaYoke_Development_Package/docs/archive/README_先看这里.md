# IGBT 视觉防呆项目资料包（V2.0）

## 先看这里

1. 打开 `IGBT_视觉防呆系统_完整开发文档.md`，先阅读第 0、1、2、3、5、8、10 章。
2. `assets/01_黄管_站立持管检测框参考图.jpg` 与 `assets/02_蓝管_站立持管检测框参考图.jpg` 是用户提供的实际场景参考图，已嵌入开发文档。
3. 这两张图中 `hand_or_pen`、`label_intact` 及置信度为示意叠加，不代表真实模型测试性能。不要直接把这两张有叠加的图片作为训练样本。
4. 第一阶段只做工艺确认、现场量测、C920 摄像头探测和光学测试，不要先写完整 AI 程序。
5. 再按文档第 18 章逐阶段把 Codex 指令交给 Codex。每个阶段完成测试后再进入下一阶段。

## V2.0 的核心决定

- 工位是站立式人工目检，员工双手握住约 53 cm 料管两端。
- 左右正常握持区不触发 C；标签接触区域单独定义。
- 料管翻面/遮挡/移出视野/模糊属于 `VIEW_INVALID`，不能被当成标签已撕掉。
- C 的 1.2 秒只在手离开标签区、标签面重新正对相机且 ROI 有效后累计。
- C920 参数必须查询、设置、读回并进行稳定性实测；`cap.set()` 返回成功不能视为锁定成功。
- 先用 C920 做 MVP，达不到像素/控制/可靠性门槛时再升级第二视角或工业相机。
- 所有 A/B/C 结论由 Python FSM 根据视觉证据决定，不由模型单帧直接放行。

## 文档结构

- `V2.0_修订与深度调研校验记录.md`：本次增删改、官方来源核验与待现场确认项。

- `IGBT_视觉防呆系统_完整开发文档.md`：完整工程规范。
- `docs/local_ui_operations.md`：本地影子工作台安装、操作、备份恢复与故障排除。

## 本地影子工作台（当前版本）

在项目根目录创建并激活 Python 虚拟环境，安装 `requirements-ui.txt`；然后在 `frontend/igbt-station-ui` 使用 Node.js 22+ 执行 `npm ci` 和 `npm run build`，再运行 `scripts/start_local_ui.ps1`。界面固定监听 `127.0.0.1`，默认端口 8765。启动时设置独立的管理员令牌与运行时令牌。状态数据默认保存在 `data/station.sqlite3`，原始证据图保存在 `data/evidence/`。前端采用 React + shadcn/ui，构建后的资源由 FastAPI 本地提供。

本地界面当前是 shadow mode：没有任何生产联锁输出接口。其相机/视觉/FSM 状态由本机采集适配器通过带 `IGBT_STATION_TOKEN` 的 `/api/runtime/snapshot` 提供；未接入时显示 NOT_CONNECTED / VIEW_INVALID。管理员可维护 ROI、阈值与版本记录，并上传不可覆盖的证据图。详见 `docs/local_ui_operations.md`。
- `assets/`：用户上传的黄管、蓝管两张参考图片；旧版概念图保留为历史资料，不应覆盖现场参考图。

## P3 标签 ROI 分类基线

- 训练脚本：`tools/train_classifier.py`
- 输入清单模板：`examples/samples_manifest_template.csv`
- 环境依赖：`requirements-training.txt`
- 操作说明：`docs/training_pipeline.md`
- 当前没有现场标注数据；需先按说明准备数据集，工具不会把两张带叠加框的参考图当训练集。

## P4 手套候选与视角门控

- `src/igbt_pokayoke/vision/hand_glove_detector.py`：独立 `hand_glove` 与 `marker` 检测器适配器。
- `src/igbt_pokayoke/vision/hand_contact.py`：两端 grip zone 排除与中央手套进入候选事件。
- `src/igbt_pokayoke/vision/view_valid.py`：fail-closed 视角与画质门控；无机器验证方向证据时返回无效。
- `tools/train_hand_glove_detector.py`：COCO 检测器训练；强制 train/val/test 都含经复核的真实现场白手套样本。
- 详细格式和边界见 `docs/hand_glove_and_view_valid.md`。
