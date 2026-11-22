# IGBT 单管透明料管视觉防呆系统

> **基于本机实时视觉追踪 + ONNX 分类器 + 确定性有限状态机（FSM）的透明料管标签防呆与 ESD 手套规范监测工作台（Loopback Shadow Mode）**

---

## 1. 项目简介与核心痛点

在 IGBT 单管透明料管（长约 **530 mm**，含黄管、蓝管等规格）站立式人工目检工位中，作业员双手握持料管两端，需按工艺规范对料管中央原始标签完成 **划除（A）、遮挡（B）或撕除（C）** 后方可流转。传统目检与朴素单帧视觉存在四大痛点：

1. **漏处理与混料风险**：人工目检易出现未处理完整标签直接流转。
2. **「无标签」误判陷阱**：单目相机极易将「料管翻面、手部遮挡、移出视野、失焦模糊或强反光过曝」误判为「标签已撕除」，导致严重假放行（False Accept）。
3. **握持手与操作手混淆**：双手握持料管两端属于正常持管动作，不应误触发撕标接触计时。
4. **防静电（ESD）作业规范缺失**：产线要求佩戴**白色防静电手套**操作，裸手作业存在静电击穿器件风险。

### 本系统的防呆闭环机制

所有放行与报警结论均由**确定性 Python FSM 状态机 + 单调时钟**依据多源视觉证据联合裁决，**严禁 AI 分类器单帧直接输出放行信号**：

| 防呆类别 | 工艺判定逻辑 | Fail-Closed 安全约束 |
| :--- | :--- | :--- |
| **A 类 · 划除 (`crossed_out`)** | 油性笔划除原始标签，连续多帧稳定识别且置信度与边际（Margin）达标后放行 | 模糊/过曝/置信度不足时降级拒识 |
| **B 类 · 遮挡 (`covered`)** | 贴纸或胶带完整覆盖原始标签，连续稳定确认后放行 | 覆盖不全或识别为 `complete` 触发报警 |
| **C 类 · 撕除 (`absent`)** | ① 先在中央 `label_contact_roi` 捕获手部接触候选（与两端 `left/right_grip_zone` 严格隔离）；② **待手完全离开、标签面正向且 `view_valid = true` 后**，采用单调时钟连续无标签满 **1.20 s（1200 ms）** 才放行 | 手遮挡期间绝不累计计时；翻面/移出/失焦立即置 `VIEW_INVALID` 并清零计时 |
| **ESD 白手套防呆** | **车间严格模式 (`production`)** 强制检测白色防静电手套，裸手入画即刻报警拦截；**桌面调试模式 (`debug`)** 允许裸手演示与联调 | 默认开启**水平镜像对齐**，确保物理左右手与画面左右一致 |

> **安全运行边界（Shadow Mode）**
> 本系统固定仅监听本机回环地址 `127.0.0.1:8765`，**不开放任何 PLC、继电器、串口或其他硬件联锁输出接口**；前端资源完全本地托管，零外部 CDN 依赖。

---

## 2. 功能界面与角色权限

系统严格分离**一线作业员视图**与**工程师维护工作台**，杜绝产线误触工程参数：

### 2.1 作业员视图（默认首页：`http://127.0.0.1:8765/`）

- **100vh × 100vw 固定纯净单屏**：零外层滚动条、零侧边栏、零视频画面叠字遮挡、零工程配置入口泄露。
- **核心呈现**：左侧 16:9 完整实时相机画面（含动态追踪框与手部合规标记），右侧精简安灯（Andon）状态、A/B/C 实时判定与 C 类 **1.20 s** 倒计时进度条。

### 2.2 工程师工作台（密码鉴权 · 4 步引导式界面）

- **鉴权方式**：点击右上角「工程师解锁」，通过 `POST /api/auth/verify` 验证口令后进入工作台；**30 分钟无操作自动锁屏**回退至作业员视图，或点击顶栏「锁定退出」即时注销。
- **口令说明**：本地演示默认口令为 **`888888`**（可通过环境变量 `IGBT_ENGINEER_PIN` 或 `IGBT_ADMIN_TOKEN` 覆盖；前端源码与构建产物中**零硬编码明文令牌**）。
- **4 步工作台布局**（48px 顶栏导航 + 28px 底部硬件/推理遥测栏 + 局部独立滚动区）：

| 步骤 | 模块名称 | 核心功能 |
| :---: | :--- | :--- |
| **Step 1** | **实时监控** | 左侧固定 16:9 全景实时流 + 右侧顶部**永久置顶锁定的「4× 动态跟随特写」**；右侧下方独立滚动查看状态机与目标坐标、光学画质仪表（Laplacian 清晰度 / 亮度 / 过曝率）及工况演练。 |
| **Step 2** | **采集标注** | 共享左侧 16:9 镜头舞台，提供两种数据入库方式：**① 实时镜头拉框 / 自动追踪 5 类一键抓拍**；**② 本地选图直接上传入库**（`POST /api/dataset/upload`）。右侧支持按类别筛选、单张改标注（`PATCH`）、勾选批量/按类删除及 COCO 标注导出。 |
| **Step 3** | **训练部署** | 基于 PyTorch **MobileNetV3-Small**（随机初始化，零外网权重下载），按物理管/批次/视频组严格防泄漏切分，支持一键训练与增量续训、自动导出 ONNX 与数值一致性校验，验证通过后一键热切换上线。 |
| **Step 4** | **记录与数据库管理** | 包含 3 个独立子页：<br>① **检测周期与证据**：证据胶卷条、周期筛选表、FSM 跃迁时间线、单条周期级联删除与演练数据清理；<br>② **数据与数据库管理**：全景存储架构视图、业务库与 AI 样本库体量监控、历史模型安全清理（运行中模型受保护不可误删）、二次确认危险区批量清理与 SQLite `VACUUM` 空间回收；<br>③ **相机镜像与系统参数**：水平镜像切换、车间严格/桌面调试模式切换、五区归一化 ROI 画布标定、9 项阈值配置与配置变更历史。 |

> **产品展示页**：访问 `http://127.0.0.1:8765/?showcase=1` 可查看独立的产品光学原理与工位架构说明页。

---

## 3. 后台数据与存储架构

系统将运行期数据清晰划分为 **生产运行业务库**、**AI 训练样本库** 与 **AI 模型产物库** 三大独立边界（统一落盘于 `IGBT_PokaYoke_Development_Package/data/`，并已被 `.gitignore` 严格屏蔽，绝不进入 Git 仓库）：

```text
IGBT_PokaYoke_Development_Package/data/          (运行期数据根目录，支持 IGBT_DATA_DIR 自定义)
│
├── [1] 生产运行业务库（产线追溯与审计台账 · 不可与训练集混用）
│   ├── station.sqlite3                          # SQLite 业务数据库（6 张核心表）
│   │   ├── cycles                               #   检测周期主表（最终结论 PASS/ALARM/VIEW_INVALID）
│   │   ├── events                               #   FSM 状态跃迁与操作事件明细表
│   │   ├── evidence                             #   证据图片索引（含相对路径与 SHA256 校验值）
│   │   ├── configuration_history                #   五区 ROI 与 9 项阈值全量变更历史（只增不覆写）
│   │   ├── model_registry                       #   已注册模型版本与许可证/指标审计台账
│   │   └── station_state                        #   工位实时状态快照单例
│   └── evidence/<cycle_id>/*.jpg                # 不可覆盖的现场抓拍证据原图（与 evidence 表一一对应）
│
├── [2] AI 训练样本库（视觉模型训练集 · 文件系统 + 结构化清单）
│   └── dataset/
│       ├── images/
│       │   ├── complete/                        #   原始完整标签 ROI 切片
│       │   ├── crossed_out/                     #   A 类：已划除标签切片
│       │   ├── covered/                         #   B 类：已遮挡标签切片
│       │   ├── absent/                          #   C 类：已撕除无标签切片
│       │   └── unknown_challenge/               #   独立拒识挑战样本（不参与训练，仅验拒识）
│       ├── samples_manifest.csv                 #   样本总清单（含物理管/批次/视频组防泄漏分组 ID）
│       └── annotations_coco.json                #   COCO 格式目标框标注文件
│
└── [3] AI 模型产物库（训练检查点与 ONNX 实时推理引擎）
    └── runs/label_classifier/<model_version>/
        ├── label_classifier.pth                 #   PyTorch 模型权重（用于增量续训）
        ├── label_classifier.onnx                #   ONNX 推理产物（LiveCameraEngine 热加载运行）
        └── metrics.json                         #   验证/测试集准确率、混淆矩阵、完整标签误放行统计与 ONNX 误差
```

### 全生命周期增删改查（CRUD）与安全销毁对照

| 存储域 | 查看 / 统计 | 新增 / 修改 | 单条或批量删除 | 全量销毁与空间回收 |
| :--- | :--- | :--- | :--- | :--- |
| **生产运行业务库**<br>`station.sqlite3`<br>`data/evidence/` | `GET /api/storage/overview`<br>`GET /api/cycles`<br>`GET /api/evidence` | 实时 FSM 自动写入；<br>`PUT /api/config` 新增参数版本；<br>`POST /api/cycles/{id}/evidence/{kind}` | `DELETE /api/cycles/{cycle_id}`（事务级联删周期 + 事件 + 证据 + 磁盘图）<br>`DELETE /api/evidence/{evidence_id}` | `POST /api/cycles/purge`<br>· `scope="simulated_only"`：仅清 `CYC_*` 演练假周期<br>· `scope="all"`：清空全部业务周期/事件/证据图并执行 SQLite **`VACUUM`** 压缩磁盘空间 |
| **AI 训练样本库**<br>`data/dataset/` | `GET /api/dataset/summary`<br>`GET /api/dataset/samples`<br>`GET /api/dataset/coco` | ① 镜头抓拍：`POST /api/dataset/capture`<br>② 本地传图：`POST /api/dataset/upload`<br>③ 改类别：`PATCH /api/dataset/sample/{id}` | `DELETE /api/dataset/sample/{id}`（删单图并重写清单）<br>`POST /api/dataset/batch_delete`（按勾选 ID 或按类别批量清理） | `POST /api/dataset/clear`（清空 5 类切片图与 manifest，需 `X-Admin-Token` 鉴权与二次确认） |
| **AI 模型产物库**<br>`data/runs/` | `GET /api/models`<br>`GET /api/training/status` | `POST /api/training/start` 训练并自动注册；<br>`POST /api/models/{id}/activate` 热加载切换 | `DELETE /api/models/{model_id}`（删除台账行并安全清理对应 `runs/<version>` 目录，含路径穿越防护） | **运行中防误删保护**：若该模型正由实时推理引擎运行，后端强制拦截并返回 **`409 Conflict`** |

> **隔离原则**：清理生产演练记录不会影响 AI 训练样本集；清空 AI 样本集也不会破坏已有生产审计台账或正在运行的 ONNX 推理模型。

---

## 4. 项目目录结构

```text
IGBT 单管透明料管视觉防呆系统/
├── README.md                                  # 项目门面文档（本文件）
├── AGENTS.md                                  # 项目工程协作与质量铁律
├── .gitignore                                 # 根仓库忽略规则（屏蔽数据/权重/依赖/凭据）
├── .agents/rules/                             # 本地常驻开发规范
├── docs/
│   └── CODE_AUDIT_REPORT.md                   # 开源标准代码审查与安全治理报告（Ruff + Bandit + OWASP）
└── IGBT_PokaYoke_Development_Package/         # 核心工程交付包
    ├── .gitignore                             # 子工程忽略规则
    ├── IGBT_视觉防呆系统_完整开发文档.md          # 完整系统工程需求、光学预算与测试矩阵规范
    ├── DATASET_PROVENANCE.md                  # 数据集来源与合规审计说明
    ├── MODEL_LICENSES.md                      # 开源框架与模型权重许可证核查清单
    ├── MODEL_CARD_TEMPLATE.md                 # 训练模型卡输出模板
    ├── THIRD_PARTY_NOTICES.md                 # 第三方开源组件声明
    ├── requirements-ui.txt                    # FastAPI / Uvicorn / Pydantic 界面服务依赖
    ├── requirements-vision.txt                # OpenCV / NumPy / Pillow / PyTorch 视觉依赖
    ├── requirements-training.txt              # PyTorch / TorchVision / ONNX / ONNX Runtime 训练推理依赖
    ├── assets/                                # 工位与黄/蓝料管视觉参考图（仅作文档参考，严禁入训练集）
    ├── docs/                                  # 专项技术手册
    │   ├── local_ui_operations.md             #   本地工作台安装、备份恢复与排障指南
    │   ├── training_pipeline.md               #   防泄漏分组切分与 ONNX 导出校验流程
    │   ├── hand_glove_and_view_valid.md       #   白手套候选接触与 Fail-Closed 视角门控规范
    │   └── archive/                           #   历史过时文档归档（V2.0 校验记录、旧版先看这里）
    ├── examples/                              # CSV 样本清单、COCO 标注与 ROI 配置模板
    ├── scripts/
    │   └── start_local_ui.ps1                 #   Windows PowerShell 本地服务启动脚本
    ├── src/igbt_pokayoke/                     # Python 后端与视觉算法源码
    │   ├── ui_app.py                          #   FastAPI 回环服务、鉴权、SQLite CRUD 与存储管理 API
    │   ├── live_camera.py                     #   实时 UVC/C920 采集、水平镜像、动态追踪、白手套合规与 ONNX 推理
    │   ├── training_manager.py                #   样本库全生命周期 CRUD、防泄漏切分训练与 ONNX 导出
    │   ├── runtime.py                         #   标准五区 ROI 常量与多场景 FSM 演练构建器
    │   ├── logic/fsm.py                       #   A/B/C 工艺防呆与 1.20s 单调时钟确定性状态机
    │   └── vision/                            #   视觉基础算子（hand_contact / hand_glove_detector / view_valid）
    ├── tools/                                 # 离线探测与 CLI 训练工具（camera_probe / train_classifier 等）
    ├── tests/                                 # 后端单元测试套件（FSM / 相机探测 / UI API / 存储 CRUD）
    ├── frontend/igbt-station-ui/              # React 19 + TypeScript + Vite 8 + Tailwind v4 + shadcn/ui 前端工程
    ├── web/dist/                              # 前端生产构建输出（已加入 .gitignore，本地构建生成）
    └── data/                                  # 运行期 SQLite / 证据图 / 样本库 / 模型产物（已加入 .gitignore）
```

---

## 5. 快速上手与常用命令（Windows PowerShell）

### 5.1 环境依赖安装

前置要求：**Python 3.11+**、**Node.js 22+**、**Logitech C920（或标准 UVC USB 相机）**。

```powershell
Set-Location ".\IGBT_PokaYoke_Development_Package"

# 1. 创建并激活 Python 虚拟环境
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 2. 安装 Web 服务、视觉与训练/ONNX 推理依赖
pip install -r requirements-ui.txt -r requirements-vision.txt -r requirements-training.txt
```

### 5.2 前端类型检查、规范检查与生产构建

修改前端代码或首次拉取仓库后，需构建静态资源至 `web/dist/`（运行时由 FastAPI 直接托管，无需常驻 Node 进程）：

```powershell
Set-Location ".\IGBT_PokaYoke_Development_Package\frontend\igbt-station-ui"
npm ci
npx tsc -b
npx eslint src
npm run build
```

### 5.3 一键启动本机工作台（FastAPI + 实时 C920 相机）

**方式 A：快速启动（推荐本地开发与实机联调）**

```powershell
Set-Location ".\IGBT_PokaYoke_Development_Package"
$env:PYTHONPATH = "src"
$env:IGBT_ENABLE_LIVE_CAMERA = "1"
$env:IGBT_UI_PORT = "8765"
$env:IGBT_ADMIN_TOKEN = "change-me-local-admin-token"
$env:IGBT_STATION_TOKEN = "change-me-station-runtime-token"
# 可选：覆盖默认演示口令 888888
# $env:IGBT_ENGINEER_PIN = "your-6-digit-pin"

python -m igbt_pokayoke.ui_app
```

**方式 B：交互式安全脚本启动（自动提示输入令牌）**

```powershell
.\IGBT_PokaYoke_Development_Package\scripts\start_local_ui.ps1 -Port 8765
```

启动后浏览器打开 **`http://127.0.0.1:8765/`** 即可进入作业员纯净单屏（点击右上角「工程师解锁」，演示口令 `888888`）。

#### 核心环境变量一览

| 变量名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `IGBT_UI_PORT` | `8765` | 本机回环监听端口（固定绑定 `127.0.0.1`，禁止外网暴露） |
| `IGBT_DATA_DIR` | `./data` | SQLite、证据图、训练样本与模型运行期根目录 |
| `IGBT_ENABLE_LIVE_CAMERA` | `1`（直接运行 `ui_app` 时） | 是否启动后台实时 USB/C920 采集线程（`0` 为纯离线/单测模式） |
| `IGBT_ENGINEER_PIN` | `888888` | 工程师模式解锁口令，生产部署务必修改 |
| `IGBT_ADMIN_TOKEN` | 未设置时需配置 | 写配置、删除样本/周期/模型等高权限操作的 `X-Admin-Token` 校验令牌 |
| `IGBT_STATION_TOKEN` | 未设置时需配置 | 外部采集适配器推送 `/api/runtime/snapshot` 的独立运行时令牌 |

### 5.4 运行单元测试（37+ 项覆盖 FSM / 相机 / API / 数据库 CRUD）

```powershell
Set-Location ".\IGBT_PokaYoke_Development_Package"
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

### 5.5 开源标准代码审查与安全扫描（Ruff + Bandit + ESLint）

```powershell
# Python 代码规范与潜在缺陷检查
ruff check IGBT_PokaYoke_Development_Package/src IGBT_PokaYoke_Development_Package/tools IGBT_PokaYoke_Development_Package/tests

# Python 安全漏洞静态审计（中/高危）
bandit -r IGBT_PokaYoke_Development_Package/src IGBT_PokaYoke_Development_Package/tools -ll

# 前端 TypeScript 严格检查与 ESLint
Set-Location ".\IGBT_PokaYoke_Development_Package\frontend\igbt-station-ui"
npx tsc -b
npx eslint src
```

完整静态安全审计、OWASP 边界评估与整改记录详见 [docs/CODE_AUDIT_REPORT.md](docs/CODE_AUDIT_REPORT.md)。

---

## 6. Git 版本控制与提交规范

本项目严格执行专业 Git 与私有 GitHub 仓库治理规范：

### 6.1 分支策略

- **`main` 主干保护**：`main` 始终保持可编译、全部单测通过、零高危审计告警的可运行状态。
- **特性分支隔离**：新功能、修复或重构均在分支开发验证通过后再合并回 `main`：
  - `feat/<topic>`：新增功能模块
  - `fix/<topic>`：缺陷与安全漏洞修复
  - `refactor/<topic>`：架构或界面重构
  - `docs/<topic>`：文档、规范与 README 维护
  - `chore/<topic>`：依赖更新、目录清理与 CI 配置

### 6.2 Conventional Commits 提交规范

```text
<type>(<scope>): <简明中文或英文摘要>

示例：
feat(storage): 新增全景存储概览与 SQLite 级联清理/VACUUM 接口
fix(camera): 修复水平镜像切换后背景差分状态残留问题
refactor(ui): 工程师监控页 4x 动态特写顶部永久置顶锁定
docs(readme): 编写项目根目录 README 与双库存储架构说明
chore(repo): 归档过时 V2.0 文档并完善私有仓库 .gitignore
test(api): 补充样本本地上传、批量删除与防误删模型单元测试
```

### 6.3 推送前强制质量门禁（Pre-Push Gate）

每次 `git commit` 与 `git push` 前必须逐项确认：

1. **仓库洁净度检查**：执行 `git status`，确认无 `.env`、无 `station.sqlite3*`、无 `data/evidence/`、无 `data/dataset/`、无 `*.pth` / `*.onnx`、无 `web/dist/`、无 `node_modules/` 混入暂存区。
2. **后端验收**：`python -m unittest discover -s tests -v` 全部 PASS，`ruff check` 与 `bandit -ll` 无中高危问题。
3. **前端验收**：`npx tsc -b`、`npx eslint src`、`npm run build` 零错误、零警告。

### 6.4 里程碑打标与安全回滚

- **版本发布 Tag**：重要交付节点使用附注标签记录变更并同步远端：
  ```powershell
  git tag -a v2.1.0 -m "release: 固定视口工作台、数据库 CRUD 管理与代码安全治理"
  git push origin main --tags
  ```
- **安全回滚红线**：对已推送至 GitHub 远端的历史提交，**严禁执行破坏性 `git push --force`**；发现线上回退需求一律使用 `git revert <commit-sha>` 生成反向提交，保留完整可追溯的工业审计日志。

---

## 7. 深入文档索引

- **工程总规范**：[IGBT_视觉防呆系统_完整开发文档.md](IGBT_PokaYoke_Development_Package/IGBT_视觉防呆系统_完整开发文档.md)
- **代码安全审查报告**：[docs/CODE_AUDIT_REPORT.md](docs/CODE_AUDIT_REPORT.md)
- **本地工作台运维与备份恢复**：[docs/local_ui_operations.md](IGBT_PokaYoke_Development_Package/docs/local_ui_operations.md)
- **AI 分类器防泄漏训练流水线**：[docs/training_pipeline.md](IGBT_PokaYoke_Development_Package/docs/training_pipeline.md)
- **白手套候选接触与视角门控**：[docs/hand_glove_and_view_valid.md](IGBT_PokaYoke_Development_Package/docs/hand_glove_and_view_valid.md)
- **合规与开源声明**：[DATASET_PROVENANCE.md](IGBT_PokaYoke_Development_Package/DATASET_PROVENANCE.md) · [MODEL_LICENSES.md](IGBT_PokaYoke_Development_Package/MODEL_LICENSES.md) · [THIRD_PARTY_NOTICES.md](IGBT_PokaYoke_Development_Package/THIRD_PARTY_NOTICES.md)
- **历史文档归档**：[docs/archive/](IGBT_PokaYoke_Development_Package/docs/archive/)
