# IGBT 单管透明料管视觉防呆系统 (v4.0 · 纯原生 C++20 架构)

> **基于 100% 原生 C++20 视觉与推理引擎（OpenCV 4 `dnn`/`ml` + DirectShow `1280×720 @ 25 FPS` 精密节拍器 + SQLite3 WAL + `cpp-httplib`）与无小字极简固定视口工作台的透明料管标签防呆与 ESD 手套规范监测系统**

---

## 1. 核心架构亮点（v4.0 纯 C++20 重构）

1. **100% 彻底零 Python 原生服务端（`cpp_backend/`）**：
   - 全量后端、UVC/C920 采集、多目标动态追踪、ESD 手套合规检测、确定性防呆状态机（FSM）、模型在线微调与推理、SQLite3 WAL 持久化及双路异步分块 MJPEG 流媒体均由 **原生 C++20（`igbt_station_server.exe`）** 单进程驱动。
2. **720P @ 稳定 25.0 FPS 精密节拍与防抖/跳帧保存**：
   - 底层锁定 **`1280×720`（MJPG 后置开启）+ `40 ms`（`25.0 FPS`）`steady_clock` 漂移补偿节拍器**。
   - **报警/证据防抖节流**：同一周期同一状态绝不重复刷盘，并内置 `2000 ms` 周期证据冷却保护。
   - **车间批量跳帧自动抓拍**：工程师端支持一键开启「每隔 $N$ 帧自动抓拍落盘」（`GET/POST /api/dataset/auto_capture`），到达张数上限自动停机。
3. **纯 C++ 双模推理与本地特征训练**：
   - 内置 64 维空间网格 + HSV + Sobel 对角笔画特征提取器与 **OpenCV `cv::ml::ANN_MLP`** 原生训练引擎（分组防泄漏切分 + 车间光学数据增强，秒级收敛并热加载 `.xml`）。
   - 同时原生支持通过 **`cv::dnn::readNetFromONNX`** 导入并热加载外部 `.onnx` 模型（`POST /api/models/import_onnx`），全程 SHA-256 完整性校验。
4. **无小字极简工业界面（≥14px 硬性排版规范）**：
   - 锁死 `16px` 基准字号与 **3 档字号体系**（`24px` 核心大数/标题、`16px` 正文/控件、`14px` 次要信息下限，全站 **0 处 `<14px` 小字**），字重仅保留 `400` 与 `600`。
   - 卡片 `24px` 留白、`>=16px` 元素间距，砍除冗余说明微文案；左 `16:9` 全景镜头与右顶 `4×` 动态特写永久置顶不动，右侧检查器纵向舒适滚动。

---

## 2. 项目目录结构

```text
IGBT 单管透明料管视觉防呆系统/
├── README.md                                  # 项目主文档（本文件）
├── AGENTS.md                                  # 项目五条开发与排版铁律
├── .coderabbit.yaml                           # CodeRabbit 严格审查配置（assertive）
├── .gitignore                                 # 根仓库忽略规则（屏蔽数据/权重/编译产物/凭据）
├── cpp_backend/                               # 100% 纯 C++20 原生后端工程
│   ├── CMakeLists.txt                         #   CMake 3.20+ 构建与 CTest 配置
│   ├── include/igbt/                          #   模块化 C++20 头文件（common / fsm / vision_engine / training_manager / database / http_server）
│   ├── src/                                   #   核心实现与 main.cpp 服务入口
│   ├── tests/test_cpp_station.cpp             #   原生 C++20 全链路单测与回环 HTTP 集成测试
│   └── third_party/                           #   MIT 单头文件库（cpp-httplib / nlohmann-json）
├── docs/
│   └── CODE_AUDIT_REPORT.md                   # 代码安全治理报告
└── IGBT_PokaYoke_Development_Package/         # 前端工程、脚本与运行期数据目录
    ├── frontend/igbt-station-ui/              #   React 19 + TypeScript + Vite 8 + Tailwind v4 极简工作台源码
    ├── scripts/start_local_ui.ps1             #   Windows PowerShell 一键拉起 C++20 服务脚本
    ├── assets/                                #   黄/蓝料管光学参考图（工况仿真使用）
    ├── web/dist/                              #   前端生产构建静态产物（由 C++ 服务直接托管）
    └── data/                                  #   运行期 SQLite / 证据图 / 训练集 / 模型（.gitignore 已屏蔽）
```

---

## 3. 编译、测试与启动（Windows PowerShell）

### 3.1 编译原生 C++20 服务端与运行 C++ 测试套件

前置环境：**CMake 3.20+**、**MSYS2 UCRT64 GCC 14+（含 `opencv` 与 `sqlite3`）**。

```powershell
$env:PATH = "D:\msys2\ucrt64\bin;D:\msys2\usr\bin;$env:PATH"
cmake -S cpp_backend -B cpp_backend/build -G "Unix Makefiles" `
  -DCMAKE_PREFIX_PATH="D:/msys2/ucrt64" `
  -DCMAKE_CXX_COMPILER="D:/msys2/ucrt64/bin/g++.exe" `
  -DCMAKE_MAKE_PROGRAM="D:/msys2/usr/bin/make.exe"

cmake --build cpp_backend/build -j 8
ctest --test-dir cpp_backend/build --output-on-failure
```

### 3.2 编译前端静态工作台

```powershell
Set-Location ".\IGBT_PokaYoke_Development_Package\frontend\igbt-station-ui"
npx tsc -b
npx eslint src --max-warnings 0
npm run build
```

### 3.3 启动本机 C++20 实机服务（`1280×720 @ 25 FPS`）

```powershell
$env:IGBT_ADMIN_TOKEN = "local-admin-token"
$env:IGBT_STATION_TOKEN = "local-station-token"
$env:IGBT_ENGINEER_PIN = "888888"
.\IGBT_PokaYoke_Development_Package\scripts\start_local_ui.ps1 -Port 8765 -WorkMode debug
```

启动后访问 **`http://127.0.0.1:8765/`**（默认进入纯净作业员单屏；右上角「工程师登录」口令由 `IGBT_ENGINEER_PIN` 指定，签发 30 分钟独立随机服务端会话令牌）。

---

## 4. 核心环境变量

| 变量名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `IGBT_UI_PORT` | `8765` | 本机回环监听端口（固定绑定 `127.0.0.1`） |
| `IGBT_DATA_DIR` | `IGBT_PokaYoke_Development_Package/data` | SQLite、证据图、训练样本与模型目录 |
| `IGBT_WEB_DIST_DIR` | `IGBT_PokaYoke_Development_Package/web/dist` | 前端静态构建目录 |
| `IGBT_ENABLE_LIVE_CAMERA` | `1` | 是否启动 C920 `1280×720 @ 25 FPS` 实时采集线程 |
| `IGBT_WORK_MODE` | `strict`（脚本默认 `debug`） | 初始工作模式（`strict` 产线严格防呆 / `debug` 桌面联调采样） |
| `IGBT_ENGINEER_PIN` | 未设置时禁用口令 | 工程师登录口令（零内置默认凭据） |
| `IGBT_ADMIN_TOKEN` | 未设置时写接口拒绝访问 | 管理端认证令牌 |
| `IGBT_STATION_TOKEN` | 未设置时快照写接口拒绝访问 | 工位外部证据上报令牌 |
