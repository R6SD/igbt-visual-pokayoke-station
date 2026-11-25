# IGBT 单管透明料管视觉防呆系统 · 项目工程约定 (AGENTS.md)

## 1. 开工必读路径（索引式读取，禁止全仓盲扫）
1. **读顺序**：先查 [项目索引.md](./项目索引.md) → [docs/模块地图.md](./docs/模块地图.md) → [项目记忆.md](./项目记忆.md)，按需配合 `serena` 按符号读取目标函数/类。
2. **严格禁区**：禁止递归遍历 `node_modules`、`build`、`dist`、`.git`，禁止整份吞读数千行大文件（如 `App.tsx`、`database.cpp`、`http_server.cpp`、`training_manager.cpp`）。

## 2. 构建、测试与本机服务
- **C++20 后端**（`cpp_backend/`）：`cmake` 构建并运行 `ctest`（保证 `g++ -Wall -Wextra` 零编译错误与原生单测全绿）。
- **前端 HMI**（`IGBT_PokaYoke_Development_Package/frontend/igbt-station-ui/`）：修改后运行 `npm run build`（`tsc -b && vite build`）刷新 `IGBT_PokaYoke_Development_Package/web/dist`。
- **本机常驻服务**：`127.0.0.1:8765`（720P @ 25 FPS）由 `IGBT_PokaYoke_Development_Package/scripts/start_local_ui.ps1` 或主协调员统一管理与重启。

## 3. UI 排版与布局硬规范（不可违背）
1. **禁止小字**：基于 `16px` 基准，正文字号 **不得小于 `14px`**，严禁 `12px` 及以下辅助说明。
2. **三档字号 / 两档字重**：全站仅允许 `24px`（大标题/核心指标）、`16px`（正文/控件）、`14px`（次要信息）与 `400` / `600` 字重。
3. **呼吸留白与固定视口**：卡片内边距 ≥ `24px`、间距 ≥ `16px`；左全景 `16:9` 与右顶 `4×` 特写置顶锁定，保留 `:focus-visible` 焦点框（滚动容器须留足内边距防裁剪）。

## 4. 分级规划与按需协作（对齐全局省额铁律）
- **单点小改直接做**：明确的小范围修复或单文件改动由单 Agent 直接完成与验证。
- **模糊/跨模块先对齐**：遇到模糊需求、架构选型或涉及 ≥3 个文件时，先用 `ask_question`（GrillMe）对齐关键决策或给 ≤5 行方案。
- **按需开 Swarm**：仅在同时跨 C++20 后端与前端大型并行开发或用户显式调用 `/teamwork` 时才拉起 `swarm`（先加载 `swarm_pane` 划清文件归属）。
- **版本控制**：正式改动验收通过后遵循 `git-version-control`（Conventional Commits、不提交运行期数据库/编译产物、按需推远端；回滚用 `git revert`）。
