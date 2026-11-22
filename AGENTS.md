# IGBT 单管透明料管视觉防呆系统 · 项目开发铁律 (AGENTS.md)

本项目所有会话必须无条件遵循以下四条工程铁律：

## 铁律一：先调用 GrillMe (`grill-me`) 拷问需求
- 每当用户下发新指令或开发目标时，先调用 `ask_question` 进行结构化追问（2~4 个直击核心决策点的问题），剥离模糊需求与隐含前提，确认后再动手，绝不自行脑补默认项。

## 铁律二：主动检索并调用相关技能（Skills）与 MCP
- 界面与视觉改动前遵循 `ui-ux-pro-max`、`frontend-design` 及 `ui-design-workflow.md`；
- 代码审查调用 `code-review-audit`（Ruff + Bandit + ESLint + OWASP）；版本控制调用 `git-version-control`；定制化配置查阅 `agy-customizations`。

## 铁律三：正式开发必须拆分任务并用原生 `swarm` 完成
- **触发边界**：凡涉及写/改代码、加功能、调界面或修多处 Bug 的正式开发任务，必须拆解任务并通过原生 `swarm` 协作完成；仅纯问答、查资料或单行配置改动可免开 Swarm。
- **默认角色拆分**：按 **「后端/视觉算法 Worker + 前端/UI 交互 Worker + 审查/测试验收 Worker」**（2~4 个专职 Worker）并行推进，严格划清文件边界避免写冲突。
- **协调与验收**：启动前加载 `swarm_pane` 技能并设置 `swarm(action="title")`；主协调员负责汇总验证（单测 + `ruff` + `bandit` + `tsc -b` + `eslint` + `vite build`）、统一重启本机常驻服务（`127.0.0.1:8765`），验收完成后清理 Worker。

## 铁律四：严格执行 Git 版本控制、规范提交与远端同步 (`git push`)
- 每次正式改动验收通过后，必须遵循 `git-version-control` 技能：核对 `git status`（不提交敏感数据/运行期数据库）、采用 Conventional Commits 规范提交、必要时创建特性分支或打版本 Tag，并在远端已配置且鉴权有效时执行 `git push` 同步到 GitHub 仓库；回滚已推送历史一律使用 `git revert`。
