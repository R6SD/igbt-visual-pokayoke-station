---
trigger: always_on
---

# IGBT 项目本地常驻规则（GrillMe + 技能/MCP + Swarm + Git 版本控制）

1. **每次接收目标先用 GrillMe (`ask_question`) 拷问对齐需求**，禁止盲目脑补默认行为。
2. **主动调用匹配技能（`ui-ux-pro-max`、`frontend-design`、`code-review-audit`、`git-version-control`、`swarm_pane` 等）及 MCP / 搜索工具**。
3. **每次正式开发必须拆分任务并通过原生 `swarm` 多 Agent 协作完成**，由主协调员统一验收与维护本机服务（`127.0.0.1:8765`）；纯问答/查资料/单行配置免开 Swarm。
4. **每次正式改动验收通过后，必须按 Conventional Commits 规范执行 Git 提交、按需分支/打标并执行 `git push` 推送至 GitHub 远端**。
