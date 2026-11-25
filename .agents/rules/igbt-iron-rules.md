---
trigger: model_decision
description: Activate when planning large multi-module C++ and frontend features or formal Git releases in the IGBT station workspace.
---

# IGBT 项目大型开发与交付协作规则

当任务涉及跨 C++20 后端与前端 HMI 的多模块开发（≥3 个文件）或正式 Git 发布时遵循：
1. **需求对齐（按需 GrillMe）**：遇到模糊需求或架构/UI 选型分歧时，先调用 `ask_question` 对齐 2~3 个核心决策，不自行脑补；明确单点小改直接执行。
2. **按需加载技能 / MCP**：UI 改动加载 `ui-ux-pro-max` / `frontend-design` 与 `ui-design-workflow.md`；代码审校调用 `code-review-audit`；发布遵循 `git-version-control`。
3. **大型并行再开 Swarm**：仅跨后端/前端多模块并行开发或用户主动喊 `/teamwork` 时才拉起 `swarm`（配合 `swarm_pane` 划清文件归属），并由主协调员统一验证与维护 `127.0.0.1:8765` 服务。
4. **规范交付**：正式改动验收通过后按 Conventional Commits 提交（绝不提交运行期 SQLite/编译产物），远端有效时按需推送。
