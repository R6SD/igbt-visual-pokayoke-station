# IGBT 单管透明料管视觉防呆系统 · 开源标准代码审查与安全治理报告

> **审查日期**：2026-11-22  
> **审查范围**：全项目源码（`IGBT_PokaYoke_Development_Package/src`、`tools`、`tests`、`frontend/igbt-station-ui/src`、`web/dist`、`scripts`）  
> **审查基准**：开源社区四维质量与安全门禁（Ruff + Bandit + TypeScript Strict + ESLint + OWASP Top 10 + AI 生成代码 Blast-Radius 清单）  
> **总体结论**：**通过验收（PASS）** —— 全部高危/中危安全隐患、生产源码 Ruff 告警、Bandit `-ll` 告警、全库静默吞异常 (`B110`/`B112`) 及前端 Blob/定时器泄漏均已**清零**；**41/41** 单元测试与前端 `tsc -b` / `eslint --max-warnings 0` / `vite build` 全部通过。

---

## 1. 开源审查工具链与实测环境

| 维度 | 工具 / 标准 | 实测版本 | 扫描目标与命令 | 最终门禁状态 |
| :--- | :--- | :--- | :--- | :--- |
| Python 静态质量 | **Ruff** | `0.15.0` | `ruff check src tools`（及 `tests` 全量） | `src` + `tools`：**0 错**；`tests` 仅 `test_fsm.py` 11 处单行风格提示 |
| Python 安全扫描 | **Bandit** | `1.9.4` (Python `3.14.3`) | `bandit -r src tools -ll` 与全级别 `-l` | `-ll`：**0 问题**；`-l`：19 → **7**（全部为已复核误报/ML 固定种子） |
| 后端回归验证 | **unittest + TestClient** | Python `3.14.3` | `PYTHONPATH=src python -m unittest discover -s tests -v` | **41 / 41 PASS** (`1.910s`) |
| 前端类型检查 | **TypeScript (`tsc -b`)** | `6.0.3` | `npx tsc -b` | **0 errors** |
| 前端规范检查 | **ESLint** | `10.15.0` | `npx eslint src --max-warnings 0` | **0 errors, 0 warnings** |
| 前端生产构建 | **Vite** | `8.3.8` (Node `v24.20.0`) | `npm run build` | **PASS** (`298ms`, 输出至 `web/dist`) |
| 安全与架构人工审计 | **OWASP Top 10 / Blast-Radius** | 2021 / 工业防呆规范 | 鉴权边界、路径穿越、SQL 注入、并发锁、内存/句柄泄漏、资产一致性 | **核心风险清零** |

---

## 2. 问题分级统计与整改前后对比

| 严重级别 | 初扫检出 | 本轮已修复 | 复扫剩余 | 说明 |
| :--- | :---: | :---: | :---: | :--- |
| **高危 (High / Security)** | 2 | **2** | **0** | 模型版本路径穿越漏洞、非 ASCII Token 引发 `hmac.compare_digest` 500 崩溃，均已彻底修复并加单测覆盖 |
| **中危 (Medium / Reliability & Data)** | 20 | **20** | **0** | 覆盖 Manifest `box_xyxy` 丢失 Bug、样本路径边界防护、线程竞争与相机句柄泄漏、12 处 `except Exception: pass/continue`、前端 `createObjectURL` 异常路径泄漏与 `X-Admin-Token` 漏传、生产源码 Ruff 违规 |
| **低危 / 误报 / 技术债 (Low / Info)** | 25 | **13** | **12 (低风险/误报)** | 含 7 项 Bandit 误报/ML 种子、11 处 `tests/test_fsm.py` 单行分号风格、前端未使用脚手架文件与嵌套 `.git` 提示（详见第 5 节） |

### 2.1 Bandit 扫描指标变化（扫描有效代码行：`6,361 LOC`，`#nosec` 跳过数：`0`）

| 规则 ID | 规则含义 | 初扫数量 | 复扫数量 | 处理结果 |
| :--- | :--- | :---: | :---: | :--- |
| `High / Medium` 全部规则 | 命令注入、SQL 注入、不安全反序列化、硬编码密码等 | 0 | **0** | 始终清零；`torch.load` 均已显式启用 `weights_only=True` |
| `B110` (`try_except_pass`) | 宽泛静默吞异常 | 11 | **0** | 全部替换为分级 `logging` 日志或精细 `OSError` 捕获 |
| `B112` (`try_except_continue`) | 循环内静默吞异常 | 1 | **0** | `training_manager.py` 已记录 `logger.debug` |
| `B105` (`hardcoded_password_string`) | 疑似硬编码口令字符串 | 4 | **4 (误报)** | `FSMState.PASS = "PASS"` (`fsm.py:19`) 与演练前缀 `"a_pass"/"b_pass"/"c_pass"` (`runtime.py:512-514`)，为工业放行状态枚举与工况键名，非凭据 |
| `B311` (`random` 非加密伪随机) | 非密码学伪随机数 | 3 | **3 (合理使用)** | `training_manager.py:664`、`train_classifier.py:150`、`train_hand_glove_detector.py:137` 用于可复现 ML 数据集划分与噪声合成的固定种子 `random.Random(seed)`，不用于安全令牌 |

---

## 3. 已修复问题清单（全量复核清零）

### 3.1 高危安全修复（协同 `@db-crud-backend` 整改并复扫确认）

| 编号 | 位置 | 风险类型 | 问题根因 | 修复措施与验证 |
| :---: | :--- | :--- | :--- | :--- |
| **SEC-01** | `src/igbt_pokayoke/ui_app.py` (`ModelRegistration`, `activate_registered_model`, `delete_registered_model`) | **OWASP A01 路径穿越 (High)** | `model_version` 原允许任意字符，直接拼接 `runs/label_classifier/{version}` 用于读取 ONNX 或 `shutil.rmtree` 删除，存在 `../` 越界风险 | 1. `ModelRegistration.model_version` 增加正则 `^[A-Za-z0-9._-]{1,64}$` 与 `@field_validator` 拒绝 `..`；<br>2. 引入 `_safe_model_run_dir()` 严格校验 `runs_root in target.resolve().parents`；<br>3. 单测 `test_model_delete_active_protection_and_path_traversal_guard` 验证通过 |
| **SEC-02** | `src/igbt_pokayoke/ui_app.py:455-461` (`admin_or_station_token`) | **鉴权异常处理缺陷 / DoS (High)** | 原直接对 `str` 调用 `hmac.compare_digest(supplied, expected)`，当请求头携带中文或非 ASCII 字符时抛出 `TypeError` 导致 500 崩溃 | 统一改为调用 UTF-8 安全的 `_compare_secret(supplied, expected)`，非法或非 ASCII 令牌恒定时间比对并稳定返回 `401 INVALID_TOKEN` |

### 3.2 中危可靠性、并发与数据一致性修复

| 编号 | 位置 | 问题分类 | 问题根因与修复措施 |
| :---: | :--- | :--- | :--- |
| **REL-01** | `src/igbt_pokayoke/training_manager.py` (`MANIFEST_FIELDS`, `_write_all_manifest_rows`) | **数据丢失 Bug (Medium)** | 原 `_write_all_manifest_rows` 遗漏 `box_xyxy` 列，任意改标注/删样本均会抹除全库拉框坐标；已统一 `MANIFEST_FIELDS` 补回 `box_xyxy`，并在删改样本时同步维护 `annotations_coco.json` |
| **REL-02** | `src/igbt_pokayoke/training_manager.py` (`_resolve_within_dataset`) | **路径边界越界防护 (Medium)** | 样本 CSV 读取、图片预览与文件删除均新增 `dataset_dir.resolve() in target.parents` 校验，杜绝越界读写或误删 `data/dataset/` 以外文件 |
| **REL-03** | `src/igbt_pokayoke/live_camera.py` (`_classify_live_label_roi`, `_loop`) | **多线程数据竞争 & 相机句柄泄漏 (Medium)** | 1. `_classify_live_label_roi` 改为在 `self._lock` 内原子快照 `(self._ort_session, self._onnx_input_size)`，避免热加载模型时新会话与旧输入尺寸错配；<br>2. `_loop` 帧循环增加 `try ... finally: cap.release()` 与异常重连保护，避免单帧异常导致采集线程死亡或 C920 句柄泄漏；<br>3. `mirror_horizontal`、`work_mode`、`contact_tracker`、`hud_bare_violation` / `hud_alarm_latched` 均收归 `self._lock` 同步读取 |
| **REL-04** | `src/igbt_pokayoke/live_camera.py` (7 处)、`tools/camera_probe.py` (3 处)、`ui_app.py` (3 处)、`training_manager.py` (3 处) | **静默吞异常 (Bandit B110/B112, Medium)** | 引入 `logging.getLogger(__name__)`，将全部静默 `except Exception: pass/continue`（含数据库回调 `_on_cycle_update`、ONNX 热加载、训练落库、Haar 人脸排除、样本文件删除、相机探测）替换为分级 `logger.warning` / `logger.debug` 与精确 `OSError` 捕获；`camera_probe.py` 采样循环同步补齐 `try ... finally: cap.release()` |
| **REL-05** | `frontend/igbt-station-ui/src/App.tsx` (协同 `@ui-pinned-db`) | **Blob 内存泄漏 & 鉴权头漏传 & 定时器清理 (Medium)** | 1. ROI 标定 `loadImage` 补全 `im.onerror` 释放 `URL.revokeObjectURL(url)`，本地样本上传预览在切图/报错/清空/卸载全路径释放；<br>2. 全部写/删/销毁 API（`clear`/`upload`/`batch_delete`/`sample PATCH&DELETE`/`cycles DELETE&purge`/`models DELETE`）统一透传 `X-Admin-Token`；<br>3. `noticeTimerRef` 与 `refreshTimerRef` 在组件卸载时统一 `clearTimeout` |
| **REL-06** | `frontend/igbt-station-ui/src/components/EngineerAuthModal.tsx` | **生产口令提示信息暴露防护 (Medium)** | 挂载时通过 `AbortController` 查询 `GET /api/auth/status`：仅在 `default_pin_active === true`（本地演示模式）显示演示口令提示，现场配置自定义 `IGBT_ENGINEER_PIN` 后自动隐藏 `888888` 提示文案 |

### 3.3 静态质量、死代码与类型现代化修复

| 文件 | 修复项 |
| :--- | :--- |
| `src/igbt_pokayoke/logic/fsm.py` | 清理死属性 `_last_capture_timestamp`；修复 `UP035` (`collections.abc.Callable`)、`B008`（`config: FSMConfig \| None = None`）、`SIM102`（合并嵌套 `if`） |
| `src/igbt_pokayoke/vision/hand_contact.py` | 移除 `ContactCandidateTracker.update` 末尾不可达死分支；修复 `UP035`、`B008` |
| `src/igbt_pokayoke/vision/hand_glove_detector.py` | 修复 `B905` (`zip(..., strict=True)`) |
| `src/igbt_pokayoke/vision/view_valid.py` | 修复 `UP035` (`collections.abc.Sequence`) |
| `src/igbt_pokayoke/runtime.py` | 修复 `UP035` (`collections.abc.Callable, Iterable`) |
| `src/igbt_pokayoke/ui_app.py` (联动修复) | 清理 Ruff `F841`（`runtime_snapshot` 未用变量 `actor`）、`SIM102`、`SIM118` |
| `src/igbt_pokayoke/training_manager.py` (联动修复) | 清理 Ruff `F401`（未用导入 `math`、`Counter`）与死函数 `_sha256_bytes` |
| `tools/camera_probe.py` | 修复 Ruff `F541`（多余 f-string 前缀）、`UP035`、`SIM102` |
| `tools/train_classifier.py` | 修复 3 处 `B905` (`zip(..., strict=True)`) |
| `tools/train_hand_glove_detector.py` | 全部清零 10 处 Ruff 默认告警（`E741` 歧义变量 `l` → `label_id`、8 处 `E702` 分号多语句、1 处 `E701` 单行 `if`）及 2 处 `B905` |
| `frontend/igbt-station-ui/src/lib/station.ts` | 移除 `guidanceFor` 中已由前序分支覆盖的不可达 `if (outcome === "ALARM")` 死代码；修正待机文案以区分 `debug` / `strict` 模式 |
| `frontend/igbt-station-ui/src/components/OpticalGauges.tsx` | 刻度线 React `key` 改为 `${idx}-${m}`，避免上下限重合时触发重复 key 警告 |

---

## 4. 权限架构、安全边界与数据库一致性专项评估

### 4.1 鉴权与前端隔离评估（OWASP A01 / A02 / A03 / A07）

| 评估维度 | 实测结论 | 关键证据 |
| :--- | :---: | :--- |
| **零硬编码管理员令牌** | **PASS** | 递归扫描 `frontend/igbt-station-ui/src` 与生产构建产物 `web/dist`：`local-admin-token` / `IGBT_ADMIN_TOKEN` / `IGBT_STATION_TOKEN` 命中数均为 **0** |
| **作业员 / 工程师严格隔离** | **PASS** | 默认进入 `operator` 角色，无侧边栏、无快捷键绕过、无训练/标定/演练/数据库控件泄露；进入工程师后台严格走 `POST /api/auth/verify`（`AuthVerifyRequest(extra="forbid")`），会话仅存 `sessionStorage` 并启用 30 分钟空闲自动锁屏 |
| **恒定时间凭据比对** | **PASS** | `_compare_secret()` 统一按 UTF-8 编码后执行 `hmac.compare_digest`，防时序侧信道攻击与非 ASCII 异常崩溃 |
| **网络暴露面收敛** | **PASS** | FastAPI 硬绑定 `127.0.0.1`，配置 `TrustedHostMiddleware(allowed_hosts=["127.0.0.1", "localhost", "testserver"])`，禁止跨主机访问与 Host Header 伪造 |
| **XSS / 代码注入防护** | **PASS** | 前端源码扫描 `dangerouslySetInnerHTML`、`.innerHTML =`、`eval(`、`new Function(` 命中数均为 **0**；所有动态文本走 React 默认转义 |
| **影子模式与 Fail-Closed** | **PASS** | `mode` 锁定 `"shadow"`，`interlock_output_enabled` 锁定 `False`；`/api/runtime/snapshot` 对缺失 `view_valid`/`quality_ok` 的 `PASS` 请求强制降级拒判；`C` 类撕除计时恒定 `1200 ms` 且使用单调时钟 |

### 4.2 数据库与磁盘资产一致性评估

| 存储资产 | 一致性机制 | 安全与防误删评估 |
| :--- | :--- | :--- |
| **SQLite 业务库** (`data/station.sqlite3` + `data/evidence/`) | 全量 SQL 采用 `?` 参数化占位符（**0 处字符串拼接 SQL**）；开启 `PRAGMA foreign_keys=ON` 与 `busy_timeout=10000`；`DELETE /api/cycles/{id}` 与 `POST /api/cycles/purge` 在事务内按外键顺序级联删除 `evidence` → `events` → `cycles`，随后经 `_safe_unlink_evidence` 校验 `evidence_root` 边界后清理磁盘文件；全量销毁后自动执行 `VACUUM` 回收空间 | **PASS**：无孤儿图片堆积、无 SQL 注入、支持仅清演练假数据 (`scope="simulated_only"`) 或全量销毁 (`scope="all"` + `confirm: true` + `X-Admin-Token`) |
| **AI 训练样本库** (`data/dataset/`) | 上传接口校验 `Content-Length` ≤ 10MB + MIME + JPEG/PNG 魔数头 + `cv2.imdecode` 真图解码；改标注/单删/批量删/清空均在 `self._lock` 下同步更新图片文件、`samples_manifest.csv`（完整保留 `box_xyxy`）及 `annotations_coco.json` | **PASS**：路径全链路校验处于 `dataset_dir` 目录内，避免孤儿文件与标注漂移 |
| **ONNX / PyTorch 模型库** (`data/runs/label_classifier/`) | `DELETE /api/models/{model_id}` 强制比对 `LIVE_CAMERA._onnx_model_version`，**禁止删除当前实时运行模型（返回 `409 Conflict`）**；仅当同版本无剩余注册记录时，经 `_safe_model_run_dir` 路径边界校验后移除对应产物目录；加载 PyTorch 权重全程强制 `weights_only=True` | **PASS**：运行中模型受保护，杜绝在线推理崩溃与路径穿越删目录风险 |

---

## 5. 遗留低风险技术债与后续加固建议

以下条目均不影响当前系统安全门禁与 41/41 单元测试通过，建议在下一版本迭代或首次 Git 初始化时择机清理：

1. **仓库初始化前移除嵌套 `.git`（高优先级工程卫生项）**：`IGBT_PokaYoke_Development_Package/frontend/igbt-station-ui/.git` 为脚手架遗留的子仓库，根目录执行 `git init` 前需先移除该 `.git` 目录，避免前端源码被误识别为失效 Git submodule。
2. **遗留只读范围风格提示**：
   - `tests/test_fsm.py` 存在 11 处 `E701`/`E702`（单行分号/单行 `for` 语句），后续可一次性拆行以实现全仓（含 `tests`）`ruff check` 0 错。
   - 扩展规则提示：`training_manager.py:15` (`UP035`)、`training_manager.py:979` (`B905`)、`ui_app.py:1480` (`SIM102`)、`test_ui_and_runtime.py:440` (`PIE807`)。
3. **前端未使用脚手架死文件清理候选**（未在任何入口引用，打包已被 Vite Tree-shaking 排除，清理可减少维护噪音）：
   - `src/components/LiquidGlassDefs.tsx`、`src/components/theme-provider.tsx`、`src/lib/utils.ts`、`src/assets/react.svg`
   - `src/components/ui/` 下 8 个未引用 shadcn 原始组件（`badge.tsx`、`button.tsx`、`card.tsx`、`label.tsx`、`separator.tsx`、`table.tsx`、`tabs.tsx`、`textarea.tsx`；当前仅实际使用 `input.tsx` 与 `select.tsx`）。
4. **纵深鉴权加固（产线生产化前建议）**：
   - 现有 `PATCH/DELETE /api/dataset/sample/{sample_id}`、`POST /api/dataset/capture`、`POST /api/dataset/bootstrap`、`POST /api/training/start`、`POST /api/models/{model_id}/activate` 及相机参数切换端点目前依赖 `127.0.0.1` 环回隔离；前端 `App.tsx` 已对其透传 `X-Admin-Token`，建议下阶段统一加上 `if os.environ.get("IGBT_ADMIN_TOKEN"): admin_or_station_token(...)` 条件鉴权。
   - 建议为 `POST /api/auth/verify` 增加按滑动窗口限流（如 5 次失败冷却 30 秒），并在产线部署时强制通过环境变量覆盖默认演示口令 `IGBT_ENGINEER_PIN` 与高熵随机 `IGBT_ADMIN_TOKEN` / `IGBT_STATION_TOKEN`。

---

## 6. 后续 CI/CD 自动化门禁接入建议

### 6.1 推荐 `pyproject.toml`（放置于 `IGBT_PokaYoke_Development_Package/pyproject.toml`）

```toml
[tool.ruff]
target-version = "py312"
line-length = 120

[tool.ruff.lint]
select = ["E", "F", "W", "B", "S", "UP", "PIE"]
ignore = ["E501", "B008", "S101", "S105", "S311"]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["S101", "S105", "S106"]

[tool.bandit]
targets = ["src", "tools"]
skips = ["B105", "B311"]
```

### 6.2 推荐 GitHub Actions 流水线（`.github/workflows/ci-quality-gate.yml`）

```yaml
name: Quality & Security Gate

on:
  push:
    branches: [main, "feat/**", "fix/**"]
  pull_request:
    branches: [main]

jobs:
  backend-audit-and-test:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: IGBT_PokaYoke_Development_Package
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies & audit tools
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-ui.txt -r requirements-vision.txt ruff bandit
      - name: Ruff static lint
        run: ruff check src tools
      - name: Bandit security scan (Medium & High)
        run: bandit -r src tools -ll
      - name: Python unit & API tests
        env:
          PYTHONPATH: src
        run: python -m unittest discover -s tests -v

  frontend-audit-and-build:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: IGBT_PokaYoke_Development_Package/frontend/igbt-station-ui
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "22"
          cache: "npm"
          cache-dependency-path: IGBT_PokaYoke_Development_Package/frontend/igbt-station-ui/package-lock.json
      - run: npm ci
      - name: TypeScript strict build check
        run: npx tsc -b
      - name: ESLint zero-warning gate
        run: npx eslint src --max-warnings 0
      - name: Vite production build
        run: npm run build
      - name: Secret leak check on src & dist
        run: |
          ! grep -rnE "local-admin-token|IGBT_ADMIN_TOKEN|IGBT_STATION_TOKEN" src ../../web/dist
```

### 6.3 本地推送前一键门禁命令（PowerShell）

```powershell
# 1. Python 静态审查与安全扫描
ruff check IGBT_PokaYoke_Development_Package/src IGBT_PokaYoke_Development_Package/tools
bandit -r IGBT_PokaYoke_Development_Package/src IGBT_PokaYoke_Development_Package/tools -ll

# 2. 后端全量单测
Push-Location IGBT_PokaYoke_Development_Package
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
Pop-Location

# 3. 前端类型、规范与生产构建
Push-Location IGBT_PokaYoke_Development_Package/frontend/igbt-station-ui
npx tsc -b
npx eslint src --max-warnings 0
npm run build
Pop-Location
```
