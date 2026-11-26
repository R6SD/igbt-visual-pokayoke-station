import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  AlertTriangle,
  CheckCircle2,
  Lock,
  RefreshCw,
  ScanBarcode,
  ShieldAlert,
  ShieldCheck,
  Unlock,
  X,
} from "lucide-react"
import { CameraFrame } from "@/components/CameraFrame"
import { type Data, labelText, reasonOf } from "@/lib/station"
import {
  RR_EVENT_TYPE_LABELS,
  RR_SAMPLE_TYPE_LABELS,
  RR_TRIGGER_LABELS,
  type RedRabbitAuditResponse,
  type RedRabbitStatus,
} from "./types"

interface RedRabbitOverlayProps {
  state: Data
  redRabbit: RedRabbitStatus | null
  adminToken: string
  manualOpen: boolean
  onCloseManual: () => void
  onOpenEngineerAuth: () => void
  onUpdated: () => Promise<void> | void
  flash: (msg: string) => void
}

async function rrRequest<T>(
  path: string,
  options: RequestInit = {},
  adminToken = ""
): Promise<T> {
  const headers = new Headers(options.headers)
  if (!headers.has("Content-Type") && options.body) {
    headers.set("Content-Type", "application/json")
  }
  if (adminToken) {
    headers.set("X-Admin-Token", adminToken)
  }
  const response = await fetch(path, { ...options, headers })
  let payload: Record<string, unknown>
  try {
    payload = (await response.json()) as Record<string, unknown>
  } catch {
    payload = {}
  }
  if (!response.ok) {
    const detail =
      typeof payload.detail === "string" && payload.detail
        ? payload.detail
        : `HTTP ${response.status}`
    throw new Error(detail)
  }
  return payload as T
}

function formatMsTimestamp(ms?: number | null): string {
  if (!ms || !Number.isFinite(ms)) return "—"
  return new Date(ms).toLocaleString("zh-CN", {
    hour12: false,
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  })
}

function formatCountdown(ms?: number | null): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return "—"
  const sec = Math.max(0, Math.floor(ms / 1000))
  const m = Math.floor(sec / 60)
  const s = sec % 60
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
}

export function RedRabbitOverlay({
  state,
  redRabbit,
  adminToken,
  manualOpen,
  onCloseManual,
  onOpenEngineerAuth,
  onUpdated,
  flash,
}: RedRabbitOverlayProps) {
  const rrState = redRabbit?.state ?? "PROD_READY"
  const isBlocking = rrState === "CHALLENGE_REQUIRED" || rrState === "CHALLENGE_HARD_LOCK"
  const visible = isBlocking || manualOpen

  const [operatorId, setOperatorId] = useState("OP_01")
  const [lotIdOverride, setLotIdOverride] = useState<string | null>(null)
  const [scanUid, setScanUid] = useState("")
  const [unlockActorId, setUnlockActorId] = useState("QE_01")
  const [unlockRole, setUnlockRole] = useState<"QE" | "ENGINEER">("QE")
  const [unlockReason, setUnlockReason] = useState("OPTICS_AND_SAMPLE_VERIFIED")
  const [unlockComment, setUnlockComment] = useState("")
  const [busyAction, setBusyAction] = useState("")
  const [localError, setLocalError] = useState("")
  const [audit, setAudit] = useState<RedRabbitAuditResponse | null>(null)
  const [auditLoading, setAuditLoading] = useState(false)

  const scanInputRef = useRef<HTMLInputElement | null>(null)
  const lotId = lotIdOverride ?? redRabbit?.active_lot_id ?? "LOT_IGBT_01"

  const loadAudit = useCallback(async () => {
    try {
      const data = await rrRequest<RedRabbitAuditResponse>("/api/red_rabbit/audit?limit=20")
      setAudit(data)
    } catch {
      // ignore transient poll errors inside overlay
    }
  }, [])

  const handleManualRefreshAudit = useCallback(async () => {
    setAuditLoading(true)
    try {
      await loadAudit()
    } finally {
      setAuditLoading(false)
    }
  }, [loadAudit])

  useEffect(() => {
    if (!visible) return
    let active = true
    void rrRequest<RedRabbitAuditResponse>("/api/red_rabbit/audit?limit=20")
      .then((data) => {
        if (active) setAudit(data)
      })
      .catch(() => {})
    return () => {
      active = false
    }
  }, [
    visible,
    rrState,
    redRabbit?.last_passed_session_id,
    redRabbit?.last_lock_event?.event_id,
  ])

  useEffect(() => {
    if (!visible || isBlocking) return
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onCloseManual()
      }
    }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [visible, isBlocking, onCloseManual])

  const activeSession =
    redRabbit?.session?.status === "IN_PROGRESS" ? redRabbit.session : null
  const currentStep = activeSession?.current_step ?? null
  const armedUid = activeSession?.armed_sample_uid ?? null

  const cameraOnline = state.camera_status === "ONLINE"
  const liveOutcome = String(state.outcome || "WAITING")
  const liveReadyForAttempt = liveOutcome === "PASS" || liveOutcome === "ALARM"

  const triggerLabel = useMemo(() => {
    const code = redRabbit?.trigger_reason || ""
    return RR_TRIGGER_LABELS[code] || reasonOf(code) || "换班 / 周期防呆点检"
  }, [redRabbit?.trigger_reason])

  if (!visible) return null

  const runAction = async (key: string, fn: () => Promise<void>) => {
    setBusyAction(key)
    setLocalError("")
    try {
      await fn()
      await onUpdated()
      await loadAudit()
    } catch (err) {
      const code = err instanceof Error ? err.message : String(err)
      setLocalError(reasonOf(code) || code)
    } finally {
      setBusyAction("")
    }
  }

  const handleStartSession = (e: React.FormEvent) => {
    e.preventDefault()
    void runAction("start", async () => {
      await rrRequest(
        "/api/red_rabbit/sessions/start",
        {
          method: "POST",
          body: JSON.stringify({
            operator_id: operatorId.trim() || "OP_01",
            lot_id: lotId.trim() || "LOT_IGBT_01",
          }),
        }
      )
      setScanUid("")
      flash("红兔随机挑战已开始，请按顺序扫码上样")
      window.requestAnimationFrame(() => scanInputRef.current?.focus())
    })
  }

  const handleArmSample = (e: React.FormEvent) => {
    e.preventDefault()
    if (!activeSession) return
    const uid = scanUid.trim()
    if (!uid) {
      setLocalError("请先扫描或输入当前步骤的金样编号")
      scanInputRef.current?.focus()
      return
    }
    void runAction("arm", async () => {
      await rrRequest(
        `/api/red_rabbit/sessions/${encodeURIComponent(activeSession.session_id)}/arm`,
        {
          method: "POST",
          body: JSON.stringify({ sample_uid: uid }),
        }
      )
      flash(`已锁定金样 ${uid}，请放入检测位完成判定`)
    })
  }

  const handleSubmitAttempt = (mode: "live" | "PASS" | "ALARM") => {
    if (!activeSession || !currentStep) return
    void runAction(`attempt_${mode}`, async () => {
      const payload =
        mode === "live"
          ? { sample_uid: currentStep.sample_uid, use_live_state: true }
          : { sample_uid: currentStep.sample_uid, actual_verdict: mode }
      const next = await rrRequest<RedRabbitStatus>(
        `/api/red_rabbit/sessions/${encodeURIComponent(activeSession.session_id)}/attempt`,
        {
          method: "POST",
          body: JSON.stringify(payload),
        }
      )
      setScanUid("")
      if (next.state === "PROD_READY") {
        flash("红兔挑战全部通过，工位已恢复生产放行")
      } else if (next.state === "CHALLENGE_HARD_LOCK") {
        flash("红兔挑战失败，工位已进入硬闭锁")
      } else {
        flash("本步金样核验通过，请继续下一步")
        window.requestAnimationFrame(() => scanInputRef.current?.focus())
      }
    })
  }

  const handleUnlockHardLock = (e: React.FormEvent) => {
    e.preventDefault()
    if (!adminToken) {
      onOpenEngineerAuth()
      return
    }
    if (unlockComment.trim().length < 6) {
      setLocalError("请填写至少 6 个字符的排查根因与解锁说明")
      return
    }
    void runAction("unlock", async () => {
      await rrRequest(
        "/api/red_rabbit/unlock",
        {
          method: "POST",
          body: JSON.stringify({
            actor_id: unlockActorId.trim() || "QE_01",
            actor_role: unlockRole,
            reason_code: unlockReason.trim() || "OPTICS_AND_SAMPLE_VERIFIED",
            comment: unlockComment.trim(),
          }),
        },
        adminToken
      )
      setUnlockComment("")
      flash("硬锁已解除，已退回至红兔重新挑战状态")
    })
  }

  const handleManualTrigger = () => {
    if (!adminToken) {
      onOpenEngineerAuth()
      return
    }
    void runAction("trigger", async () => {
      await rrRequest(
        "/api/red_rabbit/trigger",
        {
          method: "POST",
          body: JSON.stringify({
            reason: "MANUAL_AUDIT",
            lot_id: lotId.trim(),
          }),
        },
        adminToken
      )
      flash("已触发红兔强制挑战点检")
    })
  }

  const chainValid = audit?.chain_verification?.valid !== false
  const checkedRows = audit?.chain_verification?.checked_rows ?? 0

  return (
    <div
      className="rr-overlay"
      role={isBlocking ? "alertdialog" : "dialog"}
      aria-modal="true"
      aria-labelledby="rr-overlay-title"
    >
      <div className="rr-shell">
        <header className={`rr-header ${rrState === "CHALLENGE_HARD_LOCK" ? "hard-lock" : rrState === "CHALLENGE_REQUIRED" ? "challenge" : "ready"}`}>
          <div className="rr-header-title">
            {rrState === "CHALLENGE_HARD_LOCK" ? (
              <ShieldAlert size={24} aria-hidden="true" />
            ) : rrState === "CHALLENGE_REQUIRED" ? (
              <AlertTriangle size={24} aria-hidden="true" />
            ) : (
              <ShieldCheck size={24} aria-hidden="true" />
            )}
            <div>
              <h2 id="rr-overlay-title">
                {rrState === "CHALLENGE_HARD_LOCK"
                  ? "红兔挑战失败 · 工位硬件级闭锁"
                  : rrState === "CHALLENGE_REQUIRED"
                    ? "红兔强制挑战点检（未通过禁止放行）"
                    : "红兔点检审计与金样管理"}
              </h2>
              <p className="rr-header-sub">
                <span>触发原因：{triggerLabel}</span>
                <span>·</span>
                <span>策略：1 OK + 2 NG 随机挑战（AEC-Q101）</span>
              </p>
            </div>
          </div>

          <div className="rr-header-actions">
            <span className={`tag ${chainValid ? "pass" : "alarm"} mono`}>
              {chainValid ? `SHA-256 链完整 (${checkedRows})` : "审计链异常"}
            </span>
            <button
              type="button"
              className="btn ghost sm"
              onClick={() => void handleManualRefreshAudit()}
              disabled={auditLoading}
            >
              <RefreshCw size={16} aria-hidden="true" />
              刷新审计
            </button>
            {!isBlocking && (
              <button
                type="button"
                className="icon-btn"
                onClick={onCloseManual}
                aria-label="关闭红兔审计面板"
              >
                <X size={16} aria-hidden="true" />
              </button>
            )}
          </div>
        </header>

        <div className="rr-body">
          <section className="rr-card rr-optical-pane" aria-label="实时检测视场">
            <div className="rr-card-head">
              <h3>实时检测视场（点检辅助监视）</h3>
              <span className={`tag ${cameraOnline ? "pass" : "warn"}`}>
                {cameraOnline ? "相机在线 · 720P" : "相机离线（支持台架实测输入）"}
              </span>
            </div>

            <div className="rr-camera-wrap">
              <CameraFrame
                fill
                src={cameraOnline ? String(state.live_stream_url || "") : undefined}
                alt="红兔点检全景实时画面"
                focusLocked={Boolean(cameraOnline && state.view_valid)}
                emptyTitle="相机离线 · 可在右侧执行台架闭环测试"
                errorTitle="相机画面暂不可用"
              />
            </div>

            <div className="rr-optical-meta">
              <div className="rr-mini-box">
                <span>当前实时判定</span>
                <b className="mono">{liveOutcome}</b>
              </div>
              <div className="rr-mini-box">
                <span>标签识别态</span>
                <b>{labelText[String(state.label_status || "unknown")] || "未识别"}</b>
              </div>
              <div className="rr-mini-box">
                <span>4× 标签特写</span>
                <div className="rr-mini-crop">
                  <CameraFrame
                    fill
                    src={cameraOnline ? String(state.live_crop_url || "") : undefined}
                    alt="红兔点检标签 4 倍特写"
                    focusLocked={Boolean(cameraOnline && state.view_valid)}
                    emptyTitle="等待入位"
                    errorTitle="特写离线"
                  />
                </div>
              </div>
            </div>
          </section>

          <section className="rr-card rr-workflow-pane" aria-label="红兔挑战工作流">
            {localError && (
              <div className="rr-alert bad" role="alert">
                <AlertTriangle size={16} aria-hidden="true" />
                <span>{localError}</span>
              </div>
            )}

            {rrState === "CHALLENGE_HARD_LOCK" && (
              <div className="rr-stack">
                <div className="rr-alert bad">
                  <ShieldAlert size={20} aria-hidden="true" />
                  <div>
                    <strong>工位已进入硬闭锁（CHALLENGE_HARD_LOCK）</strong>
                    <p>
                      {redRabbit?.last_lock_event?.comment ||
                        reasonOf(redRabbit?.trigger_reason || "") ||
                        "金样挑战未通过或超时"}
                    </p>
                    <p>AEC-Q101 强制约束：工程师解锁后必须重新完整通过 1 OK + 2 NG 红兔挑战，严禁跳过直放。</p>
                  </div>
                </div>

                {!adminToken ? (
                  <div className="rr-stack">
                    <button
                      type="button"
                      className="btn btn-danger block"
                      onClick={onOpenEngineerAuth}
                    >
                      <Lock size={16} aria-hidden="true" />
                      验证工程师 / QE 权限以解锁
                    </button>
                  </div>
                ) : (
                  <form className="rr-form" onSubmit={handleUnlockHardLock}>
                    <div className="rr-grid-2">
                      <div className="field">
                        <label htmlFor="rr-unlock-actor">解锁人员工号</label>
                        <input
                          id="rr-unlock-actor"
                          data-slot="input"
                          value={unlockActorId}
                          onChange={(e) => setUnlockActorId(e.target.value)}
                          placeholder="例如 QE_01"
                          required
                        />
                      </div>
                      <div className="field">
                        <label htmlFor="rr-unlock-role">审批角色</label>
                        <select
                          id="rr-unlock-role"
                          className="rr-select"
                          value={unlockRole}
                          onChange={(e) =>
                            setUnlockRole(e.target.value === "ENGINEER" ? "ENGINEER" : "QE")
                          }
                        >
                          <option value="QE">QE（质量工程师）</option>
                          <option value="ENGINEER">ENGINEER（设备工程师）</option>
                        </select>
                      </div>
                    </div>

                    <div className="field">
                      <label htmlFor="rr-unlock-reason">根因代码</label>
                      <select
                        id="rr-unlock-reason"
                        className="rr-select"
                        value={unlockReason}
                        onChange={(e) => setUnlockReason(e.target.value)}
                      >
                        <option value="OPTICS_AND_SAMPLE_VERIFIED">
                          OPTICS_AND_SAMPLE_VERIFIED（已核查光学与金样）
                        </option>
                        <option value="OPERATOR_MISOPERATION_CORRECTED">
                          OPERATOR_MISOPERATION_CORRECTED（已纠正上样动作）
                        </option>
                        <option value="GOLDEN_SAMPLE_REPLACED">
                          GOLDEN_SAMPLE_REPLACED（已更换有效金样）
                        </option>
                        <option value="TIMEOUT_RETEST_APPROVED">
                          TIMEOUT_RETEST_APPROVED（超时核查后重考）
                        </option>
                      </select>
                    </div>

                    <div className="field">
                      <label htmlFor="rr-unlock-comment">排查与整改说明（至少 6 字，写入 SHA-256 哈希链）</label>
                      <textarea
                        id="rr-unlock-comment"
                        className="rr-textarea"
                        rows={3}
                        value={unlockComment}
                        onChange={(e) => setUnlockComment(e.target.value)}
                        placeholder="请写明排查根因与整改措施，例如：已清洁镜头偏振片并复检金样状态，重新发起红兔点检"
                        required
                      />
                    </div>

                    <button
                      type="submit"
                      className="btn btn-danger block"
                      disabled={busyAction === "unlock"}
                    >
                      <Unlock size={16} aria-hidden="true" />
                      {busyAction === "unlock"
                        ? "正在写入审计链…"
                        : "写入哈希链审计并退回重新点检"}
                    </button>
                  </form>
                )}
              </div>
            )}

            {rrState === "CHALLENGE_REQUIRED" && !activeSession && (
              <form className="rr-form" onSubmit={handleStartSession}>
                <div className="rr-alert warn">
                  <AlertTriangle size={18} aria-hidden="true" />
                  <div>
                    <strong>生产放行通道已闭锁</strong>
                    <p>
                      需依次完成系统随机抽取的 {redRabbit?.required_ok_count ?? 1} 支 OK 金样与{" "}
                      {redRabbit?.required_ng_count ?? 2} 支 NG 金样点检，全部匹配后自动解锁放行。
                    </p>
                  </div>
                </div>

                <div className="rr-grid-2">
                  <div className="field">
                    <label htmlFor="rr-op-id">作业员工号</label>
                    <input
                      id="rr-op-id"
                      data-slot="input"
                      value={operatorId}
                      onChange={(e) => setOperatorId(e.target.value)}
                      placeholder="OP_01"
                      required
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="rr-lot-id">当前批次 / Lot ID</label>
                    <input
                      id="rr-lot-id"
                      data-slot="input"
                      value={lotId}
                      onChange={(e) => setLotIdOverride(e.target.value)}
                      placeholder="LOT_IGBT_01"
                      required
                    />
                  </div>
                </div>

                <button
                  type="submit"
                  className="btn block"
                  disabled={busyAction === "start"}
                >
                  <ScanBarcode size={16} aria-hidden="true" />
                  {busyAction === "start"
                    ? "正在生成随机序列…"
                    : "开始随机挑战（1 OK + 2 NG）"}
                </button>
              </form>
            )}

            {rrState === "CHALLENGE_REQUIRED" && activeSession && currentStep && (
              <div className="rr-stack">
                <div className="rr-session-bar">
                  <div className="rr-steps">
                    {activeSession.required_plan.map((s, idx) => {
                      const done = Boolean(s.completed)
                      const current = idx === activeSession.current_step_index
                      return (
                        <span
                          key={s.sample_uid}
                          className={`rr-step-pill ${done ? "done" : current ? "active" : ""}`}
                        >
                          {done ? "✓ " : ""}步骤 {idx + 1}/{activeSession.required_plan.length}
                        </span>
                      )
                    })}
                  </div>
                  <span className="rr-timeout mono">
                    剩余 {formatCountdown(activeSession.timeout_remaining_ms)}
                  </span>
                </div>

                <div className="rr-sample-card">
                  <div className="rr-sample-head">
                    <span className="rr-sample-uid mono">{currentStep.sample_uid}</span>
                    <span className={`tag ${currentStep.expected_verdict === "PASS" ? "pass" : "alarm"}`}>
                      期望判定：{currentStep.expected_verdict}
                    </span>
                  </div>
                  <p className="rr-sample-desc">
                    {RR_SAMPLE_TYPE_LABELS[currentStep.sample_type] || currentStep.sample_type}
                  </p>
                </div>

                <form className="rr-arm-row" onSubmit={handleArmSample}>
                  <div className="field" style={{ flex: 1 }}>
                    <label htmlFor="rr-scan-uid">
                      第 1 步：扫码或输入金样唯一条码（防重复上样）
                    </label>
                    <input
                      ref={scanInputRef}
                      id="rr-scan-uid"
                      data-slot="input"
                      className="mono"
                      value={scanUid}
                      onChange={(e) => setScanUid(e.target.value)}
                      placeholder={`请扫描 ${currentStep.sample_uid}`}
                    />
                  </div>
                  <button
                    type="submit"
                    className="btn ghost"
                    disabled={busyAction === "arm"}
                  >
                    <ScanBarcode size={16} aria-hidden="true" />
                    {armedUid === currentStep.sample_uid ? "已锁定" : "扫码锁定"}
                  </button>
                </form>

                {armedUid === currentStep.sample_uid ? (
                  <div className="rr-alert ok">
                    <CheckCircle2 size={16} aria-hidden="true" />
                    <span>
                      已锁定金样 <strong className="mono">{armedUid}</strong>：请将金样置于检测区完成一次完整判定，或点击下方按钮提交结果。
                    </span>
                  </div>
                ) : (
                  <div className="rr-alert warn">
                    <ScanBarcode size={16} aria-hidden="true" />
                    <span>请先扫描当前步金样条码完成锁定，未锁定禁止提交判定。</span>
                  </div>
                )}

                <div className="rr-attempt-actions">
                  <button
                    type="button"
                    className="btn block"
                    disabled={
                      armedUid !== currentStep.sample_uid ||
                      !cameraOnline ||
                      !liveReadyForAttempt ||
                      busyAction !== ""
                    }
                    onClick={() => handleSubmitAttempt("live")}
                  >
                    采集当前实时视觉判定（{liveOutcome}）
                  </button>

                  <div className="rr-grid-2">
                    <button
                      type="button"
                      className="btn ghost"
                      disabled={armedUid !== currentStep.sample_uid || busyAction !== ""}
                      onClick={() => handleSubmitAttempt("PASS")}
                    >
                      台架实测提交 PASS
                    </button>
                    <button
                      type="button"
                      className="btn ghost"
                      disabled={armedUid !== currentStep.sample_uid || busyAction !== ""}
                      onClick={() => handleSubmitAttempt("ALARM")}
                    >
                      台架实测提交 ALARM
                    </button>
                  </div>
                </div>
              </div>
            )}

            {rrState === "PROD_READY" && (
              <div className="rr-stack">
                <div className="rr-alert ok">
                  <ShieldCheck size={18} aria-hidden="true" />
                  <div>
                    <strong>红兔点检状态正常（PROD_READY）</strong>
                    <p>
                      上次通过：{formatMsTimestamp(redRabbit?.last_passed_at_ms)} · 下次到期：
                      {formatMsTimestamp(redRabbit?.next_due_at_ms)}
                    </p>
                  </div>
                </div>

                <button
                  type="button"
                  className="btn block"
                  disabled={busyAction === "trigger"}
                  onClick={handleManualTrigger}
                >
                  <ShieldAlert size={16} aria-hidden="true" />
                  {adminToken ? "立即发起红兔强制点检" : "登录后可立即发起手动红兔点检"}
                </button>
              </div>
            )}

            <div className="rr-audit-section">
              <div className="rr-card-head">
                <h3>不可篡改锁机与解锁审计（SHA-256 哈希链）</h3>
                <span className="rr-muted mono">
                  {audit?.chain_verification?.latest_hash
                    ? `末链 ${audit.chain_verification.latest_hash.slice(0, 12)}…`
                    : "创世空链"}
                </span>
              </div>

              {audit && audit.lock_audit.length > 0 ? (
                <div className="rr-audit-list">
                  {audit.lock_audit.slice(0, 6).map((row) => (
                    <div key={row.audit_seq} className="rr-audit-row">
                      <div className="rr-audit-top">
                        <span
                          className={`tag ${row.event_type === "ENGINEER_UNLOCK" ? "pass" : "alarm"}`}
                        >
                          {RR_EVENT_TYPE_LABELS[row.event_type] || row.event_type}
                        </span>
                        <span className="mono">
                          {row.actor_role}:{row.actor_id}
                        </span>
                        <span className="rr-muted mono">
                          {formatMsTimestamp(row.created_at_ms)}
                        </span>
                      </div>
                      <p className="rr-audit-comment">
                        {row.reason_code} — {row.comment}
                      </p>
                      <span className="rr-audit-hash mono">
                        hash: {row.row_hash.slice(0, 24)}…
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="rr-muted">暂无锁机/解锁审计记录。</p>
              )}
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}
