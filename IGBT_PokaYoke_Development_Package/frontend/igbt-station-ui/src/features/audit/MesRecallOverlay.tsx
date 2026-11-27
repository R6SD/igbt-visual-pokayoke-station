import { useState } from "react"
import { AlertTriangle, Lock, ShieldAlert, Unlock } from "lucide-react"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { reasonOf } from "@/lib/station"
import type { MesStatus } from "./types"

interface MesRecallOverlayProps {
  mes: MesStatus | null
  adminToken: string
  onOpenEngineerAuth: () => void
  onUpdated: () => Promise<void> | void
  flash: (msg: string) => void
}

function formatRecallTime(ms?: number): string {
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

export function MesRecallOverlay({
  mes,
  adminToken,
  onOpenEngineerAuth,
  onUpdated,
  flash,
}: MesRecallOverlayProps) {
  const [actorId, setActorId] = useState("QE_01")
  const [actorRole, setActorRole] = useState<"QE" | "ENGINEER">("QE")
  const [reasonCode, setReasonCode] = useState("MES_RECALL_QUARANTINED")
  const [comment, setComment] = useState("")
  const [busy, setBusy] = useState(false)
  const [localError, setLocalError] = useState("")

  const visible = Boolean(mes?.enabled && mes?.recall_locked)
  if (!visible || !mes) return null

  const recall = mes.recall
  const affectedKeys = Array.isArray(recall.affected_keys) ? recall.affected_keys : []

  const handleUnlock = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!adminToken) {
      onOpenEngineerAuth()
      return
    }
    if (comment.trim().length < 4) {
      setLocalError("请填写至少 4 个字符的隔离清点与处置说明")
      return
    }
    setBusy(true)
    setLocalError("")
    try {
      const res = await fetch("/api/mes/recall/unlock", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Admin-Token": adminToken,
        },
        body: JSON.stringify({
          actor_id: actorId.trim() || "QE_01",
          actor_role: actorRole,
          reason_code: reasonCode.trim() || "MES_RECALL_QUARANTINED",
          comment: comment.trim(),
        }),
      })
      const payload = (await res.json().catch(() => ({}))) as Record<string, unknown>
      if (!res.ok) {
        const detail = typeof payload.detail === "string" && payload.detail ? payload.detail : `HTTP ${res.status}`
        throw new Error(detail)
      }
      setComment("")
      flash("MES 批次追回硬锁已解除并写入 SHA-256 审计链")
      await onUpdated()
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      setLocalError(reasonOf(msg) || msg)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div
      className="rr-overlay"
      role="dialog"
      aria-modal="true"
      aria-labelledby="mes-recall-title"
      style={{ zIndex: 60 }}
    >
      <div className="rr-shell" style={{ maxWidth: 920 }}>
        <header className="rr-header hard-lock">
          <div className="rr-header-title">
            <ShieldAlert size={28} aria-hidden="true" />
            <div>
              <h2 id="mes-recall-title">
                MES 批次追回硬闭锁（{recall.status === "SCRAP" ? "SCRAP 报废" : "HOLD 冻结"}）
              </h2>
              <div className="rr-header-sub">
                <span>工单：{recall.work_order_id || "—"}</span>
                <span>批次 Lot：{recall.lot_id || "—"}</span>
                <span>触发时间：{formatRecallTime(recall.triggered_at_ms)}</span>
              </div>
            </div>
          </div>
          <div className="rr-header-actions">
            <span className="tag alarm" style={{ fontSize: 14, fontWeight: 600 }}>
              <Lock size={14} aria-hidden="true" />
              不可关闭 · 待 QE 清点隔离
            </span>
          </div>
        </header>

        <section className="rr-card" style={{ padding: 24, gap: 16 }}>
          <div className="rr-alert bad" role="alert">
            <AlertTriangle size={18} aria-hidden="true" />
            <div>
              <strong>{reasonOf(mes.block_reason || "MES_LOT_HOLD_RECALL")}</strong>
              <p>
                {recall.reason || "MES 对账返回当前批次状态已变更为 HOLD/SCRAP，请立即拦截并隔离该批次全部涉事料管。"}
              </p>
            </div>
          </div>

          <div
            style={{
              background: "var(--surface-alt)",
              borderRadius: 6,
              padding: 24,
              display: "flex",
              flexDirection: "column",
              gap: 16,
            }}
          >
            <div className="rr-card-head">
              <h3 style={{ fontSize: 16, fontWeight: 600 }}>
                需核查隔离的涉事过站幂等流水（共 {affectedKeys.length} 条）
              </h3>
              <span className="mono" style={{ fontSize: 14, fontWeight: 400, color: "var(--ink-2)" }}>
                X-Idempotency-Key
              </span>
            </div>
            {affectedKeys.length > 0 ? (
              <div
                style={{
                  maxHeight: 160,
                  overflowY: "auto",
                  display: "flex",
                  flexDirection: "column",
                  gap: 8,
                }}
              >
                {affectedKeys.slice(0, 20).map((key) => (
                  <code
                    key={key}
                    className="mono"
                    style={{
                      fontSize: 14,
                      fontWeight: 400,
                      padding: "6px 10px",
                      borderRadius: 4,
                      background: "var(--surface)",
                    }}
                  >
                    {key}
                  </code>
                ))}
              </div>
            ) : (
              <p style={{ margin: 0, fontSize: 14, fontWeight: 400, color: "var(--ink-2)" }}>
                当前暂无已登记流水键，请核实工位待流转料管并由 QE 审批解锁。
              </p>
            )}
          </div>

          {localError && (
            <div className="auth-error" role="alert" style={{ fontSize: 14, fontWeight: 400 }}>
              <AlertTriangle size={16} aria-hidden="true" />
              <span>{localError}</span>
            </div>
          )}

          <form className="rr-form" onSubmit={(e) => void handleUnlock(e)}>
            <div className="rr-grid-2">
              <div className="field">
                <label htmlFor="mes-recall-actor" style={{ fontSize: 14, fontWeight: 600 }}>
                  QE / 工程师工号
                </label>
                <Input
                  id="mes-recall-actor"
                  value={actorId}
                  onChange={(e) => setActorId(e.target.value)}
                  className="mono"
                  style={{ fontSize: 16, fontWeight: 400 }}
                />
              </div>
              <div className="field">
                <label htmlFor="mes-recall-role" style={{ fontSize: 14, fontWeight: 600 }}>
                  签批角色
                </label>
                <Select value={actorRole} onValueChange={(v) => setActorRole(v === "ENGINEER" ? "ENGINEER" : "QE")}>
                  <SelectTrigger id="mes-recall-role" style={{ fontSize: 16, fontWeight: 400 }}>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="QE">质量工程师 (QE)</SelectItem>
                    <SelectItem value="ENGINEER">产线工程师 (ENGINEER)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="field">
              <label htmlFor="mes-recall-reason-code" style={{ fontSize: 14, fontWeight: 600 }}>
                处置结论代码
              </label>
              <Input
                id="mes-recall-reason-code"
                value={reasonCode}
                onChange={(e) => setReasonCode(e.target.value)}
                className="mono"
                style={{ fontSize: 16, fontWeight: 400 }}
              />
            </div>

            <div className="field">
              <label htmlFor="mes-recall-comment" style={{ fontSize: 14, fontWeight: 600 }}>
                物理隔离清点与解锁说明（将写入 SHA-256 哈希链审计）
              </label>
              <textarea
                id="mes-recall-comment"
                className="rr-textarea"
                value={comment}
                onChange={(e) => setComment(e.target.value)}
                placeholder="例如：已完成该批次 12 支料管物理隔离与 MES 工单切换确认"
                style={{ fontSize: 16, fontWeight: 400 }}
              />
            </div>

            {adminToken ? (
              <button
                type="submit"
                className="btn block"
                disabled={busy}
                style={{ fontSize: 16, fontWeight: 600 }}
              >
                <Unlock size={16} aria-hidden="true" />
                {busy ? "解锁写入审计中…" : "确认已隔离并解除 MES 追回硬锁"}
              </button>
            ) : (
              <button
                type="button"
                className="btn block"
                onClick={onOpenEngineerAuth}
                style={{ fontSize: 16, fontWeight: 600 }}
              >
                <Lock size={16} aria-hidden="true" />
                先完成工程师/QE 口令鉴权再解锁
              </button>
            )}
          </form>
        </section>
      </div>
    </div>
  )
}
