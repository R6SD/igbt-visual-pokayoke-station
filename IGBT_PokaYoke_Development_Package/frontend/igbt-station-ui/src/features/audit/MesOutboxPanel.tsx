import { useCallback, useEffect, useState } from "react"
import {
  AlertTriangle,
  CheckCircle2,
  Copy,
  RefreshCw,
  RotateCcw,
  Send,
  Server,
  ShieldAlert,
  ShieldCheck,
} from "lucide-react"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { type Data, fmt, pipeText, reasonOf } from "@/lib/station"
import {
  DEGRADATION_MODE_LABELS,
  type MesOutboxItem,
  type MesOutboxListResponse,
  type MesStatus,
  OUTBOX_STATUS_LABELS,
} from "./types"

interface MesOutboxPanelProps {
  state: Data
  token: string
  flash: (msg: string) => void
  onUpdated: () => Promise<void> | void
}

async function mesApi<T = Data>(path: string, options: RequestInit = {}, token = ""): Promise<T> {
  const headers = new Headers(options.headers)
  if (!headers.has("Content-Type") && options.body) {
    headers.set("Content-Type", "application/json")
  }
  if (token) headers.set("X-Admin-Token", token)
  const res = await fetch(path, { ...options, headers })
  const payload = (await res.json().catch(() => ({}))) as Record<string, unknown>
  if (!res.ok) {
    const detail = typeof payload.detail === "string" && payload.detail ? payload.detail : `HTTP ${res.status}`
    throw new Error(detail)
  }
  return payload as T
}

function formatMs(ms?: number | null): string {
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

export function MesOutboxPanel({ state, token, flash, onUpdated }: MesOutboxPanelProps) {
  const stateMes = (state.mes as MesStatus | undefined) ?? null
  const [outboxResp, setOutboxResp] = useState<MesOutboxListResponse | null>(null)
  const [statusFilter, setStatusFilter] = useState<string>("ALL")
  const [loading, setLoading] = useState(false)
  const [busyAction, setBusyAction] = useState("")

  // Gateway config form
  const [cfgEnabled, setCfgEnabled] = useState(() => Boolean(stateMes?.config?.enabled))
  const [cfgUrl, setCfgUrl] = useState(() => stateMes?.config?.base_url || "http://127.0.0.1:9000")
  const [cfgStationId, setCfgStationId] = useState(() => stateMes?.config?.station_id || "STATION_01")
  const [cfgFailTimeoutS, setCfgFailTimeoutS] = useState(() =>
    String(Math.round((stateMes?.config?.heartbeat_fail_lock_ms ?? 60000) / 1000))
  )

  // Work-order / Lot bind form
  const [woId, setWoId] = useState(() => stateMes?.active_lot?.work_order_id || "")
  const [lotId, setLotId] = useState(() => stateMes?.active_lot?.lot_id || "")
  const [partNumber, setPartNumber] = useState(() => stateMes?.active_lot?.part_number || "IGBT-TO247")
  const [pipeColor, setPipeColor] = useState(() => stateMes?.active_lot?.expected_pipe_color || "yellow")
  const [bindActor, setBindActor] = useState("OP_01")

  // Whitelist form
  const [wlApproved, setWlApproved] = useState(() => Boolean(stateMes?.whitelist?.approved))
  const [wlActor, setWlActor] = useState(() => stateMes?.whitelist?.actor_id || "QE_01")
  const [wlRole, setWlRole] = useState<"QE" | "ENGINEER">(() =>
    stateMes?.whitelist?.actor_role === "ENGINEER" ? "ENGINEER" : "QE"
  )
  const [wlMaxTubes, setWlMaxTubes] = useState(() => String(stateMes?.whitelist?.max_tubes ?? 50))
  const [wlMaxMinutes, setWlMaxMinutes] = useState(() =>
    String(Math.round((stateMes?.whitelist?.max_duration_ms ?? 1800000) / 60000))
  )
  const [wlReason, setWlReason] = useState(() => stateMes?.whitelist?.reason || "")

  const mes = outboxResp?.status ?? stateMes
  const counts = outboxResp?.counts ?? mes?.outbox_counts ?? {
    pending: 0,
    in_flight: 0,
    acked: 0,
    dead_letter: 0,
    total: 0,
  }
  const items: MesOutboxItem[] = outboxResp?.items ?? []

  const loadOutbox = useCallback(async () => {
    setLoading(true)
    try {
      const qs = new URLSearchParams({ limit: "50" })
      if (statusFilter !== "ALL") qs.set("status", statusFilter)
      const data = await mesApi<MesOutboxListResponse>(`/api/mes/outbox?${qs.toString()}`)
      setOutboxResp(data)
    } catch {
      /* 保留轮询快照 */
    } finally {
      setLoading(false)
    }
  }, [statusFilter])

  useEffect(() => {
    let active = true
    const qs = new URLSearchParams({ limit: "50" })
    if (statusFilter !== "ALL") qs.set("status", statusFilter)
    void mesApi<MesOutboxListResponse>(`/api/mes/outbox?${qs.toString()}`)
      .then((data) => {
        if (active) setOutboxResp(data)
      })
      .catch(() => {
        /* 保留轮询快照 */
      })
    return () => {
      active = false
    }
  }, [statusFilter])

  const runAction = async (tag: string, fn: () => Promise<void>) => {
    setBusyAction(tag)
    try {
      await fn()
      await loadOutbox()
      await onUpdated()
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      flash(`操作失败：${reasonOf(msg) || msg}`)
    } finally {
      setBusyAction("")
    }
  }

  const handleSaveConfig = (e: React.FormEvent) => {
    e.preventDefault()
    void runAction("config", async () => {
      const timeoutSec = Math.max(1, Math.min(600, Math.round(Number(cfgFailTimeoutS) || 60)))
      await mesApi(
        "/api/mes/config",
        {
          method: "POST",
          body: JSON.stringify({
            enabled: cfgEnabled,
            base_url: cfgUrl.trim(),
            station_id: cfgStationId.trim() || "STATION_01",
            heartbeat_fail_lock_ms: timeoutSec * 1000,
          }),
        },
        token
      )
      flash("MES 网关连接配置已保存")
    })
  }

  const handleBindLot = (e: React.FormEvent) => {
    e.preventDefault()
    void runAction("bind", async () => {
      const res = await mesApi<Data>(
        "/api/mes/lot/bind",
        {
          method: "POST",
          body: JSON.stringify({
            work_order_id: woId.trim(),
            lot_id: lotId.trim(),
            part_number: partNumber.trim() || "IGBT-TO247",
            expected_pipe_color: pipeColor,
            actor_id: bindActor.trim() || "OP_01",
          }),
        },
        token
      )
      flash(
        res.red_rabbit_triggered
          ? "工单/Lot 在线绑定成功，已自动触发红兔换批强制点检 (LOT_CHANGE)"
          : "工单/Lot 在线核验与本地签名刷新成功"
      )
    })
  }

  const handleSaveWhitelist = (e: React.FormEvent) => {
    e.preventDefault()
    void runAction("whitelist", async () => {
      const maxTubes = Math.max(1, Math.min(50, Math.round(Number(wlMaxTubes) || 50)))
      const maxMin = Math.max(1, Math.min(30, Math.round(Number(wlMaxMinutes) || 30)))
      await mesApi(
        "/api/mes/whitelist",
        {
          method: "POST",
          body: JSON.stringify({
            approved: wlApproved,
            actor_id: wlActor.trim() || "QE_01",
            actor_role: wlRole,
            max_tubes: maxTubes,
            max_duration_minutes: maxMin,
            reason: wlReason.trim() || (wlApproved ? "QE审批产线临时离线白名单" : "撤回离线白名单"),
          }),
        },
        token
      )
      flash(wlApproved ? `受控白名单已审批生效（上限 ${maxTubes} 管 / ${maxMin} 分钟）` : "受控白名单已撤回")
    })
  }

  const handleFlush = () => {
    void runAction("flush", async () => {
      await mesApi("/api/mes/outbox/flush", { method: "POST" }, token)
      flash("已执行 MES 心跳探活与 Outbox 补传对账")
    })
  }

  const handleRequeue = (rowId: number) => {
    void runAction(`requeue_${rowId}`, async () => {
      await mesApi(`/api/mes/outbox/${rowId}/requeue`, { method: "POST" }, token)
      flash(`发件箱记录 #${rowId} 已重新入队待补传`)
    })
  }

  const copyKey = async (key: string) => {
    try {
      await navigator.clipboard.writeText(key)
      flash("已复制 X-Idempotency-Key")
    } catch {
      flash("复制失败，请手动选中复制")
    }
  }

  const modeMeta =
    DEGRADATION_MODE_LABELS[mes?.degradation_mode || "DISABLED"] ||
    DEGRADATION_MODE_LABELS.DISABLED
  const activeLot = mes?.active_lot ?? null
  const isOfflineDisallowBind = Boolean(mes?.enabled && !mes?.connected && (mes?.offline_since_ms ?? 0) > 0)

  return (
    <div className="storage-layout" aria-label="MES 事务性发件箱与防混料中心">
      <div className="storage-arch-grid">
        {/* 卡片 1：MES 连接状态与断网分级降级 */}
        <section className="panel storage-card" style={{ padding: 24, gap: 16 }}>
          <header className="panel-head">
            <h2 className="flex items-center gap-2" style={{ fontSize: 16, fontWeight: 600 }}>
              <Server size={16} aria-hidden="true" />
              MES 实时连接与断网降级
            </h2>
            <span className={`tag ${modeMeta.tone}`} style={{ fontSize: 14, fontWeight: 600 }}>
              {modeMeta.label}
            </span>
          </header>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
              gap: 16,
              fontSize: 14,
              fontWeight: 400,
            }}
          >
            <div className="storage-metric-cell">
              <b style={{ fontSize: 24, fontWeight: 600 }}>
                {!mes?.enabled ? "旁路" : mes?.connected ? "在线" : "断联"}
              </b>
              <small style={{ fontSize: 14, fontWeight: 400 }}>
                心跳 RTT {mes?.connected ? `${mes.last_rtt_ms} ms` : "—"}
              </small>
            </div>
            <div className="storage-metric-cell">
              <b style={{ fontSize: 24, fontWeight: 600 }}>
                {mes?.whitelist?.approved
                  ? `${mes.whitelist.remaining_tubes} / ${mes.whitelist.max_tubes}`
                  : "Fail-Closed"}
              </b>
              <small style={{ fontSize: 14, fontWeight: 400 }}>
                {mes?.whitelist?.approved
                  ? `白名单剩余 ${Math.ceil((mes.whitelist.remaining_ms || 0) / 60000)} 分钟`
                  : `断联 > ${Math.round((mes?.config?.heartbeat_fail_lock_ms ?? 60000) / 1000)}s 锁机`}
              </small>
            </div>
          </div>

          {mes?.blocking && (
            <div className="auth-error" role="alert" style={{ fontSize: 14, fontWeight: 400 }}>
              <AlertTriangle size={16} aria-hidden="true" />
              <span>当前阻断放行：{reasonOf(mes.block_reason || "MES_HEARTBEAT_TIMEOUT")}</span>
            </div>
          )}

          {mes?.last_heartbeat_error && mes.enabled && !mes.connected && (
            <p style={{ margin: 0, fontSize: 14, fontWeight: 400, color: "var(--ink-2)" }}>
              最近心跳异常：{mes.last_heartbeat_error} · 断联累计{" "}
              {fmt((mes.offline_elapsed_ms || 0) / 1000, 1)}s
            </p>
          )}

          <form className="grid gap-4" onSubmit={handleSaveConfig}>
            <div className="grid grid-cols-2 gap-4">
              <div className="field">
                <label htmlFor="mes-base-url" style={{ fontSize: 14, fontWeight: 600 }}>
                  MES 网关地址 (HTTP)
                </label>
                <Input
                  id="mes-base-url"
                  value={cfgUrl}
                  onChange={(e) => setCfgUrl(e.target.value)}
                  placeholder="http://127.0.0.1:9000"
                  className="mono"
                  style={{ fontSize: 16, fontWeight: 400 }}
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="field">
                  <label htmlFor="mes-station-id" style={{ fontSize: 14, fontWeight: 600 }}>
                    工位编号
                  </label>
                  <Input
                    id="mes-station-id"
                    value={cfgStationId}
                    onChange={(e) => setCfgStationId(e.target.value)}
                    className="mono"
                    style={{ fontSize: 16, fontWeight: 400 }}
                  />
                </div>
                <div className="field">
                  <label htmlFor="mes-fail-timeout" style={{ fontSize: 14, fontWeight: 600 }}>
                    断网锁机 (秒)
                  </label>
                  <Input
                    id="mes-fail-timeout"
                    type="number"
                    min="1"
                    max="600"
                    value={cfgFailTimeoutS}
                    onChange={(e) => setCfgFailTimeoutS(e.target.value)}
                    className="mono"
                    style={{ fontSize: 16, fontWeight: 400 }}
                  />
                </div>
              </div>
            </div>

            <div className="flex items-center justify-between gap-4">
              <label className="check" style={{ fontSize: 14, fontWeight: 400 }}>
                <input
                  type="checkbox"
                  checked={cfgEnabled}
                  onChange={(e) => setCfgEnabled(e.target.checked)}
                />
                启用 MES 强管控（关闭时保持本地静默兼容）
              </label>
              <button
                type="submit"
                className="btn ghost sm"
                disabled={busyAction === "config"}
                style={{ fontSize: 14, fontWeight: 600 }}
              >
                {busyAction === "config" ? "保存中…" : "应用网关配置"}
              </button>
            </div>
          </form>
        </section>

        {/* 卡片 2：活跃工单 / Lot 防混料在线绑定 */}
        <section className="panel storage-card" style={{ padding: 24, gap: 16 }}>
          <header className="panel-head">
            <h2 className="flex items-center gap-2" style={{ fontSize: 16, fontWeight: 600 }}>
              <ShieldCheck size={16} aria-hidden="true" />
              活跃工单 / Lot 防混料核验
            </h2>
            <span
              className={`tag ${activeLot?.signature_valid ? "pass" : "warn"}`}
              style={{ fontSize: 14, fontWeight: 600 }}
            >
              {activeLot ? (activeLot.signature_valid ? "本地签名有效" : "签名失效") : "未绑定 Lot"}
            </span>
          </header>

          {activeLot ? (
            <div
              style={{
                padding: 16,
                borderRadius: 6,
                background: "var(--surface-alt)",
                display: "grid",
                gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
                gap: 12,
                fontSize: 14,
                fontWeight: 400,
              }}
            >
              <div>
                工单 / Lot：<strong className="mono" style={{ fontWeight: 600 }}>{activeLot.work_order_id} / {activeLot.lot_id}</strong>
              </div>
              <div>
                料号 / 预期管色：<strong style={{ fontWeight: 600 }}>{activeLot.part_number} · {pipeText(activeLot.expected_pipe_color)}</strong>
              </div>
              <div>
                累计管序号：<strong className="mono" style={{ fontWeight: 600 }}>#{activeLot.tube_seq}</strong>
              </div>
              <div>
                缓存到期：<strong className="mono" style={{ fontWeight: 600 }}>{formatMs(activeLot.expires_at_ms)}</strong>
              </div>
            </div>
          ) : (
            <div
              style={{
                padding: 16,
                borderRadius: 6,
                background: "var(--warn-tint)",
                color: "var(--warn)",
                fontSize: 14,
                fontWeight: 400,
              }}
            >
              启用 MES 管控后，必须先在线扫码核验工单与批次（Lot）；切换 Lot 将自动触发红兔换批点检。
            </div>
          )}

          {isOfflineDisallowBind && (
            <div className="auth-error" role="status" style={{ fontSize: 14, fontWeight: 400 }}>
              <ShieldAlert size={16} aria-hidden="true" />
              <span>离线状态严禁切换批次或工单，请先恢复 MES 网络连接。</span>
            </div>
          )}

          <form className="grid gap-4" onSubmit={handleBindLot}>
            <div className="grid grid-cols-2 gap-4">
              <div className="field">
                <label htmlFor="mes-bind-wo" style={{ fontSize: 14, fontWeight: 600 }}>
                  工单号 (Work Order)
                </label>
                <Input
                  id="mes-bind-wo"
                  value={woId}
                  onChange={(e) => setWoId(e.target.value)}
                  placeholder="WO-20261127-01"
                  className="mono"
                  style={{ fontSize: 16, fontWeight: 400 }}
                />
              </div>
              <div className="field">
                <label htmlFor="mes-bind-lot" style={{ fontSize: 14, fontWeight: 600 }}>
                  批次号 (Lot ID)
                </label>
                <Input
                  id="mes-bind-lot"
                  value={lotId}
                  onChange={(e) => setLotId(e.target.value)}
                  placeholder="LOT-A001"
                  className="mono"
                  style={{ fontSize: 16, fontWeight: 400 }}
                />
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4">
              <div className="field">
                <label htmlFor="mes-bind-part" style={{ fontSize: 14, fontWeight: 600 }}>
                  器件料号
                </label>
                <Input
                  id="mes-bind-part"
                  value={partNumber}
                  onChange={(e) => setPartNumber(e.target.value)}
                  className="mono"
                  style={{ fontSize: 16, fontWeight: 400 }}
                />
              </div>
              <div className="field">
                <label htmlFor="mes-bind-color" style={{ fontSize: 14, fontWeight: 600 }}>
                  预期管色
                </label>
                <Select value={pipeColor} onValueChange={setPipeColor}>
                  <SelectTrigger id="mes-bind-color" style={{ fontSize: 16, fontWeight: 400 }}>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="yellow">黄管 (yellow)</SelectItem>
                    <SelectItem value="blue">蓝管 (blue)</SelectItem>
                    <SelectItem value="other">其他 (other)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="field">
                <label htmlFor="mes-bind-actor" style={{ fontSize: 14, fontWeight: 600 }}>
                  操作员工号
                </label>
                <Input
                  id="mes-bind-actor"
                  value={bindActor}
                  onChange={(e) => setBindActor(e.target.value)}
                  className="mono"
                  style={{ fontSize: 16, fontWeight: 400 }}
                />
              </div>
            </div>

            <button
              type="submit"
              className="btn block"
              disabled={busyAction === "bind" || !mes?.enabled || isOfflineDisallowBind}
              style={{ fontSize: 16, fontWeight: 600 }}
            >
              <CheckCircle2 size={16} aria-hidden="true" />
              {busyAction === "bind" ? "核验绑定中…" : "在线核验绑定工单 / Lot"}
            </button>
          </form>
        </section>
      </div>

      {/* 卡片 3：QE 受控白名单审批 */}
      <section className="panel" style={{ padding: 24, gap: 16 }}>
        <header className="panel-head">
          <h2 className="flex items-center gap-2" style={{ fontSize: 16, fontWeight: 600 }}>
            <ShieldAlert size={16} aria-hidden="true" />
            QE 受控离线白名单审批（硬上限 ≤ 50 管 且 ≤ 30 分钟，先到先锁）
          </h2>
          <span
            className={`tag ${mes?.whitelist?.approved ? "warn" : "plain"}`}
            style={{ fontSize: 14, fontWeight: 600 }}
          >
            {mes?.whitelist?.approved ? "已审批开启" : "未开启（车规 Fail-Closed）"}
          </span>
        </header>

        <form className="grid gap-4" onSubmit={handleSaveWhitelist}>
          <div className="grid grid-cols-4 gap-4">
            <div className="field">
              <label htmlFor="mes-wl-actor" style={{ fontSize: 14, fontWeight: 600 }}>
                审批人工号
              </label>
              <Input
                id="mes-wl-actor"
                value={wlActor}
                onChange={(e) => setWlActor(e.target.value)}
                className="mono"
                style={{ fontSize: 16, fontWeight: 400 }}
              />
            </div>
            <div className="field">
              <label htmlFor="mes-wl-role" style={{ fontSize: 14, fontWeight: 600 }}>
                审批角色
              </label>
              <Select value={wlRole} onValueChange={(v) => setWlRole(v === "ENGINEER" ? "ENGINEER" : "QE")}>
                <SelectTrigger id="mes-wl-role" style={{ fontSize: 16, fontWeight: 400 }}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="QE">质量工程师 (QE)</SelectItem>
                  <SelectItem value="ENGINEER">工程师 (ENGINEER)</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="field">
              <label htmlFor="mes-wl-tubes" style={{ fontSize: 14, fontWeight: 600 }}>
                离线管数上限 (1–50)
              </label>
              <Input
                id="mes-wl-tubes"
                type="number"
                min="1"
                max="50"
                value={wlMaxTubes}
                onChange={(e) => setWlMaxTubes(e.target.value)}
                className="mono"
                style={{ fontSize: 16, fontWeight: 400 }}
              />
            </div>
            <div className="field">
              <label htmlFor="mes-wl-minutes" style={{ fontSize: 14, fontWeight: 600 }}>
                离线时长上限 (1–30 分钟)
              </label>
              <Input
                id="mes-wl-minutes"
                type="number"
                min="1"
                max="30"
                value={wlMaxMinutes}
                onChange={(e) => setWlMaxMinutes(e.target.value)}
                className="mono"
                style={{ fontSize: 16, fontWeight: 400 }}
              />
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="field" style={{ flex: 1 }}>
              <label htmlFor="mes-wl-reason" style={{ fontSize: 14, fontWeight: 600 }}>
                审批事由（记录至不可篡改审计链）
              </label>
              <Input
                id="mes-wl-reason"
                value={wlReason}
                onChange={(e) => setWlReason(e.target.value)}
                placeholder="例如：车间核心交换机维护期间临时允许当前已签批 Lot 离线作业"
                style={{ fontSize: 16, fontWeight: 400 }}
              />
            </div>
            <label className="check" style={{ fontSize: 14, fontWeight: 400, marginTop: 22 }}>
              <input
                type="checkbox"
                checked={wlApproved}
                onChange={(e) => setWlApproved(e.target.checked)}
              />
              批准白名单
            </label>
            <button
              type="submit"
              className="btn ghost"
              disabled={busyAction === "whitelist"}
              style={{ fontSize: 14, fontWeight: 600, marginTop: 22 }}
            >
              {busyAction === "whitelist" ? "提交中…" : "提交审批策略"}
            </button>
          </div>
        </form>
      </section>

      {/* 卡片 4：事务性 Outbox 队列监控 */}
      <section className="panel" style={{ padding: 24, gap: 16 }}>
        <header className="panel-head">
          <h2 className="flex items-center gap-2" style={{ fontSize: 16, fontWeight: 600 }}>
            <Send size={16} aria-hidden="true" />
            事务性 Outbox 幂等发件箱队列
          </h2>
          <div className="right" style={{ gap: 12 }}>
            <span className="tag warn mono" style={{ fontSize: 14, fontWeight: 600 }}>
              待补传 {counts.pending}
            </span>
            <span className="tag plain mono" style={{ fontSize: 14, fontWeight: 600 }}>
              发送中 {counts.in_flight}
            </span>
            <span className="tag pass mono" style={{ fontSize: 14, fontWeight: 600 }}>
              已对账 {counts.acked}
            </span>
            <span className="tag alarm mono" style={{ fontSize: 14, fontWeight: 600 }}>
              死信 {counts.dead_letter}
            </span>
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger aria-label="按队列状态筛选" style={{ width: 140, fontSize: 14, fontWeight: 400 }}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="ALL">全部状态</SelectItem>
                <SelectItem value="PENDING">待补传</SelectItem>
                <SelectItem value="IN_FLIGHT">发送中</SelectItem>
                <SelectItem value="ACKED">已对账</SelectItem>
                <SelectItem value="DEAD_LETTER">死信异常</SelectItem>
              </SelectContent>
            </Select>
            <button
              type="button"
              className="btn ghost sm"
              disabled={loading}
              onClick={() => void loadOutbox()}
              style={{ fontSize: 14, fontWeight: 600 }}
            >
              <RefreshCw size={14} aria-hidden="true" />
              刷新
            </button>
            <button
              type="button"
              className="btn sm"
              disabled={busyAction === "flush" || !mes?.enabled}
              onClick={handleFlush}
              style={{ fontSize: 14, fontWeight: 600 }}
            >
              <Send size={14} aria-hidden="true" />
              {busyAction === "flush" ? "对账中…" : "立即补传对账"}
            </button>
          </div>
        </header>

        {items.length > 0 ? (
          <div style={{ overflowX: "auto" }}>
            <table className="grid-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>批次 / 周期 / 管序号</th>
                  <th>X-Idempotency-Key</th>
                  <th>状态</th>
                  <th>重试 / HTTP</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {items.map((item) => {
                  const stMeta = OUTBOX_STATUS_LABELS[item.status] || OUTBOX_STATUS_LABELS.PENDING
                  return (
                    <tr key={item.id}>
                      <td className="mono">#{item.id}</td>
                      <td>
                        <div className="mono">{item.lot_id} · #{item.tube_seq}</div>
                        <div style={{ color: "var(--ink-2)", fontSize: 14 }}>{item.cycle_id || "—"}</div>
                      </td>
                      <td>
                        <div className="flex items-center gap-2">
                          <code className="mono" style={{ fontSize: 14 }}>
                            {item.idempotency_key.slice(0, 16)}…
                          </code>
                          <button
                            type="button"
                            className="icon-btn"
                            aria-label="复制幂等键"
                            title={item.idempotency_key}
                            onClick={() => void copyKey(item.idempotency_key)}
                          >
                            <Copy size={14} aria-hidden="true" />
                          </button>
                        </div>
                      </td>
                      <td>
                        <span className={`tag ${stMeta.tone}`} style={{ fontSize: 14, fontWeight: 600 }}>
                          {stMeta.label}
                        </span>
                      </td>
                      <td className="mono">
                        <div>
                          {item.retry_count} 次 · HTTP {item.last_http_code ?? "—"}
                        </div>
                        {item.last_error && (
                          <div style={{ color: "var(--bad)", fontSize: 14 }}>{item.last_error}</div>
                        )}
                      </td>
                      <td>
                        {item.status === "DEAD_LETTER" ? (
                          <button
                            type="button"
                            className="btn ghost sm"
                            disabled={busyAction === `requeue_${item.id}`}
                            onClick={() => handleRequeue(item.id)}
                            style={{ fontSize: 14, fontWeight: 600 }}
                          >
                            <RotateCcw size={14} aria-hidden="true" />
                            重新入队
                          </button>
                        ) : (
                          <span style={{ color: "var(--ink-2)" }}>
                            {item.acked_at_ms ? formatMs(item.acked_at_ms) : "自动调度"}
                          </span>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <p style={{ margin: 0, fontSize: 14, fontWeight: 400, color: "var(--ink-2)" }}>
            当前筛选条件下暂无 MES Outbox 队列记录。
          </p>
        )}
      </section>
    </div>
  )
}
