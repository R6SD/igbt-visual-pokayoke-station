import { Camera, RotateCcw, ShieldAlert, ShieldCheck } from "lucide-react"
import { Andon } from "@/components/Andon"
import { CameraFrame } from "@/components/CameraFrame"
import { C_LIMIT_MS, type Data, formatCTimerSeconds, guidanceFor, labelText, pipeText } from "@/lib/station"

interface OperatorViewProps {
  state: Data
  evidence?: Data[]
  totalCount: number
  passCount: number
  alarmCount: number
  avgTaktTime: number | null
  statsScope?: "all" | "recent"
  shutterKey?: number
  onManualTrigger?: () => void
  onResetAlarm?: () => void
}

export function OperatorView({
  state,
  evidence = [],
  totalCount,
  passCount,
  alarmCount,
  avgTaktTime,
  statsScope = "recent",
  shutterKey = 0,
  onManualTrigger,
  onResetAlarm,
}: OperatorViewProps) {
  const cTimer = Math.max(0, Number(state.c_timer_ms || 0))
  const litSegments = Math.min(12, Math.floor((cTimer / C_LIMIT_MS) * 12))
  const yieldRate = totalCount > 0 ? `${((passCount / totalCount) * 100).toFixed(1)}%` : "—"
  const online = state.camera_status === "ONLINE"
  const frameSrc = (online && state.live_stream_url) || state.latest_evidence_url || evidence[0]?.url
  const cropSrc = online && state.live_crop_url ? String(state.live_crop_url) : undefined
  const focusLocked = online && state.view_valid === true
  const isAlarmLatched = Boolean(state.alarm_latched || state.outcome === "ALARM" || state.bare_hand_violation)

  const gloveSideState = (active: unknown) =>
    active === true
      ? { lamp: "ok", text: "已佩戴" }
      : active === false
        ? { lamp: "", text: "未入画" }
        : { lamp: "", text: "待检测" }

  const leftGlove = gloveSideState(state.left_grip_active)
  const rightGlove = gloveSideState(state.right_grip_active)

  const gloveSummary = state.bare_hand_violation
    ? { tone: "bad", label: "裸手违规", short: "裸手违规" }
    : state.bare_hand_detected
      ? { tone: "warn", label: "检测到裸手", short: "裸手警告" }
      : state.glove_compliant
        ? { tone: "ok", label: "手套合规", short: "合规" }
        : { tone: "idle", label: "待核验", short: "待检" }

  const labelDisplay = state.bare_hand_violation
    ? "裸手违规"
    : state.label_status && state.label_status !== "unknown"
      ? labelText[state.label_status] || "待判定"
      : "待入位"

  const labelTone =
    state.bare_hand_violation || isAlarmLatched
      ? "alarm"
      : state.label_status === "crossed_out" || state.label_status === "covered" || state.label_status === "absent"
        ? "pass"
        : state.label_status === "complete"
          ? "warn"
          : "plain"

  const totalLabel = statsScope === "all" ? "累计总数" : "近期总数"

  return (
    <div className="op-screen">
      <Andon
        compact
        state={state}
        guide={guidanceFor(state)}
        meta={[
          { label: "手套状态", value: gloveSummary.short },
          { label: "当前料管", value: state.pipe_color ? pipeText(state.pipe_color) : "—" },
          { label: "平均节拍", value: avgTaktTime ? `${avgTaktTime.toFixed(1)}s` : "—" },
        ]}
      />

      <div className="op-stage">
        <section className="op-hero" aria-label="实时目检画面">
          <header className="op-hero-head">
            <h2>实时画面</h2>
            <div className="topbar-status" style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
              <span className={`lamp ${online ? "ok" : "bad"}`} aria-hidden="true" />
              <span style={{ fontSize: 14, color: "var(--ink-2)" }}>
                {online ? "720P · 25 FPS" : "相机离线"}
              </span>
            </div>
          </header>

          <div className="op-hero-body">
            <CameraFrame
              fill
              src={frameSrc}
              alt="16:9 全景实时目检画面"
              shutterKey={shutterKey}
              focusLocked={focusLocked}
              emptyTitle="等待相机画面"
              errorTitle="画面不可用"
            />
          </div>

          <div className="op-glove-bar" aria-label="防静电手套状态">
            <div className="op-glove-cell">
              <span className={`lamp ${leftGlove.lamp}`} aria-hidden="true" />
              <div className="op-glove-cell-text">
                <small>左手套</small>
                <b>{leftGlove.text}</b>
              </div>
            </div>
            <div className="op-glove-cell">
              <span className={`lamp ${rightGlove.lamp}`} aria-hidden="true" />
              <div className="op-glove-cell-text">
                <small>右手套</small>
                <b>{rightGlove.text}</b>
              </div>
            </div>
            <div className={`op-glove-cell op-glove-rule ${gloveSummary.tone}`}>
              {gloveSummary.tone === "bad" || gloveSummary.tone === "warn" ? (
                <ShieldAlert size={18} aria-hidden="true" />
              ) : (
                <ShieldCheck size={18} aria-hidden="true" />
              )}
              <div className="op-glove-cell-text">
                <small>防静电规范</small>
                <b>{gloveSummary.label}</b>
              </div>
            </div>
          </div>
        </section>

        <aside className="op-column" aria-label="标签特写与工位统计">
          <div className="op-card op-card-grow">
            <div style={{ display: "grid", gap: 12 }}>
              <div className="op-section-head">
                <h2 className="op-section-title">标签特写</h2>
                <span className={`tag ${labelTone}`}>{labelDisplay}</span>
              </div>
              <div className="op-crop-box">
                <CameraFrame
                  fill
                  src={cropSrc}
                  alt="标签 4 倍特写"
                  focusLocked={Boolean(focusLocked && cropSrc)}
                  emptyTitle={online ? "无特写流" : "等待入位"}
                  errorTitle="特写不可用"
                />
              </div>
            </div>

            <div style={{ display: "grid", gap: 8 }}>
              <div className="op-section-head">
                <span className="op-section-title">脱落确认</span>
                <span className="op-section-meta">{formatCTimerSeconds(cTimer, 2)} / 1.20s</span>
              </div>
              <div
                className={`segments ${litSegments >= 12 ? "done" : ""}`}
                role="progressbar"
                aria-label="脱落 1.20 秒确认进度"
                aria-valuemin={0}
                aria-valuemax={C_LIMIT_MS}
                aria-valuenow={Math.min(C_LIMIT_MS, Math.round(cTimer))}
              >
                {Array.from({ length: 12 }, (_, i) => (
                  <i key={i} className={i < litSegments ? "on" : ""} />
                ))}
              </div>
            </div>

            <div className="op-actions">
              {isAlarmLatched && onResetAlarm && (
                <button type="button" className="btn btn-danger block" onClick={onResetAlarm}>
                  <RotateCcw size={16} aria-hidden="true" />
                  报警复位
                </button>
              )}
              <button type="button" className="btn block" onClick={onManualTrigger}>
                <Camera size={16} aria-hidden="true" />
                手动抓拍
                <span className="op-kbd">空格</span>
              </button>
            </div>
          </div>

          <div className="op-card" aria-label="工位检验指标">
            <div className="op-stats">
              <div className="op-stat">
                <b>{totalCount}</b>
                <span>{totalLabel}</span>
              </div>
              <div className="op-stat">
                <b style={{ color: "var(--ok)" }}>{passCount}</b>
                <span>放行</span>
              </div>
              <div className="op-stat">
                <b style={{ color: alarmCount ? "var(--bad)" : undefined }}>{alarmCount}</b>
                <span>报警</span>
              </div>
              <div className="op-stat">
                <b>{yieldRate}</b>
                <span>良率</span>
              </div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  )
}
