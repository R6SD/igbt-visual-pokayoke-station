import { useEffect, useState } from "react"
import { Camera, RotateCcw, ShieldAlert, ShieldCheck } from "lucide-react"
import { Andon } from "@/components/Andon"
import { CameraFrame } from "@/components/CameraFrame"
import { C_LIMIT_MS, type Data, deriveOutcome, labelText, pipeText } from "@/lib/station"

interface OperatorViewProps {
  state: Data
  evidence?: Data[]
  totalCount: number
  passCount: number
  alarmCount: number
  avgTaktTime: number | null
  onManualTrigger?: () => void
  onResetAlarm?: () => void
}

function operatorGuidanceFor(state: Data): { title: string; desc?: string } {
  const outcome = deriveOutcome(state)
  const valid = state.view_valid === true
  const cTimer = Math.max(0, Number(state.c_timer_ms || 0))

  if (outcome === "FAULT" || state.camera_status === "FAULT") {
    return { title: "工位异常，已闭锁", desc: "请联系设备工程师排查相机与本机服务" }
  }
  if (state.camera_status !== "ONLINE") {
    return { title: "相机未连接", desc: "请等待设备就绪或联系工程师检查接口" }
  }
  if (state.bare_hand_violation || state.result_reason === "BARE_HAND_VIOLATION") {
    return { title: "裸手触管违规 · 已闭锁", desc: "请佩戴白色防静电手套后点击「解除报警闭锁并复位」" }
  }
  if (state.alarm_latched || outcome === "ALARM") {
    return { title: "防呆报警已闭锁，禁止流转", desc: "请将料管放入返工暂存区，人工复核后复位" }
  }
  if (state.bare_hand_detected || state.result_reason === "BARE_HAND_WARNING_NO_GLOVE") {
    return { title: "检测到裸手 · 请佩戴白色防静电手套", desc: "佩戴好白色防静电手套后再拿取料管" }
  }
  if (state.result_reason === "WAITING_FOR_PART" || state.view_valid_reasons?.[0] === "TUBE_NOT_PRESENT") {
    return { title: "等待料管入位", desc: "请佩戴白色防静电手套，双手水平持料管进入取景框" }
  }
  if (!valid) {
    if (state.left_grip_active !== true || state.right_grip_active !== true) {
      return { title: "请双手握持料管两端", desc: "保持料管水平进入取景框中央" }
    }
    return { title: "请轻转料管正对镜头", desc: "避开管壁反光与手部遮挡" }
  }
  if (cTimer > 0 && cTimer < C_LIMIT_MS) {
    return { title: "请保持稳定", desc: `C 类撕除确认 ${(cTimer / 1000).toFixed(2)} / 1.20 秒` }
  }
  if (outcome === "PASS") {
    return { title: "合格放行", desc: "请移入合格品料盒，准备下一管" }
  }
  return { title: "等待防呆处理", desc: "完成划除、遮挡或撕除后自动放行" }
}

export function OperatorView({
  state,
  evidence = [],
  totalCount,
  passCount,
  alarmCount,
  avgTaktTime,
  onManualTrigger,
  onResetAlarm,
}: OperatorViewProps) {
  const [shutterSeq, setShutterSeq] = useState(0)

  const cTimer = Math.max(0, Number(state.c_timer_ms || 0))
  const litSegments = Math.min(12, Math.floor((cTimer / C_LIMIT_MS) * 12))
  const yieldRate = totalCount > 0 ? ((passCount / totalCount) * 100).toFixed(1) : "—"
  const online = state.camera_status === "ONLINE"
  const frameSrc = (online && state.live_stream_url) || state.latest_evidence_url || evidence[0]?.url
  const cropSrc = (online && state.live_crop_url) || frameSrc
  const focusLocked = online && state.view_valid === true
  const isAlarmLatched = Boolean(state.alarm_latched || state.outcome === "ALARM" || state.bare_hand_violation)

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        document.querySelector('[role="dialog"]')
      ) {
        return
      }
      if (e.code === "Space" && !e.repeat) {
        setShutterSeq((n) => n + 1)
      }
    }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [])

  const handleManualCapture = () => {
    setShutterSeq((n) => n + 1)
    onManualTrigger?.()
  }

  const gloveSideState = (active: unknown) =>
    active === true
      ? { lamp: "ok", text: "已佩戴" }
      : active === false
        ? { lamp: "", text: "未入画" }
        : { lamp: "", text: "待检测" }

  const leftGlove = gloveSideState(state.left_grip_active)
  const rightGlove = gloveSideState(state.right_grip_active)

  const gloveSummary = state.bare_hand_violation
    ? { tone: "bad", label: "裸手触管违规 · 已闭锁", short: "裸手违规" }
    : state.bare_hand_detected
      ? { tone: "warn", label: "检测到裸手 · 请佩戴白色手套", short: "检测到裸手" }
      : state.glove_compliant
        ? { tone: "ok", label: "白色防静电手套合规", short: "手套合规" }
        : { tone: "idle", label: "请佩戴白色防静电手套入画", short: "待核验" }

  const labelDisplay = state.bare_hand_violation
    ? "裸手违规报警"
    : state.label_status && state.label_status !== "unknown"
      ? labelText[state.label_status] || "待判定"
      : "等待料管入位"

  const labelTone = state.bare_hand_violation || isAlarmLatched
    ? "alarm"
    : state.label_status === "crossed_out" || state.label_status === "covered" || state.label_status === "absent"
      ? "pass"
      : state.label_status === "complete"
        ? "warn"
        : "plain"

  return (
    <div className="op-screen">
      <Andon
        compact
        state={state}
        guide={operatorGuidanceFor(state)}
        meta={[
          { label: "防静电手套", value: gloveSummary.short },
          { label: "当前料管", value: state.pipe_color ? pipeText(state.pipe_color) : "待入位" },
          { label: "平均节拍", value: avgTaktTime ? `${avgTaktTime.toFixed(1)} 秒` : "—" },
        ]}
      />

      <div className="op-stage">
        {/* 左侧超大固定实时取景框 + 底部 3 栏手套合规指示条（零叠字、完整无裁切） */}
        <section className="op-hero" aria-label="实时目检取景框">
          <header className="panel-head op-hero-head">
            <h2>实时目检画面</h2>
            <span className="sub">{state.pipe_color ? pipeText(state.pipe_color) : "全景视野"}</span>
            <div className="right">
              <span className={`lamp ${online ? "ok" : "bad"}`} aria-hidden="true" />
              <span>{online ? "取景就绪" : "相机未连接"}</span>
            </div>
          </header>

          <div className="op-hero-body">
            <CameraFrame
              fill
              src={frameSrc}
              alt="现场 16:9 全景实时目检画面"
              shutterKey={shutterSeq}
              focusLocked={focusLocked}
              emptyTitle="等待相机画面"
            />
          </div>

          <div className="op-glove-bar" aria-label="防静电手套合规指示">
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
                <ShieldAlert size={16} aria-hidden="true" />
              ) : (
                <ShieldCheck size={16} aria-hidden="true" />
              )}
              <div className="op-glove-cell-text">
                <small>作业规范</small>
                <b>{gloveSummary.label}</b>
              </div>
            </div>
          </div>
        </section>

        {/* 右侧固定 330px 极简动作列 */}
        <aside className="op-column" aria-label="目检状态与作业操作">
          <div className="op-section op-crop-section">
            <div className="op-section-head">
              <h2 className="op-section-title">标签 4× 特写</h2>
              <span className={`tag ${labelTone}`}>{labelDisplay}</span>
            </div>
            <div className="op-crop-box">
              <CameraFrame
                fill
                src={cropSrc}
                alt="防呆标签 4 倍特写"
                focusLocked={focusLocked}
                emptyTitle="等待料管入位"
              />
            </div>
          </div>

          <div className="op-section">
            <div className="op-section-head">
              <span className="op-section-title">C 类撕除确认</span>
              <span className="op-section-meta">{(cTimer / 1000).toFixed(2)} / 1.20 秒</span>
            </div>
            <div
              className={`segments ${litSegments >= 12 ? "done" : ""}`}
              role="progressbar"
              aria-label="C 类撕除 1.20 秒确认进度"
              aria-valuemin={0}
              aria-valuemax={C_LIMIT_MS}
              aria-valuenow={Math.min(C_LIMIT_MS, Math.round(cTimer))}
            >
              {Array.from({ length: 12 }, (_, i) => (
                <i key={i} className={i < litSegments ? "on" : ""} />
              ))}
            </div>
          </div>

          <div className="op-section op-actions">
            {isAlarmLatched && onResetAlarm && (
              <button type="button" className="btn btn-danger block" onClick={onResetAlarm}>
                <RotateCcw size={15} aria-hidden="true" />
                解除报警闭锁并复位
              </button>
            )}
            <button type="button" className="btn block" onClick={handleManualCapture}>
              <Camera size={15} aria-hidden="true" />
              手动抓拍并归档
              <span className="op-kbd">空格</span>
            </button>
          </div>

          <div className="op-stats" aria-label="本班计数">
            <div className="op-stat">
              <small>本班检验</small>
              <b>
                {totalCount}
                <em>管</em>
              </b>
            </div>
            <div className="op-stat">
              <small>放行</small>
              <b style={{ color: "var(--ok)" }}>
                {passCount}
                <em>管</em>
              </b>
            </div>
            <div className="op-stat">
              <small>报警</small>
              <b style={{ color: alarmCount ? "var(--bad)" : undefined }}>
                {alarmCount}
                <em>管</em>
              </b>
            </div>
            <div className="op-stat">
              <small>良率</small>
              <b>
                {yieldRate}
                {yieldRate !== "—" && <em>%</em>}
              </b>
            </div>
          </div>
        </aside>
      </div>
    </div>
  )
}
