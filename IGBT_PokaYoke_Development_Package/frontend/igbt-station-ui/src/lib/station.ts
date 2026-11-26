/* eslint-disable @typescript-eslint/no-explicit-any -- 运行时数据结构由后端 SQLite 模式定义 */
export type Data = Record<string, any>

export const fmt = (v: unknown, digits = 1) =>
  v === null || v === undefined || v === "" || !Number.isFinite(Number(v)) ? "—" : Number(v).toFixed(digits)

const fsmText: Record<string, string> = {
  WAIT_PART: "等待入位",
  VERIFY_INITIAL: "初始核验",
  READY: "就绪",
  INSPECTING: "检验中",
  PROCESSING: "推理中",
  VERIFY_C_ABSENT: "脱落确认",
  VIEW_INVALID: "视野无效",
  PASS: "放行",
  ALARM: "报警",
  FAULT: "故障",
  REVIEW_REQUIRED: "待复核",
  BOOT: "启动中",
  NEW_CYCLE: "新建周期",
  INITIAL_COMPLETE_CONFIRMED: "初始标签确认",
  INSPECTION_STARTED: "开始检验",
  A_CROSSED_OUT: "已划除放行",
  B_COVERED: "已遮挡放行",
  C_REMOVED: "已脱落放行",
  C_ABSENT_TIMER_STARTED: "脱落计时开始",
  C_TIMER_RESET_VIEW_INVALID: "视野无效清零",
  C_TIMER_RESET_HAND_OCCLUSION: "手部遮挡清零",
  C_TIMER_RESET_LABEL_REAPPEARED: "标签重现清零",
  INITIAL_LABEL_ABSENT: "入位标签缺失",
  VIEW_OR_QUALITY_INVALID: "视野未达标",
  PART_REMOVED: "料管移出",
  AUTHORIZED_RESET: "授权复位",
  STATE_CHANGE: "状态切换",
  STATE_TRANSITION: "状态跃迁",
}

export const human = (v: unknown) => {
  const s = String(v ?? "")
  if (!s) return "—"
  return fsmText[s] || s.replaceAll("_", " ")
}

export const labelText: Record<string, string> = {
  complete: "完整",
  crossed_out: "已划除",
  covered: "已遮挡",
  absent: "已脱落",
  unknown_challenge: "挑战样本",
  unknown: "未识别",
}

export const cameraText: Record<string, string> = {
  ONLINE: "在线",
  NOT_CONNECTED: "未连接",
  OFFLINE: "离线",
  FAULT: "故障",
  UNVERIFIED: "未验证",
}

export const pipeText = (c?: string) =>
  c === "yellow" ? "黄管" : c === "blue" ? "蓝管" : c === "other" ? "其他" : c ? "未知" : "—"

export const verdicts: Record<string, { word: string; code: string; tone: string }> = {
  PASS: { word: "放行", code: "", tone: "pass" },
  ALARM: { word: "报警", code: "", tone: "alarm" },
  FAULT: { word: "故障", code: "", tone: "fault" },
  VIEW_INVALID: { word: "视野无效", code: "", tone: "warn" },
  REVIEW_REQUIRED: { word: "待复核", code: "", tone: "warn" },
  WAITING: { word: "待检", code: "", tone: "idle" },
}

export const verdictOf = (v: unknown) =>
  verdicts[String(v || "WAITING")] || { word: String(v), code: "", tone: "idle" }

export const deriveOutcome = (state: Data) => {
  if (state.alarm_latched || state.bare_hand_violation || state.outcome === "ALARM") {
    return "ALARM"
  }
  if (state.bare_hand_detected || state.result_reason === "BARE_HAND_WARNING_NO_GLOVE") {
    return "VIEW_INVALID"
  }
  if (state.result_reason === "WAITING_FOR_PART" || state.view_valid_reasons?.[0] === "TUBE_NOT_PRESENT") {
    return "WAITING"
  }
  return state.outcome === "WAITING" && !state.view_valid ? "VIEW_INVALID" : state.outcome || "WAITING"
}

export const C_LIMIT_MS = 1200

export function formatCTimerSeconds(ms: number, digits: 1 | 2 = 1): string {
  const clamped = Math.max(0, Number.isFinite(ms) ? ms : 0)
  const factor = digits === 2 ? 100 : 10
  const stepMs = 1000 / factor
  const maxSteps = Math.floor(C_LIMIT_MS / stepMs)
  const flooredSteps = Math.floor(clamped / stepMs)
  const safeSteps = clamped < C_LIMIT_MS ? Math.min(maxSteps - 1, flooredSteps) : flooredSteps
  return (Math.max(0, safeSteps) / factor).toFixed(digits)
}

const reasonText: Record<string, string> = {
  BARE_HAND_VIOLATION: "裸手触管违规闭锁",
  BARE_HAND_WARNING_NO_GLOVE: "请先佩戴白色手套",
  WAITING_FOR_PART: "等待料管入位",
  TUBE_NOT_PRESENT: "未检测到料管",
  HAND_IN_CONTACT_ROI: "手套接触标签区",
  RUNTIME_NOT_CONNECTED: "视觉服务未连接",
  API_UNAVAILABLE: "服务不可用",
  API_UNAVAILABLE_FAIL_CLOSED: "服务离线闭锁",
  LOW_SHARPNESS: "画面失焦",
  OVEREXPOSED: "管壁反光过曝",
  LOW_BRIGHTNESS: "画面过暗",
  HIGH_BRIGHTNESS: "画面过亮",
  GRIP_MISSING: "双手未就位",
  A_CROSSED_OUT: "已划除确认放行",
  B_COVERED: "已遮挡确认放行",
  C_REMOVED: "已脱落满 1.2 秒放行",
  C_ABSENT_CONFIRMED: "已脱落满 1.2 秒放行",
  INITIAL_LABEL_ABSENT: "入位缺失初始标签",
  LABEL_STILL_COMPLETE_AFTER_RELEASE: "移出时标签未处理",
  VIEW_OR_QUALITY_INVALID: "光学条件未达标",
  OVEREXPOSED_FRACTION_EXCEEDED: "反光面积超限",
  PART_REMOVED: "料管已移出",
  AUTHORIZED_RESET: "工位已复位",
  TUBE_SWAP_SUSPECTED: "疑似途中换管/掉头异常，已触发告警闭锁",
  UNDEREXPOSED: "标签 ROI 曝光不足，请检查照明",
  LABEL_ROI_DRIFT_EXCEEDED: "标签定位框漂移超限",
  OPTICAL_DRIFT: "相机光学参数/亮度漂移告警",
  CAMERA_HANDLE_LOCKED_RESTART_REQUIRED: "相机采集句柄锁死，请重启工位服务或检查 USB 连接",
  RED_RABBIT_CHALLENGE_REQUIRED: "等待完成红兔金样点检",
  RED_RABBIT_HARD_LOCK: "红兔点检失败硬闭锁（需工程师解锁）",
  CRITICAL_NG_MISSED_AS_PASS: "NG 金样误判为 PASS，工位硬闭锁",
  GOLDEN_SAMPLE_MISMATCH: "金样实测结果与预期不符",
  SESSION_TIMEOUT: "红兔点检会话超时闭锁",
  SHIFT_CHANGE: "换班时间窗强制点检",
  INTERVAL_EXPIRED: "周期超时强制点检",
  LOT_CHANGE: "批次切换强制点检",
  COLD_BOOT: "冷启动强制点检",
  CRASH_RECOVERY: "异常恢复强制点检",
  MODEL_OR_OPTICS_CHANGE: "模型或光学参数变更点检",
  MANUAL_AUDIT: "工程师手动发起点检",
  MODBUS_WATCHDOG_TIMEOUT: "ADAM DIO Modbus TCP 看门狗心跳超时闭锁",
  HARDWARE_ESTOP_ACTIVE: "急停按钮 / 安全回路断开闭锁",
  RELAY_CONTACT_WELD_FAULT: "安全继电器辅助触点粘连故障闭锁",
  VISION_LOOP_STALL_WATCHDOG_STOP: "视觉推理循环停滞，看门狗主动停喂硬锁",
  TUBE_ALIGN_FAIL: "料管透视对齐失败（已阻断放行）",
  TUBE_ALIGN_LOW_CONF: "料管姿态波动，已暂停 PASS 计时",
  WHITE_GLOVE_OVEREXPOSED: "白色手套区域过曝，请调整补光角度",
  INSUFFICIENT_MIDBAND_EDGES: "料管中段有效边缘不足（遮挡或强反光）",
  PARALLEL_PAIR_NOT_FOUND: "未检出合法平行管壁边缘对",
  COND_NUMBER_EXCEEDED: "透视变换矩阵条件数越界（几何畸变）",
  PARALLEL_DIFF_EXCEEDED: "料管上下壁面平行度偏差超限",
  WIDTH_DEV_EXCEEDED: "料管实测宽度与标定先验偏差超限",
  ANGLE_JUMP_EXCEEDED: "料管帧间倾角跳变超限",
}

export const reasonOf = (code?: string) => (code ? reasonText[code] || fsmText[code] || code.replaceAll("_", " ") : "")

export function guidanceFor(state: Data) {
  const outcome = deriveOutcome(state)
  const valid = state.view_valid === true
  const cTimer = Math.max(0, Number(state.c_timer_ms || 0))

  if (outcome === "FAULT" || state.camera_status === "FAULT") {
    return { title: "工位异常闭锁" }
  }
  if (state.camera_status !== "ONLINE") {
    return { title: "相机未连接" }
  }
  if (state.bare_hand_violation || state.result_reason === "BARE_HAND_VIOLATION") {
    return { title: "裸手违规闭锁" }
  }
  if (state.alarm_latched || outcome === "ALARM") {
    return { title: "防呆报警闭锁" }
  }
  if (state.bare_hand_detected || state.result_reason === "BARE_HAND_WARNING_NO_GLOVE") {
    return { title: "请佩戴防静电手套" }
  }
  if (state.result_reason === "WAITING_FOR_PART" || state.view_valid_reasons?.[0] === "TUBE_NOT_PRESENT") {
    return { title: "等待料管入位" }
  }
  if (!valid) {
    if (
      state.tube_align_failed ||
      state.result_reason === "TUBE_ALIGN_FAIL" ||
      state.view_valid_reasons?.includes("TUBE_ALIGN_FAIL")
    ) {
      return { title: "料管对齐失败·请摆正料管" }
    }
    if (
      state.result_reason === "TUBE_ALIGN_LOW_CONF" ||
      state.view_valid_reasons?.includes("TUBE_ALIGN_LOW_CONF")
    ) {
      return { title: "料管姿态波动·请保持平稳" }
    }
    if (state.left_grip_active !== true || state.right_grip_active !== true) {
      return { title: "请双手握持两端" }
    }
    return { title: "请轻转料管避反光" }
  }
  if (cTimer > 0 && cTimer < C_LIMIT_MS) {
    return { title: `脱落确认 ${formatCTimerSeconds(cTimer, 1)} / 1.2s` }
  }
  if (outcome === "PASS") {
    return { title: "合格放行" }
  }
  return { title: "等待防呆动作" }
}

export const timeOf = (iso?: string) =>
  iso ? new Date(iso).toLocaleTimeString("zh-CN", { hour12: false }) : "—"

export const dateTimeOf = (iso?: string) =>
  iso
    ? new Date(iso).toLocaleString("zh-CN", {
        hour12: false,
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      })
    : "—"
