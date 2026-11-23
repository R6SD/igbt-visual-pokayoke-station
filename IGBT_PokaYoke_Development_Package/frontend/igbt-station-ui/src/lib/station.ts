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
  VERIFY_C_ABSENT: "C 类确认",
  VIEW_INVALID: "视野无效",
  PASS: "放行",
  ALARM: "报警",
  FAULT: "故障",
  REVIEW_REQUIRED: "待复核",
  BOOT: "启动中",
  NEW_CYCLE: "新建检测周期",
  INITIAL_COMPLETE_CONFIRMED: "初始完整标签已确认",
  INSPECTION_STARTED: "开始防呆检验",
  A_CROSSED_OUT: "A 类划除已确认",
  B_COVERED: "B 类遮挡已确认",
  C_REMOVED: "C 类脱落满 1.2 秒放行",
  C_ABSENT_TIMER_STARTED: "开始 C 类 1.2 秒计时",
  C_TIMER_RESET_VIEW_INVALID: "视野无效，计时清零",
  C_TIMER_RESET_HAND_OCCLUSION: "手部遮挡，计时清零",
  C_TIMER_RESET_LABEL_REAPPEARED: "标签重现，计时清零",
  INITIAL_LABEL_ABSENT: "入位时标签缺失报警",
  VIEW_OR_QUALITY_INVALID: "画面或视角未达标",
  PART_REMOVED: "料管移出复位",
  AUTHORIZED_RESET: "已授权复位",
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
  crossed_out: "A 类已划除",
  covered: "B 类已遮挡",
  absent: "C 类已脱落",
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

/** 判定结果：纯中文主词 + 语义样式类（code 置空，避免中英文堆砌） */
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

/** 格式化 C 类确认计时秒数：未满 1200ms 时严格向下截断，避免 1199ms 提前显示 1.2 / 1.20 秒 */
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
  BARE_HAND_VIOLATION: "违规报警：裸手触碰料管（必须佩戴白色防静电手套）",
  BARE_HAND_WARNING_NO_GLOVE: "作业规范警告：检测到裸手，请先佩戴白色防静电手套后再拿取料管",
  WAITING_FOR_PART: "等待佩戴防静电手套并持料管入位",
  TUBE_NOT_PRESENT: "当前画面未检测到料管或标签（已过滤人像与衣物）",
  HAND_IN_CONTACT_ROI: "手套正在接触标签区",
  RUNTIME_NOT_CONNECTED: "视觉运行时未连接",
  API_UNAVAILABLE: "本机服务不可用",
  API_UNAVAILABLE_FAIL_CLOSED: "本机服务不可用，已闭锁",
  LOW_SHARPNESS: "画面失焦",
  OVEREXPOSED: "管壁反光过曝",
  LOW_BRIGHTNESS: "画面过暗",
  HIGH_BRIGHTNESS: "画面过亮",
  GRIP_MISSING: "双手未就位",
  A_CROSSED_OUT: "A 类划除已确认，允许放行",
  B_COVERED: "B 类遮挡已确认，允许放行",
  C_REMOVED: "C 类脱落满 1.20 秒，允许放行",
  C_ABSENT_CONFIRMED: "C 类脱落满 1.20 秒，允许放行",
  INITIAL_LABEL_ABSENT: "入位时未检测到初始完整标签",
  LABEL_STILL_COMPLETE_AFTER_RELEASE: "松手移出时原标签仍完整存在",
  VIEW_OR_QUALITY_INVALID: "管壁强反光或光学条件未达标",
  OVEREXPOSED_FRACTION_EXCEEDED: "管壁高光反光面积超限",
  PART_REMOVED: "料管已移出，等待下一管入位",
  AUTHORIZED_RESET: "工位已复位，等待下一管入位",
}

/** 把后端原因代码翻译成简洁中文；未知代码交由 human 兜底 */
export const reasonOf = (code?: string) => (code ? reasonText[code] || fsmText[code] || code.replaceAll("_", " ") : "")

export function guidanceFor(state: Data) {
  const outcome = deriveOutcome(state)
  const valid = state.view_valid === true
  const cTimer = Math.max(0, Number(state.c_timer_ms || 0))

  if (outcome === "FAULT" || state.camera_status === "FAULT") {
    return { title: "工位异常，已闭锁", desc: "联系设备工程师排查相机与本机服务" }
  }
  if (state.camera_status !== "ONLINE") {
    return { title: "相机未连接", desc: "检查相机接口与驱动状态" }
  }
  if (state.bare_hand_violation || state.result_reason === "BARE_HAND_VIOLATION") {
    return { title: "违规报警：裸手触碰料管（已闭锁）", desc: "作业必须佩戴白色防静电手套！请戴好手套后点击「解除报警复位」" }
  }
  if (state.alarm_latched || outcome === "ALARM") {
    return { title: "防呆报警已闭锁，禁止流转", desc: "请将不良管放入返工区，人工确认后点击「解除报警复位」" }
  }
  if (state.bare_hand_detected || state.result_reason === "BARE_HAND_WARNING_NO_GLOVE") {
    return { title: "检测到裸手 · 请佩戴防静电手套", desc: "未戴白色/浅灰防静电手套时严禁拿取或触碰 IGBT 料管" }
  }
  if (state.result_reason === "WAITING_FOR_PART" || state.view_valid_reasons?.[0] === "TUBE_NOT_PRESENT") {
    return state.work_mode === "debug"
      ? { title: "桌面调试待机 · 等待料管入位", desc: "持料管进入镜头即可动态追踪并采集调参（已过滤人像与衣物）" }
      : { title: "严格模式待机 · 等待料管入位", desc: "请佩戴白色防静电手套，双手水平持料管进入镜头（已自动过滤人像与衣物）" }
  }
  if (!valid) {
    if (state.left_grip_active !== true || state.right_grip_active !== true) {
      return { title: "双手握持料管两端", desc: "保持水平进入视野" }
    }
    return { title: "轻转料管正对镜头", desc: "避开管壁反光" }
  }
  if (cTimer > 0 && cTimer < C_LIMIT_MS) {
    return { title: "保持稳定", desc: `无标签 ${formatCTimerSeconds(cTimer, 1)} / 1.2 秒` }
  }
  if (outcome === "PASS") {
    return { title: "移入下一工序", desc: "放入合格品料盒" }
  }
  return { title: "等待防呆动作", desc: "完成划除、遮挡或撕除后自动判定" }
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
