/* eslint-disable @typescript-eslint/no-explicit-any -- 机器视觉 API 响应已定义在后端 SQLite 模式中 */
import { Fragment, useCallback, useEffect, useRef, useState } from "react"
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  ChevronRight,
  ClipboardList,
  Copy,
  Cpu,
  Crosshair,
  Database,
  Eye,
  FlipHorizontal2,
  Globe,
  HardDrive,
  ImagePlus,
  Layers,
  Lock,
  LogOut,
  Moon,
  Pin,
  Play,
  RefreshCw,
  RotateCcw,
  ShieldAlert,
  Sliders,
  Sparkles,
  Sun,
  Trash2,
  Upload,
  X,
} from "lucide-react"
import tubeBlueImg from "@/assets/showcase/tube-blue.jpg"
import tubeYellowImg from "@/assets/showcase/tube-yellow.jpg"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Andon } from "@/components/Andon"
import { CameraFrame } from "@/components/CameraFrame"
import { EngineerAuthModal, type EngineerAuthResult } from "@/components/EngineerAuthModal"
import { Filmstrip, type FilmstripItem } from "@/components/Filmstrip"
import { OpticalGauges } from "@/components/OpticalGauges"
import { OperatorView } from "@/components/OperatorView"
import { ShowcasePage } from "@/components/ShowcasePage"
import {
  C_LIMIT_MS,
  type Data,
  cameraText,
  dateTimeOf,
  deriveOutcome,
  fmt,
  human,
  labelText,
  pipeText,
  reasonOf,
  timeOf,
  verdictOf,
} from "@/lib/station"

type Role = "operator" | "engineer"
type EngineerStep = "monitor" | "collect" | "train" | "records_settings"
type RecordsSettingsTab = "records" | "storage" | "settings"
type CycleFilter = "ALL" | "PASS" | "ALARM" | "VIEW_INVALID"
type SampleLabel = "complete" | "crossed_out" | "covered" | "absent" | "unknown_challenge"
type SampleFilter = "" | SampleLabel

interface ConfirmDialogState {
  title: string
  description: string
  consequences?: string[]
  confirmLabel: string
  tone: "warn" | "danger"
  requireCheckLabel?: string
  onConfirm: () => Promise<void>
}

interface StoredEngineerSession {
  token: string
  expiresAt: number
  lastActiveAt: number
  idleTimeoutMs: number
}

const ENGINEER_SESSION_KEY = "igbt_engineer_session_v1"
const IDLE_LOCK_MS = 30 * 60 * 1000

function readStoredSession(): StoredEngineerSession | null {
  try {
    const raw = window.sessionStorage.getItem(ENGINEER_SESSION_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as Partial<StoredEngineerSession>
    if (
      typeof parsed.token !== "string" ||
      !parsed.token ||
      typeof parsed.expiresAt !== "number" ||
      typeof parsed.lastActiveAt !== "number"
    ) {
      window.sessionStorage.removeItem(ENGINEER_SESSION_KEY)
      return null
    }
    const idleLimit =
      typeof parsed.idleTimeoutMs === "number" && parsed.idleTimeoutMs > 0 ? parsed.idleTimeoutMs : IDLE_LOCK_MS
    const now = Date.now()
    if (now >= parsed.expiresAt || now - parsed.lastActiveAt >= idleLimit) {
      window.sessionStorage.removeItem(ENGINEER_SESSION_KEY)
      return null
    }
    return {
      token: parsed.token,
      expiresAt: parsed.expiresAt,
      lastActiveAt: parsed.lastActiveAt,
      idleTimeoutMs: idleLimit,
    }
  } catch {
    return null
  }
}

function saveStoredSession(token: string, expiresInSeconds: number): StoredEngineerSession {
  const now = Date.now()
  const ttlMs = Math.max(60_000, (expiresInSeconds || 1800) * 1000)
  const session: StoredEngineerSession = {
    token,
    expiresAt: now + ttlMs,
    lastActiveAt: now,
    idleTimeoutMs: Math.min(IDLE_LOCK_MS, ttlMs),
  }
  try {
    window.sessionStorage.setItem(ENGINEER_SESSION_KEY, JSON.stringify(session))
  } catch {
    /* 忽略浏览器隐私模式 sessionStorage 异常 */
  }
  return session
}

function touchStoredSession(now: number): void {
  try {
    const raw = window.sessionStorage.getItem(ENGINEER_SESSION_KEY)
    if (!raw) return
    const parsed = JSON.parse(raw) as StoredEngineerSession
    parsed.lastActiveAt = now
    window.sessionStorage.setItem(ENGINEER_SESSION_KEY, JSON.stringify(parsed))
  } catch {
    /* 忽略 sessionStorage 写入异常 */
  }
}

function clearStoredSession(): void {
  try {
    window.sessionStorage.removeItem(ENGINEER_SESSION_KEY)
  } catch {
    /* 忽略 sessionStorage 清理异常 */
  }
}

const roiNames = ["tube_roi", "label_roi", "label_contact_roi", "left_grip_zone", "right_grip_zone"]
const roiLabels: Record<string, string> = {
  tube_roi: "料管整体参考区",
  label_roi: "防呆标签参考区",
  label_contact_roi: "标签接触参考区",
  left_grip_zone: "左端握持参考区",
  right_grip_zone: "右端握持参考区",
}
const roiShortLabels: Record<string, string> = {
  tube_roi: "料管整体",
  label_roi: "防呆标签",
  label_contact_roi: "接触区",
  left_grip_zone: "左握持区",
  right_grip_zone: "右握持区",
}
const roiColors = ["#e2531b", "#2f6fdb", "#d43d8f", "#8a5cd6", "#1e9b55"]

const defaultStandardRois: Record<string, [number, number, number, number]> = {
  tube_roi: [0.16, 0.42, 0.84, 0.58],
  label_roi: [0.43, 0.45, 0.57, 0.55],
  label_contact_roi: [0.4, 0.38, 0.6, 0.62],
  left_grip_zone: [0.16, 0.34, 0.31, 0.82],
  right_grip_zone: [0.69, 0.34, 0.84, 0.82],
}

const simScenarios: { id: string; label: string; tone?: string }[] = [
  { id: "a_pass", label: "A 划除" },
  { id: "b_pass", label: "B 遮挡" },
  { id: "c_pass", label: "C 撕除" },
  { id: "alarm", label: "未处理报警", tone: "bad" },
  { id: "glare_invalid", label: "反光受阻", tone: "warn" },
  { id: "reset", label: "恢复实时" },
]

const thresholdFields: [string, string, string][] = [
  ["hand_glove_confidence", "手部/手套动态追踪置信度", "0.01"],
  ["label_min_confidence", "标签分类最小置信度", "0.01"],
  ["label_min_margin", "置信度最小差值间隔", "0.01"],
  ["stable_state_ms", "状态稳定确认时长（毫秒）", "1"],
  ["min_stable_frames", "最少连续稳定帧数", "1"],
  ["min_laplacian_variance", "标签区清晰度下限（拉普拉斯）", "0.1"],
  ["min_brightness", "标签区亮度下限", "1"],
  ["max_brightness", "标签区亮度上限", "1"],
  ["max_overexposed_fraction", "反光过曝面积上限", "0.01"],
]

const engineerSteps: { id: EngineerStep; num: string; label: string; icon: typeof Camera }[] = [
  { id: "monitor", num: "1", label: "实时监控", icon: Camera },
  { id: "collect", num: "2", label: "采集标注", icon: Database },
  { id: "train", num: "3", label: "训练部署", icon: Cpu },
  { id: "records_settings", num: "4", label: "记录与设置", icon: Sliders },
]

const sampleClassDefs: {
  id: SampleLabel
  title: string
  short: string
  desc: string
  tone: string
}[] = [
  { id: "complete", title: "拍【完整标签】", short: "完整标签", desc: "未划除、未遮挡的原始白色标签", tone: "warn" },
  { id: "crossed_out", title: "拍【A类 · 已划除】", short: "A类划除", desc: "记号笔明显划叉的合格标签", tone: "pass" },
  { id: "covered", title: "拍【B类 · 已遮挡】", short: "B类遮挡", desc: "深色胶带或标签完全覆盖", tone: "pass" },
  { id: "absent", title: "拍【C类 · 已脱落】", short: "C类脱落", desc: "已撕除标签、仅露透明管壁", tone: "pass" },
  { id: "unknown_challenge", title: "拍【挑战样本】", short: "挑战样本", desc: "反光、模糊或手部半遮挡边界", tone: "plain" },
]

const sampleFilterOptions: { id: SampleFilter; label: string }[] = [
  { id: "", label: "全部" },
  { id: "complete", label: "完整" },
  { id: "crossed_out", label: "A类划除" },
  { id: "covered", label: "B类遮挡" },
  { id: "absent", label: "C类脱落" },
  { id: "unknown_challenge", label: "挑战" },
]

function toHeaderValue(raw: string, fallback: string): string {
  const trimmed = raw.trim()
  if (!trimmed) return fallback
  if (/^[\x20-\x7E]+$/.test(trimmed)) return trimmed
  return encodeURIComponent(trimmed)
}

async function api<T = any>(path: string, options: RequestInit = {}, token = ""): Promise<T> {
  const headers = new Headers(options.headers)
  if (token) headers.set("X-Admin-Token", token)
  const response = await fetch(path, { ...options, headers })
  if (!response.ok) {
    let reason = `HTTP ${response.status}`
    try {
      reason = (await response.json()).detail || reason
    } catch {
      /* 非 JSON 响应 */
    }
    throw new Error(String(reason))
  }
  return response.json() as Promise<T>
}

function App() {
  const [showShowcase, setShowShowcase] = useState(() => {
    const sp = new URLSearchParams(window.location.search)
    return sp.get("showcase") === "1" || sp.get("page") === "showcase"
  })

  // 严格角色隔离：默认 operator，仅当 sessionStorage 中存在有效已验证会话时恢复 engineer，绝不硬编码明文口令
  const [role, setRole] = useState<Role>(() => (readStoredSession() ? "engineer" : "operator"))
  const [adminToken, setAdminToken] = useState<string>(() => readStoredSession()?.token || "")
  const [step, setStep] = useState<EngineerStep>("monitor")
  const [step4Tab, setStep4Tab] = useState<RecordsSettingsTab>("records")
  const [authModalOpen, setAuthModalOpen] = useState(false)
  const loginTriggerRef = useRef<HTMLButtonElement | null>(null)
  const lastActiveRef = useRef<number>(0)
  const idleLimitRef = useRef<number>(IDLE_LOCK_MS)

  const [theme, setTheme] = useState<"light" | "dark">(() => {
    const q = new URLSearchParams(window.location.search).get("theme")
    if (q === "dark" || q === "light") return q
    return window.localStorage.getItem("igbt-ui-theme") === "dark" ? "dark" : "light"
  })

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    window.localStorage.setItem("igbt-ui-theme", theme)
  }, [theme])

  const [currentTime, setCurrentTime] = useState(new Date())
  const [state, setState] = useState<Data>({
    camera_status: "NOT_CONNECTED",
    view_valid: false,
    label_status: "unknown",
    fsm_state: "WAIT_PART",
    outcome: "WAITING",
    c_timer_ms: 0,
    mirror_horizontal: true,
    view_valid_reasons: ["RUNTIME_NOT_CONNECTED"],
  })
  const [cycles, setCycles] = useState<Data[]>([])
  const [evidence, setEvidence] = useState<Data[]>([])
  const [config, setConfig] = useState<Data>({ rois: {}, thresholds: {} })
  const [history, setHistory] = useState<Data[]>([])
  const [models, setModels] = useState<Data[]>([])
  const [datasetSummary, setDatasetSummary] = useState<Data>({
    total_samples: 0,
    class_counts: {},
    independent_tube_groups: 0,
    ready_to_train: false,
    recent_samples: [],
  })
  const [trainingStatus, setTrainingStatus] = useState<Data>({
    state: "IDLE",
    message: "等待启动训练",
    progress_pct: 0,
    history: [],
  })
  const [storageOverview, setStorageOverview] = useState<Data | null>(null)
  const [storageLoading, setStorageLoading] = useState(false)
  const [confirmDialog, setConfirmDialog] = useState<ConfirmDialogState | null>(null)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState("")
  const [events, setEvents] = useState<Data[]>([])
  const [selectedCycleId, setSelectedCycleId] = useState<string>("")
  const [notice, setNotice] = useState("")
  const [refreshing, setRefreshing] = useState(false)
  const [simulating, setSimulating] = useState("")
  const [togglingMirror, setTogglingMirror] = useState(false)
  const [activeSnapshot, setActiveSnapshot] = useState<FilmstripItem | null>(null)
  const noticeTimerRef = useRef<number | null>(null)
  const refreshTimerRef = useRef<number | null>(null)

  const flash = useCallback((message: string) => {
    setNotice(message)
    if (noticeTimerRef.current !== null) {
      window.clearTimeout(noticeTimerRef.current)
    }
    noticeTimerRef.current = window.setTimeout(() => {
      setNotice("")
      noticeTimerRef.current = null
    }, 4500)
  }, [])

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000)
    return () => {
      clearInterval(timer)
      if (noticeTimerRef.current !== null) {
        window.clearTimeout(noticeTimerRef.current)
      }
      if (refreshTimerRef.current !== null) {
        window.clearTimeout(refreshTimerRef.current)
      }
    }
  }, [])

  const lockEngineerSession = useCallback(
    (reason?: string) => {
      clearStoredSession()
      setAdminToken("")
      setRole("operator")
      setStep("monitor")
      setActiveSnapshot(null)
      setConfirmDialog(null)
      setStorageOverview(null)
      if (reason) flash(reason)
    },
    [flash]
  )

  // 工程师会话 30 分钟无操作自动锁屏回退作业员屏
  useEffect(() => {
    if (role !== "engineer") return

    const initialStored = readStoredSession()
    lastActiveRef.current = initialStored?.lastActiveAt ?? Date.now()
    idleLimitRef.current = initialStored?.idleTimeoutMs ?? IDLE_LOCK_MS

    let lastTouchWrite = 0
    const markActivity = () => {
      const now = Date.now()
      lastActiveRef.current = now
      if (now - lastTouchWrite >= 15_000) {
        lastTouchWrite = now
        touchStoredSession(now)
      }
    }

    const activityEvents: (keyof WindowEventMap)[] = [
      "pointerdown",
      "keydown",
      "mousemove",
      "scroll",
      "touchstart",
    ]
    activityEvents.forEach((ev) => window.addEventListener(ev, markActivity, { passive: true }))

    const checkInterval = window.setInterval(() => {
      const stored = readStoredSession()
      const now = Date.now()
      if (!stored || now - lastActiveRef.current >= idleLimitRef.current || now >= stored.expiresAt) {
        lockEngineerSession("工程师会话已超时自动锁定，已返回作业员视图")
      }
    }, 15_000)

    return () => {
      activityEvents.forEach((ev) => window.removeEventListener(ev, markActivity))
      window.clearInterval(checkInterval)
    }
  }, [role, lockEngineerSession])

  const openEvents = useCallback(
    async (id: string) => {
      setSelectedCycleId(id)
      try {
        setEvents(await api<Data[]>(`/api/cycles/${encodeURIComponent(id)}/events?limit=100`))
      } catch (e) {
        flash(String(e))
      }
    },
    [flash]
  )

  const roleRef = useRef<Role>(role)
  const selectedCycleRef = useRef<string>(selectedCycleId)
  useEffect(() => {
    roleRef.current = role
  }, [role])
  useEffect(() => {
    selectedCycleRef.current = selectedCycleId
  }, [selectedCycleId])

  const refresh = useCallback(async (forceEngineerFetch = false) => {
    setRefreshing(true)
    const isEngineer = forceEngineerFetch || roleRef.current === "engineer"
    try {
      if (isEngineer) {
        const [s, c, e, cfg, h, m, ds, tr] = await Promise.all([
          api<Data>("/api/state"),
          api<Data[]>("/api/cycles?limit=30"),
          api<Data[]>("/api/evidence?limit=40"),
          api<Data>("/api/config"),
          api<Data[]>("/api/config/history?limit=20"),
          api<Data[]>("/api/models?limit=30"),
          api<Data>("/api/dataset/summary"),
          api<Data>("/api/training/status"),
        ])
        setState(s)
        setCycles(c)
        setEvidence(e)
        setConfig(cfg)
        setHistory(h)
        setModels(m)
        setDatasetSummary(ds)
        setTrainingStatus(tr)
        setConnected(true)
        setError("")
        const targetCycleId = selectedCycleRef.current || s?.cycle_id || c?.[0]?.cycle_id
        if (targetCycleId) {
          try {
            setEvents(await api<Data[]>(`/api/cycles/${encodeURIComponent(targetCycleId)}/events?limit=100`))
          } catch {
            /* 忽略周期事件查询异常 */
          }
        } else {
          setEvents([])
        }
      } else {
        const [s, c, e] = await Promise.all([
          api<Data>("/api/state"),
          api<Data[]>("/api/cycles?limit=30"),
          api<Data[]>("/api/evidence?limit=40"),
        ])
        setState(s)
        setCycles(c)
        setEvidence(e)
        setConnected(true)
        setError("")
      }
    } catch (e) {
      setConnected(false)
      setError(e instanceof Error ? e.message : "无法连接本地工作站服务")
      setState({
        camera_status: "FAULT",
        view_valid: false,
        label_status: "unknown",
        fsm_state: "FAULT",
        outcome: "FAULT",
        c_timer_ms: 0,
        mirror_horizontal: true,
        view_valid_reasons: ["API_UNAVAILABLE"],
        result_reason: "API_UNAVAILABLE_FAIL_CLOSED",
      })
    } finally {
      if (refreshTimerRef.current !== null) {
        window.clearTimeout(refreshTimerRef.current)
      }
      refreshTimerRef.current = window.setTimeout(() => {
        setRefreshing(false)
        refreshTimerRef.current = null
      }, 220)
    }
  }, [])

  const loadStorageOverview = useCallback(async () => {
    if (roleRef.current !== "engineer") return
    setStorageLoading(true)
    try {
      const ov = await api<Data>("/api/storage/overview")
      setStorageOverview(ov)
    } catch {
      /* 忽略离线时的存储统计请求异常 */
    } finally {
      setStorageLoading(false)
    }
  }, [])

  const handleAuthenticated = useCallback(
    ({ adminToken: verifiedToken, expiresInSeconds }: EngineerAuthResult) => {
      const stored = saveStoredSession(verifiedToken, expiresInSeconds)
      lastActiveRef.current = stored.lastActiveAt
      idleLimitRef.current = stored.idleTimeoutMs
      setAdminToken(verifiedToken)
      setRole("engineer")
      setStep("monitor")
      setAuthModalOpen(false)
      flash("工程师权限已解锁（30 分钟无操作将自动锁屏）")
      void refresh(true)
    },
    [flash, refresh]
  )

  const captureLiveFrame = useCallback(async () => {
    try {
      const res = await api<Data>("/api/camera/capture", { method: "POST" })
      flash(`已抓拍当前画面并归档 (${res.cycle_id})`)
      await refresh()
    } catch {
      await refresh()
    }
  }, [flash, refresh])

  const toggleAuxRois = useCallback(
    async (show: boolean) => {
      try {
        await api<Data>("/api/camera/auxiliary_rois", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ show }),
        })
        flash(show ? "已在实时画面叠加辅助参考框" : "已切换为纯动态目标跟随模式")
        await refresh(true)
      } catch (err) {
        flash(String(err))
      }
    },
    [flash, refresh]
  )

  const toggleMirrorHorizontal = useCallback(
    async (mirror: boolean) => {
      setTogglingMirror(true)
      try {
        const res = await api<Data>("/api/camera/mirror", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mirror }),
        })
        const nextMirror = res.mirror_horizontal !== false
        setState((prev) => ({ ...prev, mirror_horizontal: nextMirror }))
        flash(nextMirror ? "水平镜像对齐已开启（左右手与现实同步）" : "水平镜像已关闭（原始相机视角）")
        await refresh(true)
      } catch (err) {
        flash(`镜像切换失败：${err instanceof Error ? err.message : err}`)
      } finally {
        setTogglingMirror(false)
      }
    },
    [flash, refresh]
  )

  const switchWorkMode = useCallback(
    async (mode: "debug" | "strict") => {
      try {
        await api<Data>("/api/camera/work_mode", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ work_mode: mode }),
        })
        flash(
          mode === "debug"
            ? "已切至【桌面调试模式】：免手套防误锁，适合桌面调参"
            : "已切至【车间严格模式】：强制白色防静电手套检测与料管入位门控"
        )
        await refresh(true)
      } catch (err) {
        flash(`模式切换失败：${err instanceof Error ? err.message : err}`)
      }
    },
    [flash, refresh]
  )

  const simulateScenario = useCallback(
    async (scenario: string) => {
      setSimulating(scenario)
      try {
        await api<Data>(`/api/runtime/simulate/${encodeURIComponent(scenario)}`, {
          method: "POST",
        })
        await refresh()
      } catch (e) {
        flash(`操作失败：${e instanceof Error ? e.message : e}`)
      } finally {
        setSimulating("")
      }
    },
    [flash, refresh]
  )

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (authModalOpen || confirmDialog) return
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLSelectElement
      ) {
        return
      }
      if (e.code === "Space") {
        e.preventDefault()
        void captureLiveFrame()
      } else if (e.key.toLowerCase() === "t") {
        setTheme((cur) => (cur === "light" ? "dark" : "light"))
      } else if (e.key === "Escape") {
        setActiveSnapshot(null)
      }
    }
    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [authModalOpen, confirmDialog, captureLiveFrame])

  useEffect(() => {
    const first = window.setTimeout(() => void refresh(), 0)
    const id = window.setInterval(() => void refresh(), 750)
    return () => {
      window.clearTimeout(first)
      window.clearInterval(id)
    }
  }, [refresh])

  useEffect(() => {
    if (role !== "engineer" || step !== "records_settings" || step4Tab !== "storage") return
    const t = window.setTimeout(() => void loadStorageOverview(), 0)
    return () => window.clearTimeout(t)
  }, [role, step, step4Tab, loadStorageOverview])

  const filmstripItems: FilmstripItem[] = cycles.slice(0, 20).map((c) => {
    const matched = evidence.find((e) => e.cycle_id === c.cycle_id)
    return {
      id: c.cycle_id,
      cycleId: c.cycle_id,
      outcome: c.final_outcome || "WAITING",
      url: matched?.url,
      timestamp: c.updated_at_utc || c.created_at_utc || "",
      kind: matched?.kind,
      pipeColor: c.pipe_color,
      labelStatus: c.initial_label,
    }
  })

  const passCount = cycles.filter((c) => c.final_outcome === "PASS").length
  const alarmCount = cycles.filter((c) => c.final_outcome === "ALARM" || c.final_outcome === "FAULT").length
  const totalCount = cycles.length

  const durations = cycles
    .filter((c) => c.final_outcome && c.created_at_utc && c.updated_at_utc)
    .map((c) => (Date.parse(c.updated_at_utc) - Date.parse(c.created_at_utc)) / 1000)
    .filter((d) => Number.isFinite(d) && d > 0 && d < 120)
  const avgTakt = durations.length ? durations.reduce((a, b) => a + b, 0) / durations.length : null

  const mirrorHorizontal = state.mirror_horizontal !== false
  const workMode = state.work_mode === "strict" ? "strict" : "debug"
  const isTraining = trainingStatus.state === "TRAINING" || trainingStatus.state === "EXPORTING_ONNX"

  if (showShowcase) {
    return (
      <div className="showcase-viewport">
        <ShowcasePage
          onEnterStation={() => {
            setShowShowcase(false)
            window.history.replaceState(null, "", "/")
          }}
        />
      </div>
    )
  }

  return (
    <div className={`shell ${role === "operator" ? "operator-shell" : "studio-shell"}`}>
      {role === "operator" ? (
        <header className="topbar operator-topbar">
          <div className="op-brand">
            <span className="rail-brand-mark" aria-hidden="true" />
            <h1>晶界智检 · 一号目检工位</h1>
          </div>
          <div className="spacer" />
          <div className="meta">
            <span className={`lamp ${connected ? "ok" : "bad"}`} aria-hidden="true" />
            {connected ? "在线" : "离线"}
          </div>
          <span className="divider" aria-hidden="true" />
          <span className="clock">{currentTime.toLocaleTimeString("zh-CN", { hour12: false })}</span>
          <button
            type="button"
            className="icon-btn"
            aria-label={theme === "light" ? "切换深色模式" : "切换浅色模式"}
            title={theme === "light" ? "切换深色模式" : "切换浅色模式"}
            onClick={() => setTheme((cur) => (cur === "light" ? "dark" : "light"))}
          >
            {theme === "light" ? <Moon size={15} aria-hidden="true" /> : <Sun size={15} aria-hidden="true" />}
          </button>
          <button
            ref={loginTriggerRef}
            type="button"
            className="btn sm"
            onClick={() => setAuthModalOpen(true)}
          >
            <Lock size={14} aria-hidden="true" />
            工程师登录
          </button>
        </header>
      ) : (
        <header className="topbar studio-topbar">
          <div className="op-brand studio-brand">
            <span className="rail-brand-mark" aria-hidden="true" />
            <h1>晶界智检</h1>
          </div>

          <span className="divider" aria-hidden="true" />

          <nav className="studio-stepper" aria-label="工程师四步工作流">
            {engineerSteps.map((item, idx) => {
              const Icon = item.icon
              const active = step === item.id
              const badge =
                item.id === "collect"
                  ? `${datasetSummary.total_samples || 0}`
                  : item.id === "train"
                    ? isTraining
                      ? `${trainingStatus.progress_pct || 0}%`
                      : `${models.length}`
                    : item.id === "records_settings" && cycles.length > 0
                      ? `${cycles.length}`
                      : null

              return (
                <Fragment key={item.id}>
                  {idx > 0 && <ChevronRight size={13} className="studio-step-sep" aria-hidden="true" />}
                  <button
                    type="button"
                    className={`studio-step-btn ${active ? "on" : ""}`}
                    aria-current={active ? "step" : undefined}
                    onClick={() => setStep(item.id)}
                  >
                    <span className="studio-step-num mono">{item.num}</span>
                    <Icon size={13} aria-hidden="true" />
                    <span>{item.label}</span>
                    {badge !== null && <span className="studio-step-badge mono">{badge}</span>}
                  </button>
                </Fragment>
              )
            })}
          </nav>

          <div className="spacer" />

          <div className="meta">
            <span className={`lamp ${connected ? "ok" : "bad"}`} aria-hidden="true" />
            <span>{connected ? "在线" : "离线"}</span>
          </div>
          <span className="divider" aria-hidden="true" />
          <span className="clock">{currentTime.toLocaleTimeString("zh-CN", { hour12: false })}</span>
          <button
            type="button"
            className="btn ghost sm"
            title="打开产品展示官网"
            onClick={() => {
              setShowShowcase(true)
              window.history.replaceState(null, "", "/?showcase=1")
            }}
          >
            <Globe size={13} aria-hidden="true" />
            官网
          </button>
          <button
            type="button"
            className="icon-btn"
            aria-label={theme === "light" ? "切换深色模式" : "切换浅色模式"}
            title={theme === "light" ? "深色模式（快捷键 T）" : "浅色模式（快捷键 T）"}
            onClick={() => setTheme((cur) => (cur === "light" ? "dark" : "light"))}
          >
            {theme === "light" ? <Moon size={15} aria-hidden="true" /> : <Sun size={15} aria-hidden="true" />}
          </button>
          <button
            type="button"
            className={`icon-btn ${refreshing ? "busy" : ""}`}
            aria-label="刷新数据"
            title="刷新数据"
            onClick={() => {
              void refresh(true)
              if (step === "records_settings" && step4Tab === "storage") {
                void loadStorageOverview()
              }
            }}
          >
            <RefreshCw size={15} aria-hidden="true" />
          </button>
          <button
            type="button"
            className="btn sm"
            onClick={() => lockEngineerSession("已锁定退出，返回作业员屏")}
          >
            <LogOut size={13} aria-hidden="true" />
            锁定退出
          </button>
        </header>
      )}

      {error && (
        <div className="banner studio-banner" role="alert">
          <AlertTriangle size={15} aria-hidden="true" />
          <span>本机工位服务离线，系统保持闭锁。</span>
          <code>{error}</code>
        </div>
      )}

      {role === "operator" ? (
        <main className="content content-op">
          <OperatorView
            state={state}
            evidence={evidence}
            totalCount={totalCount}
            passCount={passCount}
            alarmCount={alarmCount}
            avgTaktTime={avgTakt}
            onManualTrigger={() => void captureLiveFrame()}
            onResetAlarm={() => void simulateScenario("reset")}
          />
        </main>
      ) : (
        <>
          <main className="studio-workspace">
            {(step === "monitor" || step === "collect") && (
              <LiveStudioWorkspace
                mode={step}
                state={state}
                config={config}
                evidence={evidence}
                cycles={cycles}
                datasetSummary={datasetSummary}
                token={adminToken}
                simulating={simulating}
                mirrorHorizontal={mirrorHorizontal}
                togglingMirror={togglingMirror}
                onSimulate={(scenario) => void simulateScenario(scenario)}
                onSwitchWorkMode={(mode) => void switchWorkMode(mode)}
                onToggleAuxRois={(show) => void toggleAuxRois(show)}
                onToggleMirror={(mirror) => void toggleMirrorHorizontal(mirror)}
                onManualCapture={() => void captureLiveFrame()}
                onRefresh={() => refresh(true)}
                onStorageChanged={loadStorageOverview}
                onConfirm={setConfirmDialog}
                flash={flash}
                onGoStep={(nextStep, subTab) => {
                  setStep(nextStep)
                  if (subTab) setStep4Tab(subTab)
                }}
              />
            )}

            {step === "train" && (
              <TrainDeployStep
                state={state}
                models={models}
                datasetSummary={datasetSummary}
                trainingStatus={trainingStatus}
                token={adminToken}
                flash={flash}
                onSaved={() => refresh(true)}
                onGoCollect={() => setStep("collect")}
              />
            )}

            {step === "records_settings" && (
              <RecordsSettingsStep
                activeSubTab={step4Tab}
                onChangeSubTab={setStep4Tab}
                state={state}
                config={config}
                history={history}
                models={models}
                datasetSummary={datasetSummary}
                storageOverview={storageOverview}
                storageLoading={storageLoading}
                cycles={cycles}
                evidence={evidence}
                events={events}
                selectedCycleId={selectedCycleId || events[0]?.cycle_id || ""}
                filmstripItems={filmstripItems}
                activeSnapshot={activeSnapshot}
                passCount={passCount}
                alarmCount={alarmCount}
                totalCount={totalCount}
                token={adminToken}
                mirrorHorizontal={mirrorHorizontal}
                togglingMirror={togglingMirror}
                flash={flash}
                onConfirm={setConfirmDialog}
                onSelectCycle={(id) => void openEvents(id)}
                onClearSelectedCycle={() => {
                  setSelectedCycleId("")
                  setEvents([])
                }}
                onOpenSnapshot={(item) => {
                  setActiveSnapshot(item)
                  void openEvents(item.cycleId)
                }}
                onManualCapture={() => void captureLiveFrame()}
                onSaved={async () => {
                  await refresh(true)
                  await loadStorageOverview()
                }}
                onRefreshStorage={() => void loadStorageOverview()}
                onToggleAuxRois={(show) => void toggleAuxRois(show)}
                onToggleMirror={(mirror) => void toggleMirrorHorizontal(mirror)}
                onSwitchWorkMode={(mode) => void switchWorkMode(mode)}
                onGoStep={(nextStep, subTab) => {
                  setStep(nextStep)
                  if (subTab) setStep4Tab(subTab)
                }}
              />
            )}
          </main>

          <footer className="studio-statusbar" aria-label="工位运行遥测状态栏">
            <span className="mono">
              {state.camera_width && state.camera_height ? `${state.camera_width}×${state.camera_height}` : "1920×1080"}
            </span>
            <span className="studio-sb-dot" aria-hidden="true" />
            <span className="mono">
              {fmt(state.fps, 1)} FPS{state.inference_ms ? ` · ${state.inference_ms} ms` : ""}
            </span>
            <span className="studio-sb-dot" aria-hidden="true" />
            <span>
              水平镜像：
              <b className={mirrorHorizontal ? "sb-ok" : "sb-muted"}>
                {mirrorHorizontal ? "已开启（左右同步）" : "已关闭（原始画面）"}
              </b>
            </span>
            <span className="studio-sb-dot" aria-hidden="true" />
            <span>运行模式：{workMode === "strict" ? "车间严格模式" : "桌面调试模式"}</span>
            <span className="studio-sb-dot" aria-hidden="true" />
            <span className="studio-sb-engine mono">
              推理引擎：{state.inference_backend || state.model_version || "动态视觉引擎"}
            </span>
            <span className="sep" />
            <span className="studio-sb-keys">
              <kbd>空格</kbd> 抓拍
              <kbd>T</kbd> 明暗
            </span>
          </footer>
        </>
      )}

      <EngineerAuthModal
        open={authModalOpen}
        onClose={() => setAuthModalOpen(false)}
        onAuthenticated={handleAuthenticated}
        triggerRef={loginTriggerRef}
      />

      {role === "engineer" && activeSnapshot && (
        <SnapshotModal item={activeSnapshot} onClose={() => setActiveSnapshot(null)} />
      )}

      {role === "engineer" && confirmDialog && (
        <ConfirmModal dialog={confirmDialog} onClose={() => setConfirmDialog(null)} />
      )}

      {notice && (
        <div className="toast" role="status">
          <span>{notice}</span>
          <button type="button" onClick={() => setNotice("")} aria-label="关闭通知">
            <X size={14} aria-hidden="true" />
          </button>
        </div>
      )}
    </div>
  )
}

function ConfirmModal({ dialog, onClose }: { dialog: ConfirmDialogState; onClose: () => void }) {
  const [acknowledged, setAcknowledged] = useState(false)
  const [busy, setBusy] = useState(false)
  const cancelBtnRef = useRef<HTMLButtonElement | null>(null)

  useEffect(() => {
    const prev = document.activeElement instanceof HTMLElement ? document.activeElement : null
    cancelBtnRef.current?.focus()
    return () => {
      if (prev && document.contains(prev)) prev.focus()
    }
  }, [])

  const canConfirm = !busy && (!dialog.requireCheckLabel || acknowledged)

  const handleConfirm = async () => {
    if (!canConfirm) return
    setBusy(true)
    try {
      await dialog.onConfirm()
      onClose()
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="modal-scrim" onClick={() => (!busy ? onClose() : undefined)}>
      <div
        className="modal confirm-modal"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-dialog-title"
        aria-describedby="confirm-dialog-desc"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={(e) => {
          if (e.key === "Escape" && !busy) onClose()
        }}
      >
        <header className="panel-head">
          <h2 id="confirm-dialog-title" className="flex items-center gap-1.5">
            <ShieldAlert
              size={15}
              aria-hidden="true"
              style={{ color: dialog.tone === "danger" ? "var(--bad)" : "var(--warn)" }}
            />
            {dialog.title}
          </h2>
          <div className="right">
            <button
              type="button"
              className="icon-btn"
              disabled={busy}
              onClick={onClose}
              aria-label="取消并关闭"
            >
              <X size={15} aria-hidden="true" />
            </button>
          </div>
        </header>
        <div className="panel-body grid gap-3">
          <div id="confirm-dialog-desc" className={`confirm-consequence ${dialog.tone === "danger" ? "danger" : ""}`}>
            <div>{dialog.description}</div>
            {dialog.consequences && dialog.consequences.length > 0 && (
              <ul>
                {dialog.consequences.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            )}
          </div>

          {dialog.requireCheckLabel && (
            <label className="check" style={{ color: "var(--ink)", fontWeight: 500 }}>
              <input
                type="checkbox"
                checked={acknowledged}
                disabled={busy}
                onChange={(e) => setAcknowledged(e.target.checked)}
              />
              <span>{dialog.requireCheckLabel}</span>
            </label>
          )}

          <div className="auth-actions">
            <button
              ref={cancelBtnRef}
              type="button"
              className="btn ghost sm"
              disabled={busy}
              onClick={onClose}
            >
              取消
            </button>
            <button
              type="button"
              className={`btn sm ${dialog.tone === "danger" ? "btn-danger" : ""}`}
              disabled={!canConfirm}
              onClick={() => void handleConfirm()}
            >
              {busy ? "执行中…" : dialog.confirmLabel}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

function SnapshotModal({ item, onClose }: { item: FilmstripItem; onClose: () => void }) {
  const v = verdictOf(item.outcome)
  const closeBtnRef = useRef<HTMLButtonElement | null>(null)

  useEffect(() => {
    const prev = document.activeElement instanceof HTMLElement ? document.activeElement : null
    closeBtnRef.current?.focus()
    return () => {
      if (prev && document.contains(prev)) prev.focus()
    }
  }, [])

  return (
    <div className="modal-scrim" onClick={onClose}>
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="snapshot-title"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={(e) => {
          if (e.key === "Escape") onClose()
        }}
      >
        <header className="panel-head">
          <h2 id="snapshot-title">检验快照</h2>
          <span className="sub mono">{item.cycleId}</span>
          <div className="right">
            <button ref={closeBtnRef} type="button" className="icon-btn" onClick={onClose} aria-label="关闭">
              <X size={15} aria-hidden="true" />
            </button>
          </div>
        </header>
        <div className="panel-body grid gap-4">
          <CameraFrame src={item.url} alt="检验周期证据快照" emptyTitle="该周期无图像" />
          <dl className="dl">
            <dt>判定</dt>
            <dd>
              <span className={`tag ${v.tone}`}>{v.word}</span>
            </dd>
            <dt>时间</dt>
            <dd className="mono">{dateTimeOf(item.timestamp)}</dd>
            <dt>料管</dt>
            <dd>{pipeText(item.pipeColor)}</dd>
            <dt>初始标签</dt>
            <dd>{labelText[item.labelStatus ?? ""] || "—"}</dd>
          </dl>
        </div>
      </div>
    </div>
  )
}

function OutcomeTag({ value }: { value: string }) {
  const v = verdictOf(value)
  return <span className={`tag ${v.tone}`}>{v.word}</span>
}

/* ==========================================================================
   步骤 1（实时监控）& 步骤 2（采集标注）共享左侧固定 16:9 镜头 + 右侧检查器
   · 步骤 1：右侧顶部永久固定置顶「4× 动态跟随特写」，仅下方诊断区局部滚动
   · 步骤 2：清晰区分「方式 A 镜头抓拍 / 方式 B 本地选图上传」+ 完整样本筛选/多选批量删/改标注
   ========================================================================== */
function LiveStudioWorkspace({
  mode,
  state,
  config,
  evidence,
  cycles,
  datasetSummary,
  token,
  simulating,
  mirrorHorizontal,
  togglingMirror,
  onSimulate,
  onSwitchWorkMode,
  onToggleAuxRois,
  onToggleMirror,
  onManualCapture,
  onRefresh,
  onStorageChanged,
  onConfirm,
  flash,
  onGoStep,
}: {
  mode: "monitor" | "collect"
  state: Data
  config: Data
  evidence: Data[]
  cycles: Data[]
  datasetSummary: Data
  token: string
  simulating: string
  mirrorHorizontal: boolean
  togglingMirror: boolean
  onSimulate: (scenario: string) => void
  onSwitchWorkMode: (mode: "debug" | "strict") => void
  onToggleAuxRois: (show: boolean) => void
  onToggleMirror: (mirror: boolean) => void
  onManualCapture: () => void
  onRefresh: () => Promise<void>
  onStorageChanged: () => Promise<void>
  onConfirm: (dialog: ConfirmDialogState) => void
  flash: (s: string) => void
  onGoStep: (step: EngineerStep, subTab?: RecordsSettingsTab) => void
}) {
  // 步骤 2 采集与拉框状态
  const [pipeColor, setPipeColor] = useState<"yellow" | "blue">("yellow")
  const [tubeGroup, setTubeGroup] = useState("TUBE_LIVE_01")
  const [batchGroup, setBatchGroup] = useState("BATCH_2026_LIVE")
  const [capturingClass, setCapturingClass] = useState("")
  const [bootstrapping, setBootstrapping] = useState(false)
  const [clearing, setClearing] = useState(false)
  const [manualBox, setManualBox] = useState<[number, number, number, number] | null>(null)
  const [dragStart, setDragStart] = useState<[number, number] | null>(null)

  // 方式 B：本地选图上传加入训练集（严格释放 URL.createObjectURL）
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [uploadPreviewUrl, setUploadPreviewUrl] = useState("")
  const [uploadLabel, setUploadLabel] = useState<SampleLabel>("complete")
  const [uploadingSample, setUploadingSample] = useState(false)
  const uploadInputRef = useRef<HTMLInputElement | null>(null)
  const previewUrlRef = useRef("")

  // 样本库筛选、加载与多选批量删除状态
  const [sampleFilter, setSampleFilter] = useState<SampleFilter>("")
  const [samples, setSamples] = useState<Data[]>([])
  const [samplesLoading, setSamplesLoading] = useState(false)
  const [selectedSampleIds, setSelectedSampleIds] = useState<string[]>([])
  const [sampleReloadTick, setSampleReloadTick] = useState(0)
  const [bulkBusy, setBulkBusy] = useState(false)

  // 离线补录业务证据帧表单状态
  const [file, setFile] = useState<File | null>(null)
  const [cycle, setCycle] = useState("")
  const [kind, setKind] = useState("initial_label")
  const [label, setLabel] = useState("complete")
  const [submitting, setSubmitting] = useState(false)

  const recentSamplesRef = useRef<Data[]>([])
  useEffect(() => {
    recentSamplesRef.current = Array.isArray(datasetSummary.recent_samples) ? datasetSummary.recent_samples : []
  }, [datasetSummary.recent_samples])

  const selectUploadFile = useCallback((nextFile: File | null) => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current)
      previewUrlRef.current = ""
    }
    setUploadFile(nextFile)
    if (nextFile) {
      const nextUrl = URL.createObjectURL(nextFile)
      previewUrlRef.current = nextUrl
      setUploadPreviewUrl(nextUrl)
    } else {
      setUploadPreviewUrl("")
    }
  }, [])

  useEffect(() => {
    return () => {
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current)
        previewUrlRef.current = ""
      }
    }
  }, [])

  const reloadSamples = useCallback(() => {
    setSampleReloadTick((n) => n + 1)
  }, [])

  useEffect(() => {
    if (mode !== "collect") return
    let cancelled = false
    const params = new URLSearchParams({ limit: "200" })
    if (sampleFilter) params.set("label", sampleFilter)
    const timer = window.setTimeout(async () => {
      if (cancelled) return
      setSamplesLoading(true)
      try {
        const res = await api<Data>(`/api/dataset/samples?${params.toString()}`)
        if (cancelled) return
        const list: Data[] = Array.isArray(res)
          ? res
          : Array.isArray(res?.samples)
            ? res.samples
            : Array.isArray(res?.items)
              ? res.items
              : []
        setSamples(list)
        const validSet = new Set(list.map((s) => String(s.sample_id)))
        setSelectedSampleIds((prev) => prev.filter((id) => validSet.has(id)))
      } catch {
        if (!cancelled) {
          const fallback = recentSamplesRef.current
          setSamples(sampleFilter ? fallback.filter((s) => s.label === sampleFilter) : fallback)
        }
      } finally {
        if (!cancelled) setSamplesLoading(false)
      }
    }, 0)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [mode, sampleFilter, sampleReloadTick, datasetSummary.total_samples])

  const valid = state.view_valid === true
  const online = state.camera_status === "ONLINE"
  const timer = Math.max(0, Number(state.c_timer_ms || 0))
  const litSegments = Math.min(12, Math.floor((timer / C_LIMIT_MS) * 12))
  const outcome = deriveOutcome(state)
  const steps = ["VERIFY_INITIAL", "READY", "INSPECTING", "PASS"]
  const stepNames = ["初始校验", "就绪", "动态检验中", "放行"]
  const currentIdx = steps.indexOf(state.fsm_state)
  const showAux = Boolean(state.show_auxiliary_rois)
  const trackedList: Data[] = Array.isArray(state.tracked_objects) ? state.tracked_objects : []
  const isAlarmLatched = Boolean(state.alarm_latched || state.outcome === "ALARM" || state.bare_hand_violation)
  const workMode = state.work_mode === "strict" ? "strict" : "debug"
  const counts = datasetSummary.class_counts || {}
  const liveSrc = (online && state.live_stream_url) || state.latest_evidence_url || evidence[0]?.url
  const cropSrc = (online && state.live_crop_url) || state.latest_evidence_url

  const visibleSampleIds = samples.map((s) => String(s.sample_id))
  const selectedIdSet = new Set(selectedSampleIds)
  const allVisibleSelected = visibleSampleIds.length > 0 && visibleSampleIds.every((id) => selectedIdSet.has(id))
  const activeFilterLabel = sampleFilterOptions.find((o) => o.id === sampleFilter)?.label || "全部"

  const gripText = (v: unknown) => (v === true ? "已动态锁定" : v === false ? "未入画" : "无信号")

  const normPoint = (e: React.PointerEvent<HTMLDivElement>): [number, number] => {
    const r = e.currentTarget.getBoundingClientRect()
    return [
      Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)),
      Math.max(0, Math.min(1, (e.clientY - r.top) / r.height)),
    ]
  }

  const toggleSampleSelection = (sampleId: string) => {
    setSelectedSampleIds((prev) =>
      prev.includes(sampleId) ? prev.filter((id) => id !== sampleId) : [...prev, sampleId]
    )
  }

  const toggleSelectAllVisible = () => {
    if (allVisibleSelected) {
      setSelectedSampleIds([])
    } else {
      setSelectedSampleIds(visibleSampleIds)
    }
  }

  const handleQuickCapture = async (targetLabel: SampleLabel, title: string) => {
    setCapturingClass(targetLabel)
    try {
      const res = await api<Data>(
        "/api/dataset/capture",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            label: targetLabel,
            pipe_color: pipeColor,
            physical_tube_id: tubeGroup.trim() || undefined,
            batch_id: batchGroup.trim() || undefined,
            box: manualBox || undefined,
          }),
        },
        token
      )
      flash(`已采集【${title}】样本 (${res.sample_id} · ${manualBox ? "手动拉框裁切+COCO标注" : "自动追踪裁切"})`)
      await onRefresh()
      reloadSamples()
      void onStorageChanged()
    } catch (err) {
      flash(`采集失败：${err instanceof Error ? err.message : err}`)
    } finally {
      setCapturingClass("")
    }
  }

  const handleLocalSampleUpload = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!uploadFile) return flash("请先选择要加入训练集的本地 JPEG/PNG 图片")
    setUploadingSample(true)
    try {
      const res = await api<Data>(
        "/api/dataset/upload",
        {
          method: "POST",
          headers: {
            "Content-Type": uploadFile.type || "image/jpeg",
            "X-Sample-Label": uploadLabel,
            "X-Pipe-Color": pipeColor,
            "X-Tube-Id": toHeaderValue(tubeGroup, "TUBE_UPLOAD_01"),
            "X-Batch-Id": toHeaderValue(batchGroup, "BATCH_UPLOAD"),
          },
          body: uploadFile,
        },
        token
      )
      const labelName = sampleClassDefs.find((c) => c.id === uploadLabel)?.short || uploadLabel
      flash(`已将本地图片加入【${labelName}】训练集 (${res.sample_id || "完成"})`)
      selectUploadFile(null)
      if (uploadInputRef.current) uploadInputRef.current.value = ""
      await onRefresh()
      reloadSamples()
      void onStorageChanged()
    } catch (err) {
      flash(`上传训练样本失败：${err instanceof Error ? err.message : err}`)
    } finally {
      setUploadingSample(false)
    }
  }

  const requestClearAllSamples = () => {
    const total = Number(datasetSummary.total_samples || 0)
    if (total === 0) return flash("当前训练集已是空的")
    onConfirm({
      title: "清空全部 AI 训练样本",
      description: `确认清空 data/dataset/ 中的全部 ${total} 张训练切片与标注索引吗？`,
      consequences: [
        "删除全部 5 类训练图片与 JSONL/COCO 标注清单（不可撤销）",
        "不影响 SQLite 检测周期业务记录及当前运行中的 ONNX 模型",
      ],
      confirmLabel: `确认清空 ${total} 张样本`,
      tone: "danger",
      onConfirm: async () => {
        setClearing(true)
        try {
          await api<Data>("/api/dataset/clear", { method: "POST" }, token)
          setSelectedSampleIds([])
          flash("已清空全部训练样本，可直接重新采集或上传车间真实数据")
          await onRefresh()
          reloadSamples()
          void onStorageChanged()
        } catch (err) {
          flash(`清空失败：${err instanceof Error ? err.message : err}`)
        } finally {
          setClearing(false)
        }
      },
    })
  }

  const requestBatchDeleteSelected = () => {
    const ids = [...selectedSampleIds]
    if (ids.length === 0) return
    onConfirm({
      title: "批量删除选中训练样本",
      description: `确认从训练集与磁盘中删除选中的 ${ids.length} 张样本吗？`,
      consequences: ["对应切片图片将被永久移除，标注清单与 COCO 索引将自动重写"],
      confirmLabel: `确认删除 ${ids.length} 张`,
      tone: "danger",
      onConfirm: async () => {
        setBulkBusy(true)
        try {
          const res = await api<Data>(
            "/api/dataset/batch_delete",
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ sample_ids: ids }),
            },
            token
          )
          setSelectedSampleIds([])
          flash(`已批量删除 ${res.deleted_count ?? ids.length} 张训练样本`)
          await onRefresh()
          reloadSamples()
          void onStorageChanged()
        } catch (err) {
          flash(`批量删除失败：${err instanceof Error ? err.message : err}`)
        } finally {
          setBulkBusy(false)
        }
      },
    })
  }

  const requestClearCurrentClass = () => {
    if (!sampleFilter) {
      requestClearAllSamples()
      return
    }
    const classCount = Number(counts[sampleFilter] || samples.length || 0)
    if (classCount === 0) return flash(`【${activeFilterLabel}】类别当前暂无样本`)
    onConfirm({
      title: `清空【${activeFilterLabel}】类别样本`,
      description: `确认清空【${activeFilterLabel}】下的全部 ${classCount} 张训练样本吗？`,
      consequences: ["仅删除该类别的切片图片与索引条目，其余 4 个类别样本保持不变"],
      confirmLabel: `确认清空该类别 (${classCount})`,
      tone: "danger",
      onConfirm: async () => {
        setBulkBusy(true)
        try {
          const res = await api<Data>(
            "/api/dataset/batch_delete",
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ label: sampleFilter }),
            },
            token
          )
          setSelectedSampleIds([])
          flash(`已清空【${activeFilterLabel}】类别共 ${res.deleted_count ?? classCount} 张样本`)
          await onRefresh()
          reloadSamples()
          void onStorageChanged()
        } catch (err) {
          flash(`清空类别失败：${err instanceof Error ? err.message : err}`)
        } finally {
          setBulkBusy(false)
        }
      },
    })
  }

  const handleBootstrapDataset = async () => {
    setBootstrapping(true)
    try {
      const res = await api<Data>("/api/dataset/bootstrap", { method: "POST" }, token)
      flash(`已生成 ${res.bootstrapped_count} 张跨 4 个独立料管组的标准训练样本`)
      await onRefresh()
      reloadSamples()
      void onStorageChanged()
    } catch (err) {
      flash(`生成失败：${err instanceof Error ? err.message : err}`)
    } finally {
      setBootstrapping(false)
    }
  }

  const handleChangeSampleLabel = async (sampleId: string, newLabel: string) => {
    try {
      await api(
        `/api/dataset/sample/${encodeURIComponent(sampleId)}`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ label: newLabel }),
        },
        token
      )
      flash(`已将样本 ${sampleId} 标注修改为【${labelText[newLabel] || newLabel}】`)
      await onRefresh()
      reloadSamples()
    } catch (err) {
      flash(`修改标注失败：${err instanceof Error ? err.message : err}`)
    }
  }

  const requestDeleteSingleSample = (smp: Data) => {
    const sampleId = String(smp.sample_id)
    onConfirm({
      title: "删除单张训练样本",
      description: `确认永久删除训练样本 ${sampleId}（${smp.physical_tube_id || "未分组"}）吗？`,
      consequences: ["该切片文件将从磁盘删除，manifest 与 COCO 索引同步更新"],
      confirmLabel: "确认删除",
      tone: "danger",
      onConfirm: async () => {
        try {
          await api(`/api/dataset/sample/${encodeURIComponent(sampleId)}`, { method: "DELETE" }, token)
          setSelectedSampleIds((prev) => prev.filter((id) => id !== sampleId))
          flash(`已删除样本 ${sampleId}`)
          await onRefresh()
          reloadSamples()
          void onStorageChanged()
        } catch (err) {
          flash(`删除失败：${err instanceof Error ? err.message : err}`)
        }
      },
    })
  }

  const submitManualEvidence = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!file) return flash("请选择要归档的图片")
    const targetCycle = cycle.trim() || `MANUAL_${Date.now().toString().slice(-6)}`
    setSubmitting(true)
    try {
      await api(
        `/api/cycles/${encodeURIComponent(targetCycle)}/evidence/${kind}`,
        {
          method: "POST",
          headers: {
            "Content-Type": file.type || "image/jpeg",
            "X-Frame-Id": "1",
            "X-Captured-At-Utc": new Date().toISOString(),
            "X-Pipe-Color": pipeColor,
            "X-Operation-Event": "MANUAL_UPLOAD",
            "X-Operator-Anon-Id": "OP_STATION_01",
            "X-Label-Status": label,
            "X-Camera-Metadata": JSON.stringify({
              camera_device_id: state.camera_device_id || "CAM_01",
              resolution: "1920x1080",
            }),
          },
          body: file,
        },
        token
      )
      flash(`已手动归档至周期 ${targetCycle}`)
      setFile(null)
      await onRefresh()
      void onStorageChanged()
    } catch (err) {
      flash(`归档失败：${err instanceof Error ? err.message : err}`)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="studio-split">
      {/* 左侧永远固定的超大 16:9 实时镜头舞台 */}
      <section className="studio-canvas" aria-label={mode === "monitor" ? "实时监控取景舞台" : "实时采集标注画布"}>
        {mode === "monitor" && (
          <div className="studio-andon-slot">
            <Andon
              compact
              state={state}
              guide={{
                title: state.result_reason
                  ? reasonOf(state.result_reason)
                  : valid
                    ? "动态追踪锁定中 · 等待周期判定"
                    : reasonOf(state.view_valid_reasons?.[0]) || "视野无效",
                desc: "",
              }}
              meta={[
                { label: "周期号", value: state.cycle_id || "—" },
                { label: "更新时间", value: timeOf(state.updated_at_utc) },
              ]}
            />
          </div>
        )}

        <header className="studio-canvas-bar">
          {mode === "monitor" ? (
            <>
              <h2>全景动态目标追踪</h2>
              <span className="sub mono">已锁定 {trackedList.length} 目标</span>
              <div className="right">
                <button
                  type="button"
                  className={`btn ghost sm ${mirrorHorizontal ? "btn-active" : ""}`}
                  disabled={togglingMirror}
                  onClick={() => onToggleMirror(!mirrorHorizontal)}
                  title="切换水平镜像对齐（左右手与现实同步）"
                >
                  <FlipHorizontal2 size={13} aria-hidden="true" />
                  {mirrorHorizontal ? "镜像同步" : "原始方向"}
                </button>
                <button type="button" className="btn ghost sm" onClick={() => onToggleAuxRois(!showAux)}>
                  <Eye size={13} aria-hidden="true" />
                  {showAux ? "隐藏辅助框" : "辅助框"}
                </button>
                <button type="button" className="btn ghost sm" onClick={onManualCapture}>
                  <Camera size={13} aria-hidden="true" />
                  抓拍
                </button>
              </div>
            </>
          ) : (
            <>
              <h2>16:9 实时手动拉框标注</h2>
              <span className="sub mono">
                {manualBox
                  ? `手动框 [${manualBox.map((v) => v.toFixed(2)).join(", ")}]`
                  : "方式 A：拖拽拉框或直接点击右侧拍样本 · 方式 B：右侧选本地图上传"}
              </span>
              <div className="right">
                {manualBox && (
                  <button type="button" className="btn ghost sm" onClick={() => setManualBox(null)}>
                    清除手动框
                  </button>
                )}
                <a className="btn ghost sm" href="/api/dataset/coco" target="_blank" rel="noreferrer">
                  导出 COCO
                </a>
                <button
                  type="button"
                  className="btn ghost sm btn-outline-bad"
                  disabled={clearing}
                  onClick={requestClearAllSamples}
                >
                  {clearing ? "清空中…" : "清空全部样本"}
                </button>
              </div>
            </>
          )}
        </header>

        <div className="studio-canvas-body">
          {mode === "collect" ? (
            <div
              className="canvas-16x9 bbox-stage"
              onPointerDown={(e) => {
                e.currentTarget.setPointerCapture(e.pointerId)
                const p = normPoint(e)
                setDragStart(p)
                setManualBox([p[0], p[1], Math.min(1, p[0] + 0.01), Math.min(1, p[1] + 0.01)])
              }}
              onPointerMove={(e) => {
                if (!dragStart) return
                const p = normPoint(e)
                setManualBox([
                  Math.min(dragStart[0], p[0]),
                  Math.min(dragStart[1], p[1]),
                  Math.max(dragStart[0], p[0]),
                  Math.max(dragStart[1], p[1]),
                ])
              }}
              onPointerUp={(e) => {
                if (!dragStart) return
                const p = normPoint(e)
                const x1 = Math.min(dragStart[0], p[0])
                const y1 = Math.min(dragStart[1], p[1])
                const x2 = Math.max(dragStart[0], p[0])
                const y2 = Math.max(dragStart[1], p[1])
                setDragStart(null)
                if (x2 - x1 < 0.02 || y2 - y1 < 0.02) {
                  setManualBox(null)
                } else {
                  setManualBox([x1, y1, x2, y2])
                }
              }}
            >
              <CameraFrame
                fill
                src={liveSrc}
                alt="16:9 实时采集与手动拉框画面"
                focusLocked={online && valid}
                emptyTitle="等待实时相机"
              />
              {manualBox && (
                <div
                  className="bbox-rect"
                  style={{
                    left: `${manualBox[0] * 100}%`,
                    top: `${manualBox[1] * 100}%`,
                    width: `${(manualBox[2] - manualBox[0]) * 100}%`,
                    height: `${(manualBox[3] - manualBox[1]) * 100}%`,
                  }}
                >
                  <span className="bbox-rect-tag">
                    手动裁切框 ({Math.round((manualBox[2] - manualBox[0]) * (Number(state.camera_width) || 1920))}×
                    {Math.round((manualBox[3] - manualBox[1]) * (Number(state.camera_height) || 1080))})
                  </span>
                </div>
              )}
            </div>
          ) : (
            <div className="canvas-16x9">
              <CameraFrame
                fill
                src={liveSrc}
                alt="全景 16:9 实时动态追踪画面"
                focusLocked={online && valid}
                emptyTitle="等待相机画面"
              />
            </div>
          )}
        </div>

        {mode === "collect" && (
          <div className="studio-class-strip" aria-label="五类样本统计">
            {sampleClassDefs.map((c) => (
              <div className="studio-class-cell" key={c.id}>
                <span>{c.short}</span>
                <b className="mono">{counts[c.id] || 0}</b>
              </div>
            ))}
            <div className="studio-class-cell studio-class-total">
              <span>总计 / 独立组</span>
              <b className="mono">
                {datasetSummary.total_samples || 0} / {datasetSummary.independent_tube_groups || 0}
              </b>
            </div>
          </div>
        )}
      </section>

      {/* 右侧检查器：监控模式将 4× 动态跟随特写永久固定置顶，下方诊断区独立滚动 */}
      <aside className="studio-inspector" aria-label={mode === "monitor" ? "实时监控检查器" : "样本采集与标注检查器"}>
        <header className="studio-inspector-head">
          {mode === "monitor" ? (
            <>
              <h2>实时诊断检查器</h2>
              <span className={`tag ${valid ? "pass" : "warn"}`}>{human(state.fsm_state || "WAIT_PART")}</span>
            </>
          ) : (
            <>
              <h2>采集标注与样本库</h2>
              <button type="button" className="btn sm" onClick={() => onGoStep("train")}>
                <Cpu size={13} aria-hidden="true" />
                前往训练部署
              </button>
            </>
          )}
        </header>

        {/* 永久固定置顶的 4× 动态跟随特写（位于滚动区外部，永不随右侧滚动移出视野） */}
        {mode === "monitor" && (
          <section className="inspector-pinned-crop" aria-label="标签特写动态跟随置顶锁定区">
            <div className="inspector-section-head">
              <div className="flex items-center gap-1.5">
                <h3>标签特写 · 动态跟随</h3>
                <span className="pin-badge">
                  <Pin size={10} aria-hidden="true" />
                  4× 置顶
                </span>
              </div>
              <span className="mono" style={{ fontSize: 12, color: "var(--ink-2)" }}>
                {state.label_confidence != null ? `${(Number(state.label_confidence) * 100).toFixed(1)}%` : "—"}
              </span>
            </div>
            <div className="inspector-crop-frame">
              <CameraFrame
                fill
                src={cropSrc}
                alt="4 倍动态跟随标签特写"
                focusLocked={online && valid}
                emptyTitle="等待锁定标签"
              />
            </div>
            <div className="inspector-verdict-row">
              <div>
                <small>实时标签判定</small>
                <b style={{ color: isAlarmLatched ? "var(--bad)" : undefined }}>
                  {state.bare_hand_violation
                    ? "裸手违规报警"
                    : labelText[state.label_status] || human(state.label_status)}
                </b>
              </div>
              <div style={{ textAlign: "right" }}>
                <small>当前料管</small>
                <b>{pipeText(state.pipe_color)}</b>
              </div>
            </div>
            <div style={{ marginTop: 8 }}>
              <div className="flex items-baseline justify-between" style={{ marginBottom: 5 }}>
                <span style={{ fontSize: 11.5, color: "var(--ink-3)" }}>C 类脱落确认计时</span>
                <span className="mono" style={{ fontSize: 11.5 }}>
                  {(timer / 1000).toFixed(2)} / 1.20 秒
                </span>
              </div>
              <div className={`segments ${litSegments >= 12 ? "done" : ""}`}>
                {Array.from({ length: 12 }, (_, i) => (
                  <i key={i} className={i < litSegments ? "on" : ""} />
                ))}
              </div>
            </div>
            {isAlarmLatched && (
              <button
                type="button"
                className="btn btn-danger sm block"
                style={{ marginTop: 8 }}
                onClick={() => onSimulate("reset")}
                disabled={simulating !== ""}
              >
                <RotateCcw size={13} aria-hidden="true" />
                解除报警闭锁并复位
              </button>
            )}
          </section>
        )}

        <div className="studio-inspector-scroll-wrap">
          <div className="studio-inspector-body">
            {mode === "monitor" ? (
              <>
                {/* 分区 1：状态机与目标坐标 */}
                <details className="inspector-group" style={{ borderTop: 0 }} open>
                  <summary>
                    <span>状态机与目标坐标</span>
                    <span className="mono">{trackedList.length} 目标</span>
                  </summary>
                  <div className="inspector-group-body grid gap-3">
                    <div className="fsm-track">
                      {steps.map((s, i) => (
                        <div key={s} className={`fsm-step ${i === currentIdx ? "now" : i < currentIdx ? "past" : ""}`}>
                          {stepNames[i]}
                        </div>
                      ))}
                    </div>

                    <div className="inspector-mini-grid">
                      <div className="inspector-mini-cell">
                        <small>物理相机</small>
                        <b>{cameraText[state.camera_status] || "未知"}</b>
                      </div>
                      <div className="inspector-mini-cell">
                        <small>手套与接触</small>
                        <b style={{ color: state.bare_hand_violation ? "var(--bad)" : undefined }}>
                          {state.bare_hand_violation
                            ? "裸手违规"
                            : state.bare_hand_detected
                              ? "裸手警告"
                              : state.glove_compliant
                                ? "手套合规"
                                : "待检测"}
                        </b>
                      </div>
                      <div className="inspector-mini-cell">
                        <small>左手动态追踪</small>
                        <b>{gripText(state.left_grip_active)}</b>
                      </div>
                      <div className="inspector-mini-cell">
                        <small>右手动态追踪</small>
                        <b>{gripText(state.right_grip_active)}</b>
                      </div>
                    </div>

                    <dl className="dl">
                      <dt>手部与标签重叠</dt>
                      <dd style={state.hand_contact_candidate ? { color: "var(--warn)", fontWeight: 600 } : undefined}>
                        {state.hand_contact_candidate ? "检测到动态接触" : "无重叠"}
                      </dd>
                      <dt>视野判定结论</dt>
                      <dd>{outcome === "VIEW_INVALID" ? "手部接触或光学受阻" : valid ? "视野有效" : "待定"}</dd>
                    </dl>

                    <div>
                      <div className="inspector-sublabel">实时追踪目标归一化坐标</div>
                      {trackedList.length > 0 ? (
                        <div className="flex flex-wrap gap-1.5">
                          {trackedList.map((obj, idx) => (
                            <span key={idx} className="tag plain mono" style={{ fontSize: 11 }}>
                              {obj.label}: [
                              {Array.isArray(obj.box) ? obj.box.map((v: number) => Number(v).toFixed(2)).join(", ") : ""}
                              ]
                            </span>
                          ))}
                        </div>
                      ) : (
                        <div className="note">当前画面未锁定活动目标框</div>
                      )}
                    </div>
                  </div>
                </details>

                {/* 分区 2：光学质量仪表 */}
                <details className="inspector-group" open>
                  <summary>
                    <span>光学质量仪表</span>
                    <span className={`tag ${valid ? "pass" : "warn"}`} style={{ height: 20, fontSize: 11 }}>
                      {valid ? "达标" : "受阻"}
                    </span>
                  </summary>
                  <div className="inspector-group-body grid gap-2.5">
                    <OpticalGauges
                      laplacianVariance={state.laplacian_variance}
                      minLaplacian={config.thresholds?.min_laplacian_variance}
                      overexposedFraction={state.overexposed_fraction}
                      maxOverexposed={config.thresholds?.max_overexposed_fraction}
                      brightness={state.brightness}
                      minBrightness={config.thresholds?.min_brightness}
                      maxBrightness={config.thresholds?.max_brightness}
                      viewValid={state.view_valid}
                    />
                    {Array.isArray(state.view_valid_reasons) && state.view_valid_reasons.length > 0 && (
                      <div className="flex flex-wrap gap-1.5">
                        {state.view_valid_reasons.map((code: string) => (
                          <span key={code} className="tag warn">
                            {reasonOf(code)}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </details>

                {/* 分区 3：工况演练与快速调试 */}
                <details className="inspector-group" open>
                  <summary>
                    <span>工况演练与快速调试</span>
                    <span className="mono">{workMode === "strict" ? "车间严格" : "桌面调试"}</span>
                  </summary>
                  <div className="inspector-group-body grid gap-3">
                    <div>
                      <div className="inspector-sublabel">影子工况回放注入</div>
                      <div className="grid grid-cols-3 gap-1.5">
                        {simScenarios.map((item) => (
                          <button
                            key={item.id}
                            type="button"
                            disabled={simulating !== ""}
                            className={`btn ghost sm ${item.tone === "bad" ? "btn-outline-bad" : ""}`}
                            style={{ padding: "0 8px", fontSize: 12 }}
                            onClick={() => onSimulate(item.id)}
                          >
                            {simulating === item.id ? "执行…" : item.label}
                          </button>
                        ))}
                      </div>
                    </div>

                    <div>
                      <div className="inspector-sublabel">工位运行模式</div>
                      <div className="sim-bar" style={{ width: "100%", display: "grid", gridTemplateColumns: "1fr 1fr" }}>
                        <button
                          type="button"
                          className={`sim-btn ${workMode === "debug" ? "busy" : ""}`}
                          onClick={() => onSwitchWorkMode("debug")}
                        >
                          桌面调试模式
                        </button>
                        <button
                          type="button"
                          className={`sim-btn ${workMode === "strict" ? "busy" : ""}`}
                          onClick={() => onSwitchWorkMode("strict")}
                        >
                          车间严格模式
                        </button>
                      </div>
                    </div>

                    <div className="grid grid-cols-3 gap-1.5">
                      <button
                        type="button"
                        className="btn ghost sm block"
                        style={{ padding: "0 6px", fontSize: 11.5 }}
                        onClick={() => onGoStep("records_settings", "records")}
                      >
                        周期与证据
                      </button>
                      <button
                        type="button"
                        className="btn ghost sm block"
                        style={{ padding: "0 6px", fontSize: 11.5 }}
                        onClick={() => onGoStep("records_settings", "storage")}
                      >
                        数据库管理
                      </button>
                      <button
                        type="button"
                        className="btn ghost sm block"
                        style={{ padding: "0 6px", fontSize: 11.5 }}
                        onClick={() => onGoStep("records_settings", "settings")}
                      >
                        镜像与参数
                      </button>
                    </div>
                  </div>
                </details>
              </>
            ) : (
              <>
                {/* 步骤 2 检查器：共用采集分组元数据 */}
                <section className="inspector-section grid gap-2.5">
                  <div className="inspector-section-head">
                    <h3>采集分组元数据（方式 A / B 共用）</h3>
                    <button
                      type="button"
                      className="btn ghost sm"
                      style={{ height: 26, fontSize: 11.5 }}
                      disabled={bootstrapping}
                      onClick={() => void handleBootstrapDataset()}
                    >
                      <Sparkles size={12} aria-hidden="true" />
                      {bootstrapping ? "生成中…" : "生成演示集"}
                    </button>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="field">
                      <label>料管颜色</label>
                      <Select value={pipeColor} onValueChange={(v) => setPipeColor(v as "yellow" | "blue")}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="yellow">黄管</SelectItem>
                          <SelectItem value="blue">蓝管</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="field">
                      <label>物理料管分组号</label>
                      <Input
                        value={tubeGroup}
                        onChange={(e) => setTubeGroup(e.target.value)}
                        className="mono"
                        placeholder="TUBE_LIVE_01"
                      />
                    </div>
                  </div>
                  <div className="field">
                    <label>生产批次号</label>
                    <Input
                      value={batchGroup}
                      onChange={(e) => setBatchGroup(e.target.value)}
                      className="mono"
                      placeholder="BATCH_2026_LIVE"
                    />
                  </div>
                </section>

                {/* 增加训练数据 · 方式 A：5 类实时镜头一键拍样本 */}
                <section className="inspector-section grid gap-2">
                  <div className="inspector-section-head">
                    <div className="flex items-center gap-1.5">
                      <span className="method-badge">方式 A</span>
                      <h3>实时镜头抓拍入库</h3>
                    </div>
                    <span className="note">{manualBox ? "手动拉框裁切" : "自动追踪裁切"}</span>
                  </div>
                  <div className="grid gap-1.5">
                    {sampleClassDefs.map((item) => (
                      <button
                        key={item.id}
                        type="button"
                        disabled={capturingClass !== ""}
                        className={`btn ${item.id === "unknown_challenge" ? "ghost" : ""} block`}
                        style={{ justifyContent: "space-between", height: 34, padding: "0 11px" }}
                        onClick={() => void handleQuickCapture(item.id, item.short)}
                      >
                        <span className="flex items-center gap-2">
                          <Camera size={13} aria-hidden="true" />
                          <span>{capturingClass === item.id ? "抓拍裁切中…" : item.title}</span>
                        </span>
                        <span className="mono" style={{ opacity: 0.8, fontSize: 12 }}>
                          {counts[item.id] || 0} 张
                        </span>
                      </button>
                    ))}
                  </div>
                </section>

                {/* 增加训练数据 · 方式 B：本地选图上传加入训练集 */}
                <section className="inspector-section grid gap-2">
                  <div className="inspector-section-head">
                    <div className="flex items-center gap-1.5">
                      <span className="method-badge alt">方式 B</span>
                      <h3>本地选图上传加入训练集</h3>
                    </div>
                    <span className="note">JPEG / PNG</span>
                  </div>
                  <form className="upload-box" onSubmit={(e) => void handleLocalSampleUpload(e)}>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="field">
                        <label>选择本地图片</label>
                        <Input
                          ref={uploadInputRef}
                          type="file"
                          accept="image/jpeg,image/png"
                          onChange={(e) => selectUploadFile(e.target.files?.[0] || null)}
                        />
                      </div>
                      <div className="field">
                        <label>目标训练类别</label>
                        <Select value={uploadLabel} onValueChange={(v) => setUploadLabel(v as SampleLabel)}>
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="complete">完整标签</SelectItem>
                            <SelectItem value="crossed_out">A类 · 已划除</SelectItem>
                            <SelectItem value="covered">B类 · 已遮挡</SelectItem>
                            <SelectItem value="absent">C类 · 已脱落</SelectItem>
                            <SelectItem value="unknown_challenge">挑战样本</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>

                    {uploadFile && uploadPreviewUrl && (
                      <div className="upload-preview-row">
                        <img
                          src={uploadPreviewUrl}
                          alt="待上传训练图片预览"
                          className="upload-preview-thumb"
                          onError={() => {
                            if (previewUrlRef.current) {
                              URL.revokeObjectURL(previewUrlRef.current)
                              previewUrlRef.current = ""
                            }
                            setUploadPreviewUrl("")
                            flash("无法预览所选图片，请确认文件为有效 JPEG/PNG")
                          }}
                        />
                        <div className="upload-preview-info">
                          <b>{uploadFile.name}</b>
                          <small className="mono">{Math.max(1, Math.round(uploadFile.size / 1024))} KB</small>
                        </div>
                        <button
                          type="button"
                          className="icon-btn"
                          style={{ width: 24, height: 24 }}
                          aria-label="清除已选图片"
                          onClick={() => {
                            selectUploadFile(null)
                            if (uploadInputRef.current) uploadInputRef.current.value = ""
                          }}
                        >
                          <X size={12} aria-hidden="true" />
                        </button>
                      </div>
                    )}

                    <button
                      type="submit"
                      className="btn sm block"
                      disabled={uploadingSample || !uploadFile}
                    >
                      <ImagePlus size={13} aria-hidden="true" />
                      {uploadingSample ? "正在写入训练集…" : "上传并加入训练集"}
                    </button>
                  </form>
                </section>

                {/* 样本图库：按类别筛选 + 多选批量删除 + 直接改标注 / 单张删除 */}
                <section className="inspector-section grid gap-2.5">
                  <div className="inspector-section-head">
                    <h3>样本管理（筛选 / 改标注 / 批量删）</h3>
                    <span className="mono" style={{ fontSize: 11.5, color: "var(--ink-3)" }}>
                      显示 {samples.length} / 总 {datasetSummary.total_samples || 0}
                    </span>
                  </div>

                  {/* 按类别筛选标签条 */}
                  <div className="filter-chip-bar" role="tablist" aria-label="按训练类别筛选样本">
                    {sampleFilterOptions.map((opt) => {
                      const cnt = opt.id === "" ? Number(datasetSummary.total_samples || 0) : Number(counts[opt.id] || 0)
                      const active = sampleFilter === opt.id
                      return (
                        <button
                          key={opt.id || "all"}
                          type="button"
                          role="tab"
                          aria-selected={active}
                          className={`filter-chip ${active ? "on" : ""}`}
                          onClick={() => {
                            setSampleFilter(opt.id)
                            setSelectedSampleIds([])
                          }}
                        >
                          <span>{opt.label}</span>
                          <em>{cnt}</em>
                        </button>
                      )
                    })}
                  </div>

                  {/* 多选与批量删除工具栏 */}
                  <div className="sample-bulk-bar">
                    <label className="check" style={{ fontSize: 11.5 }}>
                      <input
                        type="checkbox"
                        checked={allVisibleSelected}
                        disabled={samples.length === 0 || bulkBusy}
                        onChange={toggleSelectAllVisible}
                      />
                      <span>
                        全选{selectedSampleIds.length > 0 ? `（已选 ${selectedSampleIds.length}）` : ""}
                      </span>
                    </label>
                    <div className="sample-bulk-actions">
                      <button
                        type="button"
                        className="btn ghost sm btn-outline-bad"
                        style={{ height: 24, padding: "0 8px", fontSize: 11.5 }}
                        disabled={selectedSampleIds.length === 0 || bulkBusy}
                        onClick={requestBatchDeleteSelected}
                      >
                        <Trash2 size={11} aria-hidden="true" />
                        批量删除选中
                      </button>
                      <button
                        type="button"
                        className="btn ghost sm btn-outline-bad"
                        style={{ height: 24, padding: "0 8px", fontSize: 11.5 }}
                        disabled={samples.length === 0 || bulkBusy}
                        onClick={requestClearCurrentClass}
                      >
                        {sampleFilter ? `清空${activeFilterLabel}` : "清空全部"}
                      </button>
                    </div>
                  </div>

                  {samplesLoading && samples.length === 0 ? (
                    <div className="empty" style={{ minHeight: 100 }}>
                      正在加载样本列表…
                    </div>
                  ) : samples.length > 0 ? (
                    <div className="grid grid-cols-2 gap-2">
                      {samples.map((smp) => {
                        const sid = String(smp.sample_id)
                        const checked = selectedIdSet.has(sid)
                        return (
                          <div key={sid} className={`sample-thumb-card ${checked ? "selected" : ""}`}>
                            <label className="sample-thumb-check" title="勾选用于批量删除">
                              <input
                                type="checkbox"
                                checked={checked}
                                aria-label={`选择样本 ${sid}`}
                                onChange={() => toggleSampleSelection(sid)}
                              />
                            </label>
                            <img src={smp.image_url} alt={sid} loading="lazy" />
                            <div className="sample-thumb-meta">
                              <div className="flex items-center justify-between gap-1">
                                <span className="mono sample-thumb-id" title={smp.physical_tube_id || sid}>
                                  {smp.physical_tube_id || sid}
                                </span>
                                <button
                                  type="button"
                                  className="sample-del-btn"
                                  onClick={() => requestDeleteSingleSample(smp)}
                                >
                                  删除
                                </button>
                              </div>
                              <select
                                aria-label={`修改样本 ${sid} 类别`}
                                value={smp.label}
                                className="sample-label-select"
                                onChange={(e) => void handleChangeSampleLabel(sid, e.target.value)}
                              >
                                <option value="complete">完整标签</option>
                                <option value="crossed_out">A类·已划除</option>
                                <option value="covered">B类·已遮挡</option>
                                <option value="absent">C类·已脱落</option>
                                <option value="unknown_challenge">挑战样本</option>
                              </select>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  ) : (
                    <div className="empty" style={{ minHeight: 100 }}>
                      {sampleFilter
                        ? `【${activeFilterLabel}】类别暂无样本，请用上方方式 A 抓拍或方式 B 本地上传`
                        : "暂无训练样本，请用上方方式 A 镜头抓拍、方式 B 本地上传或生成演示集"}
                    </div>
                  )}
                </section>

                {/* 折叠高级本地上传（业务证据帧，区别于 AI 训练集） */}
                <details className="inspector-group">
                  <summary>
                    <span>高级：离线补录业务证据帧（SQLite 证据库）</span>
                    <Upload size={13} aria-hidden="true" />
                  </summary>
                  <form className="inspector-group-body grid gap-2.5" onSubmit={(e) => void submitManualEvidence(e)}>
                    <div className="field">
                      <label>本地 JPEG / PNG 图像</label>
                      <Input
                        type="file"
                        accept="image/jpeg,image/png"
                        onChange={(e) => setFile(e.target.files?.[0] || null)}
                      />
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="field">
                        <label>周期号（留空自动）</label>
                        <Input
                          list="cycle-options"
                          value={cycle}
                          onChange={(e) => setCycle(e.target.value)}
                          placeholder="自动生成"
                          className="mono"
                        />
                        <datalist id="cycle-options">
                          {cycles.map((c) => (
                            <option key={c.cycle_id} value={c.cycle_id} />
                          ))}
                        </datalist>
                      </div>
                      <div className="field">
                        <label>证据类别</label>
                        <Select value={kind} onValueChange={setKind}>
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="initial_label">初始标签</SelectItem>
                            <SelectItem value="A">A 类 · 划除</SelectItem>
                            <SelectItem value="B">B 类 · 遮挡</SelectItem>
                            <SelectItem value="C">C 类 · 脱落</SelectItem>
                            <SelectItem value="anomaly">异常快照</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                    <div className="field">
                      <label>标签状态</label>
                      <Select value={label} onValueChange={setLabel}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="complete">完整</SelectItem>
                          <SelectItem value="crossed_out">A 类 · 已划除</SelectItem>
                          <SelectItem value="covered">B 类 · 已遮挡</SelectItem>
                          <SelectItem value="absent">C 类 · 已脱落</SelectItem>
                          <SelectItem value="unknown">未知 / 难判</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <button type="submit" className="btn sm block" disabled={submitting}>
                      <Upload size={13} aria-hidden="true" />
                      {submitting ? "上传中…" : "上传并写入证据库"}
                    </button>
                  </form>
                </details>
              </>
            )}
          </div>
        </div>
      </aside>
    </div>
  )
}

/* ==========================================================================
   步骤 3：训练部署（左侧独立滚动看板 + 右侧独立滚动超参与模型库检查器）
   ========================================================================== */
function TrainDeployStep({
  state,
  models,
  datasetSummary,
  trainingStatus,
  token,
  flash,
  onSaved,
  onGoCollect,
}: {
  state: Data
  models: Data[]
  datasetSummary: Data
  trainingStatus: Data
  token: string
  flash: (s: string) => void
  onSaved: () => Promise<void>
  onGoCollect: () => void
}) {
  const [epochs, setEpochs] = useState("8")
  const [inputSize, setInputSize] = useState("96")
  const [batchSize, setBatchSize] = useState("8")
  const [lr, setLr] = useState("0.0025")
  const [resumeFromLatest, setResumeFromLatest] = useState(true)
  const [useAugmentation, setUseAugmentation] = useState(true)
  const [starting, setStarting] = useState(false)
  const [bootstrapping, setBootstrapping] = useState(false)

  const [name, setName] = useState("mobilenetv3-igbt-label")
  const [version, setVersion] = useState("v1.0.0")
  const [sha, setSha] = useState("")
  const [notes, setNotes] = useState("手动登记外部预训练权重")
  const [registering, setRegistering] = useState(false)

  const isTraining = trainingStatus.state === "TRAINING" || trainingStatus.state === "EXPORTING_ONNX"
  const latestRun = trainingStatus.latest_run
  const historyList: Data[] = Array.isArray(trainingStatus.history) ? trainingStatus.history : []
  const confusionMatrix = latestRun?.confusion_matrix
  const counts = datasetSummary.class_counts || {}

  const handleBootstrap = async () => {
    setBootstrapping(true)
    try {
      await api<Data>("/api/dataset/bootstrap", { method: "POST" }, token)
      await onSaved()
      flash("标准演示训练集已就绪，可立即启动 MobileNetV3 微调")
    } catch (err) {
      flash(`数据集生成失败：${err instanceof Error ? err.message : err}`)
    } finally {
      setBootstrapping(false)
    }
  }

  const handleStartTraining = async () => {
    setStarting(true)
    try {
      await api<Data>(
        "/api/training/start",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            epochs: Number(epochs) || 8,
            input_size: Number(inputSize) || 96,
            batch_size: Number(batchSize) || 8,
            learning_rate: Number(lr) || 0.0025,
            resume_from_latest: resumeFromLatest,
            use_augmentation: useAugmentation,
          }),
        },
        token
      )
      flash("已启动 PyTorch MobileNetV3-Small 训练任务，完成后将自动导出 ONNX 并热加载")
      await onSaved()
    } catch (err) {
      flash(`启动训练失败：${err instanceof Error ? err.message : err}`)
    } finally {
      setStarting(false)
    }
  }

  const handleActivateModel = async (modelId: string, modelVer: string) => {
    try {
      await api<Data>(`/api/models/${encodeURIComponent(modelId)}/activate`, { method: "POST" }, token)
      flash(`已将模型 ${modelVer} 热加载至实时相机引擎`)
      await onSaved()
    } catch (err) {
      flash(`切换失败：${err instanceof Error ? err.message : err}`)
    }
  }

  const registerManual = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!token) return flash("当前工程师会话无效，请重新登录")
    setRegistering(true)
    try {
      await api(
        "/api/models",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model_name: name,
            model_version: version,
            sha256: sha.toLowerCase(),
            license_status: "reviewed",
            notes,
          }),
        },
        token
      )
      flash("外部模型版本已登记")
      setSha("")
      await onSaved()
    } catch (err) {
      flash(`登记失败：${err instanceof Error ? err.message : err}`)
    } finally {
      setRegistering(false)
    }
  }

  const copyHash = (hash: string) => {
    navigator.clipboard
      .writeText(hash)
      .then(() => flash("已复制 SHA256"))
      .catch(() => flash("复制失败，请手动选中复制"))
  }

  return (
    <div className="studio-split">
      {/* 左侧独立滚动训练看板 */}
      <div className="studio-board" aria-label="训练指标与数据集评估看板">
        {/* 数据集分组就绪卡片 */}
        <section className="panel">
          <header className="panel-head">
            <h2>数据集分组就绪状态</h2>
            <span className={`tag ${datasetSummary.ready_to_train ? "pass" : "warn"}`}>
              {datasetSummary.ready_to_train ? "可启动训练" : "待补充样本"}
            </span>
            <div className="right">
              <button
                type="button"
                className="btn ghost sm"
                disabled={bootstrapping || isTraining}
                onClick={() => void handleBootstrap()}
              >
                <Sparkles size={13} aria-hidden="true" />
                {bootstrapping ? "生成中…" : "一键生成演示集"}
              </button>
              <button type="button" className="btn ghost sm" onClick={onGoCollect}>
                返回采集标注
              </button>
            </div>
          </header>
          <div className="panel-body grid gap-3">
            <div className="callout">
              <div>
                <b style={{ display: "block", color: "var(--ink)", fontSize: 13 }}>
                  已入库 {datasetSummary.total_samples || 0} 张样本 · 覆盖 {datasetSummary.independent_tube_groups || 0}{" "}
                  个独立物理料管组
                </b>
                <span>{datasetSummary.readiness_reason || "按物理料管 ID 严格隔离训练 / 验证 / 测试集"}</span>
              </div>
            </div>
            <div className="kpi-row kpi-row-5 drawer-kpi">
              {sampleClassDefs.map((c) => (
                <div className="readout" key={c.id}>
                  <small>{c.short}</small>
                  <b style={{ fontSize: 20 }}>
                    {counts[c.id] || 0}
                    <em>张</em>
                  </b>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* 实时训练进度条与 Epoch 曲线表 */}
        <section className="panel">
          <header className="panel-head">
            <h2>实时训练进度与 Epoch 曲线</h2>
            <div className="right">
              <span className={`tag ${isTraining ? "warn" : trainingStatus.state === "FAILED" ? "alarm" : "plain"} mono`}>
                {trainingStatus.progress_pct || 0}%
              </span>
            </div>
          </header>
          <div className="panel-body grid gap-3">
            <div className="flex items-center justify-between gap-2">
              <b style={{ fontSize: 13 }}>{trainingStatus.message || "等待启动训练"}</b>
              <span className="mono" style={{ fontSize: 12, color: "var(--ink-3)" }}>
                {trainingStatus.state || "IDLE"}
              </span>
            </div>
            <div className="meter">
              <i
                className={trainingStatus.state === "FAILED" ? "bad" : ""}
                style={{ width: `${Math.min(100, Number(trainingStatus.progress_pct || 0))}%` }}
              />
            </div>

            {historyList.length > 0 ? (
              <div style={{ border: "1px solid var(--rule)", borderRadius: 4, overflowX: "auto" }}>
                <table className="grid-table">
                  <thead>
                    <tr>
                      <th>轮次 (Epoch)</th>
                      <th>训练集 Loss</th>
                      <th>验证集准确率</th>
                    </tr>
                  </thead>
                  <tbody>
                    {historyList.map((h) => (
                      <tr key={h.epoch}>
                        <td className="mono">#{h.epoch}</td>
                        <td className="mono">{Number(h.train_loss).toFixed(4)}</td>
                        <td className="mono">{(Number(h.val_accuracy) * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="note">在右侧配置超参并点击「启动 MobileNetV3 训练并导出 ONNX」后，此处将实时刷新每轮 Loss 与准确率。</div>
            )}
          </div>
        </section>

        {/* 4×4 混淆矩阵与 ONNX 校验报告 */}
        <section className="panel">
          <header className="panel-head">
            <h2>4×4 混淆矩阵与 ONNX 数值校验报告</h2>
            {latestRun && <span className="tag pass mono">最大绝对误差 {latestRun.onnx_max_abs_diff}</span>}
          </header>
          <div className="panel-body grid gap-3">
            {latestRun ? (
              <>
                <div className="callout" style={{ background: "var(--ok-tint)", borderColor: "var(--ok-border)" }}>
                  <CheckCircle2 size={16} aria-hidden="true" style={{ color: "var(--ok)" }} />
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <b style={{ color: "var(--ok)", fontSize: 13.5, display: "block" }}>
                      最新产出：{latestRun.model_version}
                    </b>
                    <div className="flex flex-wrap gap-x-5 gap-y-1 mono" style={{ fontSize: 12, marginTop: 4 }}>
                      <span>
                        跨管测试准确率：<b>{(Number(latestRun.test_accuracy) * 100).toFixed(1)}%</b> ({latestRun.test_samples} 张)
                      </span>
                      <span>
                        完整标签误放行 (FAR)：<b>{latestRun.complete_false_accept_total} 次</b>
                      </span>
                    </div>
                    <div className="mono" style={{ fontSize: 11, color: "var(--ink-2)", marginTop: 4, wordBreak: "break-all" }}>
                      SHA256: {latestRun.onnx_sha256}
                    </div>
                  </div>
                </div>

                {confusionMatrix && Array.isArray(confusionMatrix.labels) && Array.isArray(confusionMatrix.matrix) && (
                  <div style={{ border: "1px solid var(--rule)", borderRadius: 4, overflowX: "auto" }}>
                    <table className="grid-table">
                      <thead>
                        <tr>
                          <th>真实 \ 预测</th>
                          {confusionMatrix.labels.map((col: string) => (
                            <th key={col}>{labelText[col] || col}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {confusionMatrix.matrix.map((rowVals: number[], rIdx: number) => (
                          <tr key={confusionMatrix.labels[rIdx]}>
                            <td style={{ fontWeight: 600 }}>
                              {labelText[confusionMatrix.labels[rIdx]] || confusionMatrix.labels[rIdx]}
                            </td>
                            {rowVals.map((val, cIdx) => (
                              <td
                                key={cIdx}
                                className="mono"
                                style={{
                                  fontWeight: rIdx === cIdx ? 700 : 400,
                                  color: rIdx === cIdx ? "var(--ok)" : val > 0 ? "var(--bad)" : "var(--ink-3)",
                                }}
                              >
                                {val}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </>
            ) : (
              <div className="empty" style={{ minHeight: 120 }}>
                尚未产出训练校验报告，完成一次模型训练后自动展示 4×4 混淆矩阵与 ONNX 误差校验
              </div>
            )}
          </div>
        </section>
      </div>

      {/* 右侧独立滚动超参与模型库检查器 */}
      <aside className="studio-inspector" aria-label="训练超参与模型版本库">
        <header className="studio-inspector-head">
          <h2>训练超参与模型部署</h2>
          <span className="tag pass mono" style={{ maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis" }}>
            {state.model_version || "动态引擎"}
          </span>
        </header>

        <div className="studio-inspector-scroll-wrap">
          <div className="studio-inspector-body">
            {/* 训练超参配置 */}
            <section className="inspector-section grid gap-3">
              <div className="inspector-section-head">
                <h3>MobileNetV3-Small 超参配置</h3>
                <span className={`tag ${datasetSummary.ready_to_train ? "pass" : "warn"}`}>
                  {datasetSummary.ready_to_train ? "就绪" : "缺样本"}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2.5">
                <div className="field">
                  <label>训练轮数 (Epochs)</label>
                  <Input type="number" min="2" max="50" value={epochs} onChange={(e) => setEpochs(e.target.value)} className="mono" />
                </div>
                <div className="field">
                  <label>输入尺寸 (px)</label>
                  <Input
                    type="number"
                    min="64"
                    max="320"
                    value={inputSize}
                    onChange={(e) => setInputSize(e.target.value)}
                    className="mono"
                  />
                </div>
                <div className="field">
                  <label>批次大小 (Batch)</label>
                  <Input
                    type="number"
                    min="2"
                    max="64"
                    value={batchSize}
                    onChange={(e) => setBatchSize(e.target.value)}
                    className="mono"
                  />
                </div>
                <div className="field">
                  <label>学习率 (LR)</label>
                  <Input type="number" step="0.0005" value={lr} onChange={(e) => setLr(e.target.value)} className="mono" />
                </div>
              </div>

              <div className="grid gap-2">
                <label className="check">
                  <input
                    type="checkbox"
                    checked={resumeFromLatest}
                    onChange={(e) => setResumeFromLatest(e.target.checked)}
                  />
                  基于上一版最优权重增量微调
                </label>
                <label className="check">
                  <input
                    type="checkbox"
                    checked={useAugmentation}
                    onChange={(e) => setUseAugmentation(e.target.checked)}
                  />
                  启用车间光学增强（光照 / 反光 / 微旋）
                </label>
              </div>

              <button
                type="button"
                className="btn block"
                style={{ height: 40 }}
                disabled={isTraining || starting || !datasetSummary.ready_to_train}
                onClick={() => void handleStartTraining()}
              >
                <Play size={14} aria-hidden="true" />
                {isTraining
                  ? `训练中 ${trainingStatus.progress_pct || 0}%`
                  : "启动 MobileNetV3 训练并导出 ONNX"}
              </button>
            </section>

            {/* 已登记模型版本库（一键热加载） */}
            <section className="inspector-section" style={{ padding: 0 }}>
              <div className="inspector-section-head" style={{ padding: "12px 14px 8px" }}>
                <h3>已登记模型版本库</h3>
                <span className="mono" style={{ fontSize: 11.5, color: "var(--ink-3)" }}>
                  {models.length} 版
                </span>
              </div>
              {models.length ? (
                models.map((m) => {
                  const isActive = state.model_version === m.model_version
                  return (
                    <div className="model-row" key={m.model_id}>
                      <div className="name">
                        {m.model_name}
                        <span>{m.model_version}</span>
                      </div>
                      <div>
                        {isActive ? (
                          <span className="tag pass">运行中</span>
                        ) : (
                          <button
                            type="button"
                            className="btn ghost sm"
                            style={{ height: 26, fontSize: 11.5 }}
                            onClick={() => void handleActivateModel(m.model_id, m.model_version)}
                          >
                            一键热加载
                          </button>
                        )}
                      </div>
                      <div className="hash">
                        <code>{m.sha256}</code>
                        <button type="button" onClick={() => copyHash(m.sha256)} title="复制 SHA256" aria-label="复制 SHA256">
                          <Copy size={12} aria-hidden="true" />
                        </button>
                      </div>
                      <div className="meta">
                        {dateTimeOf(m.created_at_utc)} · {m.registered_by}
                        {m.notes ? ` · ${m.notes}` : ""}
                      </div>
                    </div>
                  )
                })
              ) : (
                <div className="empty" style={{ minHeight: 110, padding: "0 14px" }}>
                  尚未登记模型，点击上方启动训练自动导出并部署
                </div>
              )}
            </section>

            {/* 折叠外部 SHA256 手动登记 */}
            <details className="inspector-group">
              <summary>
                <span>外部预训练权重 SHA256 手动登记</span>
              </summary>
              <form className="inspector-group-body grid gap-2.5" onSubmit={(e) => void registerManual(e)}>
                <div className="grid grid-cols-2 gap-2">
                  <div className="field">
                    <label>模型名称</label>
                    <Input value={name} onChange={(e) => setName(e.target.value)} required className="mono" />
                  </div>
                  <div className="field">
                    <label>版本号</label>
                    <Input value={version} onChange={(e) => setVersion(e.target.value)} required className="mono" />
                  </div>
                </div>
                <div className="field">
                  <label>权重文件 SHA256（64 位十六进制）</label>
                  <Input
                    className="mono"
                    value={sha}
                    onChange={(e) => setSha(e.target.value)}
                    required
                    pattern="[0-9a-fA-F]{64}"
                    maxLength={64}
                    placeholder="64 位十六进制 SHA256"
                  />
                </div>
                <div className="field">
                  <label>备注</label>
                  <Input value={notes} onChange={(e) => setNotes(e.target.value)} />
                </div>
                <button type="submit" className="btn sm block" disabled={registering}>
                  {registering ? "登记中…" : "使用当前工程师会话登记"}
                </button>
              </form>
            </details>
          </div>
        </div>
      </aside>
    </div>
  )
}

/* ==========================================================================
   步骤 4：记录与设置（3 子页：检测周期与证据 | 数据与数据库管理 | 相机镜像与系统参数）
   ========================================================================== */
function RecordsSettingsStep({
  activeSubTab,
  onChangeSubTab,
  state,
  config,
  history,
  models,
  datasetSummary,
  storageOverview,
  storageLoading,
  cycles,
  evidence,
  events,
  selectedCycleId,
  filmstripItems,
  activeSnapshot,
  passCount,
  alarmCount,
  totalCount,
  token,
  mirrorHorizontal,
  togglingMirror,
  flash,
  onConfirm,
  onSelectCycle,
  onClearSelectedCycle,
  onOpenSnapshot,
  onManualCapture,
  onSaved,
  onRefreshStorage,
  onToggleAuxRois,
  onToggleMirror,
  onSwitchWorkMode,
  onGoStep,
}: {
  activeSubTab: RecordsSettingsTab
  onChangeSubTab: (tab: RecordsSettingsTab) => void
  state: Data
  config: Data
  history: Data[]
  models: Data[]
  datasetSummary: Data
  storageOverview: Data | null
  storageLoading: boolean
  cycles: Data[]
  evidence: Data[]
  events: Data[]
  selectedCycleId: string
  filmstripItems: FilmstripItem[]
  activeSnapshot: FilmstripItem | null
  passCount: number
  alarmCount: number
  totalCount: number
  token: string
  mirrorHorizontal: boolean
  togglingMirror: boolean
  flash: (s: string) => void
  onConfirm: (dialog: ConfirmDialogState) => void
  onSelectCycle: (cycleId: string) => void
  onClearSelectedCycle: () => void
  onOpenSnapshot: (item: FilmstripItem) => void
  onManualCapture: () => void
  onSaved: () => Promise<void>
  onRefreshStorage: () => void
  onToggleAuxRois: (show: boolean) => void
  onToggleMirror: (mirror: boolean) => void
  onSwitchWorkMode: (mode: "debug" | "strict") => void
  onGoStep: (step: EngineerStep, subTab?: RecordsSettingsTab) => void
}) {
  const yieldRate = totalCount > 0 ? `${((passCount / totalCount) * 100).toFixed(1)}%` : "—"

  const requestPurgeSimulated = () => {
    onConfirm({
      title: "仅清理演练假周期",
      description: "确认清理 SQLite 中全部模拟演练周期（CYC_*）及其关联状态事件与磁盘证据图片吗？",
      consequences: [
        "仅删除模拟演练产生的 CYC_* 周期、事件日志与对应证据帧文件",
        "保留所有真实产线检测周期、系统参数历史、AI 训练样本与 ONNX 模型",
      ],
      confirmLabel: "确认清理演练数据",
      tone: "warn",
      onConfirm: async () => {
        try {
          const res = await api<Data>(
            "/api/cycles/purge",
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ scope: "simulated_only", confirm: true }),
            },
            token
          )
          onClearSelectedCycle()
          flash(
            `已清理 ${res.deleted_cycles ?? 0} 条演练周期、${res.deleted_events ?? 0} 条事件，释放 ${res.freed_files ?? 0} 个证据文件`
          )
          await onSaved()
        } catch (err) {
          flash(`清理演练数据失败：${err instanceof Error ? err.message : err}`)
        }
      },
    })
  }

  return (
    <div className="studio-step4">
      <div className="studio-subbar">
        <div className="studio-subtabs" role="tablist" aria-label="记录与设置子页">
          <button
            type="button"
            role="tab"
            aria-selected={activeSubTab === "records"}
            className={`studio-subtab ${activeSubTab === "records" ? "on" : ""}`}
            onClick={() => onChangeSubTab("records")}
          >
            <ClipboardList size={14} aria-hidden="true" />
            检测周期与证据
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeSubTab === "storage"}
            className={`studio-subtab ${activeSubTab === "storage" ? "on" : ""}`}
            onClick={() => onChangeSubTab("storage")}
          >
            <HardDrive size={14} aria-hidden="true" />
            数据与数据库管理
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeSubTab === "settings"}
            className={`studio-subtab ${activeSubTab === "settings" ? "on" : ""}`}
            onClick={() => onChangeSubTab("settings")}
          >
            <Sliders size={14} aria-hidden="true" />
            相机镜像与系统参数
          </button>
        </div>

        <div className="studio-subbar-right">
          {activeSubTab === "records" ? (
            <>
              <span className="tag plain mono">总 {totalCount}</span>
              <span className="tag pass mono">放行 {passCount}</span>
              <span className="tag alarm mono">报警 {alarmCount}</span>
              <span className="tag plain mono">良率 {yieldRate}</span>
              <button
                type="button"
                className="btn ghost sm btn-outline-bad"
                onClick={requestPurgeSimulated}
              >
                <Trash2 size={12} aria-hidden="true" />
                仅清空演练数据
              </button>
              <button type="button" className="btn sm" onClick={onManualCapture}>
                <Camera size={13} aria-hidden="true" />
                抓拍当前帧
              </button>
            </>
          ) : activeSubTab === "storage" ? (
            <>
              {storageOverview?.business_db && (
                <span className="tag plain mono">
                  SQLite {fmt(storageOverview.business_db.db_size_kb, 1)} KB
                </span>
              )}
              {storageOverview?.training_dataset && (
                <span className="tag plain mono">
                  样本 {storageOverview.training_dataset.total_samples ?? 0} 张 ·{" "}
                  {fmt(storageOverview.training_dataset.dataset_size_mb, 2)} MB
                </span>
              )}
              <button
                type="button"
                className="btn ghost sm"
                disabled={storageLoading}
                onClick={onRefreshStorage}
              >
                <RefreshCw size={12} aria-hidden="true" />
                {storageLoading ? "统计中…" : "刷新存储统计"}
              </button>
            </>
          ) : (
            <span className="tag plain mono">
              {!config.roi_config_version || config.roi_config_version === "ROI_UNCALIBRATED"
                ? "默认五区配置"
                : config.roi_config_version}
            </span>
          )}
        </div>
      </div>

      {activeSubTab === "records" ? (
        <RecordsSubView
          cycles={cycles}
          evidence={evidence}
          events={events}
          selectedCycleId={selectedCycleId}
          filmstripItems={filmstripItems}
          activeSnapshot={activeSnapshot}
          token={token}
          flash={flash}
          onConfirm={onConfirm}
          onSelectCycle={onSelectCycle}
          onClearSelectedCycle={onClearSelectedCycle}
          onOpenSnapshot={onOpenSnapshot}
          onSaved={onSaved}
        />
      ) : activeSubTab === "storage" ? (
        <StorageManagerSubView
          state={state}
          models={models}
          datasetSummary={datasetSummary}
          storageOverview={storageOverview}
          storageLoading={storageLoading}
          token={token}
          flash={flash}
          onConfirm={onConfirm}
          onSaved={onSaved}
          onClearSelectedCycle={onClearSelectedCycle}
          onChangeSubTab={onChangeSubTab}
          onGoStep={onGoStep}
        />
      ) : (
        <SettingsSubView
          key={config.config_version ?? "initial"}
          state={state}
          config={config}
          history={history}
          token={token}
          mirrorHorizontal={mirrorHorizontal}
          togglingMirror={togglingMirror}
          flash={flash}
          onSaved={() => void onSaved()}
          onToggleAuxRois={onToggleAuxRois}
          onToggleMirror={onToggleMirror}
          onSwitchWorkMode={onSwitchWorkMode}
        />
      )}
    </div>
  )
}

function RecordsSubView({
  cycles,
  evidence,
  events,
  selectedCycleId,
  filmstripItems,
  activeSnapshot,
  token,
  flash,
  onConfirm,
  onSelectCycle,
  onClearSelectedCycle,
  onOpenSnapshot,
  onSaved,
}: {
  cycles: Data[]
  evidence: Data[]
  events: Data[]
  selectedCycleId: string
  filmstripItems: FilmstripItem[]
  activeSnapshot: FilmstripItem | null
  token: string
  flash: (s: string) => void
  onConfirm: (dialog: ConfirmDialogState) => void
  onSelectCycle: (cycleId: string) => void
  onClearSelectedCycle: () => void
  onOpenSnapshot: (item: FilmstripItem) => void
  onSaved: () => Promise<void>
}) {
  const [filter, setFilter] = useState<CycleFilter>("ALL")

  const filteredCycles = cycles.filter((c) => {
    if (filter === "ALL") return true
    if (filter === "ALARM") return c.final_outcome === "ALARM" || c.final_outcome === "FAULT"
    return c.final_outcome === filter
  })

  const requestDeleteCycle = (cycleId: string) => {
    onConfirm({
      title: "删除单条检测周期",
      description: `确认永久删除检测周期 ${cycleId} 及其全部关联记录吗？`,
      consequences: [
        "事务内级联删除 SQLite 中该周期的 cycles 行、events 状态跃迁日志与 evidence 记录",
        "同步安全清理磁盘 data/evidence/ 中该周期对应的快照图片文件",
      ],
      confirmLabel: "确认删除该周期",
      tone: "danger",
      onConfirm: async () => {
        try {
          const res = await api<Data>(`/api/cycles/${encodeURIComponent(cycleId)}`, { method: "DELETE" }, token)
          if (selectedCycleId === cycleId) {
            onClearSelectedCycle()
          }
          flash(
            `已删除周期 ${cycleId}（清理 ${res.deleted_events ?? 0} 条事件，释放 ${res.freed_files ?? 0} 个证据文件）`
          )
          await onSaved()
        } catch (err) {
          flash(`删除周期失败：${err instanceof Error ? err.message : err}`)
        }
      },
    })
  }

  return (
    <div className="records-layout">
      <div className="records-filmstrip">
        <Filmstrip items={filmstripItems} selectedId={activeSnapshot?.id || selectedCycleId} onSelect={onOpenSnapshot} />
      </div>

      <div className="records-split">
        <section className="panel records-pane">
          <header className="panel-head">
            <h2>检测周期记录</h2>
            <span className="sub mono">{filteredCycles.length} 条</span>
            <div className="right">
              <div className="sim-bar" role="group" aria-label="按判定结果筛选">
                {(
                  [
                    ["ALL", "全部"],
                    ["PASS", "放行"],
                    ["ALARM", "报警"],
                    ["VIEW_INVALID", "视野无效"],
                  ] as [CycleFilter, string][]
                ).map(([val, label]) => (
                  <button
                    key={val}
                    type="button"
                    className={`sim-btn ${filter === val ? "busy" : ""}`}
                    onClick={() => setFilter(val)}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          </header>
          <div className="records-scroll">
            <table className="grid-table records-table">
              <thead>
                <tr>
                  <th>周期号</th>
                  <th>更新时间</th>
                  <th>料管</th>
                  <th>判定</th>
                  <th>原因</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {filteredCycles.length ? (
                  filteredCycles.map((row) => {
                    const matched = evidence.find((e) => e.cycle_id === row.cycle_id)
                    return (
                      <tr
                        key={row.cycle_id}
                        className={`click ${selectedCycleId === row.cycle_id ? "sel" : ""}`}
                        onClick={() => onSelectCycle(row.cycle_id)}
                      >
                        <td className="mono">{row.cycle_id}</td>
                        <td className="mono">{dateTimeOf(row.updated_at_utc)}</td>
                        <td>{pipeText(row.pipe_color)}</td>
                        <td>
                          <OutcomeTag value={row.final_outcome} />
                        </td>
                        <td className="reason">{reasonOf(row.result_reason) || "—"}</td>
                        <td>
                          <div className="row-actions">
                            <button
                              type="button"
                              className="btn ghost sm"
                              style={{ height: 24, padding: "0 8px", fontSize: 11.5 }}
                              onClick={(e) => {
                                e.stopPropagation()
                                onOpenSnapshot({
                                  id: row.cycle_id,
                                  cycleId: row.cycle_id,
                                  outcome: row.final_outcome || "WAITING",
                                  url: matched?.url,
                                  timestamp: row.updated_at_utc || row.created_at_utc || "",
                                  kind: matched?.kind,
                                  pipeColor: row.pipe_color,
                                  labelStatus: row.initial_label,
                                })
                              }}
                            >
                              查看
                            </button>
                            <button
                              type="button"
                              className="btn ghost sm btn-outline-bad"
                              style={{ height: 24, padding: "0 7px", fontSize: 11.5 }}
                              aria-label={`删除周期 ${row.cycle_id}`}
                              onClick={(e) => {
                                e.stopPropagation()
                                requestDeleteCycle(String(row.cycle_id))
                              }}
                            >
                              删除
                            </button>
                          </div>
                        </td>
                      </tr>
                    )
                  })
                ) : (
                  <tr>
                    <td colSpan={6}>
                      <div className="empty">暂无符合条件的周期记录</div>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>

        <section className="panel records-pane">
          <header className="panel-head">
            <h2>单周期状态跃迁时间线</h2>
            <span className="sub mono">{selectedCycleId || ""}</span>
            <div className="right mono">{events.length} 步</div>
          </header>
          <div className="records-scroll">
            {events.length ? (
              <ol className="timeline records-timeline">
                {events.map((row) => (
                  <li key={row.event_id}>
                    <div className="row">
                      <b>{human(row.event_type)}</b>
                      <time>{timeOf(row.created_at_utc)}</time>
                    </div>
                    <div className="flow">
                      {human(row.state_before)}{" "}
                      <ChevronRight size={11} aria-hidden="true" style={{ display: "inline", verticalAlign: -1 }} />{" "}
                      <b style={{ color: "var(--ink)" }}>{human(row.state_after)}</b>
                    </div>
                    {row.reason && <div className="why">{reasonOf(row.reason)}</div>}
                  </li>
                ))}
              </ol>
            ) : (
              <div className="empty">点击左侧周期行查看完整状态跃迁事件</div>
            )}
          </div>
        </section>
      </div>
    </div>
  )
}

/* ==========================================================================
   步骤 4B：数据与数据库管理中心（两套存储架构对比 + 模型销毁保护 + 危险区压缩）
   ========================================================================== */
function StorageManagerSubView({
  state,
  models,
  datasetSummary,
  storageOverview,
  storageLoading,
  token,
  flash,
  onConfirm,
  onSaved,
  onClearSelectedCycle,
  onChangeSubTab,
  onGoStep,
}: {
  state: Data
  models: Data[]
  datasetSummary: Data
  storageOverview: Data | null
  storageLoading: boolean
  token: string
  flash: (s: string) => void
  onConfirm: (dialog: ConfirmDialogState) => void
  onSaved: () => Promise<void>
  onClearSelectedCycle: () => void
  onChangeSubTab: (tab: RecordsSettingsTab) => void
  onGoStep: (step: EngineerStep, subTab?: RecordsSettingsTab) => void
}) {
  const bdb = storageOverview?.business_db || {}
  const tables = bdb.tables || {}
  const ds = storageOverview?.training_dataset || {}
  const ma = storageOverview?.model_artifacts || {}
  const classCounts = ds.class_counts || datasetSummary.class_counts || {}
  const activeModelVersion = ma.active_model_version || state.model_version || ""

  const sqliteTableItems = [
    { key: "cycles", name: "检测周期", count: tables.cycles },
    { key: "events", name: "状态事件", count: tables.events },
    { key: "evidence", name: "证据快照", count: tables.evidence },
    { key: "configuration_history", name: "参数历史", count: tables.configuration_history },
    { key: "model_registry", name: "模型版本", count: tables.model_registry },
  ]

  const copyHash = (hash: string) => {
    navigator.clipboard
      .writeText(hash)
      .then(() => flash("已复制 SHA256"))
      .catch(() => flash("复制失败，请手动选中复制"))
  }

  const handleActivateModel = async (modelId: string, modelVer: string) => {
    try {
      await api<Data>(`/api/models/${encodeURIComponent(modelId)}/activate`, { method: "POST" }, token)
      flash(`已切换实时引擎至模型 ${modelVer}`)
      await onSaved()
    } catch (err) {
      flash(`热加载切换失败：${err instanceof Error ? err.message : err}`)
    }
  }

  const requestDeleteModel = (m: Data) => {
    const modelId = String(m.model_id)
    const modelVer = String(m.model_version)
    onConfirm({
      title: "销毁历史模型版本",
      description: `确认永久销毁模型 ${m.model_name} (${modelVer}) 吗？`,
      consequences: [
        "删除 SQLite model_registry 表中的该模型登记记录",
        `安全清理磁盘 data/runs/label_classifier/${modelVer} 下的 ONNX / PyTorch 权重目录`,
        "当前正在实时运行的模型受系统锁保护，不会被误删",
      ],
      confirmLabel: "确认永久销毁模型",
      tone: "danger",
      onConfirm: async () => {
        try {
          await api<Data>(`/api/models/${encodeURIComponent(modelId)}`, { method: "DELETE" }, token)
          flash(`已销毁废弃模型 ${modelVer} 并清理磁盘产物目录`)
          await onSaved()
        } catch (err) {
          flash(`销毁模型失败：${err instanceof Error ? err.message : err}`)
        }
      },
    })
  }

  const requestPurgeSimulated = () => {
    onConfirm({
      title: "仅清理演练假周期",
      description: "确认清理全部演练模拟周期（CYC_*）及其关联事件和证据图片吗？",
      consequences: [
        "仅删除 cycle_id 以 CYC_ 开头的演练周期、对应 events 与 evidence 磁盘图片",
        "真实产线记录、参数配置、AI 训练样本与 ONNX 模型均完整保留",
      ],
      confirmLabel: "确认清理演练假周期",
      tone: "warn",
      onConfirm: async () => {
        try {
          const res = await api<Data>(
            "/api/cycles/purge",
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ scope: "simulated_only", confirm: true }),
            },
            token
          )
          onClearSelectedCycle()
          flash(
            `已清理 ${res.deleted_cycles ?? 0} 条演练周期、${res.deleted_events ?? 0} 条事件，释放 ${res.freed_files ?? 0} 个证据文件`
          )
          await onSaved()
        } catch (err) {
          flash(`清理演练周期失败：${err instanceof Error ? err.message : err}`)
        }
      },
    })
  }

  const requestClearTrainingSamples = () => {
    const totalSamples = Number(ds.total_samples ?? datasetSummary.total_samples ?? 0)
    onConfirm({
      title: "清空全部 AI 训练样本",
      description: `确认永久清空 data/dataset/ 下全部 ${totalSamples} 张训练切片与标注索引吗？`,
      consequences: [
        "删除 5 个类别下的全部切片图像及 manifest / COCO 标注清单",
        "不影响 SQLite 业务检测周期记录，也不影响当前已训练部署的 ONNX 模型",
      ],
      confirmLabel: `确认清空 ${totalSamples} 张训练样本`,
      tone: "danger",
      onConfirm: async () => {
        try {
          await api<Data>("/api/dataset/clear", { method: "POST" }, token)
          flash("已清空全部 AI 训练样本")
          await onSaved()
        } catch (err) {
          flash(`清空训练样本失败：${err instanceof Error ? err.message : err}`)
        }
      },
    })
  }

  const requestDestroyAllBusiness = () => {
    onConfirm({
      title: "销毁全部业务周期与证据 + VACUUM 压缩空间",
      description: "高危操作：即将永久清空 SQLite 中全部检测周期、状态跃迁事件与磁盘证据快照，并执行 VACUUM 立即压缩数据库文件。",
      consequences: [
        `删除全部 ${tables.cycles ?? 0} 条检测周期、${tables.events ?? 0} 条状态事件与 ${tables.evidence ?? 0} 条证据索引`,
        `永久删除磁盘 data/evidence/ 下 ${bdb.evidence_files ?? 0} 个证据快照图片（释放约 ${fmt(bdb.evidence_size_mb, 2)} MB）`,
        "执行 SQLite VACUUM 收缩 station.sqlite3 磁盘占用；保留系统参数历史与模型版本库",
      ],
      confirmLabel: "确认永久销毁并 VACUUM",
      tone: "danger",
      requireCheckLabel: "我已知晓全部业务检测周期与证据快照将永久删除且不可恢复",
      onConfirm: async () => {
        try {
          const res = await api<Data>(
            "/api/cycles/purge",
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ scope: "all", confirm: true }),
            },
            token
          )
          onClearSelectedCycle()
          flash(
            `已销毁全部 ${res.deleted_cycles ?? 0} 条周期、释放 ${res.freed_files ?? 0} 个证据文件并完成 SQLite VACUUM 压缩`
          )
          await onSaved()
        } catch (err) {
          flash(`销毁业务库失败：${err instanceof Error ? err.message : err}`)
        }
      },
    })
  }

  return (
    <div className="storage-layout" aria-label="数据与数据库管理中心">
      {/* 顶部两列：后台存储逻辑架构卡（一眼看懂两套数据） */}
      <div className="storage-arch-grid">
        {/* 左卡：生产运行业务库 */}
        <section className="panel storage-card">
          <header className="panel-head">
            <h2 className="flex items-center gap-1.5">
              <HardDrive size={14} aria-hidden="true" />
              生产运行业务库
            </h2>
            <span className="tag plain mono">SQLite + 证据图</span>
            <div className="right mono">
              {storageLoading && !storageOverview
                ? "统计中…"
                : `${fmt(bdb.db_size_kb, 1)} KB · 证据 ${fmt(bdb.evidence_size_mb, 2)} MB`}
            </div>
          </header>

          <div className="storage-card-body">
            {/* 5 张表实时行数 */}
            <div className="storage-metric-grid" aria-label="SQLite 五张业务表实时行数">
              {sqliteTableItems.map((t) => (
                <div className="storage-metric-cell" key={t.key}>
                  <small>{t.name}</small>
                  <b>{t.count ?? "—"}</b>
                  <code title={t.key}>{t.key}</code>
                </div>
              ))}
            </div>

            {/* 真实文件与路径 */}
            <div className="storage-path-list">
              <div className="storage-path-row">
                <span>SQLite 库</span>
                <code>{bdb.db_path || "data/station.sqlite3"}</code>
                <em>{fmt(bdb.db_size_kb, 1)} KB</em>
              </div>
              <div className="storage-path-row">
                <span>证据目录</span>
                <code>{bdb.evidence_dir || "data/evidence"}</code>
                <em>
                  {bdb.evidence_files ?? 0} 张 · {fmt(bdb.evidence_size_mb, 2)} MB
                </em>
              </div>
            </div>

            {/* 一眼看懂：存什么 / 怎么删 / 怎么销毁 */}
            <div className="storage-logic-list">
              <div className="storage-logic-row">
                <span className="storage-logic-pill">存什么</span>
                <span>实时产线检测周期、状态机跃迁日志、抓拍证据帧、参数版本历史与已登记模型表。</span>
              </div>
              <div className="storage-logic-row">
                <span className="storage-logic-pill">怎么删</span>
                <span>前往「检测周期与证据」逐行级联删除单条周期，或在下方销毁非运行中的废弃模型。</span>
              </div>
              <div className="storage-logic-row">
                <span className="storage-logic-pill">怎么销毁</span>
                <span>在下方危险区一键清理演练假周期，或清空全部周期证据并执行 VACUUM 收缩体积。</span>
              </div>
            </div>
          </div>

          <footer className="storage-card-foot">
            <button type="button" className="btn ghost sm" onClick={() => onChangeSubTab("records")}>
              <ClipboardList size={13} aria-hidden="true" />
              前往检测周期与证据
            </button>
          </footer>
        </section>

        {/* 右卡：AI 训练样本与模型库 */}
        <section className="panel storage-card">
          <header className="panel-head">
            <h2 className="flex items-center gap-1.5">
              <Layers size={14} aria-hidden="true" />
              AI 训练样本与模型库
            </h2>
            <span className="tag plain mono">切片 + ONNX</span>
            <div className="right mono">
              {storageLoading && !storageOverview
                ? "统计中…"
                : `样本 ${fmt(ds.dataset_size_mb, 2)} MB · 模型 ${fmt(ma.models_size_mb, 2)} MB`}
            </div>
          </header>

          <div className="storage-card-body">
            {/* 5 类训练切片实时数量 */}
            <div className="storage-metric-grid" aria-label="五类 AI 训练切片统计">
              {sampleClassDefs.map((c) => (
                <div className="storage-metric-cell" key={c.id}>
                  <small>{c.short}</small>
                  <b>{classCounts[c.id] ?? 0}</b>
                  <code>{c.id}</code>
                </div>
              ))}
            </div>

            {/* 汇总指标 */}
            <div className="storage-summary-strip">
              <div className="storage-summary-item">
                <small>训练切片 / 独立料管组</small>
                <b className="mono">
                  {ds.total_samples ?? datasetSummary.total_samples ?? 0} 张 /{" "}
                  {ds.independent_tube_groups ?? datasetSummary.independent_tube_groups ?? 0} 组
                </b>
              </div>
              <div className="storage-summary-item">
                <small>COCO 框标注数</small>
                <b className="mono">{ds.coco_annotations ?? 0} 条</b>
              </div>
              <div className="storage-summary-item">
                <small>模型产出目录 / 实时运行</small>
                <b className="mono" title={activeModelVersion || "未加载 ONNX"}>
                  {ma.run_count ?? 0} 版 · {activeModelVersion || "动态引擎"}
                </b>
              </div>
            </div>

            {/* 真实文件与路径 */}
            <div className="storage-path-list">
              <div className="storage-path-row">
                <span>样本目录</span>
                <code>{ds.dataset_dir || "data/dataset"}</code>
                <em>{fmt(ds.dataset_size_mb, 2)} MB</em>
              </div>
              <div className="storage-path-row">
                <span>索引清单</span>
                <code>{ds.manifest_path || "data/dataset/manifest.csv"}</code>
                <em>{ds.coco_annotations ?? 0} COCO</em>
              </div>
              <div className="storage-path-row">
                <span>模型目录</span>
                <code>{ma.runs_dir || "data/runs/label_classifier"}</code>
                <em>{fmt(ma.models_size_mb, 2)} MB</em>
              </div>
            </div>

            {/* 一眼看懂：存什么 / 怎么删 / 怎么销毁 */}
            <div className="storage-logic-list">
              <div className="storage-logic-row">
                <span className="storage-logic-pill">存什么</span>
                <span>5 类标签训练切片、拉框 COCO 标注清单，以及 PyTorch 训练产出的 ONNX 推理权重。</span>
              </div>
              <div className="storage-logic-row">
                <span className="storage-logic-pill">怎么删</span>
                <span>在「2. 采集标注」按类筛选、修改标注、单删或勾选多张样本一键批量删除。</span>
              </div>
              <div className="storage-logic-row">
                <span className="storage-logic-pill">怎么销毁</span>
                <span>在下方清空全部 AI 训练样本，或按版本销毁非运行中的废弃 ONNX 模型目录。</span>
              </div>
            </div>
          </div>

          <footer className="storage-card-foot">
            <button type="button" className="btn ghost sm" onClick={() => onGoStep("collect")}>
              <Database size={13} aria-hidden="true" />
              前往采集标注
            </button>
            <button type="button" className="btn sm" onClick={() => onGoStep("train")}>
              <Cpu size={13} aria-hidden="true" />
              前往训练部署
            </button>
          </footer>
        </section>
      </div>

      {/* 中部：模型版本清理与销毁（当前实时运行模型绿锁保护） */}
      <section className="panel" aria-label="模型版本清理与销毁">
        <header className="panel-head">
          <h2>模型版本清理与销毁</h2>
          <span className="sub mono">
            已登记 {models.length} 版 · 实时引擎：{activeModelVersion || "动态启发式引擎"}
          </span>
          <div className="right">
            <span className="protected-lock-badge">
              <Lock size={11} aria-hidden="true" />
              运行中模型自动受保护，防误删
            </span>
          </div>
        </header>

        <div style={{ overflowX: "auto" }}>
          <table className="grid-table">
            <thead>
              <tr>
                <th>模型名称与版本</th>
                <th>SHA256 校验摘要</th>
                <th>登记时间与备注</th>
                <th>状态</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {models.length > 0 ? (
                models.map((m) => {
                  const isActive = Boolean(activeModelVersion && m.model_version === activeModelVersion)
                  return (
                    <tr key={m.model_id}>
                      <td>
                        <b>{m.model_name}</b>{" "}
                        <span className="mono" style={{ color: "var(--ink-2)", fontSize: 12 }}>
                          {m.model_version}
                        </span>
                      </td>
                      <td className="mono">
                        <span className="inline-flex items-center gap-1.5">
                          <code style={{ fontSize: 11.5 }}>{String(m.sha256 || "").slice(0, 16)}…</code>
                          <button
                            type="button"
                            className="icon-btn"
                            style={{ width: 22, height: 22 }}
                            onClick={() => copyHash(String(m.sha256 || ""))}
                            title="复制完整 SHA256"
                            aria-label={`复制模型 ${m.model_version} SHA256`}
                          >
                            <Copy size={11} aria-hidden="true" />
                          </button>
                        </span>
                      </td>
                      <td className="reason">
                        {dateTimeOf(m.created_at_utc)}
                        {m.notes ? ` · ${m.notes}` : ""}
                      </td>
                      <td>
                        {isActive ? (
                          <span className="protected-lock-badge">
                            <Lock size={11} aria-hidden="true" />
                            运行中·受保护
                          </span>
                        ) : (
                          <span className="tag plain">历史归档</span>
                        )}
                      </td>
                      <td>
                        {isActive ? (
                          <span className="note" style={{ fontSize: 11.5 }}>
                            请先切换其他版本再销毁
                          </span>
                        ) : (
                          <div className="row-actions">
                            <button
                              type="button"
                              className="btn ghost sm"
                              style={{ height: 24, padding: "0 8px", fontSize: 11.5 }}
                              onClick={() => void handleActivateModel(String(m.model_id), String(m.model_version))}
                            >
                              热加载
                            </button>
                            <button
                              type="button"
                              className="btn ghost sm btn-outline-bad"
                              style={{ height: 24, padding: "0 8px", fontSize: 11.5 }}
                              onClick={() => requestDeleteModel(m)}
                            >
                              <Trash2 size={11} aria-hidden="true" />
                              销毁
                            </button>
                          </div>
                        )}
                      </td>
                    </tr>
                  )
                })
              ) : (
                <tr>
                  <td colSpan={5}>
                    <div className="empty" style={{ minHeight: 84 }}>
                      尚未登记训练模型，可前往「3. 训练部署」训练并导出 ONNX
                    </div>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      {/* 底部：危险区 · 数据库批量清理与空间回收（二次确认保护） */}
      <section className="panel danger-panel" aria-label="危险区：数据库批量清理与空间回收">
        <header className="panel-head">
          <h2>
            <ShieldAlert size={15} aria-hidden="true" />
            危险区：数据库批量清理与空间回收（二次确认保护）
          </h2>
          <span className="sub">每项操作均需携带工程师会话令牌并通过显式二次确认，绝不盲目清空</span>
        </header>
        <div className="panel-body">
          <div className="danger-action-grid">
            <div className="danger-action-card">
              <div>
                <h3>1. 仅清理演练假周期</h3>
                <p>
                  仅删除影子演练产生的模拟周期（<code>CYC_*</code>）及对应事件与证据图；完整保留真实产线记录、参数历史与训练集。
                </p>
              </div>
              <button
                type="button"
                className="btn ghost sm btn-outline-bad block"
                onClick={requestPurgeSimulated}
              >
                <Trash2 size={13} aria-hidden="true" />
                仅清理演练假周期
              </button>
            </div>

            <div className="danger-action-card">
              <div>
                <h3>2. 清空全部 AI 训练样本</h3>
                <p>
                  删除 <code>data/dataset/</code> 下全部{" "}
                  <b className="mono">{ds.total_samples ?? datasetSummary.total_samples ?? 0}</b>{" "}
                  张训练切片与标注索引；不影响业务库及已部署模型。
                </p>
              </div>
              <button
                type="button"
                className="btn ghost sm btn-outline-bad block"
                onClick={requestClearTrainingSamples}
              >
                <Trash2 size={13} aria-hidden="true" />
                清空全部 AI 训练样本
              </button>
            </div>

            <div className="danger-action-card severe">
              <div>
                <h3>3. 销毁全部业务周期与证据 + VACUUM</h3>
                <p>
                  永久清空全部检测周期（<b className="mono">{tables.cycles ?? 0}</b> 条）、状态事件与证据快照，并执行 SQLite{" "}
                  <code>VACUUM</code> 立即回收磁盘空间。
                </p>
              </div>
              <button
                type="button"
                className="btn btn-danger sm block"
                onClick={requestDestroyAllBusiness}
              >
                <ShieldAlert size={13} aria-hidden="true" />
                销毁全部业务周期与证据 + VACUUM 压缩空间
              </button>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

function SettingsSubView({
  state,
  config,
  history,
  token,
  mirrorHorizontal,
  togglingMirror,
  flash,
  onSaved,
  onToggleAuxRois,
  onToggleMirror,
  onSwitchWorkMode,
}: {
  state: Data
  config: Data
  history: Data[]
  token: string
  mirrorHorizontal: boolean
  togglingMirror: boolean
  flash: (s: string) => void
  onSaved: () => void
  onToggleAuxRois: (show: boolean) => void
  onToggleMirror: (mirror: boolean) => void
  onSwitchWorkMode: (mode: "debug" | "strict") => void
}) {
  const [rois, setRois] = useState<Data>(() =>
    config.rois && Object.keys(config.rois).length > 0 ? config.rois : defaultStandardRois
  )
  const [width, setWidth] = useState(String(config.camera_width || 1920))
  const [height, setHeight] = useState(String(config.camera_height || 1080))
  const [device, setDevice] = useState(config.camera_device_id || "USB_CAM_00_C920")
  const [both, setBoth] = useState(config.camera_covers_both_ends !== false)
  const [thresholds, setThresholds] = useState<Data>(config.thresholds || {})
  const [target, setTarget] = useState(roiNames[0])
  const [image, setImage] = useState<HTMLImageElement | null>(null)
  const [drag, setDrag] = useState<[number, number] | null>(null)
  const [canvasSize, setCanvasSize] = useState<[number, number]>([960, 540])
  const canvasRefElem = useRef<HTMLCanvasElement | null>(null)

  const activeRois = both ? roiNames : roiNames.slice(0, 3)
  const showAux = Boolean(state.show_auxiliary_rois)
  const workMode = state.work_mode === "strict" ? "strict" : "debug"

  const loadPresetFrame = useCallback((src: string) => {
    const im = new Image()
    im.onload = () => {
      setImage(im)
      setCanvasSize([im.naturalWidth || 1920, im.naturalHeight || 1080])
      setWidth(String(im.naturalWidth || 1920))
      setHeight(String(im.naturalHeight || 1080))
    }
    im.src = src
  }, [])

  const prevMirrorRef = useRef<boolean>(mirrorHorizontal)

  useEffect(() => {
    const im = new Image()
    im.onload = () => {
      setImage(im)
      setCanvasSize([im.naturalWidth || 1920, im.naturalHeight || 1080])
      setWidth(String(im.naturalWidth || 1920))
      setHeight(String(im.naturalHeight || 1080))
    }
    im.onerror = () => loadPresetFrame(tubeYellowImg)
    im.src = `/api/camera/frame.jpg?t=${Date.now()}`
  }, [loadPresetFrame])

  useEffect(() => {
    if (prevMirrorRef.current === mirrorHorizontal) return
    prevMirrorRef.current = mirrorHorizontal
    const timer = window.setTimeout(() => {
      loadPresetFrame(`/api/camera/frame.jpg?t=${Date.now()}`)
    }, 180)
    return () => window.clearTimeout(timer)
  }, [mirrorHorizontal, loadPresetFrame])

  useEffect(() => {
    const canvas = canvasRefElem.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    const W = canvas.width
    const H = canvas.height
    const u = Math.max(1, W / 960)

    ctx.clearRect(0, 0, W, H)
    ctx.fillStyle = "#14171c"
    ctx.fillRect(0, 0, W, H)

    if (image) {
      ctx.drawImage(image, 0, 0, W, H)
    }

    Object.entries(rois).forEach(([k, b]: [string, any]) => {
      if (!Array.isArray(b) || b.length !== 4) return
      const i = roiNames.indexOf(k)
      const color = roiColors[i < 0 ? 0 : i]
      const rx = b[0] * W
      const ry = b[1] * H
      const rw = (b[2] - b[0]) * W
      const rh = (b[3] - b[1]) * H
      const on = target === k

      ctx.fillStyle = `${color}${on ? "26" : "14"}`
      ctx.fillRect(rx, ry, rw, rh)
      ctx.strokeStyle = color
      ctx.lineWidth = (on ? 2.5 : 1.5) * u
      ctx.setLineDash(on ? [] : [6 * u, 4 * u])
      ctx.strokeRect(rx, ry, rw, rh)
      ctx.setLineDash([])

      const text = roiLabels[k] || k
      ctx.font = `600 ${12 * u}px "Geist Variable", "Microsoft YaHei UI", sans-serif`
      const tw = ctx.measureText(text).width + 12 * u
      ctx.fillStyle = color
      ctx.fillRect(rx, ry - 20 * u < 0 ? ry : ry - 20 * u, tw, 20 * u)
      ctx.fillStyle = "#fff"
      ctx.fillText(text, rx + 6 * u, (ry - 20 * u < 0 ? ry : ry - 20 * u) + 14 * u)
    })
  }, [rois, image, target, canvasSize])

  const point = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    return [
      Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)),
      Math.max(0, Math.min(1, (e.clientY - r.top) / r.height)),
    ] as [number, number]
  }

  const loadImage = (f?: File) => {
    if (!f) return
    const url = URL.createObjectURL(f)
    const im = new Image()
    im.onload = () => {
      setImage(im)
      setCanvasSize([im.naturalWidth, im.naturalHeight])
      setWidth(String(im.naturalWidth))
      setHeight(String(im.naturalHeight))
      URL.revokeObjectURL(url)
    }
    im.onerror = () => {
      URL.revokeObjectURL(url)
      flash("无法解析所选图片，请选择有效的 JPEG/PNG 文件")
    }
    im.src = url
  }

  const save = async () => {
    if (!token) return flash("当前工程师会话无效，请重新验证登录")
    const required = both ? roiNames : roiNames.slice(0, 3)
    if (required.some((k) => !rois[k])) {
      return flash(`尚未标定：${required.filter((k) => !rois[k]).map((k) => roiLabels[k]).join("、")}`)
    }
    try {
      await api(
        "/api/config",
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            camera_device_id: device || "CAM_DEFAULT",
            camera_width: Number(width),
            camera_height: Number(height),
            camera_covers_both_ends: both,
            rois: Object.fromEntries(Object.entries(rois).filter(([k]) => required.includes(k))),
            thresholds,
          }),
        },
        token
      )
      flash("参数与五区 ROI 已保存为新版本")
      onSaved()
    } catch (e) {
      flash(`保存失败：${e instanceof Error ? e.message : e}`)
    }
  }

  return (
    <div className="studio-split">
      {/* 左侧固定 16:9 五区 ROI 标定画布（不随右侧检查器滚动丢失） */}
      <section className="studio-canvas" aria-label="五区 ROI 交互式标定画布">
        <header className="studio-canvas-bar">
          <h2>五区 ROI 标定画布</h2>
          <span className="sub">
            正在编辑：<b style={{ color: "var(--ink)" }}>{roiLabels[target]}</b>
          </span>
          <div className="right">
            <button
              type="button"
              className="btn sm"
              onClick={() => loadPresetFrame(`/api/camera/frame.jpg?t=${Date.now()}`)}
            >
              抓取相机帧
            </button>
            <button type="button" className="btn ghost sm" onClick={() => loadPresetFrame(tubeYellowImg)}>
              黄管参考
            </button>
            <button type="button" className="btn ghost sm" onClick={() => loadPresetFrame(tubeBlueImg)}>
              蓝管参考
            </button>
            <button
              type="button"
              className="btn ghost sm"
              onClick={() => {
                setBoth(true)
                setRois(defaultStandardRois)
              }}
            >
              重置五区
            </button>
            <label className="btn ghost sm file-btn">
              <Upload size={13} aria-hidden="true" />
              载入
              <input type="file" accept="image/*" onChange={(e) => loadImage(e.target.files?.[0])} />
            </label>
          </div>
        </header>

        <div className="studio-canvas-body">
          <div className="canvas-16x9 canvas-box">
            <canvas
              ref={canvasRefElem}
              width={canvasSize[0]}
              height={canvasSize[1]}
              onPointerDown={(e) => {
                if (!image) return flash("请先载入参考帧")
                e.currentTarget.setPointerCapture(e.pointerId)
                setDrag(point(e))
              }}
              onPointerUp={(e) => {
                if (!drag) return
                const p = point(e)
                const box = [
                  Math.min(drag[0], p[0]),
                  Math.min(drag[1], p[1]),
                  Math.max(drag[0], p[0]),
                  Math.max(drag[1], p[1]),
                ]
                setRois((r: Data) => ({ ...r, [target]: box }))
                setDrag(null)
              }}
            />
          </div>
        </div>

        <div className="roi-bar" aria-label="选择编辑 ROI 区域">
          {activeRois.map((r, i) => (
            <button key={r} type="button" className={target === r ? "on" : ""} onClick={() => setTarget(r)}>
              <div className="roi-bar-top">
                <i style={{ background: roiColors[i] }} />
                <span>{roiShortLabels[r] || roiLabels[r]}</span>
              </div>
              <code className="mono">
                {Array.isArray(rois[r]) ? rois[r].map((v: number) => v.toFixed(2)).join(" ") : "未标定"}
              </code>
            </button>
          ))}
        </div>
      </section>

      {/* 右侧独立局部滚动参数检查器 */}
      <aside className="studio-inspector" aria-label="相机镜像与系统参数检查器">
        <header className="studio-inspector-head">
          <h2>相机镜像与系统参数</h2>
          <button type="button" className="btn sm" onClick={() => void save()}>
            保存参数
          </button>
        </header>

        <div className="studio-inspector-scroll-wrap">
          <div className="studio-inspector-body">
            {/* 水平镜像 + 运行模式 + 辅助标定框 */}
            <section className="inspector-section grid gap-3">
              <div className="inspector-section-head">
                <h3>画面方向与防呆运行模式</h3>
              </div>

              <div className="toggle-row">
                <div className="toggle-copy">
                  <b>
                    <FlipHorizontal2 size={14} aria-hidden="true" style={{ display: "inline", marginRight: 6, verticalAlign: -2 }} />
                    水平镜像对齐（左右手与现实同步）
                  </b>
                  <small>{mirrorHorizontal ? "已开启：抬真实左手命中画面左侧" : "已关闭：输出原始未翻转相机视角"}</small>
                </div>
                <button
                  type="button"
                  role="switch"
                  aria-checked={mirrorHorizontal}
                  aria-label="水平镜像对齐开关"
                  disabled={togglingMirror}
                  className={`studio-switch ${mirrorHorizontal ? "on" : ""}`}
                  onClick={() => onToggleMirror(!mirrorHorizontal)}
                >
                  <span className="studio-switch-thumb" />
                </button>
              </div>

              <div className="toggle-row">
                <div className="toggle-copy">
                  <b>
                    <Crosshair size={14} aria-hidden="true" style={{ display: "inline", marginRight: 6, verticalAlign: -2 }} />
                    叠加辅助标定框
                  </b>
                  <small>{showAux ? "实时画面已叠加五区参考线" : "纯动态追踪画面，无静态参考框"}</small>
                </div>
                <button
                  type="button"
                  role="switch"
                  aria-checked={showAux}
                  aria-label="叠加辅助标定框开关"
                  className={`studio-switch ${showAux ? "on" : ""}`}
                  onClick={() => onToggleAuxRois(!showAux)}
                >
                  <span className="studio-switch-thumb" />
                </button>
              </div>

              <div className="grid gap-1.5">
                <div className="inspector-sublabel">防呆运行模式</div>
                <div className="sim-bar" style={{ width: "100%", display: "grid", gridTemplateColumns: "1fr 1fr" }}>
                  <button
                    type="button"
                    className={`sim-btn ${workMode === "debug" ? "busy" : ""}`}
                    onClick={() => onSwitchWorkMode("debug")}
                  >
                    桌面调试模式
                  </button>
                  <button
                    type="button"
                    className={`sim-btn ${workMode === "strict" ? "busy" : ""}`}
                    onClick={() => onSwitchWorkMode("strict")}
                  >
                    车间严格模式
                  </button>
                </div>
                <span className="note">
                  {workMode === "strict"
                    ? "强制白色防静电手套合规检测与细长料管入位门控"
                    : "免手套防误锁，适合桌面调试与样本抓拍"}
                </span>
              </div>
            </section>

            {/* 9 项阈值 */}
            <details className="inspector-group" open>
              <summary>
                <span>识别与光学阈值参数</span>
                <span className="mono">9 项</span>
              </summary>
              <div className="inspector-group-body">
                {thresholdFields.map(([key, label, step]) => (
                  <div className="th-row" key={key}>
                    <div>
                      <b>{label}</b>
                    </div>
                    <Input
                      type="number"
                      step={step}
                      value={thresholds[key] ?? ""}
                      placeholder="—"
                      onChange={(e) =>
                        setThresholds((t: Data) => ({
                          ...t,
                          [key]: e.target.value === "" ? null : Number(e.target.value),
                        }))
                      }
                    />
                  </div>
                ))}
              </div>
            </details>

            {/* 硬件绑定 */}
            <details className="inspector-group" open>
              <summary>
                <span>相机硬件绑定</span>
                <span className="mono">
                  {width}×{height}
                </span>
              </summary>
              <div className="inspector-group-body grid gap-3">
                <div className="field">
                  <label>设备 ID</label>
                  <Input value={device} onChange={(e) => setDevice(e.target.value)} className="mono" />
                </div>
                <div className="grid grid-cols-2 gap-2.5">
                  <div className="field">
                    <label>采集宽度</label>
                    <Input type="number" value={width} onChange={(e) => setWidth(e.target.value)} className="mono" />
                  </div>
                  <div className="field">
                    <label>采集高度</label>
                    <Input type="number" value={height} onChange={(e) => setHeight(e.target.value)} className="mono" />
                  </div>
                </div>
                <label className="check">
                  <input type="checkbox" checked={both} onChange={(e) => setBoth(e.target.checked)} />
                  单相机覆盖料管左右双端握持区（五区标定）
                </label>
                <button type="button" className="btn block" onClick={() => void save()}>
                  保存参数为新版本
                </button>
              </div>
            </details>

            {/* 版本历史 */}
            <details className="inspector-group">
              <summary>
                <span>参数变更历史</span>
                <span className="mono">{history.length} 条</span>
              </summary>
              <div className="inspector-group-body" style={{ padding: 0 }}>
                {history.length ? (
                  <ol className="timeline" style={{ maxHeight: 240 }}>
                    {history.slice(0, 10).map((h) => (
                      <li key={h.config_version}>
                        <div className="row">
                          <b className="mono">#{h.config_version}</b>
                          <time>{dateTimeOf(h.created_at_utc)}</time>
                        </div>
                        <div className="flow mono">{h.roi_config_version}</div>
                      </li>
                    ))}
                  </ol>
                ) : (
                  <div className="empty" style={{ minHeight: 72 }}>
                    暂无变更记录
                  </div>
                )}
              </div>
            </details>
          </div>
        </div>
      </aside>
    </div>
  )
}

export default App
