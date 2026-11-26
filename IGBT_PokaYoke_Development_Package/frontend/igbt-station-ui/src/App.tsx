/* eslint-disable @typescript-eslint/no-explicit-any -- 机器视觉 API 响应已定义在后端模式中 */
import { useCallback, useEffect, useRef, useState } from "react"
import { AlertDialog, Dialog } from "radix-ui"
import {
  AlertTriangle,
  Camera,
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
  Lock,
  LogOut,
  Moon,
  Pin,
  Play,
  RefreshCw,
  RotateCcw,
  ShieldAlert,
  ShieldCheck,
  Sliders,
  Sparkles,
  Square,
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
import { RedRabbitOverlay } from "@/features/red-rabbit/RedRabbitOverlay"
import type { RedRabbitStatus } from "@/features/red-rabbit/types"
import {
  C_LIMIT_MS,
  type Data,
  cameraText,
  dateTimeOf,
  fmt,
  formatCTimerSeconds,
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
type AutoCaptureLabel = "complete" | "crossed_out" | "covered" | "absent"
type SampleFilter = "" | SampleLabel

interface AutoCaptureState {
  enabled: boolean
  every_n_frames: number
  label: AutoCaptureLabel
  pipe_color: string
  physical_tube_id: string
  max_samples: number
  captured_count: number
}

interface ConfirmDialogState {
  title: string
  description: string
  consequences?: string[]
  confirmLabel: string
  tone: "warn" | "danger"
  requireCheckLabel?: string
  requiresInventory?: boolean
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
const DEFAULT_CAMERA_WIDTH = 1280
const DEFAULT_CAMERA_HEIGHT = 720

function parseAutoCapture(raw: unknown): AutoCaptureState | null {
  if (!raw || typeof raw !== "object") return null
  const src =
    "auto_capture" in (raw as Record<string, unknown>) &&
    typeof (raw as Record<string, unknown>).auto_capture === "object" &&
    (raw as Record<string, unknown>).auto_capture !== null
      ? ((raw as Record<string, unknown>).auto_capture as Record<string, unknown>)
      : (raw as Record<string, unknown>)
  if (typeof src.enabled !== "boolean" && src.captured_count === undefined && src.every_n_frames === undefined) {
    return null
  }
  const rawLabel = String(src.label || "complete")
  const validLabel: AutoCaptureLabel =
    rawLabel === "crossed_out" || rawLabel === "covered" || rawLabel === "absent" ? rawLabel : "complete"
  return {
    enabled: Boolean(src.enabled),
    every_n_frames: Math.max(1, Number(src.every_n_frames) || 10),
    label: validLabel,
    pipe_color: String(src.pipe_color || "yellow"),
    physical_tube_id: String(src.physical_tube_id || "UNGROUPED"),
    max_samples: Math.max(1, Number(src.max_samples) || 200),
    captured_count: Math.max(0, Number(src.captured_count) || 0),
  }
}

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
    /* 忽略隐私模式 sessionStorage 异常 */
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
    /* 忽略 sessionStorage 异常 */
  }
}

function clearStoredSession(): void {
  try {
    window.sessionStorage.removeItem(ENGINEER_SESSION_KEY)
  } catch {
    /* 忽略 sessionStorage 异常 */
  }
}

const roiNames = ["tube_roi", "label_roi", "label_contact_roi", "left_grip_zone", "right_grip_zone"]
const roiLabels: Record<string, string> = {
  tube_roi: "料管整体",
  label_roi: "防呆标签",
  label_contact_roi: "标签接触",
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
  { id: "a_pass", label: "划除放行" },
  { id: "b_pass", label: "遮挡放行" },
  { id: "c_pass", label: "脱落放行" },
  { id: "alarm", label: "未处理报警", tone: "bad" },
  { id: "glare_invalid", label: "反光受阻", tone: "warn" },
  { id: "reset", label: "恢复实时" },
]

const thresholdFields: [string, string, string][] = [
  ["hand_glove_confidence", "手套置信", "0.01"],
  ["label_min_confidence", "分类置信", "0.01"],
  ["label_min_margin", "置信差值", "0.01"],
  ["stable_state_ms", "稳定毫秒", "1"],
  ["min_stable_frames", "稳定帧数", "1"],
  ["min_laplacian_variance", "清晰下限", "0.1"],
  ["min_brightness", "亮度下限", "1"],
  ["max_brightness", "亮度上限", "1"],
  ["max_overexposed_fraction", "反光上限", "0.01"],
]

const engineerSteps: { id: EngineerStep; label: string; icon: typeof Camera }[] = [
  { id: "monitor", label: "实时监控", icon: Camera },
  { id: "collect", label: "采集标注", icon: Database },
  { id: "train", label: "训练部署", icon: Cpu },
  { id: "records_settings", label: "记录设置", icon: Sliders },
]

const sampleClassDefs: {
  id: SampleLabel
  title: string
  short: string
}[] = [
  { id: "complete", title: "抓拍 · 完整", short: "完整" },
  { id: "crossed_out", title: "抓拍 · 已划除", short: "已划除" },
  { id: "covered", title: "抓拍 · 已遮挡", short: "已遮挡" },
  { id: "absent", title: "抓拍 · 已脱落", short: "已脱落" },
  { id: "unknown_challenge", title: "抓拍 · 挑战", short: "挑战" },
]

const sampleFilterOptions: { id: SampleFilter; label: string }[] = [
  { id: "", label: "全部" },
  { id: "complete", label: "完整" },
  { id: "crossed_out", label: "已划除" },
  { id: "covered", label: "已遮挡" },
  { id: "absent", label: "已脱落" },
  { id: "unknown_challenge", label: "挑战" },
]

function toHeaderValue(raw: string, fallback: string): string {
  const trimmed = raw.trim()
  if (!trimmed) return fallback
  if (/^[\x20-\x7E]+$/.test(trimmed)) return trimmed
  return encodeURIComponent(trimmed)
}

class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.name = "ApiError"
    this.status = status
  }
}

class RequestTimeoutError extends Error {
  constructor(message = "请求超时") {
    super(message)
    this.name = "RequestTimeoutError"
  }
}

const POLL_TIMEOUT_MS = 4000
const MIN_ROI_SPAN = 0.005

function isValidRoiBox(box: unknown): box is [number, number, number, number] {
  if (!Array.isArray(box) || box.length !== 4) return false
  const [x1, y1, x2, y2] = box.map(Number)
  if (![x1, y1, x2, y2].every(Number.isFinite)) return false
  return x1 >= 0 && y1 >= 0 && x2 <= 1 && y2 <= 1 && x2 - x1 >= MIN_ROI_SPAN && y2 - y1 >= MIN_ROI_SPAN
}

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  if (target.isContentEditable) return true
  return Boolean(target.closest('input, textarea, select, [contenteditable="true"]'))
}

function isInteractiveTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  if (target.isContentEditable) return true
  return Boolean(
    target.closest(
      'button, a[href], input, select, textarea, summary, details, [contenteditable="true"], [role="button"], [role="link"], [role="tab"], [role="switch"], [role="checkbox"], [role="radio"], [role="menuitem"], [role="combobox"], [role="listbox"], [role="option"], [role="slider"]'
    )
  )
}

function readStoredThemePreference(): "light" | "dark" {
  const q = new URLSearchParams(window.location.search).get("theme")
  if (q === "dark" || q === "light") return q
  try {
    return window.localStorage.getItem("igbt-ui-theme") === "dark" ? "dark" : "light"
  } catch {
    return "light"
  }
}

function writeStoredThemePreference(nextTheme: "light" | "dark"): void {
  try {
    window.localStorage.setItem("igbt-ui-theme", nextTheme)
  } catch {
    /* 保留内存状态 */
  }
}

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError"
}

async function api<T = any>(
  path: string,
  options: RequestInit = {},
  token = "",
  timeoutMs?: number
): Promise<T> {
  const headers = new Headers(options.headers)
  if (token) headers.set("X-Admin-Token", token)

  let controller: AbortController | null = null
  let timeoutId: number | null = null
  let onExternalAbort: (() => void) | null = null
  let timedOut = false
  const externalSignal = options.signal

  if (timeoutMs && timeoutMs > 0) {
    controller = new AbortController()
    if (externalSignal) {
      if (externalSignal.aborted) {
        controller.abort()
      } else {
        onExternalAbort = () => controller?.abort()
        externalSignal.addEventListener("abort", onExternalAbort, { once: true })
      }
    }
    timeoutId = window.setTimeout(() => {
      timedOut = true
      controller?.abort()
    }, timeoutMs)
  }

  try {
    const response = await fetch(path, {
      ...options,
      headers,
      signal: controller ? controller.signal : externalSignal,
    })
    if (!response.ok) {
      let reason = `HTTP ${response.status}`
      try {
        reason = (await response.json()).detail || reason
      } catch (parseErr) {
        if (timedOut && !externalSignal?.aborted) {
          throw new RequestTimeoutError()
        }
        if (externalSignal?.aborted || isAbortError(parseErr)) {
          throw parseErr
        }
      }
      throw new ApiError(String(reason), response.status)
    }
    return (await response.json()) as T
  } catch (err) {
    if (timedOut && !externalSignal?.aborted) {
      throw new RequestTimeoutError()
    }
    throw err
  } finally {
    if (timeoutId !== null) window.clearTimeout(timeoutId)
    if (externalSignal && onExternalAbort) {
      externalSignal.removeEventListener("abort", onExternalAbort)
    }
  }
}

function getClassCounts(ds: Data): Record<string, number> {
  return ds?.class_counts || ds?.counts_by_class || {}
}

function isDatasetReady(ds: Data): boolean {
  return Boolean(ds?.ready_to_train ?? ds?.ready_for_training)
}

function App() {
  const [showShowcase, setShowShowcase] = useState(() => {
    const sp = new URLSearchParams(window.location.search)
    return sp.get("showcase") === "1" || sp.get("page") === "showcase"
  })

  const [engineerSession, setEngineerSession] = useState<StoredEngineerSession | null>(() => readStoredSession())
  const role: Role = engineerSession ? "engineer" : "operator"
  const adminToken = engineerSession?.token || ""

  const [step, setStep] = useState<EngineerStep>("monitor")
  const [step4Tab, setStep4Tab] = useState<RecordsSettingsTab>("records")
  const [authModalOpen, setAuthModalOpen] = useState(false)
  const loginTriggerRef = useRef<HTMLButtonElement | null>(null)
  const engineerEntryFocusRef = useRef<HTMLButtonElement | null>(null)
  const shouldFocusEngineerRef = useRef(false)
  const lastActiveRef = useRef<number>(0)

  const [theme, setTheme] = useState<"light" | "dark">(() => readStoredThemePreference())

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    writeStoredThemePreference(theme)
  }, [theme])

  const [currentTime, setCurrentTime] = useState(new Date())
  const [state, setState] = useState<Data>({
    camera_status: "NOT_CONNECTED",
    camera_width: DEFAULT_CAMERA_WIDTH,
    camera_height: DEFAULT_CAMERA_HEIGHT,
    fps: 25,
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
  const [config, setConfig] = useState<Data>({
    camera_width: DEFAULT_CAMERA_WIDTH,
    camera_height: DEFAULT_CAMERA_HEIGHT,
    rois: {},
    thresholds: {},
  })
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
    message: "待启动",
    progress_pct: 0,
    history: [],
  })
  const [trainingError, setTrainingError] = useState("")
  const [storageOverview, setStorageOverview] = useState<Data | null>(null)
  const [storageLoading, setStorageLoading] = useState(false)
  const [storageError, setStorageError] = useState("")
  const [confirmDialog, setConfirmDialog] = useState<ConfirmDialogState | null>(null)
  const [rrAuditOpen, setRrAuditOpen] = useState(false)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState("")
  const [selectedCycleId, setSelectedCycleId] = useState<string>("")
  const [cycleTimeline, setCycleTimeline] = useState<{
    cycleId: string
    events: Data[]
    loading: boolean
    error: string
  }>({ cycleId: "", events: [], loading: false, error: "" })
  const [notice, setNotice] = useState("")
  const [refreshing, setRefreshing] = useState(false)
  const [simulating, setSimulating] = useState("")
  const [togglingMirror, setTogglingMirror] = useState(false)
  const [captureShutterSeq, setCaptureShutterSeq] = useState(0)
  const [activeSnapshot, setActiveSnapshot] = useState<FilmstripItem | null>(null)
  const noticeTimerRef = useRef<number | null>(null)
  const refreshTimerRef = useRef<number | null>(null)

  const roleRef = useRef<Role>(role)
  const adminTokenRef = useRef<string>(adminToken)
  const selectedCycleRef = useRef<string>(selectedCycleId)
  const refreshSeqRef = useRef(0)
  const manualRefreshSeqRef = useRef<number | null>(null)
  const scheduledInFlightRef = useRef(false)
  const refreshAbortRef = useRef<AbortController | null>(null)
  const eventsReqSeqRef = useRef(0)
  const eventsAbortRef = useRef<AbortController | null>(null)
  const eventsInFlightRef = useRef<{ cycleId: string; explicit: boolean } | null>(null)
  const storageReqSeqRef = useRef(0)
  const storageAbortRef = useRef<AbortController | null>(null)
  const hasInventoryRef = useRef(false)
  const confirmDialogRef = useRef<ConfirmDialogState | null>(confirmDialog)

  useEffect(() => {
    roleRef.current = role
  }, [role])
  useEffect(() => {
    adminTokenRef.current = adminToken
  }, [adminToken])
  useEffect(() => {
    selectedCycleRef.current = selectedCycleId
  }, [selectedCycleId])
  useEffect(() => {
    confirmDialogRef.current = confirmDialog
  }, [confirmDialog])
  useEffect(() => {
    hasInventoryRef.current = Boolean(
      storageOverview?.business_db && storageOverview.business_db.tables && !storageError
    )
  }, [storageOverview, storageError])

  const isBusinessInventoryAvailable = useCallback(() => hasInventoryRef.current, [])

  useEffect(() => {
    if (role !== "engineer" || !shouldFocusEngineerRef.current) return
    shouldFocusEngineerRef.current = false
    const raf = window.requestAnimationFrame(() => {
      engineerEntryFocusRef.current?.focus()
    })
    return () => window.cancelAnimationFrame(raf)
  }, [role])

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
      refreshAbortRef.current?.abort()
      eventsAbortRef.current?.abort()
      storageAbortRef.current?.abort()
    }
  }, [])

  const clearSelectedCycle = useCallback(() => {
    selectedCycleRef.current = ""
    setSelectedCycleId("")
    eventsReqSeqRef.current += 1
    eventsInFlightRef.current = null
    eventsAbortRef.current?.abort()
    eventsAbortRef.current = null
    setCycleTimeline({ cycleId: "", events: [], loading: false, error: "" })
  }, [])

  const lockEngineerSession = useCallback(
    (reason?: string) => {
      clearStoredSession()
      storageReqSeqRef.current += 1
      storageAbortRef.current?.abort()
      storageAbortRef.current = null
      hasInventoryRef.current = false
      setEngineerSession(null)
      setStep("monitor")
      setActiveSnapshot(null)
      setConfirmDialog(null)
      setStorageOverview(null)
      setStorageLoading(false)
      setStorageError("")
      setTrainingError("")
      clearSelectedCycle()
      if (reason) flash(reason)
    },
    [clearSelectedCycle, flash]
  )

  useEffect(() => {
    if (!engineerSession) return

    lastActiveRef.current = Math.max(lastActiveRef.current, engineerSession.lastActiveAt || Date.now())

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
      const now = Date.now()
      const stored = readStoredSession()
      if (stored && stored.token === engineerSession.token && stored.lastActiveAt > lastActiveRef.current) {
        lastActiveRef.current = stored.lastActiveAt
      }
      if (now - lastActiveRef.current >= engineerSession.idleTimeoutMs || now >= engineerSession.expiresAt) {
        lockEngineerSession("会话超时锁定")
      }
    }, 15_000)

    return () => {
      activityEvents.forEach((ev) => window.removeEventListener(ev, markActivity))
      window.clearInterval(checkInterval)
    }
  }, [engineerSession, lockEngineerSession])

  const loadCycleEvents = useCallback(
    async (
      id: string,
      opts: { silent?: boolean; onlyIfNoExplicitSelection?: boolean } = {}
    ) => {
      if (!id) {
        if (!selectedCycleRef.current) {
          eventsReqSeqRef.current += 1
          eventsInFlightRef.current = null
          eventsAbortRef.current?.abort()
          eventsAbortRef.current = null
          setCycleTimeline({ cycleId: "", events: [], loading: false, error: "" })
        }
        return
      }
      if (opts.silent && eventsInFlightRef.current) {
        if (eventsInFlightRef.current.explicit || eventsInFlightRef.current.cycleId === id) {
          return
        }
      }

      const reqSeq = ++eventsReqSeqRef.current
      eventsAbortRef.current?.abort()
      const controller = new AbortController()
      eventsAbortRef.current = controller
      eventsInFlightRef.current = { cycleId: id, explicit: !opts.silent }

      if (!opts.silent) {
        setCycleTimeline((prev) => ({
          cycleId: id,
          events: prev.cycleId === id ? prev.events : [],
          loading: true,
          error: "",
        }))
      }

      try {
        const list = await api<Data[]>(
          `/api/cycles/${encodeURIComponent(id)}/events?limit=100`,
          { signal: controller.signal },
          "",
          POLL_TIMEOUT_MS
        )
        if (reqSeq !== eventsReqSeqRef.current || controller.signal.aborted) return
        if (opts.onlyIfNoExplicitSelection && selectedCycleRef.current) return
        if (selectedCycleRef.current && selectedCycleRef.current !== id) return
        setCycleTimeline({ cycleId: id, events: list, loading: false, error: "" })
      } catch (err) {
        if (reqSeq !== eventsReqSeqRef.current || controller.signal.aborted || isAbortError(err)) return
        if (opts.onlyIfNoExplicitSelection && selectedCycleRef.current) return
        if (selectedCycleRef.current && selectedCycleRef.current !== id) return
        const msg = err instanceof Error ? err.message : String(err)
        if (!opts.silent) {
          setCycleTimeline((prev) => ({
            cycleId: id,
            events: prev.cycleId === id ? prev.events : [],
            loading: false,
            error: msg,
          }))
          flash(`加载失败：${msg}`)
        }
      } finally {
        if (eventsAbortRef.current === controller) {
          eventsAbortRef.current = null
        }
        if (reqSeq === eventsReqSeqRef.current) {
          eventsInFlightRef.current = null
        }
      }
    },
    [flash]
  )

  const openEvents = useCallback(
    async (id: string) => {
      selectedCycleRef.current = id
      setSelectedCycleId(id)
      await loadCycleEvents(id, { silent: false })
    },
    [loadCycleEvents]
  )

  const refresh = useCallback(
    async (forceEngineerFetch = false, scheduled = false) => {
      if (scheduled && (scheduledInFlightRef.current || manualRefreshSeqRef.current !== null)) return
      if (!scheduled) {
        refreshAbortRef.current?.abort()
      }

      const seq = ++refreshSeqRef.current
      if (scheduled) {
        scheduledInFlightRef.current = true
      } else {
        manualRefreshSeqRef.current = seq
        if (refreshTimerRef.current !== null) {
          window.clearTimeout(refreshTimerRef.current)
          refreshTimerRef.current = null
        }
        setRefreshing(true)
      }

      const controller = new AbortController()
      refreshAbortRef.current = controller
      const isEngineer = forceEngineerFetch || roleRef.current === "engineer"
      const isCurrent = () => seq === refreshSeqRef.current && !controller.signal.aborted

      try {
        const signalOpt: RequestInit = { signal: controller.signal }

        const statePromise = api<Data>("/api/state", signalOpt, "", POLL_TIMEOUT_MS).then(
          (s) => {
            if (!isCurrent()) return undefined
            setState(s)
            setConnected(true)
            setError("")
            return s
          },
          (err) => {
            if (!isCurrent() || isAbortError(err)) return undefined
            setConnected(false)
            setError(err instanceof Error ? err.message : "服务未连接")
            setState({
              camera_status: "FAULT",
              camera_width: DEFAULT_CAMERA_WIDTH,
              camera_height: DEFAULT_CAMERA_HEIGHT,
              fps: 0,
              view_valid: false,
              label_status: "unknown",
              fsm_state: "FAULT",
              outcome: "FAULT",
              c_timer_ms: 0,
              mirror_horizontal: true,
              view_valid_reasons: ["API_UNAVAILABLE"],
              result_reason: "API_UNAVAILABLE_FAIL_CLOSED",
            })
            return undefined
          }
        )

        const cyclesPromise = api<Data[]>("/api/cycles?limit=30", signalOpt, "", POLL_TIMEOUT_MS).then(
          (c) => {
            if (!isCurrent()) return undefined
            setCycles(c)
            return c
          },
          () => undefined
        )

        const evidencePromise = api<Data[]>("/api/evidence?limit=40", signalOpt, "", POLL_TIMEOUT_MS).then(
          (e) => {
            if (isCurrent()) setEvidence(e)
          },
          () => {}
        )

        const tasks: Promise<unknown>[] = [statePromise, cyclesPromise, evidencePromise]

        if (isEngineer) {
          tasks.push(
            api<Data>("/api/config", signalOpt, "", POLL_TIMEOUT_MS).then(
              (cfg) => {
                if (isCurrent()) setConfig(cfg)
              },
              () => {}
            ),
            api<Data[]>("/api/config/history?limit=20", signalOpt, "", POLL_TIMEOUT_MS).then(
              (h) => {
                if (isCurrent()) setHistory(h)
              },
              () => {}
            ),
            api<Data[]>("/api/models?limit=30", signalOpt, "", POLL_TIMEOUT_MS).then(
              (m) => {
                if (isCurrent()) setModels(m)
              },
              () => {}
            ),
            api<Data>("/api/dataset/summary", signalOpt, "", POLL_TIMEOUT_MS).then(
              (ds) => {
                if (isCurrent()) setDatasetSummary(ds)
              },
              () => {}
            ),
            api<Data>("/api/training/status", signalOpt, "", POLL_TIMEOUT_MS).then(
              (tr) => {
                if (!isCurrent()) return
                setTrainingStatus(tr)
                setTrainingError("")
              },
              (err) => {
                if (!isCurrent() || isAbortError(err)) return
                setTrainingError(err instanceof Error ? err.message : "训练状态获取失败")
              }
            ),
            Promise.all([statePromise, cyclesPromise]).then(([latestState, latestCycles]) => {
              if (!isCurrent()) return
              const explicitId = selectedCycleRef.current
              const fallbackId = latestState?.cycle_id || latestCycles?.[0]?.cycle_id
              const targetCycleId = explicitId || fallbackId
              if (targetCycleId) {
                void loadCycleEvents(String(targetCycleId), {
                  silent: true,
                  onlyIfNoExplicitSelection: !explicitId,
                })
              } else if (!explicitId) {
                void loadCycleEvents("")
              }
            })
          )
        }

        await Promise.allSettled(tasks)
      } finally {
        if (scheduled) {
          scheduledInFlightRef.current = false
        }
        if (refreshAbortRef.current === controller) {
          refreshAbortRef.current = null
        }
        if (!scheduled && manualRefreshSeqRef.current === seq) {
          manualRefreshSeqRef.current = null
          if (refreshTimerRef.current !== null) {
            window.clearTimeout(refreshTimerRef.current)
          }
          refreshTimerRef.current = window.setTimeout(() => {
            if (manualRefreshSeqRef.current === null) {
              setRefreshing(false)
            }
            refreshTimerRef.current = null
          }, 220)
        }
      }
    },
    [loadCycleEvents]
  )

  const loadStorageOverview = useCallback(async () => {
    if (roleRef.current !== "engineer") return
    const sessionToken = adminTokenRef.current
    if (!sessionToken) return

    const reqSeq = ++storageReqSeqRef.current
    storageAbortRef.current?.abort()
    const controller = new AbortController()
    storageAbortRef.current = controller

    setStorageLoading(true)
    setStorageError("")

    const isLatestForSession = () =>
      reqSeq === storageReqSeqRef.current &&
      roleRef.current === "engineer" &&
      adminTokenRef.current === sessionToken &&
      !controller.signal.aborted

    const invalidateInventoryDialog = () => {
      hasInventoryRef.current = false
      if (confirmDialogRef.current?.requiresInventory) {
        setConfirmDialog((cur) => (cur?.requiresInventory ? null : cur))
        flash("统计已失效，请刷新存储统计后重试")
      }
    }

    try {
      const ov = await api<Data>(
        "/api/storage/overview",
        { signal: controller.signal },
        sessionToken,
        POLL_TIMEOUT_MS
      )
      if (!isLatestForSession()) return
      if (ov?.business_db && ov.business_db.tables) {
        hasInventoryRef.current = true
        setStorageOverview(ov)
        setStorageError("")
      } else {
        setStorageOverview(null)
        setStorageError("统计数据不完整")
        invalidateInventoryDialog()
      }
    } catch (err) {
      if (!isLatestForSession() || isAbortError(err)) return
      setStorageOverview(null)
      setStorageError(err instanceof Error ? err.message : "统计获取失败")
      invalidateInventoryDialog()
    } finally {
      if (storageAbortRef.current === controller) {
        storageAbortRef.current = null
      }
      if (
        reqSeq === storageReqSeqRef.current &&
        roleRef.current === "engineer" &&
        adminTokenRef.current === sessionToken
      ) {
        setStorageLoading(false)
      }
    }
  }, [flash])

  const handleAuthenticated = useCallback(
    ({ adminToken: verifiedToken, expiresInSeconds }: EngineerAuthResult) => {
      const stored = saveStoredSession(verifiedToken, expiresInSeconds)
      lastActiveRef.current = stored.lastActiveAt
      adminTokenRef.current = stored.token
      roleRef.current = "engineer"
      shouldFocusEngineerRef.current = true
      setEngineerSession(stored)
      setStep("monitor")
      setAuthModalOpen(false)
      flash("已进入工程师工作台")
      void refresh(true)
    },
    [flash, refresh]
  )

  const captureLiveFrame = useCallback(async () => {
    try {
      const res = await api<Data>("/api/camera/capture", { method: "POST" }, adminTokenRef.current)
      setCaptureShutterSeq((n) => n + 1)
      flash(`已抓拍归档 (${res.cycle_id})`)
    } catch (err) {
      flash(`抓拍失败：${err instanceof Error ? err.message : String(err)}`)
    } finally {
      await refresh()
    }
  }, [flash, refresh])

  const toggleAuxRois = useCallback(
    async (show: boolean) => {
      try {
        await api<Data>(
          "/api/camera/auxiliary_rois",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ show }),
          },
          adminTokenRef.current
        )
        flash(show ? "已显示辅助框" : "已隐藏辅助框")
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
        const res = await api<Data>(
          "/api/camera/mirror",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mirror }),
          },
          adminTokenRef.current
        )
        const nextMirror = res.mirror_horizontal !== false
        setState((prev) => ({ ...prev, mirror_horizontal: nextMirror }))
        flash(nextMirror ? "镜像已开启" : "镜像已关闭")
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
        await api<Data>(
          "/api/camera/work_mode",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ work_mode: mode }),
          },
          adminTokenRef.current
        )
        flash(mode === "debug" ? "已切换桌面调试" : "已切换车间严格")
        await refresh(true)
      } catch (err) {
        flash(`切换失败：${err instanceof Error ? err.message : err}`)
      }
    },
    [flash, refresh]
  )

  const simulateScenario = useCallback(
    async (scenario: string) => {
      setSimulating(scenario)
      try {
        await api<Data>(
          `/api/runtime/simulate/${encodeURIComponent(scenario)}`,
          { method: "POST" },
          adminTokenRef.current
        )
        await refresh()
      } catch (e) {
        if (
          scenario === "reset" &&
          roleRef.current === "operator" &&
          e instanceof ApiError &&
          (e.status === 401 || e.status === 403)
        ) {
          flash("报警复位需工程师登录")
        } else {
          flash(`执行失败：${e instanceof Error ? e.message : e}`)
        }
      } finally {
        setSimulating("")
      }
    },
    [flash, refresh]
  )

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (showShowcase || e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey) return
      if (
        authModalOpen ||
        confirmDialog ||
        activeSnapshot ||
        document.querySelector('[role="dialog"], [role="alertdialog"]')
      ) {
        return
      }
      if (e.code === "Space") {
        if (e.repeat || isInteractiveTarget(e.target)) return
        e.preventDefault()
        void captureLiveFrame()
      } else if (e.key.toLowerCase() === "t") {
        if (e.repeat || isEditableTarget(e.target)) return
        setTheme((cur) => (cur === "light" ? "dark" : "light"))
      }
    }
    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [showShowcase, authModalOpen, confirmDialog, activeSnapshot, captureLiveFrame])

  useEffect(() => {
    const first = window.setTimeout(() => void refresh(false, true), 0)
    const id = window.setInterval(() => void refresh(false, true), 750)
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
      reason: c.result_reason,
    }
  })

  const allSummary =
    state.cycle_summary_all &&
    typeof state.cycle_summary_all === "object" &&
    typeof state.cycle_summary_all.total === "number" &&
    Number.isFinite(state.cycle_summary_all.total)
      ? state.cycle_summary_all
      : null
  const statsScope: "all" | "recent" = allSummary ? "all" : "recent"
  const totalCount = allSummary ? Number(allSummary.total) : cycles.length
  const passCount = allSummary
    ? Number(allSummary.pass || 0)
    : cycles.filter((c) => c.final_outcome === "PASS").length
  const alarmCount = allSummary
    ? Number(allSummary.alarm || 0) + Number(allSummary.fault || 0)
    : cycles.filter((c) => c.final_outcome === "ALARM" || c.final_outcome === "FAULT").length

  const durations = cycles
    .filter((c) => c.final_outcome && c.created_at_utc && c.updated_at_utc)
    .map((c) => (Date.parse(c.updated_at_utc) - Date.parse(c.created_at_utc)) / 1000)
    .filter((d) => Number.isFinite(d) && d > 0 && d < 120)
  const avgTakt = durations.length ? durations.reduce((a, b) => a + b, 0) / durations.length : null

  const displayedCycleId = selectedCycleId || cycleTimeline.cycleId
  const displayedEvents = displayedCycleId && cycleTimeline.cycleId === displayedCycleId ? cycleTimeline.events : []
  const timelineLoading =
    Boolean(selectedCycleId && cycleTimeline.cycleId !== selectedCycleId) ||
    (cycleTimeline.cycleId === displayedCycleId && cycleTimeline.loading)
  const timelineError = cycleTimeline.cycleId === displayedCycleId ? cycleTimeline.error : ""

  const mirrorHorizontal = state.mirror_horizontal !== false
  const workMode = state.work_mode === "strict" ? "strict" : "debug"
  const isTraining = trainingStatus.state === "TRAINING" || trainingStatus.state === "EXPORTING_ONNX"
  const camW = Number(state.camera_width) || DEFAULT_CAMERA_WIDTH
  const camH = Number(state.camera_height) || DEFAULT_CAMERA_HEIGHT
  const redRabbit = (state.red_rabbit as RedRabbitStatus | undefined) ?? null
  const rrState = redRabbit?.state ?? "PROD_READY"
  const rrBadgeTone = rrState === "CHALLENGE_HARD_LOCK" ? "bad" : rrState === "CHALLENGE_REQUIRED" ? "warn" : "ok"
  const rrBadgeLabel = rrState === "CHALLENGE_HARD_LOCK" ? "红兔硬锁" : rrState === "CHALLENGE_REQUIRED" ? "红兔待检" : "红兔合格"

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
            <h1>晶界智检</h1>
          </div>
          <div className="spacer" />
          <div className="meta" title="Red Rabbit 换班挑战状态">
            {rrState === "PROD_READY" ? (
              <ShieldCheck size={16} aria-hidden="true" />
            ) : (
              <ShieldAlert size={16} aria-hidden="true" />
            )}
            <span className={`lamp ${rrBadgeTone}`} aria-hidden="true" />
            <span>{rrBadgeLabel}</span>
          </div>
          <div className="meta">
            <span className={`lamp ${connected ? "ok" : "bad"}`} aria-hidden="true" />
            <span>{connected ? "在线" : "离线"}</span>
          </div>
          <span className="clock">{currentTime.toLocaleTimeString("zh-CN", { hour12: false })}</span>
          <button
            type="button"
            className="icon-btn"
            aria-label={theme === "light" ? "切换深色模式" : "切换浅色模式"}
            title={theme === "light" ? "深色模式" : "浅色模式"}
            onClick={() => setTheme((cur) => (cur === "light" ? "dark" : "light"))}
          >
            {theme === "light" ? <Moon size={16} aria-hidden="true" /> : <Sun size={16} aria-hidden="true" />}
          </button>
          <button
            ref={loginTriggerRef}
            type="button"
            className="btn sm"
            onClick={() => setAuthModalOpen(true)}
          >
            <Lock size={16} aria-hidden="true" />
            工程师登录
          </button>
        </header>
      ) : (
        <header className="topbar studio-topbar">
          <div className="op-brand studio-brand">
            <span className="rail-brand-mark" aria-hidden="true" />
            <h1>晶界智检</h1>
          </div>

          <nav className="studio-stepper" aria-label="工程师工作流">
            {engineerSteps.map((item) => {
              const Icon = item.icon
              const active = step === item.id
              const badge =
                item.id === "collect"
                  ? `${datasetSummary.total_samples || 0}`
                  : item.id === "train"
                    ? isTraining
                      ? `${trainingStatus.progress_pct || 0}%`
                      : `${models.length}`
                    : item.id === "records_settings" && totalCount > 0
                      ? `${totalCount}`
                      : null

              return (
                <button
                  key={item.id}
                  ref={item.id === "monitor" ? engineerEntryFocusRef : undefined}
                  type="button"
                  className={`studio-step-btn ${active ? "on" : ""}`}
                  aria-current={active ? "step" : undefined}
                  onClick={() => setStep(item.id)}
                >
                  <Icon size={16} aria-hidden="true" />
                  <span>{item.label}</span>
                  {badge !== null && <span className="studio-step-badge mono">{badge}</span>}
                </button>
              )
            })}
          </nav>

          <div className="spacer" />

          <span className="tag plain mono">C++20 · 720P 25FPS</span>
          <button
            type="button"
            className="btn ghost sm"
            title="查看红兔点检与 SHA-256 哈希链审计"
            onClick={() => setRrAuditOpen(true)}
          >
            {rrState === "PROD_READY" ? (
              <ShieldCheck size={16} aria-hidden="true" />
            ) : (
              <ShieldAlert size={16} aria-hidden="true" />
            )}
            红兔审计
          </button>
          <div className="meta">
            <span className={`lamp ${connected ? "ok" : "bad"}`} aria-hidden="true" />
            <span>{connected ? "在线" : "离线"}</span>
          </div>
          <span className="clock">{currentTime.toLocaleTimeString("zh-CN", { hour12: false })}</span>
          <button
            type="button"
            className="btn ghost sm"
            title="产品官网"
            onClick={() => {
              setShowShowcase(true)
              window.history.replaceState(null, "", "/?showcase=1")
            }}
          >
            <Globe size={16} aria-hidden="true" />
            官网
          </button>
          <button
            type="button"
            className="icon-btn"
            aria-label={theme === "light" ? "切换深色模式" : "切换浅色模式"}
            title="切换明暗 (T)"
            onClick={() => setTheme((cur) => (cur === "light" ? "dark" : "light"))}
          >
            {theme === "light" ? <Moon size={16} aria-hidden="true" /> : <Sun size={16} aria-hidden="true" />}
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
            <RefreshCw size={16} aria-hidden="true" />
          </button>
          <button
            type="button"
            className="btn sm"
            onClick={() => lockEngineerSession("已返回作业员视图")}
          >
            <LogOut size={16} aria-hidden="true" />
            退出
          </button>
        </header>
      )}

      {error && (
        <div className="banner studio-banner" role="alert">
          <AlertTriangle size={16} aria-hidden="true" />
          <span>工位服务离线闭锁</span>
          <code>{error}</code>
        </div>
      )}

      {!error && state.hardware_fault_locked && (
        <div className="banner studio-banner" role="alert">
          <AlertTriangle size={16} aria-hidden="true" />
          <span>
            DIO 硬件看门狗故障闭锁：
            {reasonOf(state.hal_dio?.fault_reason || state.result_reason || "MODBUS_WATCHDOG_TIMEOUT")}
          </span>
          <code>{String(state.hal_dio?.fault_reason || state.result_reason || "FAULT")}</code>
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
            statsScope={statsScope}
            shutterKey={captureShutterSeq}
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
                shutterKey={captureShutterSeq}
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
                trainingError={trainingError}
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
                storageError={storageError}
                isInventoryAvailable={isBusinessInventoryAvailable}
                cycles={cycles}
                evidence={evidence}
                events={displayedEvents}
                eventsLoading={timelineLoading}
                eventsError={timelineError}
                selectedCycleId={displayedCycleId}
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
                onRetryEvents={() => {
                  if (displayedCycleId) void loadCycleEvents(displayedCycleId, { silent: false })
                }}
                onClearSelectedCycle={clearSelectedCycle}
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

          <footer className="studio-statusbar" aria-label="运行遥测栏">
            <span className="mono">{`${camW} × ${camH}`}</span>
            <span className="mono">
              {state.fps != null ? `${fmt(state.fps, 1)} FPS` : "25.0 FPS"}
              {state.inference_ms ? ` · ${state.inference_ms} ms` : ""}
            </span>
            <span>
              镜像：<b className={mirrorHorizontal ? "sb-ok" : "sb-muted"}>{mirrorHorizontal ? "开" : "关"}</b>
            </span>
            <span>{workMode === "strict" ? "车间严格" : "桌面调试"}</span>
            <span className="studio-sb-engine mono">
              {state.inference_backend || state.model_version || "C++20 · 720P 25FPS"}
            </span>
            <span className="sep" />
            <span className="studio-sb-keys">
              <kbd>空格</kbd> 抓拍
              <kbd>T</kbd> 明暗
            </span>
          </footer>
        </>
      )}

      <RedRabbitOverlay
        state={state}
        redRabbit={redRabbit}
        adminToken={adminToken}
        manualOpen={rrAuditOpen}
        onCloseManual={() => setRrAuditOpen(false)}
        onOpenEngineerAuth={() => setAuthModalOpen(true)}
        onUpdated={() => refresh(true)}
        flash={flash}
      />

      <EngineerAuthModal
        open={authModalOpen}
        onClose={() => setAuthModalOpen(false)}
        onAuthenticated={handleAuthenticated}
        triggerRef={loginTriggerRef}
      />

      <SnapshotModal
        item={role === "engineer" ? activeSnapshot : null}
        onClose={() => setActiveSnapshot(null)}
      />

      <ConfirmModal
        dialog={role === "engineer" ? confirmDialog : null}
        onClose={() => setConfirmDialog(null)}
      />

      {notice && (
        <div className="toast" role="status">
          <span>{notice}</span>
          <button type="button" onClick={() => setNotice("")} aria-label="关闭通知">
            <X size={16} aria-hidden="true" />
          </button>
        </div>
      )}
    </div>
  )
}

function ConfirmModal({ dialog, onClose }: { dialog: ConfirmDialogState | null; onClose: () => void }) {
  const busyRef = useRef(false)
  return (
    <AlertDialog.Root
      open={Boolean(dialog)}
      onOpenChange={(nextOpen) => {
        if (!nextOpen && !busyRef.current) onClose()
      }}
    >
      {dialog && (
        <ConfirmModalContent
          key={`${dialog.title}:${dialog.description}`}
          dialog={dialog}
          busyRef={busyRef}
          onClose={onClose}
        />
      )}
    </AlertDialog.Root>
  )
}

function ConfirmModalContent({
  dialog,
  busyRef,
  onClose,
}: {
  dialog: ConfirmDialogState
  busyRef: React.MutableRefObject<boolean>
  onClose: () => void
}) {
  const [acknowledged, setAcknowledged] = useState(false)
  const [busy, setBusy] = useState(false)
  const cancelBtnRef = useRef<HTMLButtonElement | null>(null)

  useEffect(() => {
    busyRef.current = busy
    return () => {
      busyRef.current = false
    }
  }, [busy, busyRef])

  const canConfirm = !busy && (!dialog.requireCheckLabel || acknowledged)

  const handleConfirm = async () => {
    if (!canConfirm) return
    busyRef.current = true
    setBusy(true)
    try {
      await dialog.onConfirm()
      busyRef.current = false
      onClose()
    } finally {
      busyRef.current = false
      setBusy(false)
    }
  }

  return (
    <AlertDialog.Portal>
      <AlertDialog.Overlay
        className="modal-scrim"
        onClick={() => {
          if (!busy) onClose()
        }}
      />
      <AlertDialog.Content
        className="modal confirm-modal"
        onOpenAutoFocus={(e) => {
          e.preventDefault()
          cancelBtnRef.current?.focus()
        }}
        onEscapeKeyDown={(e) => {
          if (busy) e.preventDefault()
        }}
      >
        <header className="panel-head">
          <AlertDialog.Title asChild>
            <h2 id="confirm-dialog-title" className="flex items-center gap-2">
              <ShieldAlert
                size={18}
                aria-hidden="true"
                style={{ color: dialog.tone === "danger" ? "var(--bad)" : "var(--warn)" }}
              />
              {dialog.title}
            </h2>
          </AlertDialog.Title>
          <div className="right">
            <button
              type="button"
              className="icon-btn"
              disabled={busy}
              onClick={onClose}
              aria-label="关闭"
            >
              <X size={16} aria-hidden="true" />
            </button>
          </div>
        </header>
        <div className="panel-body">
          <div className={`confirm-consequence ${dialog.tone === "danger" ? "danger" : ""}`}>
            <AlertDialog.Description asChild>
              <div id="confirm-dialog-desc">{dialog.description}</div>
            </AlertDialog.Description>
            {dialog.consequences && dialog.consequences.length > 0 && (
              <ul>
                {dialog.consequences.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            )}
          </div>

          {dialog.requireCheckLabel && (
            <label className="check" style={{ fontWeight: 600 }}>
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
              className="btn ghost"
              disabled={busy}
              onClick={onClose}
            >
              取消
            </button>
            <button
              type="button"
              className={`btn ${dialog.tone === "danger" ? "btn-danger" : ""}`}
              disabled={!canConfirm}
              onClick={() => void handleConfirm()}
            >
              {busy ? "执行中…" : dialog.confirmLabel}
            </button>
          </div>
        </div>
      </AlertDialog.Content>
    </AlertDialog.Portal>
  )
}

function SnapshotModal({ item, onClose }: { item: FilmstripItem | null; onClose: () => void }) {
  const closeBtnRef = useRef<HTMLButtonElement | null>(null)
  const v = item ? verdictOf(item.outcome) : null

  return (
    <Dialog.Root
      open={Boolean(item)}
      onOpenChange={(nextOpen) => {
        if (!nextOpen) onClose()
      }}
    >
      {item && v && (
        <Dialog.Portal>
          <Dialog.Overlay className="modal-scrim" />
          <Dialog.Content
            className="modal"
            onOpenAutoFocus={(e) => {
              e.preventDefault()
              closeBtnRef.current?.focus()
            }}
          >
            <header className="panel-head">
              <Dialog.Title asChild>
                <h2 id="snapshot-title">周期详情</h2>
              </Dialog.Title>
              <span className="sub mono">{item.cycleId}</span>
              <div className="right">
                <button ref={closeBtnRef} type="button" className="icon-btn" onClick={onClose} aria-label="关闭">
                  <X size={16} aria-hidden="true" />
                </button>
              </div>
            </header>
            <Dialog.Description className="sr-only">
              周期 {item.cycleId} 快照详情
            </Dialog.Description>
            <div className="panel-body">
              <CameraFrame
                src={item.url}
                alt="周期证据快照"
                emptyTitle="无快照图像"
                errorTitle="快照不可用"
              />
              <dl className="dl">
                <dt>判定结果</dt>
                <dd>
                  <span className={`tag ${v.tone}`}>{v.word}</span>
                </dd>
                <dt>判定原因</dt>
                <dd>{reasonOf(item.reason) || "—"}</dd>
                <dt>记录时间</dt>
                <dd className="mono">{dateTimeOf(item.timestamp)}</dd>
                <dt>料管颜色</dt>
                <dd>{pipeText(item.pipeColor)}</dd>
                <dt>初始标签</dt>
                <dd>{labelText[item.labelStatus ?? ""] || "—"}</dd>
              </dl>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      )}
    </Dialog.Root>
  )
}

function OutcomeTag({ value }: { value: string }) {
  const v = verdictOf(value)
  return <span className={`tag ${v.tone}`}>{v.word}</span>
}

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
  shutterKey = 0,
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
  shutterKey?: number
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
  const [pipeColor, setPipeColor] = useState<"yellow" | "blue">("yellow")
  const [tubeGroup, setTubeGroup] = useState("T01")
  const [batchGroup, setBatchGroup] = useState("B01")
  const [capturingClass, setCapturingClass] = useState("")
  const [bootstrapping, setBootstrapping] = useState(false)
  const [clearing, setClearing] = useState(false)
  const [manualBox, setManualBox] = useState<[number, number, number, number] | null>(null)
  const [dragStart, setDragStart] = useState<[number, number] | null>(null)

  // 跳帧自动抓拍 (GET/POST /api/dataset/auto_capture + state.auto_capture)
  const [autoLabel, setAutoLabel] = useState<AutoCaptureLabel>("complete")
  const [everyNFrames, setEveryNFrames] = useState("10")
  const [maxSamples, setMaxSamples] = useState("200")
  const [autoServerState, setAutoServerState] = useState<AutoCaptureState | null>(null)
  const [autoBusy, setAutoBusy] = useState(false)
  const autoInitializedRef = useRef(false)

  // 本地图片上传
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [uploadPreviewUrl, setUploadPreviewUrl] = useState("")
  const [uploadLabel, setUploadLabel] = useState<SampleLabel>("complete")
  const [uploadingSample, setUploadingSample] = useState(false)
  const uploadInputRef = useRef<HTMLInputElement | null>(null)
  const uploadFileRef = useRef<File | null>(null)
  const previewUrlRef = useRef("")

  // 样本库筛选与多选
  const [sampleFilter, setSampleFilter] = useState<SampleFilter>("")
  const [samplesLoadedFilter, setSamplesLoadedFilter] = useState<SampleFilter | null>(null)
  const [samples, setSamples] = useState<Data[]>([])
  const [samplesLoading, setSamplesLoading] = useState(false)
  const [selectedSampleIds, setSelectedSampleIds] = useState<string[]>([])
  const [sampleReloadTick, setSampleReloadTick] = useState(0)
  const [bulkBusy, setBulkBusy] = useState(false)

  // 证据补录
  const [file, setFile] = useState<File | null>(null)
  const [cycle, setCycle] = useState("")
  const [kind, setKind] = useState("initial_label")
  const [label, setLabel] = useState("complete")
  const [submitting, setSubmitting] = useState(false)

  // 硬件光学 3A 与防反光控制
  const [opticsBusy, setOpticsBusy] = useState(false)
  const [draftOptics, setDraftOptics] = useState<{ exposure?: number; gain?: number; focus?: number }>({})
  const opticsCommitTimerRef = useRef<number | null>(null)
  const opticsPendingPatchRef = useRef<{ exposure?: number; gain?: number; focus?: number }>({})

  useEffect(() => {
    return () => {
      if (opticsCommitTimerRef.current !== null) {
        window.clearTimeout(opticsCommitTimerRef.current)
      }
    }
  }, [])

  const recentSamplesRef = useRef<Data[]>([])
  useEffect(() => {
    recentSamplesRef.current = Array.isArray(datasetSummary.recent_samples) ? datasetSummary.recent_samples : []
  }, [datasetSummary.recent_samples])

  useEffect(() => {
    if (mode !== "collect" || !token) return
    let cancelled = false
    const controller = new AbortController()
    void api<Data>("/api/dataset/auto_capture", { signal: controller.signal }, token, POLL_TIMEOUT_MS)
      .then((res) => {
        if (cancelled) return
        const parsed = parseAutoCapture(res)
        if (!parsed) return
        setAutoServerState(parsed)
        if (!autoInitializedRef.current) {
          autoInitializedRef.current = true
          setAutoLabel(parsed.label)
          setEveryNFrames(String(parsed.every_n_frames))
          setMaxSamples(String(parsed.max_samples))
        }
      })
      .catch(() => {})
    return () => {
      cancelled = true
      controller.abort()
    }
  }, [mode, token])

  const stateAutoCapture = parseAutoCapture(state.auto_capture)
  const effectiveAutoCapture = stateAutoCapture ?? autoServerState
  const autoEnabled = Boolean(effectiveAutoCapture?.enabled)
  const autoCapturedCount = effectiveAutoCapture?.captured_count ?? 0

  const selectUploadFile = useCallback((nextFile: File | null) => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current)
      previewUrlRef.current = ""
    }
    uploadFileRef.current = nextFile
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
    const controller = new AbortController()
    const requestedFilter = sampleFilter
    const params = new URLSearchParams({ limit: "200" })
    if (requestedFilter) params.set("label", requestedFilter)
    const timer = window.setTimeout(async () => {
      if (cancelled) return
      setSamplesLoading(true)
      try {
        const res = await api<Data>(
          `/api/dataset/samples?${params.toString()}`,
          { signal: controller.signal },
          "",
          POLL_TIMEOUT_MS
        )
        if (cancelled) return
        const list: Data[] = Array.isArray(res)
          ? res
          : Array.isArray(res?.samples)
            ? res.samples
            : Array.isArray(res?.items)
              ? res.items
              : []
        setSamples(list)
        setSamplesLoadedFilter(requestedFilter)
        const validSet = new Set(list.map((s) => String(s.sample_id)))
        setSelectedSampleIds((prev) => prev.filter((id) => validSet.has(id)))
      } catch (err) {
        if (!cancelled && !isAbortError(err)) {
          const fallback = recentSamplesRef.current
          const filtered = requestedFilter ? fallback.filter((s) => s.label === requestedFilter) : fallback
          setSamples(filtered)
          setSamplesLoadedFilter(requestedFilter)
          const validSet = new Set(filtered.map((s) => String(s.sample_id)))
          setSelectedSampleIds((prev) => prev.filter((id) => validSet.has(id)))
        }
      } finally {
        if (!cancelled) setSamplesLoading(false)
      }
    }, 0)
    return () => {
      cancelled = true
      controller.abort()
      window.clearTimeout(timer)
    }
  }, [mode, sampleFilter, sampleReloadTick, datasetSummary.total_samples, autoCapturedCount])

  const valid = state.view_valid === true
  const online = state.camera_status === "ONLINE"
  const timer = Math.max(0, Number(state.c_timer_ms || 0))
  const litSegments = Math.min(12, Math.floor((timer / C_LIMIT_MS) * 12))
  const steps = ["VERIFY_INITIAL", "READY", "INSPECTING", "PASS"]
  const stepNames = ["核验", "就绪", "检验", "放行"]
  const currentIdx = steps.indexOf(state.fsm_state)
  const showAux = Boolean(state.show_auxiliary_rois)
  const trackedList: Data[] = Array.isArray(state.tracked_objects) ? state.tracked_objects : []
  const isAlarmLatched = Boolean(state.alarm_latched || state.outcome === "ALARM" || state.bare_hand_violation)
  const workMode = state.work_mode === "strict" ? "strict" : "debug"
  const counts = getClassCounts(datasetSummary)
  const liveSrc = (online && state.live_stream_url) || state.latest_evidence_url || evidence[0]?.url
  const cropSrc = online && state.live_crop_url ? String(state.live_crop_url) : undefined

  const samplesSynced = !samplesLoading && samplesLoadedFilter === sampleFilter
  const visibleSampleIds = samplesSynced ? samples.map((s) => String(s.sample_id)) : []
  const selectedIdSet = new Set(selectedSampleIds)
  const allVisibleSelected =
    samplesSynced && visibleSampleIds.length > 0 && visibleSampleIds.every((id) => selectedIdSet.has(id))
  const activeFilterLabel = sampleFilterOptions.find((o) => o.id === sampleFilter)?.label || "全部"

  const gripText = (v: unknown) => (v === true ? "在位" : v === false ? "未入画" : "—")

  const normPoint = (e: React.PointerEvent<HTMLDivElement>): [number, number] => {
    const r = e.currentTarget.getBoundingClientRect()
    return [
      Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)),
      Math.max(0, Math.min(1, (e.clientY - r.top) / r.height)),
    ]
  }

  const toggleSampleSelection = (sampleId: string) => {
    if (!samplesSynced) return
    setSelectedSampleIds((prev) =>
      prev.includes(sampleId) ? prev.filter((id) => id !== sampleId) : [...prev, sampleId]
    )
  }

  const toggleSelectAllVisible = () => {
    if (!samplesSynced) return
    setSelectedSampleIds(allVisibleSelected ? [] : visibleSampleIds)
  }

  const handleToggleAutoCapture = async () => {
    const nextEnabled = !autoEnabled
    const intervalVal = Math.max(1, Math.min(300, Math.round(Number(everyNFrames) || 10)))
    const maxVal = Math.max(1, Math.min(5000, Math.round(Number(maxSamples) || 200)))
    setEveryNFrames(String(intervalVal))
    setMaxSamples(String(maxVal))
    setAutoBusy(true)
    try {
      const res = await api<Data>(
        "/api/dataset/auto_capture",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            enabled: nextEnabled,
            label: autoLabel,
            every_n_frames: intervalVal,
            max_samples: maxVal,
            pipe_color: pipeColor,
            physical_tube_id: tubeGroup.trim() || "UNGROUPED",
          }),
        },
        token
      )
      const parsed = parseAutoCapture(res)
      if (parsed) {
        setAutoServerState(parsed)
      } else {
        setAutoServerState({
          enabled: nextEnabled,
          every_n_frames: intervalVal,
          label: autoLabel,
          pipe_color: pipeColor,
          physical_tube_id: tubeGroup.trim() || "UNGROUPED",
          max_samples: maxVal,
          captured_count: nextEnabled ? 0 : autoCapturedCount,
        })
      }
      flash(nextEnabled ? `已开启跳帧抓拍（每 ${intervalVal} 帧）` : "已停止抓拍")
      await onRefresh()
      reloadSamples()
    } catch (err) {
      flash(`跳帧抓拍设置失败：${err instanceof Error ? err.message : err}`)
    } finally {
      setAutoBusy(false)
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
      flash(`已抓拍 ${title} (${res.sample_id})`)
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
    const submittedFile = uploadFile
    if (!submittedFile) return flash("请选择图片")
    setUploadingSample(true)
    try {
      const res = await api<Data>(
        "/api/dataset/upload",
        {
          method: "POST",
          headers: {
            "Content-Type": submittedFile.type || "image/jpeg",
            "X-Sample-Label": uploadLabel,
            "X-Pipe-Color": pipeColor,
            "X-Tube-Id": toHeaderValue(tubeGroup, "TUBE_UPLOAD_01"),
            "X-Batch-Id": toHeaderValue(batchGroup, "BATCH_UPLOAD"),
          },
          body: submittedFile,
        },
        token
      )
      flash(`已入库 (${res.sample_id || "完成"})`)
      if (uploadFileRef.current === submittedFile) {
        selectUploadFile(null)
        if (uploadInputRef.current) uploadInputRef.current.value = ""
      }
      await onRefresh()
      reloadSamples()
      void onStorageChanged()
    } catch (err) {
      flash(`上传失败：${err instanceof Error ? err.message : err}`)
    } finally {
      setUploadingSample(false)
    }
  }

  const requestClearAllSamples = () => {
    const total = Number(datasetSummary.total_samples || 0)
    if (total === 0) return flash("样本库已空")
    onConfirm({
      title: "清空训练样本",
      description: `确认清空全部 ${total} 张训练样本吗？`,
      consequences: ["删除全部训练切片与 COCO 标注索引", "不影响业务周期与运行模型"],
      confirmLabel: "确认清空",
      tone: "danger",
      onConfirm: async () => {
        setClearing(true)
        try {
          await api<Data>("/api/dataset/clear", { method: "POST" }, token)
          setSelectedSampleIds([])
          flash("已清空训练样本")
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
    if (!samplesSynced) return
    const ids = [...selectedSampleIds]
    if (ids.length === 0) return
    onConfirm({
      title: "批量删除样本",
      description: `确认删除选中的 ${ids.length} 张样本吗？`,
      confirmLabel: `删除 ${ids.length} 张`,
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
          flash(`已删除 ${res.deleted_count ?? ids.length} 张样本`)
          await onRefresh()
          reloadSamples()
          void onStorageChanged()
        } catch (err) {
          flash(`删除失败：${err instanceof Error ? err.message : err}`)
        } finally {
          setBulkBusy(false)
        }
      },
    })
  }

  const requestClearCurrentClass = () => {
    if (!samplesSynced) return
    if (!sampleFilter) {
      requestClearAllSamples()
      return
    }
    const classCount = Number(counts[sampleFilter] || samples.length || 0)
    if (classCount === 0) return flash("该类别暂无样本")
    onConfirm({
      title: `清空 ${activeFilterLabel}`,
      description: `确认清空 ${activeFilterLabel} 共 ${classCount} 张样本吗？`,
      confirmLabel: "确认清空",
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
          flash(`已清空 ${res.deleted_count ?? classCount} 张样本`)
          await onRefresh()
          reloadSamples()
          void onStorageChanged()
        } catch (err) {
          flash(`清空失败：${err instanceof Error ? err.message : err}`)
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
      flash(`已生成 ${res.bootstrapped_count} 张演示样本`)
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
      flash(`已改为 ${labelText[newLabel] || newLabel}`)
      await onRefresh()
      reloadSamples()
    } catch (err) {
      flash(`修改失败：${err instanceof Error ? err.message : err}`)
    }
  }

  const requestDeleteSingleSample = (smp: Data) => {
    const sampleId = String(smp.sample_id)
    onConfirm({
      title: "删除样本",
      description: `确认删除样本 ${sampleId} 吗？`,
      confirmLabel: "删除",
      tone: "danger",
      onConfirm: async () => {
        try {
          await api(`/api/dataset/sample/${encodeURIComponent(sampleId)}`, { method: "DELETE" }, token)
          setSelectedSampleIds((prev) => prev.filter((id) => id !== sampleId))
          flash(`已删除 ${sampleId}`)
          await onRefresh()
          reloadSamples()
          void onStorageChanged()
        } catch (err) {
          flash(`删除失败：${err instanceof Error ? err.message : err}`)
        }
      },
    })
  }

  const submitOpticsPatch = useCallback(
    async (patch: Record<string, unknown>, quiet = false) => {
      if (!token) {
        flash("请先解锁工程师权限")
        return
      }
      setOpticsBusy(true)
      try {
        await api<Data>(
          "/api/camera/optics",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(patch),
          },
          token
        )
        if (!quiet) flash("光学参数已应用并锁定")
        await onRefresh()
      } catch (err) {
        flash(`光学参数设置失败：${err instanceof Error ? err.message : err}`)
      } finally {
        setOpticsBusy(false)
      }
    },
    [token, flash, onRefresh]
  )

  const queueOpticsSliderPatch = useCallback(
    (key: "exposure" | "gain" | "focus", val: number) => {
      setDraftOptics((prev) => ({ ...prev, [key]: val }))
      opticsPendingPatchRef.current = { ...opticsPendingPatchRef.current, [key]: val }
      if (opticsCommitTimerRef.current !== null) {
        window.clearTimeout(opticsCommitTimerRef.current)
      }
      opticsCommitTimerRef.current = window.setTimeout(() => {
        const pending = { ...opticsPendingPatchRef.current }
        opticsPendingPatchRef.current = {}
        opticsCommitTimerRef.current = null
        if (Object.keys(pending).length > 0) {
          void submitOpticsPatch({ ...pending, lock_3a: true }, true).finally(() => {
            setDraftOptics({})
          })
        }
      }, 180)
    },
    [submitOpticsPatch]
  )

  const flushOpticsSliderPatch = useCallback(() => {
    if (opticsCommitTimerRef.current !== null) {
      window.clearTimeout(opticsCommitTimerRef.current)
      opticsCommitTimerRef.current = null
    }
    const pending = { ...opticsPendingPatchRef.current }
    opticsPendingPatchRef.current = {}
    if (Object.keys(pending).length > 0) {
      void submitOpticsPatch({ ...pending, lock_3a: true }, true).finally(() => {
        setDraftOptics({})
      })
    }
  }, [submitOpticsPatch])

  const handleAutoCalibrate = useCallback(async () => {
    if (!token) {
      flash("请先解锁工程师权限")
      return
    }
    setOpticsBusy(true)
    try {
      await api<Data>("/api/camera/optics/auto_calibrate", { method: "POST" }, token)
      flash("已启动闭环防反光校准…")
      await onRefresh()
    } catch (err) {
      flash(`启动防反光校准失败：${err instanceof Error ? err.message : err}`)
    } finally {
      setOpticsBusy(false)
    }
  }, [token, flash, onRefresh])

  const submitManualEvidence = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!file) return flash("请选择图片")
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
              resolution: "1280x720",
            }),
          },
          body: file,
        },
        token
      )
      flash(`已归档至 ${targetCycle}`)
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
      {/* 左侧永久固定的 16:9 实时镜头舞台 */}
      <section className="studio-canvas" aria-label={mode === "monitor" ? "实时监控舞台" : "采集标注画布"}>
        {mode === "monitor" && (
          <div className="studio-andon-slot">
            <Andon
              compact
              state={state}
              guide={{
                title: state.result_reason
                  ? reasonOf(state.result_reason)
                  : valid
                    ? "动态锁定中"
                    : reasonOf(state.view_valid_reasons?.[0]) || "视野无效",
              }}
              meta={[
                { label: "周期编号", value: state.cycle_id || "—" },
                { label: "更新时间", value: timeOf(state.updated_at_utc) },
              ]}
            />
          </div>
        )}

        <header className="studio-canvas-bar">
          {mode === "monitor" ? (
            <>
              <h2>全景画面</h2>
              <span className="sub mono">720P · 25 FPS</span>
              <div className="right">
                <button
                  type="button"
                  className={`btn ghost sm ${mirrorHorizontal ? "btn-active" : ""}`}
                  disabled={togglingMirror}
                  onClick={() => onToggleMirror(!mirrorHorizontal)}
                >
                  <FlipHorizontal2 size={16} aria-hidden="true" />
                  {mirrorHorizontal ? "镜像开" : "镜像关"}
                </button>
                <button type="button" className="btn ghost sm" onClick={() => onToggleAuxRois(!showAux)}>
                  <Eye size={16} aria-hidden="true" />
                  {showAux ? "隐藏框" : "辅助框"}
                </button>
                <button type="button" className="btn ghost sm" onClick={onManualCapture}>
                  <Camera size={16} aria-hidden="true" />
                  抓拍
                </button>
              </div>
            </>
          ) : (
            <>
              <h2>拉框采集</h2>
              {manualBox && (
                <span className="sub mono">
                  [{manualBox.map((v) => v.toFixed(2)).join(", ")}]
                </span>
              )}
              <div className="right">
                {manualBox && (
                  <button type="button" className="btn ghost sm" onClick={() => setManualBox(null)}>
                    清除选框
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
                  {clearing ? "清空中…" : "清空样本"}
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
                alt="16:9 实时采集画面"
                focusLocked={online && valid}
                emptyTitle="等待相机画面"
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
                    {Math.round((manualBox[2] - manualBox[0]) * (Number(state.camera_width) || DEFAULT_CAMERA_WIDTH))}×
                    {Math.round((manualBox[3] - manualBox[1]) * (Number(state.camera_height) || DEFAULT_CAMERA_HEIGHT))}
                  </span>
                </div>
              )}
            </div>
          ) : (
            <div className="canvas-16x9">
              <CameraFrame
                fill
                src={liveSrc}
                alt="16:9 实时追踪画面"
                shutterKey={shutterKey}
                focusLocked={online && valid}
                emptyTitle="等待相机画面"
                errorTitle="画面不可用"
              />
            </div>
          )}
        </div>

        {mode === "collect" && (
          <div className="studio-class-strip" aria-label="类别样本计数">
            {sampleClassDefs.map((c) => (
              <div className="studio-class-cell" key={c.id}>
                <span>{c.short}</span>
                <b className="mono">{counts[c.id] || 0}</b>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* 右侧检查器：监控模式顶部永久置顶 4× 标签特写，下方卡片栈宽敞滚动 */}
      <aside className="studio-inspector" aria-label={mode === "monitor" ? "实时诊断栏" : "样本采集栏"}>
        {mode === "monitor" && (
          <section className="inspector-pinned-crop" aria-label="标签特写置顶区">
            <div className="inspector-section-head">
              <div className="flex items-center gap-2">
                <h3>标签特写</h3>
                <span className="pin-badge">
                  <Pin size={14} aria-hidden="true" />
                  4× 置顶
                </span>
              </div>
              <span className="mono" style={{ fontSize: 14, color: "var(--ink-2)" }}>
                {state.label_confidence != null ? `${(Number(state.label_confidence) * 100).toFixed(1)}%` : "—"}
              </span>
            </div>
            <div className="inspector-crop-frame">
              <CameraFrame
                fill
                src={cropSrc}
                alt="标签 4 倍特写"
                focusLocked={Boolean(online && valid && cropSrc)}
                emptyTitle={online ? "无特写流" : "等待入位"}
                errorTitle="特写不可用"
              />
            </div>
            <div className="inspector-verdict-row">
              <div>
                <small>标签状态</small>
                <b style={{ color: isAlarmLatched ? "var(--bad)" : undefined }}>
                  {state.bare_hand_violation ? "裸手违规" : labelText[state.label_status] || human(state.label_status)}
                </b>
              </div>
              <div style={{ textAlign: "right" }}>
                <small>当前料管</small>
                <b>{pipeText(state.pipe_color)}</b>
              </div>
            </div>
            <div style={{ display: "grid", gap: 8 }}>
              <div className="flex items-baseline justify-between">
                <span style={{ fontSize: 14, color: "var(--ink-2)" }}>脱落确认</span>
                <span className="mono" style={{ fontSize: 14, fontWeight: 600 }}>
                  {formatCTimerSeconds(timer, 2)} / 1.20s
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
                className="btn btn-danger block"
                onClick={() => onSimulate("reset")}
                disabled={simulating !== ""}
              >
                <RotateCcw size={16} aria-hidden="true" />
                报警复位
              </button>
            )}
          </section>
        )}

        <div className="studio-inspector-scroll-wrap">
          <div className="studio-inspector-body">
            {mode === "monitor" ? (
              <>
                <section className="inspector-section">
                  <div className="inspector-section-head">
                    <h3>运行状态</h3>
                    <span className={`tag ${valid ? "pass" : "warn"}`}>{human(state.fsm_state || "WAIT_PART")}</span>
                  </div>
                  <div className="fsm-track">
                    {steps.map((s, i) => (
                      <div key={s} className={`fsm-step ${i === currentIdx ? "now" : i < currentIdx ? "past" : ""}`}>
                        {stepNames[i]}
                      </div>
                    ))}
                  </div>
                  <div className="inspector-mini-grid">
                    <div className="inspector-mini-cell">
                      <small>相机连接</small>
                      <b>{cameraText[state.camera_status] || "未知"}</b>
                    </div>
                    <div className="inspector-mini-cell">
                      <small>手套状态</small>
                      <b style={{ color: state.bare_hand_violation ? "var(--bad)" : undefined }}>
                        {state.bare_hand_violation
                          ? "裸手违规"
                          : state.bare_hand_detected
                            ? "裸手警告"
                            : state.glove_compliant
                              ? "合规"
                              : "待检"}
                      </b>
                    </div>
                    <div className="inspector-mini-cell">
                      <small>左手握持</small>
                      <b>{gripText(state.left_grip_active)}</b>
                    </div>
                    <div className="inspector-mini-cell">
                      <small>右手握持</small>
                      <b>{gripText(state.right_grip_active)}</b>
                    </div>
                  </div>
                  {trackedList.length > 0 && (
                    <div className="flex flex-wrap gap-2">
                      {trackedList.map((obj, idx) => (
                        <span
                          key={idx}
                          className="tag plain mono"
                          title={
                            Array.isArray(obj.box)
                              ? `[${obj.box.map((v: number) => Number(v).toFixed(2)).join(", ")}]`
                              : undefined
                          }
                        >
                          {obj.label}
                        </span>
                      ))}
                    </div>
                  )}
                </section>

                <section className="inspector-section">
                  <div className="inspector-section-head">
                    <h3>光学指标</h3>
                    <span className={`tag ${valid ? "pass" : "warn"}`}>{valid ? "有效" : "受阻"}</span>
                  </div>
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
                  {(() => {
                    const optics: Data =
                      state.camera_optics && typeof state.camera_optics === "object" ? state.camera_optics : {}
                    const hw3a: Data =
                      state.hardware_3a && typeof state.hardware_3a === "object" ? state.hardware_3a : {}
                    const autoCalibrating = Boolean(optics.auto_calibrating ?? hw3a.auto_calibrating)
                    const lock3a = optics.lock_3a !== false
                    const backlightOff = Number(optics.backlight_comp ?? 0) === 0
                    const liveExp =
                      typeof hw3a.exposure?.value === "number"
                        ? Number(hw3a.exposure.value)
                        : Number(optics.exposure ?? -7)
                    const liveGain =
                      typeof hw3a.gain?.value === "number"
                        ? Number(hw3a.gain.value)
                        : Number(optics.gain ?? 0)
                    const liveFocusRaw =
                      typeof hw3a.focus?.value === "number"
                        ? Number(hw3a.focus.value)
                        : Number(optics.focus ?? 35)
                    const liveFocus = liveFocusRaw >= 0 ? liveFocusRaw : 35
                    const expVal =
                      draftOptics.exposure ??
                      (Number.isFinite(Number(optics.exposure)) ? Number(optics.exposure) : liveExp)
                    const gainVal =
                      draftOptics.gain ??
                      (Number.isFinite(Number(optics.gain)) ? Number(optics.gain) : liveGain)
                    const focusVal =
                      draftOptics.focus ??
                      (Number.isFinite(Number(optics.focus)) && Number(optics.focus) >= 0
                        ? Number(optics.focus)
                        : liveFocus)
                    return (
                      <div className="optics-panel">
                        <div className="optics-meta">
                          <span className="mono">
                            {`曝光 ${liveExp.toFixed(0)} EV · 增益 ${liveGain.toFixed(0)} · 焦距 ${liveFocus.toFixed(0)}`}
                          </span>
                          <span className={`tag ${autoCalibrating ? "warn" : lock3a ? "pass" : "plain"}`}>
                            {autoCalibrating ? "校准中…" : lock3a ? "已锁 3A" : "自动 3A"}
                          </span>
                        </div>
                        <button
                          type="button"
                          className="btn sm block"
                          disabled={opticsBusy || autoCalibrating}
                          onClick={() => void handleAutoCalibrate()}
                        >
                          <Sparkles size={16} aria-hidden="true" />
                          {autoCalibrating ? "防反光校准中…" : "一键防反光校准"}
                        </button>
                        <details className="optics-details">
                          <summary>
                            <span>硬件 3A 参数微调</span>
                            <span>{lock3a ? "手动锁定" : "自动模式"}</span>
                          </summary>
                          <div className="optics-details-body">
                            <div className="toggle-row">
                              <div className="toggle-copy">
                                <b>手动锁死 3A</b>
                              </div>
                              <button
                                type="button"
                                role="switch"
                                aria-checked={lock3a}
                                aria-label="手动锁死 3A 开关"
                                disabled={opticsBusy}
                                className={`studio-switch ${lock3a ? "on" : ""}`}
                                onClick={() => void submitOpticsPatch({ lock_3a: !lock3a })}
                              >
                                <span className="studio-switch-thumb" />
                              </button>
                            </div>
                            <div className="toggle-row">
                              <div className="toggle-copy">
                                <b>关闭背光补偿</b>
                              </div>
                              <button
                                type="button"
                                role="switch"
                                aria-checked={backlightOff}
                                aria-label="关闭背光补偿开关"
                                disabled={opticsBusy}
                                className={`studio-switch ${backlightOff ? "on" : ""}`}
                                onClick={() =>
                                  void submitOpticsPatch({ backlight_comp: backlightOff ? 1 : 0 })
                                }
                              >
                                <span className="studio-switch-thumb" />
                              </button>
                            </div>
                            {lock3a && (
                              <div className="optics-sliders">
                                <div className="optics-slider-row">
                                  <div className="optics-slider-head">
                                    <label htmlFor="optics-exposure">曝光 (EV)</label>
                                    <span className="mono">{expVal.toFixed(0)}</span>
                                  </div>
                                  <input
                                    id="optics-exposure"
                                    type="range"
                                    min={-11}
                                    max={-2}
                                    step={1}
                                    value={Math.max(-11, Math.min(-2, Math.round(expVal)))}
                                    className="optics-range"
                                    onChange={(e) => queueOpticsSliderPatch("exposure", Number(e.target.value))}
                                    onPointerUp={flushOpticsSliderPatch}
                                    onKeyUp={flushOpticsSliderPatch}
                                    onBlur={flushOpticsSliderPatch}
                                  />
                                </div>
                                <div className="optics-slider-row">
                                  <div className="optics-slider-head">
                                    <label htmlFor="optics-gain">增益</label>
                                    <span className="mono">{gainVal.toFixed(0)}</span>
                                  </div>
                                  <input
                                    id="optics-gain"
                                    type="range"
                                    min={0}
                                    max={255}
                                    step={8}
                                    value={Math.max(0, Math.min(255, Math.round(gainVal)))}
                                    className="optics-range"
                                    onChange={(e) => queueOpticsSliderPatch("gain", Number(e.target.value))}
                                    onPointerUp={flushOpticsSliderPatch}
                                    onKeyUp={flushOpticsSliderPatch}
                                    onBlur={flushOpticsSliderPatch}
                                  />
                                </div>
                                <div className="optics-slider-row">
                                  <div className="optics-slider-head">
                                    <label htmlFor="optics-focus">焦距</label>
                                    <span className="mono">{focusVal.toFixed(0)}</span>
                                  </div>
                                  <input
                                    id="optics-focus"
                                    type="range"
                                    min={0}
                                    max={255}
                                    step={5}
                                    value={Math.max(0, Math.min(255, Math.round(focusVal)))}
                                    className="optics-range"
                                    onChange={(e) => queueOpticsSliderPatch("focus", Number(e.target.value))}
                                    onPointerUp={flushOpticsSliderPatch}
                                    onKeyUp={flushOpticsSliderPatch}
                                    onBlur={flushOpticsSliderPatch}
                                  />
                                </div>
                              </div>
                            )}
                          </div>
                        </details>
                      </div>
                    )
                  })()}
                </section>

                <section className="inspector-section">
                  <div className="inspector-section-head">
                    <h3>工况演练</h3>
                    <span className="tag plain">{workMode === "strict" ? "车间严格" : "桌面调试"}</span>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    {simScenarios.map((item) => (
                      <button
                        key={item.id}
                        type="button"
                        disabled={simulating !== ""}
                        className={`btn ghost sm ${item.tone === "bad" ? "btn-outline-bad" : ""}`}
                        onClick={() => onSimulate(item.id)}
                      >
                        {simulating === item.id ? "执行中…" : item.label}
                      </button>
                    ))}
                  </div>
                  <div className="sim-bar" style={{ width: "100%", display: "grid", gridTemplateColumns: "1fr 1fr" }}>
                    <button
                      type="button"
                      className={`sim-btn ${workMode === "debug" ? "busy" : ""}`}
                      onClick={() => onSwitchWorkMode("debug")}
                    >
                      桌面调试
                    </button>
                    <button
                      type="button"
                      className={`sim-btn ${workMode === "strict" ? "busy" : ""}`}
                      onClick={() => onSwitchWorkMode("strict")}
                    >
                      车间严格
                    </button>
                  </div>
                  <div className="grid grid-cols-3 gap-3">
                    <button
                      type="button"
                      className="btn ghost sm block"
                      onClick={() => onGoStep("records_settings", "records")}
                    >
                      周期记录
                    </button>
                    <button
                      type="button"
                      className="btn ghost sm block"
                      onClick={() => onGoStep("records_settings", "storage")}
                    >
                      存储管理
                    </button>
                    <button
                      type="button"
                      className="btn ghost sm block"
                      onClick={() => onGoStep("records_settings", "settings")}
                    >
                      系统参数
                    </button>
                  </div>
                </section>
              </>
            ) : (
              <>
                {/* 卡片 1：分组标识 + 前往训练 */}
                <section className="inspector-section">
                  <div className="inspector-section-head">
                    <h3>样本分组</h3>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        className="btn ghost sm"
                        disabled={bootstrapping}
                        onClick={() => void handleBootstrapDataset()}
                      >
                        <Sparkles size={14} aria-hidden="true" />
                        {bootstrapping ? "生成中…" : "演示集"}
                      </button>
                      <button type="button" className="btn sm" onClick={() => onGoStep("train")}>
                        <Cpu size={14} aria-hidden="true" />
                        去训练
                      </button>
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="field">
                      <label htmlFor="collect-pipe-color">料管颜色</label>
                      <Select value={pipeColor} onValueChange={(v) => setPipeColor(v as "yellow" | "blue")}>
                        <SelectTrigger id="collect-pipe-color">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="yellow">黄管</SelectItem>
                          <SelectItem value="blue">蓝管</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="field">
                      <label htmlFor="collect-tube-group">料管编号</label>
                      <Input
                        id="collect-tube-group"
                        value={tubeGroup}
                        onChange={(e) => setTubeGroup(e.target.value)}
                        className="mono"
                        placeholder="如T01"
                      />
                    </div>
                    <div className="field">
                      <label htmlFor="collect-batch-group">批次编号</label>
                      <Input
                        id="collect-batch-group"
                        value={batchGroup}
                        onChange={(e) => setBatchGroup(e.target.value)}
                        className="mono"
                        placeholder="如B01"
                      />
                    </div>
                  </div>
                </section>

                {/* 卡片 2：跳帧自动抓拍 (A3 核心功能) */}
                <section className="inspector-section" aria-label="跳帧自动抓拍">
                  <div className="inspector-section-head">
                    <h3>跳帧自动抓拍</h3>
                    <span className={`tag ${autoEnabled ? "pass" : "plain"}`}>
                      {autoEnabled ? "抓拍中" : "已停止"}
                    </span>
                  </div>

                  <div className="grid grid-cols-3 gap-4">
                    <div className="field">
                      <label htmlFor="auto-capture-label">目标类别</label>
                      <Select
                        value={autoLabel}
                        disabled={autoEnabled || autoBusy}
                        onValueChange={(v) => setAutoLabel(v as AutoCaptureLabel)}
                      >
                        <SelectTrigger id="auto-capture-label">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="complete">完整</SelectItem>
                          <SelectItem value="crossed_out">已划除</SelectItem>
                          <SelectItem value="covered">已遮挡</SelectItem>
                          <SelectItem value="absent">已脱落</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="field">
                      <label htmlFor="auto-capture-interval">帧间隔</label>
                      <Input
                        id="auto-capture-interval"
                        type="number"
                        min="1"
                        max="300"
                        value={everyNFrames}
                        disabled={autoEnabled || autoBusy}
                        onChange={(e) => setEveryNFrames(e.target.value)}
                        placeholder="默认10"
                        className="mono"
                      />
                    </div>
                    <div className="field">
                      <label htmlFor="auto-capture-max">张数上限</label>
                      <Input
                        id="auto-capture-max"
                        type="number"
                        min="1"
                        max="5000"
                        value={maxSamples}
                        disabled={autoEnabled || autoBusy}
                        onChange={(e) => setMaxSamples(e.target.value)}
                        placeholder="默认200"
                        className="mono"
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4" style={{ alignItems: "stretch" }}>
                    <div className="kpi-tile">
                      <b className="mono">{autoCapturedCount}</b>
                      <span>已抓拍</span>
                    </div>
                    <button
                      type="button"
                      className={`btn block ${autoEnabled ? "btn-danger" : ""}`}
                      style={{ height: "100%" }}
                      disabled={autoBusy}
                      onClick={() => void handleToggleAutoCapture()}
                    >
                      {autoEnabled ? <Square size={16} aria-hidden="true" /> : <Play size={16} aria-hidden="true" />}
                      {autoEnabled ? "停止抓拍" : "开启跳帧抓拍"}
                    </button>
                  </div>
                </section>

                {/* 卡片 3：单帧手动抓拍 */}
                <section className="inspector-section">
                  <div className="inspector-section-head">
                    <h3>单帧抓拍</h3>
                    <span className="tag plain">{manualBox ? "选框裁切" : "自动裁切"}</span>
                  </div>
                  <div className="grid gap-3">
                    {sampleClassDefs.map((item) => (
                      <button
                        key={item.id}
                        type="button"
                        disabled={capturingClass !== ""}
                        className={`btn ${item.id === "unknown_challenge" ? "ghost" : ""} block`}
                        style={{ justifyContent: "space-between" }}
                        onClick={() => void handleQuickCapture(item.id, item.short)}
                      >
                        <span className="flex items-center gap-2">
                          <Camera size={16} aria-hidden="true" />
                          <span>{capturingClass === item.id ? "抓拍中…" : item.title}</span>
                        </span>
                        <span className="mono">{counts[item.id] || 0}</span>
                      </button>
                    ))}
                  </div>
                </section>

                {/* 卡片 4：本地上传 */}
                <section className="inspector-section">
                  <div className="inspector-section-head">
                    <h3>本地上传</h3>
                  </div>
                  <form className="upload-box" onSubmit={(e) => void handleLocalSampleUpload(e)}>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="field">
                        <label htmlFor="collect-upload-file">选择图片</label>
                        <Input
                          id="collect-upload-file"
                          ref={uploadInputRef}
                          type="file"
                          accept="image/jpeg,image/png"
                          disabled={uploadingSample}
                          onChange={(e) => selectUploadFile(e.target.files?.[0] || null)}
                        />
                      </div>
                      <div className="field">
                        <label htmlFor="collect-upload-label">目标类别</label>
                        <Select value={uploadLabel} onValueChange={(v) => setUploadLabel(v as SampleLabel)}>
                          <SelectTrigger id="collect-upload-label">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="complete">完整</SelectItem>
                            <SelectItem value="crossed_out">已划除</SelectItem>
                            <SelectItem value="covered">已遮挡</SelectItem>
                            <SelectItem value="absent">已脱落</SelectItem>
                            <SelectItem value="unknown_challenge">挑战</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>

                    {uploadFile && uploadPreviewUrl && (
                      <div className="upload-preview-row">
                        <img
                          src={uploadPreviewUrl}
                          alt="预览"
                          className="upload-preview-thumb"
                          onError={() => {
                            if (previewUrlRef.current) {
                              URL.revokeObjectURL(previewUrlRef.current)
                              previewUrlRef.current = ""
                            }
                            setUploadPreviewUrl("")
                            flash("无法预览所选图片")
                          }}
                        />
                        <div className="upload-preview-info">
                          <b>{uploadFile.name}</b>
                        </div>
                        <button
                          type="button"
                          className="icon-btn"
                          disabled={uploadingSample}
                          aria-label="清除图片"
                          onClick={() => {
                            selectUploadFile(null)
                            if (uploadInputRef.current) uploadInputRef.current.value = ""
                          }}
                        >
                          <X size={16} aria-hidden="true" />
                        </button>
                      </div>
                    )}

                    <button type="submit" className="btn block" disabled={uploadingSample || !uploadFile}>
                      <ImagePlus size={16} aria-hidden="true" />
                      {uploadingSample ? "上传中…" : "上传入库"}
                    </button>
                  </form>
                </section>

                {/* 卡片 5：样本图库 */}
                <section className="inspector-section">
                  <div className="inspector-section-head">
                    <h3>样本图库</h3>
                    <span className="mono" style={{ fontSize: 14, color: "var(--ink-2)" }}>
                      {!samplesSynced ? "加载中…" : `${samples.length} / ${datasetSummary.total_samples || 0}`}
                    </span>
                  </div>

                  <div className="filter-chip-bar" role="group" aria-label="类别筛选">
                    {sampleFilterOptions.map((opt) => {
                      const cnt = opt.id === "" ? Number(datasetSummary.total_samples || 0) : Number(counts[opt.id] || 0)
                      const active = sampleFilter === opt.id
                      return (
                        <button
                          key={opt.id || "all"}
                          type="button"
                          aria-pressed={active}
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

                  <div className="sample-bulk-bar">
                    <label className="check" style={{ fontSize: 14 }}>
                      <input
                        type="checkbox"
                        checked={allVisibleSelected}
                        disabled={!samplesSynced || samples.length === 0 || bulkBusy}
                        onChange={toggleSelectAllVisible}
                      />
                      <span>{selectedSampleIds.length > 0 ? `已选 ${selectedSampleIds.length}` : "全选"}</span>
                    </label>
                    <div className="sample-bulk-actions">
                      <button
                        type="button"
                        className="btn ghost sm btn-outline-bad"
                        disabled={!samplesSynced || selectedSampleIds.length === 0 || bulkBusy}
                        onClick={requestBatchDeleteSelected}
                      >
                        <Trash2 size={14} aria-hidden="true" />
                        删除选中
                      </button>
                      <button
                        type="button"
                        className="btn ghost sm btn-outline-bad"
                        disabled={!samplesSynced || samples.length === 0 || bulkBusy}
                        onClick={requestClearCurrentClass}
                      >
                        {sampleFilter ? `清空${activeFilterLabel}` : "清空全部"}
                      </button>
                    </div>
                  </div>

                  {!samplesSynced ? (
                    <div className="empty">加载中…</div>
                  ) : samples.length > 0 ? (
                    <div className="sample-grid">
                      {samples.map((smp) => {
                        const sid = String(smp.sample_id)
                        const checked = selectedIdSet.has(sid)
                        const tip = `${sid} · ${smp.physical_tube_id || "未分组"}`
                        return (
                          <div key={sid} className={`sample-thumb-card ${checked ? "selected" : ""}`} title={tip}>
                            <label className="sample-thumb-check">
                              <input
                                type="checkbox"
                                checked={checked}
                                disabled={!samplesSynced || bulkBusy}
                                aria-label={`选择 ${sid}`}
                                onChange={() => toggleSampleSelection(sid)}
                              />
                            </label>
                            <img src={smp.image_url} alt={sid} loading="lazy" />
                            <div className="sample-thumb-meta">
                              <select
                                aria-label={`类别 ${sid}`}
                                value={smp.label}
                                className="sample-label-select"
                                onChange={(e) => void handleChangeSampleLabel(sid, e.target.value)}
                              >
                                <option value="complete">完整</option>
                                <option value="crossed_out">已划除</option>
                                <option value="covered">已遮挡</option>
                                <option value="absent">已脱落</option>
                                <option value="unknown_challenge">挑战</option>
                              </select>
                              <button
                                type="button"
                                className="sample-del-btn"
                                onClick={() => requestDeleteSingleSample(smp)}
                              >
                                删除
                              </button>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  ) : (
                    <div className="empty">暂无样本</div>
                  )}
                </section>

                {/* 折叠：离线证据补录 */}
                <details className="inspector-group">
                  <summary>
                    <span>证据补录</span>
                    <Upload size={16} aria-hidden="true" />
                  </summary>
                  <form className="inspector-group-body" onSubmit={(e) => void submitManualEvidence(e)}>
                    <div className="field">
                      <label htmlFor="evidence-upload-file">证据图片</label>
                      <Input
                        id="evidence-upload-file"
                        type="file"
                        accept="image/jpeg,image/png"
                        onChange={(e) => setFile(e.target.files?.[0] || null)}
                      />
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="field">
                        <label htmlFor="evidence-cycle-id">周期编号</label>
                        <Input
                          id="evidence-cycle-id"
                          list="cycle-options"
                          value={cycle}
                          onChange={(e) => setCycle(e.target.value)}
                          placeholder="自动"
                          className="mono"
                        />
                        <datalist id="cycle-options">
                          {cycles.map((c) => (
                            <option key={c.cycle_id} value={c.cycle_id} />
                          ))}
                        </datalist>
                      </div>
                      <div className="field">
                        <label htmlFor="evidence-kind">证据类型</label>
                        <Select value={kind} onValueChange={setKind}>
                          <SelectTrigger id="evidence-kind">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="initial_label">初始标签</SelectItem>
                            <SelectItem value="A">划除</SelectItem>
                            <SelectItem value="B">遮挡</SelectItem>
                            <SelectItem value="C">脱落</SelectItem>
                            <SelectItem value="anomaly">异常</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                    <div className="field">
                      <label htmlFor="evidence-label-status">标签状态</label>
                      <Select value={label} onValueChange={setLabel}>
                        <SelectTrigger id="evidence-label-status">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="complete">完整</SelectItem>
                          <SelectItem value="crossed_out">已划除</SelectItem>
                          <SelectItem value="covered">已遮挡</SelectItem>
                          <SelectItem value="absent">已脱落</SelectItem>
                          <SelectItem value="unknown">未知</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <button type="submit" className="btn block" disabled={submitting}>
                      <Upload size={16} aria-hidden="true" />
                      {submitting ? "上传中…" : "归档证据"}
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

function TrainDeployStep({
  state,
  models,
  datasetSummary,
  trainingStatus,
  trainingError,
  token,
  flash,
  onSaved,
  onGoCollect,
}: {
  state: Data
  models: Data[]
  datasetSummary: Data
  trainingStatus: Data
  trainingError: string
  token: string
  flash: (s: string) => void
  onSaved: () => Promise<void>
  onGoCollect: () => void
}) {
  const [epochs, setEpochs] = useState("12")
  const [inputSize, setInputSize] = useState("96")
  const [batchSize, setBatchSize] = useState("8")
  const [lr, setLr] = useState("0.0025")
  const [resumeFromLatest, setResumeFromLatest] = useState(true)
  const [useAugmentation, setUseAugmentation] = useState(true)
  const [trainFormError, setTrainFormError] = useState("")
  const [starting, setStarting] = useState(false)
  const [bootstrapping, setBootstrapping] = useState(false)

  const [onnxFile, setOnnxFile] = useState<File | null>(null)
  const [importingOnnx, setImportingOnnx] = useState(false)

  const [name, setName] = useState("opencv-mlp-cpp-igbt")
  const [version, setVersion] = useState("v1.0.0")
  const [sha, setSha] = useState("")
  const [notes, setNotes] = useState("")
  const [registering, setRegistering] = useState(false)

  const isTraining = trainingStatus.state === "TRAINING" || trainingStatus.state === "EXPORTING_ONNX"
  const latestRun = trainingStatus.latest_run
  const historyList: Data[] = Array.isArray(trainingStatus.history) ? trainingStatus.history : []
  const confusionMatrix = latestRun?.confusion_matrix
  const counts = getClassCounts(datasetSummary)
  const readyToTrain = isDatasetReady(datasetSummary)

  const handleBootstrap = async () => {
    setBootstrapping(true)
    try {
      await api<Data>("/api/dataset/bootstrap", { method: "POST" }, token)
      await onSaved()
      flash("演示训练集已就绪")
    } catch (err) {
      flash(`生成失败：${err instanceof Error ? err.message : err}`)
    } finally {
      setBootstrapping(false)
    }
  }

  const handleStartTraining = async (e: React.FormEvent) => {
    e.preventDefault()
    const epochsVal = Number(epochs)
    if (!epochs.trim() || !Number.isInteger(epochsVal) || epochsVal < 2 || epochsVal > 50) {
      const msg = "训练轮数须为 2 ~ 50 整数"
      setTrainFormError(msg)
      flash(msg)
      return
    }
    const inputSizeVal = Number(inputSize)
    if (!inputSize.trim() || !Number.isInteger(inputSizeVal) || inputSizeVal < 64 || inputSizeVal > 320) {
      const msg = "输入尺寸须为 64 ~ 320 整数"
      setTrainFormError(msg)
      flash(msg)
      return
    }
    const batchSizeVal = Number(batchSize)
    if (!batchSize.trim() || !Number.isInteger(batchSizeVal) || batchSizeVal < 2 || batchSizeVal > 64) {
      const msg = "批次大小须为 2 ~ 64 整数"
      setTrainFormError(msg)
      flash(msg)
      return
    }
    const lrVal = Number(lr)
    if (!lr.trim() || !Number.isFinite(lrVal) || lrVal <= 0 || lrVal > 0.1) {
      const msg = "学习率须在 (0, 0.1] 之间"
      setTrainFormError(msg)
      flash(msg)
      return
    }
    setTrainFormError("")
    setStarting(true)
    try {
      await api<Data>(
        "/api/training/start",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            epochs: epochsVal,
            input_size: inputSizeVal,
            batch_size: batchSizeVal,
            learning_rate: lrVal,
            resume_from_latest: resumeFromLatest,
            use_augmentation: useAugmentation,
          }),
        },
        token
      )
      flash("已启动训练任务")
      await onSaved()
    } catch (err) {
      flash(`启动失败：${err instanceof Error ? err.message : err}`)
    } finally {
      setStarting(false)
    }
  }

  const handleImportOnnx = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!onnxFile) return flash("请选择 ONNX 文件")
    const safeVersion = onnxFile.name
      .replace(/\.onnx$/i, "")
      .replace(/[^A-Za-z0-9._-]/g, "_")
      .replace(/^[_.-]+|[_.-]+$/g, "")
      .slice(0, 60)
    const headers: Record<string, string> = {
      "Content-Type": "application/octet-stream",
      "X-Input-Size": String(Math.max(64, Math.min(320, Math.round(Number(inputSize) || 96)))),
    }
    if (safeVersion) {
      headers["X-Model-Version"] = safeVersion
    }
    setImportingOnnx(true)
    try {
      const res = await api<Data>(
        "/api/models/import_onnx",
        {
          method: "POST",
          headers,
          body: onnxFile,
        },
        token
      )
      setOnnxFile(null)
      flash(`已导入并加载 ${res.model_version || "ONNX 模型"}`)
      await onSaved()
    } catch (err) {
      flash(`导入失败：${err instanceof Error ? err.message : err}`)
    } finally {
      setImportingOnnx(false)
    }
  }

  const handleActivateModel = async (modelId: string, modelVer: string) => {
    try {
      await api<Data>(`/api/models/${encodeURIComponent(modelId)}/activate`, { method: "POST" }, token)
      flash(`已启用 ${modelVer}`)
      await onSaved()
    } catch (err) {
      flash(`切换失败：${err instanceof Error ? err.message : err}`)
    }
  }

  const registerManual = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!token) return flash("请重新登录")
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
      flash("模型已登记")
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
      .catch(() => flash("复制失败"))
  }

  return (
    <div className="studio-split">
      {/* 左侧宽敞滚动训练看板 */}
      <div className="studio-board" aria-label="训练看板">
        <section className="panel">
          <header className="panel-head">
            <h2>训练数据</h2>
            <span className={`tag ${readyToTrain ? "pass" : "warn"}`}>
              {readyToTrain ? "可训练" : "缺样本"}
            </span>
            <div className="right">
              <button
                type="button"
                className="btn ghost sm"
                disabled={bootstrapping || isTraining}
                onClick={() => void handleBootstrap()}
              >
                <Sparkles size={14} aria-hidden="true" />
                {bootstrapping ? "生成中…" : "生成演示集"}
              </button>
              <button type="button" className="btn ghost sm" onClick={onGoCollect}>
                返回采集
              </button>
            </div>
          </header>
          <div className="kpi-grid-2">
            <div className="kpi-tile">
              <b className="mono">{datasetSummary.total_samples || 0}</b>
              <span>总样本数</span>
            </div>
            <div className="kpi-tile">
              <b className="mono">{datasetSummary.independent_tube_groups || 0}</b>
              <span>独立料管组</span>
            </div>
          </div>
          <div className="kpi-row kpi-row-5">
            {sampleClassDefs.map((c) => (
              <div className="readout" key={c.id}>
                <b className="mono">{counts[c.id] || 0}</b>
                <small>{c.short}</small>
              </div>
            ))}
          </div>
        </section>

        <section className="panel">
          <header className="panel-head">
            <h2>训练进度</h2>
            <div className="right">
              <span className={`tag ${isTraining ? "warn" : trainingStatus.state === "FAILED" ? "alarm" : "plain"} mono`}>
                {trainingStatus.progress_pct || 0}%
              </span>
            </div>
          </header>
          <div className="panel-body">
            {trainingError && (
              <div className="auth-error" role="alert">
                <AlertTriangle size={16} aria-hidden="true" />
                <span>{trainingError}</span>
              </div>
            )}
            <div className="flex items-center justify-between gap-4">
              <b style={{ fontSize: 16, fontWeight: 600 }}>{trainingStatus.message || "待启动"}</b>
              <span className="mono" style={{ fontSize: 14, color: "var(--ink-2)" }}>
                {trainingStatus.state || "IDLE"}
              </span>
            </div>
            <div className="meter">
              <i
                className={trainingStatus.state === "FAILED" ? "bad" : ""}
                style={{ width: `${Math.min(100, Number(trainingStatus.progress_pct || 0))}%` }}
              />
            </div>

            {historyList.length > 0 && (
              <div style={{ overflowX: "auto" }}>
                <table className="grid-table">
                  <thead>
                    <tr>
                      <th>轮次</th>
                      <th>训练损失</th>
                      <th>验证准确率</th>
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
            )}
          </div>
        </section>

        <section className="panel">
          <header className="panel-head">
            <h2>评估报告</h2>
            {latestRun && <span className="tag pass mono">{latestRun.model_version}</span>}
          </header>
          <div className="panel-body">
            {latestRun ? (
              <>
                <div className="kpi-grid-2">
                  <div className="kpi-tile">
                    <b className="mono" style={{ color: "var(--ok)" }}>
                      {latestRun.test_accuracy != null ? `${(Number(latestRun.test_accuracy) * 100).toFixed(1)}%` : "—"}
                    </b>
                    <span>测试准确率</span>
                  </div>
                  <div className="kpi-tile">
                    <b className="mono">
                      {latestRun.val_accuracy != null
                        ? `${(Number(latestRun.val_accuracy) * 100).toFixed(1)}%`
                        : `${latestRun.complete_false_accept_total ?? 0}`}
                    </b>
                    <span>{latestRun.val_accuracy != null ? "验证准确率" : "误放行次数"}</span>
                  </div>
                </div>

                {confusionMatrix && Array.isArray(confusionMatrix.labels) && Array.isArray(confusionMatrix.matrix) && (
                  <div style={{ overflowX: "auto" }}>
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
                                  fontWeight: rIdx === cIdx ? 600 : 400,
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
              <div className="empty">暂无评估报告</div>
            )}
          </div>
        </section>
      </div>

      {/* 右侧超参与模型部署检查器 */}
      <aside className="studio-inspector" aria-label="超参与模型库">
        <div className="studio-inspector-scroll-wrap">
          <div className="studio-inspector-body">
            <section className="inspector-section">
              <div className="inspector-section-head">
                <h3>训练参数</h3>
                <span className="tag pass mono" style={{ maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis" }}>
                  {state.model_version || "C++20"}
                </span>
              </div>

              <form className="grid gap-4" onSubmit={(e) => void handleStartTraining(e)} noValidate>
                <div className="grid grid-cols-2 gap-4">
                  <div className="field">
                    <label htmlFor="train-epochs">训练轮数</label>
                    <Input
                      id="train-epochs"
                      type="number"
                      min="2"
                      max="50"
                      step="1"
                      value={epochs}
                      placeholder="2-50"
                      onChange={(e) => {
                        setEpochs(e.target.value)
                        if (trainFormError) setTrainFormError("")
                      }}
                      className="mono"
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="train-input-size">输入尺寸</label>
                    <Input
                      id="train-input-size"
                      type="number"
                      min="64"
                      max="320"
                      step="1"
                      value={inputSize}
                      placeholder="64-320"
                      onChange={(e) => {
                        setInputSize(e.target.value)
                        if (trainFormError) setTrainFormError("")
                      }}
                      className="mono"
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="train-batch-size">批次大小</label>
                    <Input
                      id="train-batch-size"
                      type="number"
                      min="2"
                      max="64"
                      step="1"
                      value={batchSize}
                      placeholder="2-64"
                      onChange={(e) => {
                        setBatchSize(e.target.value)
                        if (trainFormError) setTrainFormError("")
                      }}
                      className="mono"
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="train-lr">学习率</label>
                    <Input
                      id="train-lr"
                      type="number"
                      min="0.0001"
                      max="0.1"
                      step="0.0005"
                      value={lr}
                      placeholder="默认"
                      onChange={(e) => {
                        setLr(e.target.value)
                        if (trainFormError) setTrainFormError("")
                      }}
                      className="mono"
                    />
                  </div>
                </div>

                <div className="grid gap-3">
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={resumeFromLatest}
                      onChange={(e) => setResumeFromLatest(e.target.checked)}
                    />
                    增量微调
                  </label>
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={useAugmentation}
                      onChange={(e) => setUseAugmentation(e.target.checked)}
                    />
                    光学增强
                  </label>
                </div>

                {trainFormError && (
                  <div className="auth-error" role="alert">
                    <AlertTriangle size={16} aria-hidden="true" />
                    <span>{trainFormError}</span>
                  </div>
                )}

                <button
                  type="submit"
                  className="btn block"
                  disabled={isTraining || starting || !readyToTrain}
                >
                  <Play size={16} aria-hidden="true" />
                  {isTraining ? `训练中 ${trainingStatus.progress_pct || 0}%` : "开始训练"}
                </button>
              </form>
            </section>

            <section className="inspector-section">
              <div className="inspector-section-head">
                <h3>模型版本</h3>
                <span className="mono" style={{ fontSize: 14, color: "var(--ink-2)" }}>
                  {models.length}
                </span>
              </div>
              {models.length ? (
                <div className="model-list">
                  {models.map((m) => {
                    const isActive = state.model_version === m.model_version
                    const tip = `${m.model_name} · ${m.sha256 || ""} · ${dateTimeOf(m.created_at_utc)}`
                    return (
                      <div className="model-row" key={m.model_id} title={tip}>
                        <div className="name mono">{m.model_version}</div>
                        <div className="row-actions">
                          <button
                            type="button"
                            className="icon-btn"
                            onClick={() => copyHash(String(m.sha256 || ""))}
                            title="复制 SHA256"
                            aria-label={`复制 ${m.model_version} 哈希`}
                          >
                            <Copy size={14} aria-hidden="true" />
                          </button>
                          {isActive ? (
                            <span className="tag pass">运行中</span>
                          ) : (
                            <button
                              type="button"
                              className="btn ghost sm"
                              onClick={() => void handleActivateModel(m.model_id, m.model_version)}
                            >
                              启用
                            </button>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              ) : (
                <div className="empty">暂无模型</div>
              )}
            </section>

            <details className="inspector-group">
              <summary>
                <span>导入 ONNX 权重</span>
              </summary>
              <form className="inspector-group-body" onSubmit={(e) => void handleImportOnnx(e)}>
                <div className="field">
                  <label htmlFor="onnx-import-file">权重文件</label>
                  <Input
                    id="onnx-import-file"
                    type="file"
                    accept=".onnx,application/octet-stream"
                    onChange={(e) => setOnnxFile(e.target.files?.[0] || null)}
                  />
                </div>
                <button type="submit" className="btn block" disabled={importingOnnx || !onnxFile}>
                  <Upload size={16} aria-hidden="true" />
                  {importingOnnx ? "导入中…" : "导入并启用"}
                </button>
              </form>
            </details>

            <details className="inspector-group">
              <summary>
                <span>手动登记哈希</span>
              </summary>
              <form className="inspector-group-body" onSubmit={(e) => void registerManual(e)}>
                <div className="grid grid-cols-2 gap-4">
                  <div className="field">
                    <label htmlFor="model-reg-name">模型名称</label>
                    <Input
                      id="model-reg-name"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      required
                      placeholder="名称"
                      className="mono"
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="model-reg-version">版本号</label>
                    <Input
                      id="model-reg-version"
                      value={version}
                      onChange={(e) => setVersion(e.target.value)}
                      required
                      placeholder="如v1"
                      className="mono"
                    />
                  </div>
                </div>
                <div className="field">
                  <label htmlFor="model-reg-sha256">校验哈希</label>
                  <Input
                    id="model-reg-sha256"
                    className="mono"
                    value={sha}
                    onChange={(e) => setSha(e.target.value)}
                    required
                    pattern="[0-9a-fA-F]{64}"
                    maxLength={64}
                    placeholder="哈希"
                  />
                </div>
                <div className="field">
                  <label htmlFor="model-reg-notes">备注说明</label>
                  <Input
                    id="model-reg-notes"
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    placeholder="选填"
                  />
                </div>
                <button type="submit" className="btn block" disabled={registering}>
                  {registering ? "登记中…" : "确认登记"}
                </button>
              </form>
            </details>
          </div>
        </div>
      </aside>
    </div>
  )
}

const step4SubTabs: { id: RecordsSettingsTab; label: string; icon: typeof ClipboardList }[] = [
  { id: "records", label: "周期记录", icon: ClipboardList },
  { id: "storage", label: "存储管理", icon: HardDrive },
  { id: "settings", label: "系统参数", icon: Sliders },
]

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
  storageError,
  isInventoryAvailable,
  cycles,
  evidence,
  events,
  eventsLoading,
  eventsError,
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
  onRetryEvents,
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
  storageError: string
  isInventoryAvailable: () => boolean
  cycles: Data[]
  evidence: Data[]
  events: Data[]
  eventsLoading: boolean
  eventsError: string
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
  onRetryEvents: () => void
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
  const subTabRefs = useRef<Partial<Record<RecordsSettingsTab, HTMLButtonElement | null>>>({})

  const handleSubTabKeyDown = (e: React.KeyboardEvent<HTMLButtonElement>, currentId: RecordsSettingsTab) => {
    const idx = step4SubTabs.findIndex((t) => t.id === currentId)
    let nextIdx = -1
    if (e.key === "ArrowRight") nextIdx = (idx + 1) % step4SubTabs.length
    else if (e.key === "ArrowLeft") nextIdx = (idx - 1 + step4SubTabs.length) % step4SubTabs.length
    else if (e.key === "Home") nextIdx = 0
    else if (e.key === "End") nextIdx = step4SubTabs.length - 1
    if (nextIdx >= 0) {
      e.preventDefault()
      const nextTab = step4SubTabs[nextIdx].id
      onChangeSubTab(nextTab)
      subTabRefs.current[nextTab]?.focus()
    }
  }

  const requestPurgeSimulated = () => {
    onConfirm({
      title: "清理演练数据",
      description: "确认清理全部模拟演练周期（CYC_*）及其关联事件与证据图吗？",
      consequences: ["仅删除模拟演练产生的记录", "保留真实产线周期、训练样本与模型"],
      confirmLabel: "确认清理",
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
          flash(`已清理 ${res.deleted_cycles ?? 0} 条演练记录`)
          await onSaved()
        } catch (err) {
          flash(`清理失败：${err instanceof Error ? err.message : err}`)
        }
      },
    })
  }

  return (
    <div className="studio-step4">
      <div className="studio-subbar">
        <div className="studio-subtabs" role="tablist" aria-label="记录与设置">
          {step4SubTabs.map((tab) => {
            const Icon = tab.icon
            const selected = activeSubTab === tab.id
            return (
              <button
                key={tab.id}
                ref={(el) => {
                  subTabRefs.current[tab.id] = el
                }}
                id={`step4-tab-${tab.id}`}
                type="button"
                role="tab"
                tabIndex={selected ? 0 : -1}
                aria-selected={selected}
                aria-controls="step4-tabpanel"
                className={`studio-subtab ${selected ? "on" : ""}`}
                onClick={() => onChangeSubTab(tab.id)}
                onKeyDown={(e) => handleSubTabKeyDown(e, tab.id)}
              >
                <Icon size={16} aria-hidden="true" />
                {tab.label}
              </button>
            )
          })}
        </div>

        <div className="studio-subbar-right">
          {activeSubTab === "records" ? (
            <>
              <span className="tag plain mono">总计 {totalCount}</span>
              <span className="tag pass mono">放行 {passCount}</span>
              <span className="tag alarm mono">报警 {alarmCount}</span>
              <button
                type="button"
                className="btn ghost sm btn-outline-bad"
                onClick={requestPurgeSimulated}
              >
                <Trash2 size={14} aria-hidden="true" />
                清理演练
              </button>
              <button type="button" className="btn sm" onClick={onManualCapture}>
                <Camera size={14} aria-hidden="true" />
                抓拍
              </button>
            </>
          ) : activeSubTab === "storage" ? (
            <button
              type="button"
              className="btn ghost sm"
              disabled={storageLoading}
              onClick={onRefreshStorage}
            >
              <RefreshCw size={14} aria-hidden="true" />
              {storageLoading ? "刷新中…" : "刷新统计"}
            </button>
          ) : (
            <span className="tag plain mono">
              {!config.roi_config_version || config.roi_config_version === "ROI_UNCALIBRATED"
                ? "五区标定"
                : config.roi_config_version}
            </span>
          )}
        </div>
      </div>

      <div
        id="step4-tabpanel"
        role="tabpanel"
        aria-labelledby={`step4-tab-${activeSubTab}`}
        style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column", overflow: "hidden" }}
      >
        {activeSubTab === "records" ? (
          <RecordsSubView
            cycles={cycles}
            evidence={evidence}
            events={events}
            eventsLoading={eventsLoading}
            eventsError={eventsError}
            selectedCycleId={selectedCycleId}
            filmstripItems={filmstripItems}
            activeSnapshot={activeSnapshot}
            token={token}
            flash={flash}
            onConfirm={onConfirm}
            onSelectCycle={onSelectCycle}
            onRetryEvents={onRetryEvents}
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
            storageError={storageError}
            isInventoryAvailable={isInventoryAvailable}
            token={token}
            flash={flash}
            onConfirm={onConfirm}
            onSaved={onSaved}
            onRefreshStorage={onRefreshStorage}
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
    </div>
  )
}

function RecordsSubView({
  cycles,
  evidence,
  events,
  eventsLoading,
  eventsError,
  selectedCycleId,
  filmstripItems,
  activeSnapshot,
  token,
  flash,
  onConfirm,
  onSelectCycle,
  onRetryEvents,
  onClearSelectedCycle,
  onOpenSnapshot,
  onSaved,
}: {
  cycles: Data[]
  evidence: Data[]
  events: Data[]
  eventsLoading: boolean
  eventsError: string
  selectedCycleId: string
  filmstripItems: FilmstripItem[]
  activeSnapshot: FilmstripItem | null
  token: string
  flash: (s: string) => void
  onConfirm: (dialog: ConfirmDialogState) => void
  onSelectCycle: (cycleId: string) => void
  onRetryEvents: () => void
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
      title: "删除周期",
      description: `确认永久删除周期 ${cycleId} 及其证据文件吗？`,
      confirmLabel: "删除",
      tone: "danger",
      onConfirm: async () => {
        try {
          await api<Data>(`/api/cycles/${encodeURIComponent(cycleId)}`, { method: "DELETE" }, token)
          if (selectedCycleId === cycleId) {
            onClearSelectedCycle()
          }
          flash(`已删除 ${cycleId}`)
          await onSaved()
        } catch (err) {
          flash(`删除失败：${err instanceof Error ? err.message : err}`)
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
            <h2>周期列表</h2>
            <span className="sub mono">{filteredCycles.length}</span>
            <div className="right">
              <div className="sim-bar" role="group" aria-label="判定筛选">
                {(
                  [
                    ["ALL", "全部"],
                    ["PASS", "放行"],
                    ["ALARM", "报警"],
                    ["VIEW_INVALID", "受阻"],
                  ] as [CycleFilter, string][]
                ).map(([val, lbl]) => (
                  <button
                    key={val}
                    type="button"
                    className={`sim-btn ${filter === val ? "busy" : ""}`}
                    onClick={() => setFilter(val)}
                  >
                    {lbl}
                  </button>
                ))}
              </div>
            </div>
          </header>
          <div className="records-scroll">
            <table className="grid-table records-table">
              <thead>
                <tr>
                  <th>周期编号</th>
                  <th>判定结果</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {filteredCycles.length ? (
                  filteredCycles.map((row) => {
                    const matched = evidence.find((e) => e.cycle_id === row.cycle_id)
                    const rowTip = `${pipeText(row.pipe_color)} · ${dateTimeOf(row.updated_at_utc)} · ${reasonOf(row.result_reason) || "—"}`
                    return (
                      <tr
                        key={row.cycle_id}
                        className={`click ${selectedCycleId === row.cycle_id ? "sel" : ""}`}
                        title={rowTip}
                        onClick={() => onSelectCycle(row.cycle_id)}
                      >
                        <td className="mono">{row.cycle_id}</td>
                        <td>
                          <OutcomeTag value={row.final_outcome} />
                        </td>
                        <td>
                          <div className="row-actions">
                            <button
                              type="button"
                              className="btn ghost sm"
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
                                  reason: row.result_reason,
                                })
                              }}
                            >
                              查看
                            </button>
                            <button
                              type="button"
                              className="btn ghost sm btn-outline-bad"
                              aria-label={`删除 ${row.cycle_id}`}
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
                    <td colSpan={3}>
                      <div className="empty">暂无记录</div>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>

        <section className="panel records-pane">
          <header className="panel-head">
            <h2>状态时间线</h2>
            <span className="sub mono">{selectedCycleId || ""}</span>
            <div className="right mono">
              {eventsLoading ? "加载中…" : `${events.length}`}
            </div>
          </header>
          <div className="records-scroll">
            {eventsLoading && events.length === 0 ? (
              <div className="empty">加载中…</div>
            ) : eventsError ? (
              <div className="panel-body">
                <div className="auth-error" role="alert">
                  <AlertTriangle size={16} aria-hidden="true" />
                  <span>{eventsError}</span>
                </div>
                <div>
                  <button type="button" className="btn ghost sm" onClick={onRetryEvents}>
                    <RefreshCw size={14} aria-hidden="true" />
                    重试
                  </button>
                </div>
              </div>
            ) : events.length ? (
              <ol className="timeline">
                {events.map((row) => (
                  <li key={row.event_id} title={reasonOf(row.reason) || undefined}>
                    <div className="row-main">
                      <b>{human(row.event_type)}</b>
                      <span>{`${human(row.state_before)} → ${human(row.state_after)}`}</span>
                    </div>
                    <time>{timeOf(row.created_at_utc)}</time>
                  </li>
                ))}
              </ol>
            ) : (
              <div className="empty">{selectedCycleId ? "暂无事件" : "选择左侧周期"}</div>
            )}
          </div>
        </section>
      </div>
    </div>
  )
}

function StorageManagerSubView({
  state,
  models,
  datasetSummary,
  storageOverview,
  storageLoading,
  storageError,
  isInventoryAvailable,
  token,
  flash,
  onConfirm,
  onSaved,
  onRefreshStorage,
  onClearSelectedCycle,
  onChangeSubTab,
  onGoStep,
}: {
  state: Data
  models: Data[]
  datasetSummary: Data
  storageOverview: Data | null
  storageLoading: boolean
  storageError: string
  isInventoryAvailable: () => boolean
  token: string
  flash: (s: string) => void
  onConfirm: (dialog: ConfirmDialogState) => void
  onSaved: () => Promise<void>
  onRefreshStorage: () => void
  onClearSelectedCycle: () => void
  onChangeSubTab: (tab: RecordsSettingsTab) => void
  onGoStep: (step: EngineerStep, subTab?: RecordsSettingsTab) => void
}) {
  const bdb = storageOverview?.business_db || {}
  const tables = bdb.tables || {}
  const ds = storageOverview?.training_dataset || {}
  const ma = storageOverview?.model_artifacts || {}
  const hasInventory = Boolean(storageOverview?.business_db && storageOverview.business_db.tables && !storageError)
  const formatInventoryCount = (value: unknown) =>
    typeof value === "number" && Number.isFinite(value) ? String(value) : "—"
  const classCounts = ds.class_counts || getClassCounts(datasetSummary)
  const activeModelVersion = ma.active_model_version || state.model_version || ""

  const sqliteTableItems = [
    { key: "cycles", name: "周期", count: hasInventory ? tables.cycles : undefined },
    { key: "events", name: "事件", count: hasInventory ? tables.events : undefined },
    { key: "evidence", name: "证据", count: hasInventory ? tables.evidence : undefined },
    { key: "configuration_history", name: "参数", count: hasInventory ? tables.configuration_history : undefined },
    { key: "model_registry", name: "模型", count: hasInventory ? tables.model_registry : undefined },
  ]

  const copyHash = (hash: string) => {
    navigator.clipboard
      .writeText(hash)
      .then(() => flash("已复制 SHA256"))
      .catch(() => flash("复制失败"))
  }

  const handleActivateModel = async (modelId: string, modelVer: string) => {
    try {
      await api<Data>(`/api/models/${encodeURIComponent(modelId)}/activate`, { method: "POST" }, token)
      flash(`已启用 ${modelVer}`)
      await onSaved()
    } catch (err) {
      flash(`切换失败：${err instanceof Error ? err.message : err}`)
    }
  }

  const requestDeleteModel = (m: Data) => {
    const modelId = String(m.model_id)
    const modelVer = String(m.model_version)
    onConfirm({
      title: "销毁模型版本",
      description: `确认永久销毁模型 ${m.model_name} (${modelVer}) 吗？`,
      consequences: ["删除数据库模型记录及磁盘权重目录", "运行中模型受保护不会被误删"],
      confirmLabel: "确认销毁",
      tone: "danger",
      onConfirm: async () => {
        try {
          await api<Data>(`/api/models/${encodeURIComponent(modelId)}`, { method: "DELETE" }, token)
          flash(`已销毁 ${modelVer}`)
          await onSaved()
        } catch (err) {
          flash(`销毁失败：${err instanceof Error ? err.message : err}`)
        }
      },
    })
  }

  const requestPurgeSimulated = () => {
    onConfirm({
      title: "清理演练周期",
      description: "确认清理全部演练周期（CYC_*）及对应事件与证据图片吗？",
      confirmLabel: "确认清理",
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
          flash(`已清理 ${res.deleted_cycles ?? 0} 条演练周期`)
          await onSaved()
        } catch (err) {
          flash(`清理失败：${err instanceof Error ? err.message : err}`)
        }
      },
    })
  }

  const requestClearTrainingSamples = () => {
    const totalSamples = Number(ds.total_samples ?? datasetSummary.total_samples ?? 0)
    onConfirm({
      title: "清空训练样本",
      description: `确认清空全部 ${totalSamples} 张训练样本吗？`,
      confirmLabel: "确认清空",
      tone: "danger",
      onConfirm: async () => {
        try {
          await api<Data>("/api/dataset/clear", { method: "POST" }, token)
          flash("已清空训练样本")
          await onSaved()
        } catch (err) {
          flash(`清空失败：${err instanceof Error ? err.message : err}`)
        }
      },
    })
  }

  const requestDestroyAllBusiness = () => {
    if (!hasInventory) {
      flash("请先刷新统计")
      return
    }
    const cyclesText = formatInventoryCount(tables.cycles)
    const eventsText = formatInventoryCount(tables.events)
    const evidenceText = formatInventoryCount(tables.evidence)
    onConfirm({
      title: "重置业务数据库",
      description: "高危操作：永久清空全部检测周期、状态事件与磁盘证据快照，并执行 VACUUM 压缩。",
      consequences: [
        `删除 ${cyclesText} 条周期、${eventsText} 条事件与 ${evidenceText} 条证据`,
        "保留系统参数历史与模型版本库",
      ],
      confirmLabel: "确认销毁并压缩",
      tone: "danger",
      requireCheckLabel: "确认永久删除全部业务周期与证据快照",
      requiresInventory: true,
      onConfirm: async () => {
        if (!isInventoryAvailable()) {
          flash("存储统计已失效，已中止")
          return
        }
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
          flash(`已清空 ${res.deleted_cycles ?? 0} 条周期并完成压缩`)
          await onSaved()
        } catch (err) {
          flash(`执行失败：${err instanceof Error ? err.message : err}`)
        }
      },
    })
  }

  return (
    <div className="storage-layout" aria-label="存储管理中心">
      {storageError && (
        <div className="auth-error" role="alert" style={{ justifyContent: "space-between" }}>
          <span className="inline-flex items-center gap-2">
            <AlertTriangle size={16} aria-hidden="true" />
            <span>统计获取失败：{storageError}</span>
          </span>
          <button
            type="button"
            className="btn ghost sm"
            disabled={storageLoading}
            onClick={onRefreshStorage}
          >
            <RefreshCw size={14} aria-hidden="true" />
            重试
          </button>
        </div>
      )}

      <div className="storage-arch-grid">
        <section
          className="panel storage-card"
          title={`${bdb.db_path || "data/station.db"} · ${bdb.evidence_dir || "data/evidence"}`}
        >
          <header className="panel-head">
            <h2 className="flex items-center gap-2">
              <HardDrive size={16} aria-hidden="true" />
              业务数据库
            </h2>
            <div className="right mono">
              {hasInventory ? `${fmt(bdb.db_size_kb, 1)} KB` : "—"}
            </div>
          </header>

          <div className="storage-metric-grid" aria-label="业务表行数">
            {sqliteTableItems.map((t) => (
              <div className="storage-metric-cell" key={t.key} title={t.key}>
                <b>{hasInventory ? (t.count ?? "—") : "—"}</b>
                <small>{t.name}</small>
              </div>
            ))}
          </div>

          <footer className="storage-card-foot">
            <button type="button" className="btn ghost sm" onClick={() => onChangeSubTab("records")}>
              周期记录
            </button>
          </footer>
        </section>

        <section
          className="panel storage-card"
          title={`${ds.dataset_dir || "data/dataset"} · ${ma.runs_dir || "data/runs/label_classifier"}`}
        >
          <header className="panel-head">
            <h2 className="flex items-center gap-2">
              <Database size={16} aria-hidden="true" />
              样本与模型库
            </h2>
            <div className="right mono">
              {`${fmt(ds.dataset_size_mb, 2)} MB`}
            </div>
          </header>

          <div className="storage-metric-grid" aria-label="类别切片数">
            {sampleClassDefs.map((c) => (
              <div className="storage-metric-cell" key={c.id}>
                <b>{classCounts[c.id] ?? 0}</b>
                <small>{c.short}</small>
              </div>
            ))}
          </div>

          <footer className="storage-card-foot">
            <button type="button" className="btn ghost sm" onClick={() => onGoStep("collect")}>
              采集标注
            </button>
            <button type="button" className="btn sm" onClick={() => onGoStep("train")}>
              训练部署
            </button>
          </footer>
        </section>
      </div>

      <section className="panel" aria-label="模型管理">
        <header className="panel-head">
          <h2>模型管理</h2>
          <div className="right">
            <span className="protected-lock-badge">
              <Lock size={14} aria-hidden="true" />
              运行模型受保护
            </span>
          </div>
        </header>

        <div style={{ overflowX: "auto" }}>
          <table className="grid-table">
            <thead>
              <tr>
                <th>模型版本</th>
                <th>状态</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {models.length > 0 ? (
                models.map((m) => {
                  const isActive = Boolean(activeModelVersion && m.model_version === activeModelVersion)
                  const rowTip = `${m.model_name} · SHA256: ${m.sha256 || "—"} · ${dateTimeOf(m.created_at_utc)}${m.notes ? ` · ${m.notes}` : ""}`
                  return (
                    <tr key={m.model_id} title={rowTip}>
                      <td className="mono" style={{ fontWeight: 600 }}>
                        {m.model_version}
                      </td>
                      <td>
                        {isActive ? (
                          <span className="protected-lock-badge">
                            <Lock size={14} aria-hidden="true" />
                            运行中
                          </span>
                        ) : (
                          <span className="tag plain">已归档</span>
                        )}
                      </td>
                      <td>
                        <div className="row-actions">
                          <button
                            type="button"
                            className="icon-btn"
                            onClick={() => copyHash(String(m.sha256 || ""))}
                            title="复制 SHA256"
                            aria-label={`复制 ${m.model_version} 哈希`}
                          >
                            <Copy size={14} aria-hidden="true" />
                          </button>
                          {!isActive && (
                            <>
                              <button
                                type="button"
                                className="btn ghost sm"
                                onClick={() => void handleActivateModel(String(m.model_id), String(m.model_version))}
                              >
                                启用
                              </button>
                              <button
                                type="button"
                                className="btn ghost sm btn-outline-bad"
                                onClick={() => requestDeleteModel(m)}
                              >
                                销毁
                              </button>
                            </>
                          )}
                        </div>
                      </td>
                    </tr>
                  )
                })
              ) : (
                <tr>
                  <td colSpan={3}>
                    <div className="empty">暂无模型</div>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel" aria-label="数据清理">
        <header className="panel-head">
          <h2>数据清理</h2>
        </header>
        <div className="danger-action-grid">
          <div className="danger-action-card">
            <h3>清理演练周期</h3>
            <button
              type="button"
              className="btn ghost btn-outline-bad block"
              onClick={requestPurgeSimulated}
            >
              <Trash2 size={16} aria-hidden="true" />
              清理演练
            </button>
          </div>

          <div className="danger-action-card">
            <h3>清空训练样本</h3>
            <button
              type="button"
              className="btn ghost btn-outline-bad block"
              onClick={requestClearTrainingSamples}
            >
              <Trash2 size={16} aria-hidden="true" />
              清空样本
            </button>
          </div>

          <div className="danger-action-card severe">
            <h3>重置业务数据库</h3>
            <button
              type="button"
              className="btn btn-danger block"
              disabled={storageLoading || !hasInventory}
              onClick={requestDestroyAllBusiness}
            >
              <ShieldAlert size={16} aria-hidden="true" />
              {hasInventory ? "全量销毁并压缩" : "请先刷新统计"}
            </button>
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
  const [width, setWidth] = useState(String(config.camera_width || DEFAULT_CAMERA_WIDTH))
  const [height, setHeight] = useState(String(config.camera_height || DEFAULT_CAMERA_HEIGHT))
  const [device, setDevice] = useState(config.camera_device_id || "CAM_0")
  const [both, setBoth] = useState(config.camera_covers_both_ends !== false)
  const [thresholds, setThresholds] = useState<Data>(config.thresholds || {})
  const [target, setTarget] = useState(roiNames[0])
  const [image, setImage] = useState<HTMLImageElement | null>(null)
  const [drag, setDrag] = useState<[number, number] | null>(null)
  const [canvasSize, setCanvasSize] = useState<[number, number]>([DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT])
  const canvasRefElem = useRef<HTMLCanvasElement | null>(null)

  const activeRois = both ? roiNames : roiNames.slice(0, 3)
  const showAux = Boolean(state.show_auxiliary_rois)
  const workMode = state.work_mode === "strict" ? "strict" : "debug"

  const loadPresetFrame = useCallback((src: string) => {
    const im = new Image()
    im.onload = () => {
      setImage(im)
      setCanvasSize([im.naturalWidth || DEFAULT_CAMERA_WIDTH, im.naturalHeight || DEFAULT_CAMERA_HEIGHT])
    }
    im.src = src
  }, [])

  const prevMirrorRef = useRef<boolean>(mirrorHorizontal)

  useEffect(() => {
    const im = new Image()
    im.onload = () => {
      setImage(im)
      setCanvasSize([im.naturalWidth || DEFAULT_CAMERA_WIDTH, im.naturalHeight || DEFAULT_CAMERA_HEIGHT])
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
      ctx.font = '600 16px "Geist Variable", "Microsoft YaHei UI", sans-serif'
      const tw = ctx.measureText(text).width + 16
      const boxH = 24
      ctx.fillStyle = color
      ctx.fillRect(rx, ry - boxH < 0 ? ry : ry - boxH, tw, boxH)
      ctx.fillStyle = "#fff"
      ctx.fillText(text, rx + 8, (ry - boxH < 0 ? ry : ry - boxH) + 18)
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
      setCanvasSize([im.naturalWidth || DEFAULT_CAMERA_WIDTH, im.naturalHeight || DEFAULT_CAMERA_HEIGHT])
      URL.revokeObjectURL(url)
    }
    im.onerror = () => {
      URL.revokeObjectURL(url)
      flash("无法解析所选图片")
    }
    im.src = url
  }

  const save = async () => {
    if (!token) return flash("请重新登录")
    const required = both ? roiNames : roiNames.slice(0, 3)
    const invalidRois = required.filter((k) => !isValidRoiBox(rois[k]))
    if (invalidRois.length > 0) {
      return flash(`区域无效：${invalidRois.map((k) => roiLabels[k]).join("、")}`)
    }
    try {
      await api(
        "/api/config",
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            camera_device_id: device || "CAM_DEFAULT",
            camera_width: Number(width) || DEFAULT_CAMERA_WIDTH,
            camera_height: Number(height) || DEFAULT_CAMERA_HEIGHT,
            camera_covers_both_ends: both,
            rois: Object.fromEntries(Object.entries(rois).filter(([k]) => required.includes(k))),
            thresholds,
          }),
        },
        token
      )
      flash("参数已保存")
      onSaved()
    } catch (e) {
      flash(`保存失败：${e instanceof Error ? e.message : e}`)
    }
  }

  return (
    <div className="studio-split">
      <section className="studio-canvas" aria-label="五区标定画布">
        <header className="studio-canvas-bar">
          <h2>五区标定</h2>
          <span className="sub">{roiLabels[target]}</span>
          <div className="right">
            <button
              type="button"
              className="btn sm"
              onClick={() => loadPresetFrame(`/api/camera/frame.jpg?t=${Date.now()}`)}
            >
              实时帧
            </button>
            <button type="button" className="btn ghost sm" onClick={() => loadPresetFrame(tubeYellowImg)}>
              黄管
            </button>
            <button type="button" className="btn ghost sm" onClick={() => loadPresetFrame(tubeBlueImg)}>
              蓝管
            </button>
            <button
              type="button"
              className="btn ghost sm"
              onClick={() => {
                setBoth(true)
                setRois(defaultStandardRois)
              }}
            >
              重置
            </button>
            <label htmlFor="settings-roi-image" className="btn ghost sm file-btn">
              <Upload size={14} aria-hidden="true" />
              载入
              <input
                id="settings-roi-image"
                type="file"
                accept="image/*"
                onChange={(e) => loadImage(e.target.files?.[0])}
              />
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
                const box: [number, number, number, number] = [
                  Math.min(drag[0], p[0]),
                  Math.min(drag[1], p[1]),
                  Math.max(drag[0], p[0]),
                  Math.max(drag[1], p[1]),
                ]
                setDrag(null)
                if (!isValidRoiBox(box)) {
                  flash("选区过小")
                  return
                }
                setRois((r: Data) => ({ ...r, [target]: box }))
              }}
              onPointerCancel={() => setDrag(null)}
            />
          </div>
        </div>

        <div className="roi-bar" aria-label="选择标定区">
          {activeRois.map((r, i) => (
            <button key={r} type="button" className={target === r ? "on" : ""} onClick={() => setTarget(r)}>
              <div className="roi-bar-top">
                <i style={{ background: roiColors[i] }} />
                <span>{roiLabels[r]}</span>
              </div>
              <code className="mono">
                {Array.isArray(rois[r]) ? rois[r].map((v: number) => v.toFixed(2)).join(" ") : "未标定"}
              </code>
            </button>
          ))}
        </div>
      </section>

      <aside className="studio-inspector" aria-label="参数设置栏">
        <div className="studio-inspector-scroll-wrap">
          <div className="studio-inspector-body">
            <section className="inspector-section">
              <div className="inspector-section-head">
                <h3>画面与模式</h3>
                <button type="button" className="btn sm" onClick={() => void save()}>
                  保存
                </button>
              </div>

              <div className="toggle-row">
                <div className="toggle-copy">
                  <b>
                    <FlipHorizontal2 size={16} aria-hidden="true" />
                    水平镜像
                  </b>
                </div>
                <button
                  type="button"
                  role="switch"
                  aria-checked={mirrorHorizontal}
                  aria-label="水平镜像开关"
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
                    <Crosshair size={16} aria-hidden="true" />
                    辅助标定框
                  </b>
                </div>
                <button
                  type="button"
                  role="switch"
                  aria-checked={showAux}
                  aria-label="辅助标定框开关"
                  className={`studio-switch ${showAux ? "on" : ""}`}
                  onClick={() => onToggleAuxRois(!showAux)}
                >
                  <span className="studio-switch-thumb" />
                </button>
              </div>

              <div className="sim-bar" style={{ width: "100%", display: "grid", gridTemplateColumns: "1fr 1fr" }}>
                <button
                  type="button"
                  className={`sim-btn ${workMode === "debug" ? "busy" : ""}`}
                  onClick={() => onSwitchWorkMode("debug")}
                >
                  桌面调试
                </button>
                <button
                  type="button"
                  className={`sim-btn ${workMode === "strict" ? "busy" : ""}`}
                  onClick={() => onSwitchWorkMode("strict")}
                >
                  车间严格
                </button>
              </div>
            </section>

            <section className="inspector-section">
              <div className="inspector-section-head">
                <h3>光学阈值</h3>
              </div>
              <div className="th-grid">
                {thresholdFields.map(([key, lbl, step]) => (
                  <div className="th-row" key={key}>
                    <label htmlFor={`threshold-${key}`}>{lbl}</label>
                    <Input
                      id={`threshold-${key}`}
                      type="number"
                      step={step}
                      value={thresholds[key] ?? ""}
                      placeholder="默认"
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
            </section>

            <section className="inspector-section">
              <div className="inspector-section-head">
                <h3>相机规格</h3>
                <span className="tag plain mono">720P · 25 FPS</span>
              </div>
              <div className="field">
                <label htmlFor="settings-camera-device">设备编号</label>
                <Input
                  id="settings-camera-device"
                  value={device}
                  placeholder="如C0"
                  onChange={(e) => setDevice(e.target.value)}
                  className="mono"
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="field">
                  <label htmlFor="settings-camera-width">画面宽度</label>
                  <Input
                    id="settings-camera-width"
                    type="number"
                    value={width}
                    placeholder="1280"
                    onChange={(e) => setWidth(e.target.value)}
                    className="mono"
                  />
                </div>
                <div className="field">
                  <label htmlFor="settings-camera-height">画面高度</label>
                  <Input
                    id="settings-camera-height"
                    type="number"
                    value={height}
                    placeholder="720"
                    onChange={(e) => setHeight(e.target.value)}
                    className="mono"
                  />
                </div>
              </div>
              <label className="check">
                <input type="checkbox" checked={both} onChange={(e) => setBoth(e.target.checked)} />
                双端覆盖
              </label>
              <button type="button" className="btn block" onClick={() => void save()}>
                保存配置
              </button>
            </section>

            <details className="inspector-group">
              <summary>
                <span>版本历史</span>
                <span className="mono">{history.length}</span>
              </summary>
              <div className="inspector-group-body">
                {history.length ? (
                  <ol className="timeline">
                    {history.slice(0, 10).map((h) => (
                      <li key={h.config_version}>
                        <div className="row-main">
                          <b className="mono">#{h.config_version}</b>
                          <span className="mono">{h.roi_config_version}</span>
                        </div>
                        <time>{dateTimeOf(h.created_at_utc)}</time>
                      </li>
                    ))}
                  </ol>
                ) : (
                  <div className="empty">暂无历史</div>
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
