import { useCallback, useEffect, useState } from "react"
import { RefreshCw, ShieldAlert, ShieldCheck, Sliders, Upload } from "lucide-react"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { type Data, fmt, reasonOf } from "@/lib/station"

interface LabelOodCardProps {
  state: Data
  token: string
  flash: (msg: string) => void
  onUpdated: () => Promise<void> | void
}

const MODE_LABELS: Record<string, { text: string; tone: string }> = {
  onnx_dual_ood: { text: "ONNX 双门控已标定", tone: "pass" },
  onnx_energy_only: { text: "ONNX Energy 单门控", tone: "pass" },
  onnx_uncalibrated: { text: "ONNX 未标定", tone: "warn" },
  mlp_ood: { text: "ANN_MLP 双门控已标定", tone: "pass" },
  mlp_uncalibrated: { text: "ANN_MLP 未标定", tone: "warn" },
  heuristic_fallback: { text: "启发式回退", tone: "plain" },
}

async function oodApi<T = Data>(path: string, options: RequestInit = {}, token = ""): Promise<T> {
  const headers = new Headers(options.headers)
  if (token) headers.set("X-Admin-Token", token)
  const res = await fetch(path, { ...options, headers })
  let body: Record<string, unknown>
  try {
    body = (await res.json()) as Record<string, unknown>
  } catch {
    body = {}
  }
  if (!res.ok) {
    const detail = typeof body.detail === "string" && body.detail ? body.detail : `HTTP ${res.status}`
    throw new Error(detail)
  }
  return body as T
}

export function LabelOodCard({ state, token, flash, onUpdated }: LabelOodCardProps) {
  const initialClassifier = state.label_ood_classifier as Data | undefined
  const [liveStatus, setLiveStatus] = useState<Data | null>(null)
  const [quantile, setQuantile] = useState("0.98")
  const [probThreshold, setProbThreshold] = useState(() =>
    initialClassifier?.calibration?.prob_threshold != null
      ? String(initialClassifier.calibration.prob_threshold)
      : "0.85"
  )
  const [gateEnabled, setGateEnabled] = useState(() =>
    typeof initialClassifier?.gate_config?.gate_enabled === "boolean"
      ? initialClassifier.gate_config.gate_enabled
      : true
  )
  const [alarmStreak, setAlarmStreak] = useState(() =>
    initialClassifier?.gate_config?.alarm_streak_threshold != null
      ? String(initialClassifier.gate_config.alarm_streak_threshold)
      : "5"
  )
  const [calFile, setCalFile] = useState<File | null>(null)

  const [calibrating, setCalibrating] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [savingCfg, setSavingCfg] = useState(false)
  const [refreshing, setRefreshing] = useState(false)

  const statusObj: Data = liveStatus || state.label_ood_classifier || {}
  const calObj: Data | null = statusObj.calibration || null
  const gateCfg: Data = statusObj.gate_config || {}
  const lastEval: Data | null = statusObj.last_eval || null

  const syncFromStatus = useCallback((st: Data) => {
    const gc = st?.gate_config
    if (gc && typeof gc === "object") {
      if (typeof gc.gate_enabled === "boolean") setGateEnabled(gc.gate_enabled)
      if (gc.alarm_streak_threshold != null) setAlarmStreak(String(gc.alarm_streak_threshold))
    }
    const cal = st?.calibration
    if (cal && typeof cal === "object" && cal.prob_threshold != null) {
      setProbThreshold(String(cal.prob_threshold))
    }
  }, [])

  const fetchOodStatus = useCallback(async () => {
    setRefreshing(true)
    try {
      const data = await oodApi<Data>("/api/models/ood/status")
      setLiveStatus(data)
      syncFromStatus(data)
    } catch {
      /* 保留 state 轮询快照 */
    } finally {
      setRefreshing(false)
    }
  }, [syncFromStatus])

  useEffect(() => {
    let active = true
    void (async () => {
      try {
        const data = await oodApi<Data>("/api/models/ood/status")
        if (active) {
          setLiveStatus(data)
          syncFromStatus(data)
        }
      } catch {
        /* 保留 state 轮询快照 */
      }
    })()
    return () => {
      active = false
    }
  }, [syncFromStatus])

  const backendMode = String(statusObj.backend_mode || "heuristic_fallback")
  const modeMeta = MODE_LABELS[backendMode] || { text: backendMode, tone: "plain" }
  const modelKind = String(statusObj.model_kind || "heuristic")
  const hasNeuralModel = modelKind !== "heuristic"

  const curEnergy = state.ood_energy ?? lastEval?.energy_score
  const curMaha = state.ood_mahalanobis ?? lastEval?.mahalanobis_distance
  const effEnergyThresh = gateCfg.energy_threshold_override ?? calObj?.energy_threshold
  const effMahaThresh = gateCfg.mahalanobis_threshold_override ?? calObj?.mahalanobis_threshold
  const effProbThresh = gateCfg.prob_threshold_override ?? calObj?.prob_threshold
  const oodStreak = Number(state.ood_streak ?? 0)
  const streakLimit = Number(gateCfg.alarm_streak_threshold ?? 5)

  const handleAutoCalibrate = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!token) return flash("请先解锁工程师权限")
    const qVal = Number(quantile)
    const pVal = Number(probThreshold)
    if (!Number.isFinite(qVal) || qVal < 0.5 || qVal >= 1.0) {
      return flash("分位数须在 [0.50, 0.999] 之间")
    }
    if (!Number.isFinite(pVal) || pVal < 0.0 || pVal > 1.0) {
      return flash("概率阈值须在 [0, 1] 之间")
    }
    setCalibrating(true)
    try {
      const res = await oodApi<Data>(
        "/api/models/ood/calibrate",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            quantile: qVal,
            prob_threshold: pVal,
            fit_temperature: true,
            persist: true,
          }),
        },
        token
      )
      if (res.label_ood_classifier) {
        setLiveStatus(res.label_ood_classifier)
        syncFromStatus(res.label_ood_classifier)
      }
      flash("OOD 分布标定完成，已联动触发红兔点检")
      await onUpdated()
    } catch (err) {
      flash(`标定失败：${err instanceof Error ? err.message : err}`)
    } finally {
      setCalibrating(false)
    }
  }

  const handleUploadCalibration = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!token) return flash("请先解锁工程师权限")
    if (!calFile) return flash("请选择 ood_calibration.json 文件")
    setUploading(true)
    try {
      const rawText = await calFile.text()
      const parsed = JSON.parse(rawText) as Record<string, unknown>
      const res = await oodApi<Data>(
        "/api/models/ood/calibration",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(parsed),
        },
        token
      )
      if (res.label_ood_classifier) {
        setLiveStatus(res.label_ood_classifier)
        syncFromStatus(res.label_ood_classifier)
      }
      setCalFile(null)
      flash("已导入 OOD 标定参数，已触发红兔点检")
      await onUpdated()
    } catch (err) {
      flash(`导入失败：${err instanceof Error ? err.message : err}`)
    } finally {
      setUploading(false)
    }
  }

  const handleSaveConfig = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!token) return flash("请先解锁工程师权限")
    const streakVal = Number(alarmStreak)
    if (!Number.isInteger(streakVal) || streakVal < 1 || streakVal > 30) {
      return flash("连续闭锁帧数须为 1 ~ 30 整数")
    }
    setSavingCfg(true)
    try {
      const res = await oodApi<Data>(
        "/api/models/ood/config",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            gate_enabled: gateEnabled,
            alarm_streak_threshold: streakVal,
          }),
        },
        token
      )
      if (res.label_ood_classifier) {
        setLiveStatus(res.label_ood_classifier)
        syncFromStatus(res.label_ood_classifier)
      }
      flash("已更新 OOD 双门控策略")
      await onUpdated()
    } catch (err) {
      flash(`保存失败：${err instanceof Error ? err.message : err}`)
    } finally {
      setSavingCfg(false)
    }
  }

  return (
    <section className="panel" aria-label="OOD 未知异常双门控">
      <header className="panel-head">
        <h2>OOD 未知异常双门控</h2>
        <span className={`tag ${modeMeta.tone}`}>{modeMeta.text}</span>
        {state.ood_rejected && (
          <span className="tag alarm">
            <ShieldAlert size={14} aria-hidden="true" />
            拒识中 {oodStreak}/{streakLimit}
          </span>
        )}
        <div className="right">
          <button
            type="button"
            className="btn ghost sm"
            disabled={refreshing}
            onClick={() => void fetchOodStatus()}
          >
            <RefreshCw size={14} aria-hidden="true" />
            刷新指标
          </button>
        </div>
      </header>

      <div className="panel-body">
        <div className="kpi-grid-2">
          <div className="kpi-tile">
            <b
              className="mono"
              style={{
                fontSize: 24,
                fontWeight: 600,
                color:
                  curEnergy != null && effEnergyThresh != null && Number(curEnergy) > Number(effEnergyThresh)
                    ? "var(--bad)"
                    : "var(--ink)",
              }}
            >
              {fmt(curEnergy, 2)} / {fmt(effEnergyThresh, 2)}
            </b>
            <span style={{ fontSize: 14, fontWeight: 400 }}>Energy 分数 / 阈值 τₑ</span>
          </div>

          <div className="kpi-tile">
            <b
              className="mono"
              style={{
                fontSize: 24,
                fontWeight: 600,
                color:
                  curMaha != null && effMahaThresh != null && Number(curMaha) > Number(effMahaThresh)
                    ? "var(--bad)"
                    : "var(--ink)",
              }}
            >
              {statusObj.dual_output_available
                ? `${fmt(curMaha, 2)} / ${fmt(effMahaThresh, 2)}`
                : "单输出停用"}
            </b>
            <span style={{ fontSize: 14, fontWeight: 400 }}>Mahalanobis 距离 / 阈值 τₘ</span>
          </div>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))",
            gap: 16,
            padding: 16,
            borderRadius: 6,
            background: "var(--surface-alt)",
            fontSize: 14,
            fontWeight: 400,
          }}
        >
          <div>
            <span style={{ color: "var(--ink-2)" }}>温度缩放 T：</span>
            <span className="mono" style={{ fontWeight: 600 }}>
              {fmt(calObj?.temperature, 3)}
            </span>
          </div>
          <div>
            <span style={{ color: "var(--ink-2)" }}>概率下限 τₚ：</span>
            <span className="mono" style={{ fontWeight: 600 }}>
              {fmt(effProbThresh, 2)}
            </span>
          </div>
          <div>
            <span style={{ color: "var(--ink-2)" }}>AUROC / FPR95：</span>
            <span className="mono" style={{ fontWeight: 600 }}>
              {calObj?.validation_auroc != null ? fmt(calObj.validation_auroc, 3) : "—"} /{" "}
              {calObj?.validation_fpr95 != null ? fmt(calObj.validation_fpr95, 3) : "—"}
            </span>
          </div>
          <div>
            <span style={{ color: "var(--ink-2)" }}>输入维度：</span>
            <span className="mono" style={{ fontWeight: 600 }}>
              {statusObj.input_width || 96}×{statusObj.input_height || 96}
            </span>
          </div>
        </div>

        {(statusObj.calibration_error || lastEval?.reject_reason) && (
          <div className="auth-error" role="status" style={{ fontSize: 14, fontWeight: 400 }}>
            <ShieldAlert size={16} aria-hidden="true" />
            <span>
              {reasonOf(statusObj.calibration_error || lastEval?.reject_reason)}
            </span>
          </div>
        )}

        {!hasNeuralModel && (
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
            当前为启发式回退模式；请先训练 ANN_MLP 或导入 ONNX 模型后再执行 OOD 分布标定。
          </div>
        )}

        <form className="grid gap-4" onSubmit={(e) => void handleAutoCalibrate(e)}>
          <div className="grid grid-cols-2 gap-4">
            <div className="field">
              <label htmlFor="ood-quantile-select" style={{ fontSize: 14, fontWeight: 600 }}>
                ID 分布分位数
              </label>
              <Select value={quantile} onValueChange={setQuantile}>
                <SelectTrigger id="ood-quantile-select" style={{ fontSize: 16 }}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="0.95">95% 分位数（严格）</SelectItem>
                  <SelectItem value="0.98">98% 分位数（推荐）</SelectItem>
                  <SelectItem value="0.99">99% 分位数（宽松）</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="field">
              <label htmlFor="ood-prob-threshold" style={{ fontSize: 14, fontWeight: 600 }}>
                置信概率下限 τₚ
              </label>
              <Input
                id="ood-prob-threshold"
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={probThreshold}
                onChange={(e) => setProbThreshold(e.target.value)}
                className="mono"
              />
            </div>
          </div>
          <button
            type="submit"
            className="btn block"
            disabled={calibrating || !hasNeuralModel}
            style={{ fontSize: 16, fontWeight: 600 }}
          >
            <ShieldCheck size={16} aria-hidden="true" />
            {calibrating ? "标定中…" : "基于当前样本自动标定 OOD 阈值"}
          </button>
        </form>

        <div className="grid grid-cols-2 gap-4">
          <form className="grid gap-4" onSubmit={(e) => void handleUploadCalibration(e)}>
            <div className="field">
              <label htmlFor="ood-cal-file" style={{ fontSize: 14, fontWeight: 600 }}>
                导入 ood_calibration.json
              </label>
              <Input
                id="ood-cal-file"
                type="file"
                accept=".json,application/json"
                onChange={(e) => setCalFile(e.target.files?.[0] || null)}
              />
            </div>
            <button
              type="submit"
              className="btn ghost block"
              disabled={uploading || !calFile || !hasNeuralModel}
              style={{ fontSize: 14, fontWeight: 600 }}
            >
              <Upload size={14} aria-hidden="true" />
              {uploading ? "导入中…" : "热加载标定文件"}
            </button>
          </form>

          <form className="grid gap-4" onSubmit={(e) => void handleSaveConfig(e)}>
            <div className="field">
              <label htmlFor="ood-streak-limit" style={{ fontSize: 14, fontWeight: 600 }}>
                连续闭锁帧数 (1~30)
              </label>
              <Input
                id="ood-streak-limit"
                type="number"
                min="1"
                max="30"
                step="1"
                value={alarmStreak}
                onChange={(e) => setAlarmStreak(e.target.value)}
                className="mono"
              />
            </div>
            <div className="flex items-center justify-between gap-4">
              <label className="check" style={{ fontSize: 14, fontWeight: 400 }}>
                <input
                  type="checkbox"
                  checked={gateEnabled}
                  onChange={(e) => setGateEnabled(e.target.checked)}
                />
                启用双门控
              </label>
              <button
                type="submit"
                className="btn ghost sm"
                disabled={savingCfg}
                style={{ fontSize: 14, fontWeight: 600 }}
              >
                <Sliders size={14} aria-hidden="true" />
                {savingCfg ? "保存中…" : "应用策略"}
              </button>
            </div>
          </form>
        </div>
      </div>
    </section>
  )
}
