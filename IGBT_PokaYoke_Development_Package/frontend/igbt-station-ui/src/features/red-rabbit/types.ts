export type RedRabbitState = "PROD_READY" | "CHALLENGE_REQUIRED" | "CHALLENGE_HARD_LOCK"

export interface GoldenSample {
  sample_uid: string
  sample_type: "OK_GOLDEN" | "NG_UNMARKED" | "NG_PARTIAL" | "NG_REVERSED" | "NG_OOD"
  expected_verdict: "PASS" | "ALARM"
  product_family: string
  is_active: boolean
  calibrated_by: string
  expires_at_ms: number
  created_at_ms: number
}

export interface RedRabbitStep {
  step_index: number
  sample_uid: string
  sample_type: GoldenSample["sample_type"]
  expected_verdict: "PASS" | "ALARM"
  product_family: string
  completed?: boolean
  is_match?: boolean | null
  actual_verdict?: "PASS" | "ALARM" | null
}

export interface RedRabbitSession {
  session_id: string
  trigger_reason: string
  operator_id: string
  lot_id: string
  status: "IN_PROGRESS" | "PASSED" | "FAILED" | "TIMEOUT" | "ABORTED"
  started_at_ms: number
  completed_at_ms?: number | null
  timeout_remaining_ms?: number
  current_step_index: number
  total_steps?: number
  armed_sample_uid?: string | null
  current_step?: RedRabbitStep | null
  required_plan: RedRabbitStep[]
}

export interface StationLockAuditRow {
  audit_seq: number
  event_id: string
  event_type:
    | "CHALLENGE_FAIL_LOCK"
    | "WATCHDOG_FAULT_LOCK"
    | "MES_OFFLINE_LOCK"
    | "ENGINEER_UNLOCK"
    | "POLICY_OVERRIDE"
  actor_id: string
  actor_role: "SYSTEM" | "OPERATOR" | "ENGINEER" | "QE"
  reason_code: string
  comment: string
  related_session?: string | null
  created_at_ms: number
  prev_hash: string
  row_hash: string
}

export interface RedRabbitStatus {
  station_id: string
  strict_enabled: boolean
  state: RedRabbitState
  blocking: boolean
  hard_locked: boolean
  trigger_reason?: string | null
  active_lot_id?: string | null
  required_ok_count: number
  required_ng_count: number
  interval_ms: number
  session_timeout_ms: number
  last_passed_session_id?: string | null
  last_passed_at_ms?: number | null
  next_due_at_ms?: number | null
  now_ms: number
  session?: RedRabbitSession | null
  last_attempt?: Record<string, unknown> | null
  last_lock_event?: StationLockAuditRow | null
}

export interface RedRabbitAuditResponse {
  chain_verification: {
    valid: boolean
    checked_rows: number
    broken_seq: number | null
    latest_hash: string
  }
  lock_audit: StationLockAuditRow[]
  recent_sessions: RedRabbitSession[]
  samples: GoldenSample[]
}

export const RR_TRIGGER_LABELS: Record<string, string> = {
  COLD_BOOT: "冷启动强制首检",
  CRASH_RECOVERY: "异常恢复强检",
  SHIFT_CHANGE: "换班时间窗触发",
  INTERVAL_EXPIRED: "4 小时周期到期",
  LOT_CHANGE: "批次 / 工单切换",
  MODEL_OR_OPTICS_CHANGE: "光学 3A 或模型变更",
  MANUAL_AUDIT: "工程师发起核检",
  CRITICAL_NG_MISSED_AS_PASS: "致命风险：NG 金样误判放行",
  GOLDEN_SAMPLE_MISMATCH: "金样实测与期望不符",
  SESSION_TIMEOUT: "红兔点检超时锁机",
}

export const RR_SAMPLE_TYPE_LABELS: Record<string, string> = {
  OK_GOLDEN: "OK 良品金样（需完成防呆 action 判 PASS）",
  NG_UNMARKED: "NG 完整白标未划除（应拦截 ALARM）",
  NG_PARTIAL: "NG 半划除 / 遮挡异常（应拦截 ALARM）",
  NG_REVERSED: "NG 180° 倒置异常（应拦截 ALARM）",
  NG_OOD: "NG 未知异型 OOD（应拦截 ALARM）",
}

export const RR_EVENT_TYPE_LABELS: Record<string, string> = {
  CHALLENGE_FAIL_LOCK: "挑战失败硬锁",
  WATCHDOG_FAULT_LOCK: "看门狗故障硬锁",
  MES_OFFLINE_LOCK: "MES 断网锁机",
  ENGINEER_UNLOCK: "工程师 / QE 解锁",
  POLICY_OVERRIDE: "策略变更",
}
