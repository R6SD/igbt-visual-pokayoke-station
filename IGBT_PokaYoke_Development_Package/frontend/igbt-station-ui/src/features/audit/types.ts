export interface MesActiveLot {
  work_order_id: string
  lot_id: string
  part_number: string
  expected_pipe_color: string
  lot_status: "ACTIVE" | "HOLD" | "SCRAP"
  signature_sha256: string
  signature_valid?: boolean
  expired?: boolean
  validated_at_ms: number
  expires_at_ms: number
  tube_seq: number
  offline_used_count: number
  offline_started_at_ms: number | null
  updated_at_ms: number
}

export interface MesOutboxCounts {
  pending: number
  in_flight: number
  acked: number
  dead_letter: number
  total: number
}

export interface MesOutboxItem {
  id: number
  idempotency_key: string
  cycle_id: string
  work_order_id: string
  lot_id: string
  tube_seq: number
  endpoint_path: string
  payload: Record<string, unknown>
  status: "PENDING" | "IN_FLIGHT" | "ACKED" | "DEAD_LETTER"
  retry_count: number
  next_retry_at_ms: number
  last_http_code: number | null
  last_error: string | null
  response_body: string | null
  created_at_ms: number
  acked_at_ms: number | null
}

export interface MesWhitelistInfo {
  approved: boolean
  actor_id: string
  actor_role: string
  reason: string
  max_tubes: number
  max_duration_ms: number
  used_tubes: number
  remaining_tubes: number
  remaining_ms: number
  approved_at_ms: number
}

export interface MesRecallInfo {
  locked: boolean
  status: string
  lot_id: string
  work_order_id: string
  reason: string
  affected_keys: string[]
  triggered_at_ms: number
}

export interface MesConfigSummary {
  enabled: boolean
  base_url: string
  station_id: string
  api_key_configured: boolean
  api_key_preview: string
  heartbeat_path: string
  validate_lot_path: string
  pass_record_path: string
  connect_timeout_ms: number
  io_timeout_ms: number
  heartbeat_period_ms: number
  heartbeat_fail_lock_ms: number
  outbox_poll_ms: number
  base_backoff_ms: number
  max_backoff_ms: number
  max_retries: number
  whitelist_max_tubes: number
  whitelist_max_duration_ms: number
}

export interface MesStatus {
  backend: string
  enabled: boolean
  connected: boolean
  ever_connected: boolean
  blocking: boolean
  block_reason: string
  degradation_mode: string
  lot_bound: boolean
  signature_valid: boolean
  lot_expired: boolean
  recall_locked: boolean
  offline_locked: boolean
  offline_since_ms: number
  offline_elapsed_ms: number
  last_heartbeat_attempt_ms: number
  last_heartbeat_ok_ms: number
  last_rtt_ms: number
  last_heartbeat_error: string
  config: MesConfigSummary
  active_lot: MesActiveLot | null
  whitelist: MesWhitelistInfo
  recall: MesRecallInfo
  outbox_counts: MesOutboxCounts
}

export interface MesOutboxListResponse {
  items: MesOutboxItem[]
  counts: MesOutboxCounts
  status: MesStatus
}

export const DEGRADATION_MODE_LABELS: Record<string, { label: string; tone: string }> = {
  DISABLED: { label: "未启用（本地兼容）", tone: "plain" },
  ONLINE: { label: "在线直连（车规 Fail-Closed）", tone: "pass" },
  ONLINE_WHITELIST_ARMED: { label: "在线直连（已备受控白名单）", tone: "pass" },
  OFFLINE_GRACE: { label: "离线宽限窗口", tone: "warn" },
  WHITELIST_ACTIVE: { label: "受控白名单离线作业中", tone: "warn" },
  FAIL_CLOSED_LOCKED: { label: "断网 Fail-Closed 锁机", tone: "alarm" },
  WHITELIST_LOCKED: { label: "白名单配额/超时锁机", tone: "alarm" },
  LOT_UNBOUND: { label: "未在线绑定工单/Lot", tone: "warn" },
  RECALL_LOCKED: { label: "批次 HOLD/SCRAP 追回闭锁", tone: "alarm" },
}

export const OUTBOX_STATUS_LABELS: Record<string, { label: string; tone: string }> = {
  PENDING: { label: "待补传", tone: "warn" },
  IN_FLIGHT: { label: "发送中", tone: "plain" },
  ACKED: { label: "已对账", tone: "pass" },
  DEAD_LETTER: { label: "死信异常", tone: "alarm" },
}
