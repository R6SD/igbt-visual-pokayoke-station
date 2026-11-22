import { useEffect, useRef, useState } from "react"
import { AlertCircle, Eye, EyeOff, Lock, X } from "lucide-react"

export interface EngineerAuthResult {
  adminToken: string
  expiresInSeconds: number
}

interface EngineerAuthModalProps {
  open: boolean
  onClose: () => void
  onAuthenticated: (result: EngineerAuthResult) => void
  triggerRef?: React.RefObject<HTMLButtonElement | null>
}

export function EngineerAuthModal({
  open,
  onClose,
  onAuthenticated,
  triggerRef,
}: EngineerAuthModalProps) {
  if (!open) return null
  return (
    <EngineerAuthDialog
      onClose={onClose}
      onAuthenticated={onAuthenticated}
      triggerRef={triggerRef}
    />
  )
}

function EngineerAuthDialog({
  onClose,
  onAuthenticated,
  triggerRef,
}: Omit<EngineerAuthModalProps, "open">) {
  const [passcode, setPasscode] = useState("")
  const [showPasscode, setShowPasscode] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState("")
  const [defaultPinActive, setDefaultPinActive] = useState(true)

  const modalRef = useRef<HTMLDivElement | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)
  const authenticatedRef = useRef(false)

  useEffect(() => {
    const controller = new AbortController()
    void fetch("/api/auth/status", { signal: controller.signal })
      .then((res) => (res.ok ? res.json() : null))
      .then((data: Record<string, unknown> | null) => {
        if (data && typeof data.default_pin_active === "boolean") {
          setDefaultPinActive(data.default_pin_active)
        }
      })
      .catch(() => {
        /* 忽略弹窗关闭中止或离线查询失败 */
      })
    return () => controller.abort()
  }, [])

  useEffect(() => {
    const prevFocused = document.activeElement instanceof HTMLElement ? document.activeElement : null
    const triggerEl = triggerRef?.current ?? null
    const raf = window.requestAnimationFrame(() => {
      inputRef.current?.focus()
    })

    return () => {
      window.cancelAnimationFrame(raf)
      if (!authenticatedRef.current) {
        const target = triggerEl ?? prevFocused
        if (target && document.contains(target)) {
          window.requestAnimationFrame(() => target.focus())
        }
      }
    }
  }, [triggerRef])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Escape") {
      e.preventDefault()
      e.stopPropagation()
      if (!submitting) onClose()
      return
    }

    if (e.key === "Tab" && modalRef.current) {
      const focusables = Array.from(
        modalRef.current.querySelectorAll<HTMLElement>(
          'button:not(:disabled), input:not(:disabled), [href], select:not(:disabled), textarea:not(:disabled), [tabindex]:not([tabindex="-1"])'
        )
      )
      if (focusables.length === 0) return
      const first = focusables[0]
      const last = focusables[focusables.length - 1]
      const active = document.activeElement
      if (e.shiftKey) {
        if (active === first || !modalRef.current.contains(active)) {
          e.preventDefault()
          last.focus()
        }
      } else if (active === last || !modalRef.current.contains(active)) {
        e.preventDefault()
        first.focus()
      }
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const trimmed = passcode.trim()
    if (!trimmed) {
      setError("请输入工程师口令")
      inputRef.current?.focus()
      return
    }

    setSubmitting(true)
    setError("")
    try {
      const response = await fetch("/api/auth/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ passcode: trimmed }),
      })

      let data: Record<string, unknown> = {}
      try {
        data = (await response.json()) as Record<string, unknown>
      } catch {
        data = {}
      }

      if (!response.ok) {
        const detail =
          typeof data.detail === "string" && data.detail ? data.detail : "工程师口令错误，请重试"
        setError(detail)
        window.requestAnimationFrame(() => {
          inputRef.current?.focus()
          inputRef.current?.select()
        })
        return
      }

      if (
        data.authenticated === true &&
        typeof data.admin_token === "string" &&
        data.admin_token.length > 0
      ) {
        authenticatedRef.current = true
        onAuthenticated({
          adminToken: data.admin_token,
          expiresInSeconds: Number(data.expires_in_seconds) || 1800,
        })
      } else {
        setError("验证响应无效，请联系设备工程师")
      }
    } catch {
      setError("无法连接本机工位服务，请稍后重试")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div
      className="modal-scrim"
      onClick={() => {
        if (!submitting) onClose()
      }}
    >
      <div
        ref={modalRef}
        className="modal auth-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="engineer-auth-title"
        aria-describedby="engineer-auth-hint"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        <header className="panel-head">
          <Lock size={15} aria-hidden="true" />
          <h2 id="engineer-auth-title">工程师权限验证</h2>
          <div className="right">
            <button
              type="button"
              className="icon-btn"
              onClick={onClose}
              disabled={submitting}
              aria-label="关闭弹窗"
            >
              <X size={15} aria-hidden="true" />
            </button>
          </div>
        </header>

        <form className="panel-body auth-form" onSubmit={(e) => void handleSubmit(e)} noValidate>
          <div className="field">
            <label htmlFor="engineer-passcode-input">工程师口令</label>
            <div className="auth-input-wrap">
              <input
                ref={inputRef}
                id="engineer-passcode-input"
                type={showPasscode ? "text" : "password"}
                className="auth-input mono"
                value={passcode}
                onChange={(e) => {
                  setPasscode(e.target.value)
                  if (error) setError("")
                }}
                autoComplete="current-password"
                placeholder="请输入工程师口令"
                aria-invalid={error ? "true" : "false"}
                aria-describedby={error ? "engineer-auth-error engineer-auth-hint" : "engineer-auth-hint"}
                disabled={submitting}
              />
              <button
                type="button"
                className="auth-eye-btn"
                onClick={() => setShowPasscode((v) => !v)}
                aria-label={showPasscode ? "隐藏口令" : "显示口令"}
                disabled={submitting}
              >
                {showPasscode ? <EyeOff size={15} aria-hidden="true" /> : <Eye size={15} aria-hidden="true" />}
              </button>
            </div>
            <small id="engineer-auth-hint" className="auth-hint">
              {defaultPinActive ? "请输入工程师口令（演示口令：888888）" : "请输入现场配置的工程师口令"}
            </small>
          </div>

          {error && (
            <div id="engineer-auth-error" className="auth-error" role="alert">
              <AlertCircle size={15} aria-hidden="true" />
              <span>{error}</span>
            </div>
          )}

          <div className="auth-actions">
            <button type="button" className="btn ghost" onClick={onClose} disabled={submitting}>
              取消
            </button>
            <button type="submit" className="btn" disabled={submitting}>
              {submitting ? "验证中…" : "验证并进入工程师后台"}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
