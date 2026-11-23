import { useCallback, useEffect, useRef, useState } from "react"
import { Dialog } from "radix-ui"
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

const VERIFY_TIMEOUT_MS = 10_000

export function EngineerAuthModal({
  open,
  onClose,
  onAuthenticated,
  triggerRef,
}: EngineerAuthModalProps) {
  const authenticatedRef = useRef(false)
  const abortControllerRef = useRef<AbortController | null>(null)
  const requestSeqRef = useRef(0)

  const handleCancel = useCallback(() => {
    requestSeqRef.current += 1
    abortControllerRef.current?.abort()
    abortControllerRef.current = null
    onClose()
  }, [onClose])

  return (
    <Dialog.Root
      open={open}
      onOpenChange={(nextOpen) => {
        if (!nextOpen) {
          handleCancel()
        }
      }}
    >
      {open && (
        <EngineerAuthDialogContent
          authenticatedRef={authenticatedRef}
          abortControllerRef={abortControllerRef}
          requestSeqRef={requestSeqRef}
          onCancel={handleCancel}
          onAuthenticated={onAuthenticated}
          triggerRef={triggerRef}
        />
      )}
    </Dialog.Root>
  )
}

function EngineerAuthDialogContent({
  authenticatedRef,
  abortControllerRef,
  requestSeqRef,
  onCancel,
  onAuthenticated,
  triggerRef,
}: {
  authenticatedRef: React.MutableRefObject<boolean>
  abortControllerRef: React.MutableRefObject<AbortController | null>
  requestSeqRef: React.MutableRefObject<number>
  onCancel: () => void
  onAuthenticated: (result: EngineerAuthResult) => void
  triggerRef?: React.RefObject<HTMLButtonElement | null>
}) {
  const [passcode, setPasscode] = useState("")
  const [showPasscode, setShowPasscode] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState("")
  const [defaultPinActive, setDefaultPinActive] = useState(false)
  const [authConfigured, setAuthConfigured] = useState<boolean | null>(null)

  const inputRef = useRef<HTMLInputElement | null>(null)
  const unmountedRef = useRef(false)

  useEffect(() => {
    authenticatedRef.current = false
    unmountedRef.current = false
    const statusController = new AbortController()
    void fetch("/api/auth/status", { signal: statusController.signal })
      .then((res) => (res.ok ? res.json() : null))
      .then((data: Record<string, unknown> | null) => {
        if (unmountedRef.current || !data) return
        setDefaultPinActive(data.default_pin_active === true)
        if (typeof data.auth_configured === "boolean") {
          setAuthConfigured(data.auth_configured)
        } else if (typeof data.pin_configured === "boolean") {
          setAuthConfigured(data.pin_configured)
        }
      })
      .catch(() => {
        /* 忽略弹窗关闭中止或离线查询失败；保持默认通用提示 */
      })
    return () => {
      unmountedRef.current = true
      statusController.abort()
      requestSeqRef.current += 1
      abortControllerRef.current?.abort()
      abortControllerRef.current = null
    }
  }, [authenticatedRef, abortControllerRef, requestSeqRef])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const trimmed = passcode.trim()
    if (!trimmed) {
      setError("请输入工程师口令")
      inputRef.current?.focus()
      return
    }

    abortControllerRef.current?.abort()
    const controller = new AbortController()
    abortControllerRef.current = controller
    const reqSeq = ++requestSeqRef.current
    let timedOut = false
    const timeoutId = window.setTimeout(() => {
      timedOut = true
      controller.abort()
    }, VERIFY_TIMEOUT_MS)

    setSubmitting(true)
    setError("")
    try {
      const response = await fetch("/api/auth/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ passcode: trimmed }),
        signal: controller.signal,
      })

      if (unmountedRef.current || reqSeq !== requestSeqRef.current) {
        return
      }
      if (timedOut) {
        throw new DOMException("Verification timed out", "AbortError")
      }
      if (controller.signal.aborted) {
        return
      }

      let data: Record<string, unknown> = {}
      try {
        data = (await response.json()) as Record<string, unknown>
      } catch (parseErr) {
        if (timedOut || controller.signal.aborted) {
          throw parseErr
        }
        data = {}
      }

      if (unmountedRef.current || reqSeq !== requestSeqRef.current) {
        return
      }
      if (timedOut) {
        throw new DOMException("Verification timed out", "AbortError")
      }
      if (controller.signal.aborted) {
        return
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
    } catch (err) {
      if (unmountedRef.current || reqSeq !== requestSeqRef.current) {
        return
      }
      if (timedOut) {
        setError("验证请求超时，请检查本机工位服务后重试")
        window.requestAnimationFrame(() => {
          inputRef.current?.focus()
        })
        return
      }
      if (err instanceof DOMException && err.name === "AbortError") {
        return
      }
      setError("无法连接本机工位服务，请稍后重试")
    } finally {
      window.clearTimeout(timeoutId)
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null
      }
      if (!unmountedRef.current && reqSeq === requestSeqRef.current) {
        setSubmitting(false)
      }
    }
  }

  const hintText = defaultPinActive
    ? "本机处于演示模式，请输入本机配置的演示工程师口令"
    : authConfigured === false
      ? "本机尚未配置工程师口令，请先设置环境变量 IGBT_ENGINEER_PIN 或 IGBT_ADMIN_TOKEN"
      : "请输入现场配置的工程师口令"

  return (
    <Dialog.Portal>
      <Dialog.Overlay className="modal-scrim" />
      <Dialog.Content
        className="modal auth-modal"
        aria-describedby="engineer-auth-hint"
        onOpenAutoFocus={(e) => {
          e.preventDefault()
          inputRef.current?.focus()
        }}
        onCloseAutoFocus={(e) => {
          if (authenticatedRef.current) {
            e.preventDefault()
            return
          }
          if (triggerRef?.current && document.contains(triggerRef.current)) {
            e.preventDefault()
            triggerRef.current.focus()
          }
        }}
      >
        <header className="panel-head">
          <Lock size={15} aria-hidden="true" />
          <Dialog.Title asChild>
            <h2 id="engineer-auth-title">工程师权限验证</h2>
          </Dialog.Title>
          <div className="right">
            <button
              type="button"
              className="icon-btn"
              onClick={onCancel}
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
            <Dialog.Description asChild>
              <small id="engineer-auth-hint" className="auth-hint">
                {hintText}
              </small>
            </Dialog.Description>
          </div>

          {error && (
            <div id="engineer-auth-error" className="auth-error" role="alert">
              <AlertCircle size={15} aria-hidden="true" />
              <span>{error}</span>
            </div>
          )}

          <div className="auth-actions">
            <button type="button" className="btn ghost" onClick={onCancel}>
              取消
            </button>
            <button type="submit" className="btn" disabled={submitting}>
              {submitting ? "验证中…" : "验证并进入工程师后台"}
            </button>
          </div>
        </form>
      </Dialog.Content>
    </Dialog.Portal>
  )
}
