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
        if (typeof data.auth_configured === "boolean") {
          setAuthConfigured(data.auth_configured)
        } else if (typeof data.pin_configured === "boolean") {
          setAuthConfigured(data.pin_configured)
        }
      })
      .catch(() => {})
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
      setError("请输入口令")
      inputRef.current?.focus()
      return
    }
    if (authConfigured === false) {
      setError("服务端未配置工程师口令环境变量")
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

      if (unmountedRef.current || reqSeq !== requestSeqRef.current) return
      if (timedOut) throw new DOMException("Verification timed out", "AbortError")
      if (controller.signal.aborted) return

      let data: Record<string, unknown> = {}
      try {
        data = (await response.json()) as Record<string, unknown>
      } catch (parseErr) {
        if (timedOut || controller.signal.aborted) throw parseErr
        data = {}
      }

      if (unmountedRef.current || reqSeq !== requestSeqRef.current) return
      if (timedOut) throw new DOMException("Verification timed out", "AbortError")
      if (controller.signal.aborted) return

      if (!response.ok) {
        const detail = typeof data.detail === "string" && data.detail ? data.detail : "口令错误"
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
        setError("验证响应无效")
      }
    } catch (err) {
      if (unmountedRef.current || reqSeq !== requestSeqRef.current) return
      if (timedOut) {
        setError("请求超时")
        window.requestAnimationFrame(() => {
          inputRef.current?.focus()
        })
        return
      }
      if (err instanceof DOMException && err.name === "AbortError") return
      setError("服务未连接")
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

  return (
    <Dialog.Portal>
      <Dialog.Overlay className="modal-scrim" />
      <Dialog.Content
        className="modal auth-modal"
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
          <Lock size={16} aria-hidden="true" />
          <Dialog.Title asChild>
            <h2 id="engineer-auth-title">工程师登录</h2>
          </Dialog.Title>
          <div className="right">
            <button
              type="button"
              className="icon-btn"
              onClick={onCancel}
              aria-label="关闭弹窗"
            >
              <X size={16} aria-hidden="true" />
            </button>
          </div>
        </header>

        <Dialog.Description className="sr-only">工程师权限验证</Dialog.Description>

        <form className="auth-form" onSubmit={(e) => void handleSubmit(e)} noValidate>
          <div className="field">
            <label htmlFor="engineer-passcode-input">登录口令</label>
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
                placeholder="输入口令"
                aria-invalid={error ? "true" : "false"}
                aria-describedby={error ? "engineer-auth-error" : undefined}
                disabled={submitting}
              />
              <button
                type="button"
                className="auth-eye-btn"
                onClick={() => setShowPasscode((v) => !v)}
                aria-label={showPasscode ? "隐藏口令" : "显示口令"}
                disabled={submitting}
              >
                {showPasscode ? <EyeOff size={16} aria-hidden="true" /> : <Eye size={16} aria-hidden="true" />}
              </button>
            </div>
          </div>

          {error && (
            <div id="engineer-auth-error" className="auth-error" role="alert">
              <AlertCircle size={16} aria-hidden="true" />
              <span>{error}</span>
            </div>
          )}

          <div className="auth-actions">
            <button type="button" className="btn ghost" onClick={onCancel}>
              取消
            </button>
            <button type="submit" className="btn" disabled={submitting}>
              {submitting ? "验证中…" : "登录"}
            </button>
          </div>
        </form>
      </Dialog.Content>
    </Dialog.Portal>
  )
}
