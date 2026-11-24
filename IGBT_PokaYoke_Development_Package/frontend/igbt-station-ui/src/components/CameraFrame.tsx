import { useEffect, useState, type ReactNode } from "react"

interface CameraFrameProps {
  src?: string
  alt: string
  fill?: boolean
  captions?: Partial<Record<"tl" | "tr" | "bl" | "br", ReactNode>>
  emptyTitle: string
  errorTitle?: string
  zoom?: number
  shutterKey?: number
  focusLocked?: boolean
  showTicks?: boolean
}

const STREAM_RETRY_MS = 3000

function withRetryParam(src: string, retryTick: number): string {
  if (retryTick <= 0) return src
  if (src.startsWith("data:") || src.startsWith("blob:")) return src
  const sep = src.includes("?") ? "&" : "?"
  return `${src}${sep}_retry=${retryTick}`
}

function CameraFrameImage({
  src,
  alt,
  errorTitle,
  zoom,
  focusLocked,
  showTicks,
}: {
  src: string
  alt: string
  errorTitle?: string
  zoom?: number
  focusLocked: boolean
  showTicks: boolean
}) {
  const [phase, setPhase] = useState<"loading" | "loaded" | "error" | "retrying">("loading")
  const [retryTick, setRetryTick] = useState(0)

  useEffect(() => {
    if (phase !== "error") return
    const timer = window.setTimeout(() => {
      setRetryTick((n) => n + 1)
      setPhase("retrying")
    }, STREAM_RETRY_MS)
    return () => window.clearTimeout(timer)
  }, [phase])

  const effectiveSrc = withRetryParam(src, retryTick)
  const isUnavailable = phase === "error" || phase === "retrying"
  const effectiveFocusLocked = focusLocked && phase === "loaded"
  const imgStyle =
    phase === "retrying"
      ? ({
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          opacity: 0,
          pointerEvents: "none",
          ...(zoom ? { transform: `scale(${zoom})`, objectFit: "cover" as const } : {}),
        } as const)
      : zoom
        ? ({ transform: `scale(${zoom})`, objectFit: "cover" as const } as const)
        : undefined

  return (
    <>
      {isUnavailable && (
        <div className="frame-empty" role="status">
          <div className="crosshair" aria-hidden="true" />
          <b>{errorTitle || "画面不可用"}</b>
        </div>
      )}
      {phase !== "error" && (
        <img
          key={retryTick}
          src={effectiveSrc}
          alt={phase === "retrying" ? "" : alt}
          aria-hidden={phase === "retrying" ? true : undefined}
          draggable={false}
          style={imgStyle}
          onLoad={() => setPhase("loaded")}
          onError={() => setPhase("error")}
        />
      )}
      {showTicks && <div className={`ticks ${effectiveFocusLocked ? "locked" : ""}`} aria-hidden="true" />}
    </>
  )
}

export function CameraFrame({
  src,
  alt,
  fill = false,
  captions,
  emptyTitle,
  errorTitle,
  zoom,
  shutterKey = 0,
  focusLocked = false,
  showTicks = true,
}: CameraFrameProps) {
  const captionEntries = captions
    ? (Object.keys(captions) as (keyof typeof captions)[]).filter((k) => Boolean(captions[k]))
    : []

  return (
    <div className={`frame ${fill ? "frame-fill" : ""}`}>
      {src ? (
        <CameraFrameImage
          key={src}
          src={src}
          alt={alt}
          errorTitle={errorTitle}
          zoom={zoom}
          focusLocked={focusLocked}
          showTicks={showTicks}
        />
      ) : (
        <>
          <div className="frame-empty">
            <div className="crosshair" aria-hidden="true" />
            <b>{emptyTitle}</b>
          </div>
          {showTicks && <div className="ticks" aria-hidden="true" />}
        </>
      )}
      {shutterKey > 0 && <div key={shutterKey} className="shutter-flash" aria-hidden="true" />}
      {captionEntries.map((k) => (
        <span key={k} className={`cap ${k}`}>
          {captions?.[k]}
        </span>
      ))}
    </div>
  )
}
