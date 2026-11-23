import { useEffect, useState, type ReactNode } from "react"

interface CameraFrameProps {
  src?: string
  alt: string
  /** 固定视口自适应铺满模式（保持 16:9 完整无裁切，干净深色背景） */
  fill?: boolean
  /** 四角说明文字（仅在显式传入时渲染，默认零叠字） */
  captions?: Partial<Record<"tl" | "tr" | "bl" | "br", ReactNode>>
  emptyTitle: string
  emptyHint?: string
  /** 图片或视频流加载失败时的标题 */
  errorTitle?: string
  /** 图片或视频流加载失败时的提示 */
  errorHint?: string
  /** 对图片进行中心放大（用于标签特写） */
  zoom?: number
  /** 手动抓拍触发计数：每次递增播放一次光学快门闪光 */
  shutterKey?: number
  /** 相机在线且视野有效时启用取景框四角对焦锁定微亮 */
  focusLocked?: boolean
  /** 是否显示取景框四角标线 */
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
  emptyTitle,
  emptyHint,
  errorTitle,
  errorHint,
  zoom,
  focusLocked,
  showTicks,
}: {
  src: string
  alt: string
  emptyTitle: string
  emptyHint?: string
  errorTitle?: string
  errorHint?: string
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
          <span>{errorHint || emptyHint || `${emptyTitle} · 正在尝试重新连接`}</span>
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
  emptyHint,
  errorTitle,
  errorHint,
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
          emptyTitle={emptyTitle}
          emptyHint={emptyHint}
          errorTitle={errorTitle}
          errorHint={errorHint}
          zoom={zoom}
          focusLocked={focusLocked}
          showTicks={showTicks}
        />
      ) : (
        <>
          <div className="frame-empty">
            <div className="crosshair" aria-hidden="true" />
            <b>{emptyTitle}</b>
            {emptyHint && <span>{emptyHint}</span>}
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
