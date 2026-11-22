import type { ReactNode } from "react"

interface CameraFrameProps {
  src?: string
  alt: string
  /** 固定视口自适应铺满模式（保持 16:9 完整无裁切，干净深色背景） */
  fill?: boolean
  /** 四角说明文字（仅在显式传入时渲染，默认零叠字） */
  captions?: Partial<Record<"tl" | "tr" | "bl" | "br", ReactNode>>
  emptyTitle: string
  emptyHint?: string
  /** 对图片进行中心放大（用于标签特写） */
  zoom?: number
  /** 手动抓拍触发计数：每次递增播放一次光学快门闪光 */
  shutterKey?: number
  /** 相机在线且视野有效时启用取景框四角对焦锁定微亮 */
  focusLocked?: boolean
  /** 是否显示取景框四角标线 */
  showTicks?: boolean
}

export function CameraFrame({
  src,
  alt,
  fill = false,
  captions,
  emptyTitle,
  emptyHint,
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
        <img
          src={src}
          alt={alt}
          draggable={false}
          style={zoom ? { transform: `scale(${zoom})`, objectFit: "cover" } : undefined}
        />
      ) : (
        <div className="frame-empty">
          <div className="crosshair" aria-hidden="true" />
          <b>{emptyTitle}</b>
          {emptyHint && <span>{emptyHint}</span>}
        </div>
      )}
      {showTicks && <div className={`ticks ${focusLocked ? "locked" : ""}`} aria-hidden="true" />}
      {shutterKey > 0 && <div key={shutterKey} className="shutter-flash" aria-hidden="true" />}
      {captionEntries.map((k) => (
        <span key={k} className={`cap ${k}`}>
          {captions?.[k]}
        </span>
      ))}
    </div>
  )
}
