/**
 * 全局 SVG 光学折射滤镜（供 CSS backdrop-filter: url(#...) 调用）
 * - liquid-glass-refract: 轻度平滑透镜折射，适用于悬浮栏、控制胶囊、状态灯条
 * - liquid-glass-lens: 略强的流体边缘折射，适用于弹窗与聚焦徽章
 * 注意：必须声明 color-interpolation-filters="sRGB"，避免线性色彩空间偏色。
 */
export function LiquidGlassDefs() {
  return (
    <svg
      aria-hidden="true"
      width="0"
      height="0"
      style={{ position: "fixed", width: 0, height: 0, pointerEvents: "none", overflow: "hidden" }}
    >
      <defs>
        <filter id="liquid-glass-refract" x="-10%" y="-10%" width="120%" height="120%" colorInterpolationFilters="sRGB">
          <feTurbulence type="fractalNoise" baseFrequency="0.008 0.012" numOctaves="2" seed="7" result="noise" />
          <feGaussianBlur in="noise" stdDeviation="4" result="smoothNoise" />
          <feDisplacementMap
            in="SourceGraphic"
            in2="smoothNoise"
            scale="10"
            xChannelSelector="R"
            yChannelSelector="G"
          />
        </filter>

        <filter id="liquid-glass-lens" x="-10%" y="-10%" width="120%" height="120%" colorInterpolationFilters="sRGB">
          <feTurbulence type="fractalNoise" baseFrequency="0.012 0.012" numOctaves="2" seed="19" result="noise" />
          <feGaussianBlur in="noise" stdDeviation="3.5" result="smoothNoise" />
          <feDisplacementMap
            in="SourceGraphic"
            in2="smoothNoise"
            scale="16"
            xChannelSelector="R"
            yChannelSelector="G"
          />
        </filter>
      </defs>
    </svg>
  )
}
