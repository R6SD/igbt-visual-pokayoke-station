import { fmt } from "@/lib/station"

interface OpticalGaugesProps {
  laplacianVariance?: number | null
  minLaplacian?: number | null
  overexposedFraction?: number | null
  maxOverexposed?: number | null
  brightness?: number | null
  minBrightness?: number | null
  maxBrightness?: number | null
  viewValid?: boolean | null
}

const has = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v)

export function OpticalGauges({
  laplacianVariance,
  minLaplacian,
  overexposedFraction,
  maxOverexposed,
  brightness,
  minBrightness,
  maxBrightness,
}: OpticalGaugesProps) {
  const focusBad = has(laplacianVariance) && has(minLaplacian) && laplacianVariance < minLaplacian
  const glare = has(overexposedFraction) ? overexposedFraction * 100 : null
  const glareLimit = has(maxOverexposed) ? maxOverexposed * 100 : null
  const glareBad = glare !== null && glareLimit !== null && glare > glareLimit
  const lumBad =
    has(brightness) &&
    ((has(minBrightness) && brightness < minBrightness) || (has(maxBrightness) && brightness > maxBrightness))

  const items = [
    {
      label: "清晰度",
      value: fmt(laplacianVariance, 1),
      rule: has(minLaplacian) ? `下限 ${fmt(minLaplacian, 1)}` : undefined,
      bad: focusBad,
    },
    {
      label: "管壁反光",
      value: glare === null ? "—" : `${glare.toFixed(1)}%`,
      rule: glareLimit !== null ? `上限 ${glareLimit.toFixed(1)}%` : undefined,
      bad: glareBad,
    },
    {
      label: "平均亮度",
      value: fmt(brightness, 0),
      rule:
        has(minBrightness) || has(maxBrightness)
          ? `区间 ${fmt(minBrightness, 0)} - ${fmt(maxBrightness, 0)}`
          : undefined,
      bad: lumBad,
    },
  ]

  return (
    <div className="kpi-row">
      {items.map((it) => (
        <div key={it.label} className="readout" title={it.rule}>
          <b style={it.bad ? { color: "var(--bad)" } : undefined}>{it.value}</b>
          <small>{it.label}</small>
        </div>
      ))}
    </div>
  )
}
