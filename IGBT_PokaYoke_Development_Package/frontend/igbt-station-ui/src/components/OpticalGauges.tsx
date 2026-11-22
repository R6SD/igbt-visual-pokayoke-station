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

const pct = (v: number, max: number) => `${Math.max(0, Math.min(100, (v / max) * 100))}%`
const has = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v)

/** 光学条件读数：清晰度 / 反光 / 亮度。无运行时数据时显示 “—”，不伪造数值 */
export function OpticalGauges({
  laplacianVariance,
  minLaplacian,
  overexposedFraction,
  maxOverexposed,
  brightness,
  minBrightness,
  maxBrightness,
  viewValid,
}: OpticalGaugesProps) {
  const focusMax = Math.max(80, has(minLaplacian) ? minLaplacian * 3 : 0)
  const focusBad = has(laplacianVariance) && has(minLaplacian) && laplacianVariance < minLaplacian

  const glare = has(overexposedFraction) ? overexposedFraction * 100 : null
  const glareLimit = has(maxOverexposed) ? maxOverexposed * 100 : null
  const glareMax = Math.max(15, glareLimit ? glareLimit * 2 : 0)
  const glareBad = glare !== null && glareLimit !== null && glare > glareLimit

  const lumBad =
    has(brightness) &&
    ((has(minBrightness) && brightness < minBrightness) || (has(maxBrightness) && brightness > maxBrightness))

  const items = [
    {
      label: "清晰度",
      value: fmt(laplacianVariance, 1),
      unit: "",
      width: has(laplacianVariance) ? pct(laplacianVariance, focusMax) : "0%",
      marks: has(minLaplacian) ? [pct(minLaplacian, focusMax)] : [],
      rule: has(minLaplacian) ? `下限 ${fmt(minLaplacian, 1)}` : "下限未配置",
      bad: focusBad,
    },
    {
      label: "管壁反光",
      value: glare === null ? "—" : glare.toFixed(1),
      unit: "%",
      width: glare === null ? "0%" : pct(glare, glareMax),
      marks: glareLimit !== null ? [pct(glareLimit, glareMax)] : [],
      rule: glareLimit !== null ? `上限 ${glareLimit.toFixed(1)}%` : "上限未配置",
      bad: glareBad,
    },
    {
      label: "平均亮度",
      value: fmt(brightness, 0),
      unit: "/ 255",
      width: has(brightness) ? pct(brightness, 255) : "0%",
      marks: [minBrightness, maxBrightness].filter(has).map((v) => pct(v, 255)),
      rule:
        has(minBrightness) || has(maxBrightness)
          ? `区间 ${fmt(minBrightness, 0)} – ${fmt(maxBrightness, 0)}`
          : "区间未配置",
      bad: lumBad,
    },
  ]

  return (
    <section className="panel">
      <header className="panel-head">
        <h2>光学条件</h2>
        <div className="right">
          <span className={`tag ${viewValid ? "pass" : "warn"}`}>{viewValid ? "视野有效" : "视野无效"}</span>
        </div>
      </header>
      <div className="kpi-row" style={{ gridTemplateColumns: "repeat(3, minmax(0, 1fr))" }}>
        {items.map((it) => (
          <div key={it.label} className="readout">
            <small>{it.label}</small>
            <b style={it.bad ? { color: "var(--bad)" } : undefined}>
              {it.value}
              {it.value !== "—" && it.unit && <em>{it.unit}</em>}
            </b>
            <div className="meter" style={{ marginTop: 12 }}>
              <i className={it.bad ? "bad" : ""} style={{ width: it.width }} />
              {it.marks.map((m, idx) => (
                <s key={`${idx}-${m}`} style={{ left: m }} />
              ))}
            </div>
            <p>{it.rule}</p>
          </div>
        ))}
      </div>
    </section>
  )
}
