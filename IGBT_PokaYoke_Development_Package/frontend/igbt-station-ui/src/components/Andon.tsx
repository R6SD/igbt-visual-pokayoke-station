import { type Data, deriveOutcome, guidanceFor, verdictOf } from "@/lib/station"

interface AndonProps {
  state: Data
  compact?: boolean
  meta: { label: string; value: string }[]
  /** 覆盖默认的引导文案（工程师视图展示判定原因） */
  guide?: { title: string; desc?: string }
}

/** 工位判定灯条：整块底色即状态，远距离也能一眼看清 */
export function Andon({ state, compact, meta, guide }: AndonProps) {
  const outcome = deriveOutcome(state)
  const v = verdictOf(outcome)
  const g = guide ?? guidanceFor(state)

  return (
    <section className={`andon ${v.tone} ${compact ? "compact" : ""}`} aria-live="polite">
      <div className="andon-word">
        <strong>{v.word}</strong>
      </div>
      <div className="andon-guide">
        <b>{g.title}</b>
        {g.desc && <p>{g.desc}</p>}
      </div>
      <div className="andon-meta">
        {meta.map((m) => (
          <div key={m.label}>
            <small>{m.label}</small>
            <b>{m.value}</b>
          </div>
        ))}
      </div>
    </section>
  )
}
