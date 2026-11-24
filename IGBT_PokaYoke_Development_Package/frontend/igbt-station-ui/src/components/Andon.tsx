import { type Data, deriveOutcome, guidanceFor, verdictOf } from "@/lib/station"

interface AndonProps {
  state: Data
  compact?: boolean
  meta: { label: string; value: string }[]
  guide?: { title: string }
}

export function Andon({ state, compact, meta, guide }: AndonProps) {
  const outcome = deriveOutcome(state)
  const v = verdictOf(outcome)
  const g = guide ?? guidanceFor(state)

  return (
    <section className={`andon ${v.tone} ${compact ? "compact" : ""}`} aria-label="工位判定状态">
      <div className="andon-live" aria-live="polite" aria-atomic="true">
        <div className="andon-word">
          <strong>{v.word}</strong>
        </div>
        <div className="andon-guide">
          <b>{g.title}</b>
        </div>
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
