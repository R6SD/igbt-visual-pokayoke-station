import { dateTimeOf, pipeText, verdictOf } from "@/lib/station"

export interface FilmstripItem {
  id: string
  cycleId: string
  kind?: string
  outcome: string
  url?: string
  timestamp: string
  labelStatus?: string
  pipeColor?: string
  reason?: string
}

interface FilmstripProps {
  items: FilmstripItem[]
  selectedId?: string
  onSelect: (item: FilmstripItem) => void
}

export function Filmstrip({ items, selectedId, onSelect }: FilmstripProps) {
  return (
    <section className="panel">
      <header className="panel-head">
        <h2>最近记录</h2>
        <div className="right mono">{items.length}</div>
      </header>
      {items.length ? (
        <div className="strip">
          {items.map((item) => {
            const v = verdictOf(item.outcome)
            const sel = selectedId === item.id || selectedId === item.cycleId
            return (
              <button
                key={item.id}
                type="button"
                aria-pressed={sel}
                className={sel ? "sel" : ""}
                title={`${item.cycleId} · ${pipeText(item.pipeColor)} · ${dateTimeOf(item.timestamp)}`}
                onClick={() => onSelect(item)}
              >
                <div className="thumb">
                  {item.url ? <img src={item.url} alt={item.cycleId} /> : <span>{pipeText(item.pipeColor)}</span>}
                  <i className={`bar ${v.tone}`} />
                </div>
                <div className="info">
                  <b>{item.cycleId}</b>
                  <span>{v.word}</span>
                </div>
              </button>
            )
          })}
        </div>
      ) : (
        <div className="empty">暂无记录</div>
      )}
    </section>
  )
}
