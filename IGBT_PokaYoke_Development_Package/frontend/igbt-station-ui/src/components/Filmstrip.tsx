import { pipeText, verdictOf } from "@/lib/station"

export interface FilmstripItem {
  id: string
  cycleId: string
  kind?: string
  outcome: string
  url?: string
  timestamp: string
  labelStatus?: string
  pipeColor?: string
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
        <h2>最近检验</h2>
        <div className="right mono">{items.length} 条</div>
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
                onClick={() => onSelect(item)}
              >
                <div className="thumb">
                  {item.url ? <img src={item.url} alt={item.cycleId} /> : <span>{pipeText(item.pipeColor)} 无图</span>}
                  <i className={`bar ${v.tone}`} />
                </div>
                <div className="info">
                  <b>{item.cycleId}</b>
                  <span>
                    <span className="mono">{item.timestamp ? new Date(item.timestamp).toLocaleTimeString("zh-CN", { hour12: false }) : "—"}</span>
                    <span>{v.word}</span>
                  </span>
                </div>
              </button>
            )
          })}
        </div>
      ) : (
        <div className="empty" style={{ minHeight: 96 }}>
          暂无检验记录
        </div>
      )}
    </section>
  )
}
