import { useRef, useState } from "react"
import stationHeroImg from "@/assets/showcase/station-hero.png"
import stationConceptImg from "@/assets/showcase/station-concept.png"
import tubeYellowImg from "@/assets/showcase/tube-yellow.jpg"
import tubeBlueImg from "@/assets/showcase/tube-blue.jpg"
import "@/showcase.css"

interface ShowcasePageProps {
  onEnterStation: () => void
}

type TubeKind = "yellow" | "blue"
type VerdictKey = "pass" | "alarm" | "blocked"

const verdictCases: Record<
  VerdictKey,
  {
    name: string
    code: string
    tone: "pass" | "alarm" | "warn"
    word: string
    rows: [string, string][]
  }
> = {
  pass: {
    name: "防呆动作完成",
    code: "PASS",
    tone: "pass",
    word: "放行",
    rows: [
      ["双手握持", "左右均在位"],
      ["标签状态", "划除 / 遮挡 / 脱落"],
      ["确认计时", "1.20 s"],
      ["存证校验", "SHA256 归档"],
    ],
  },
  alarm: {
    name: "未处理或违规触管",
    code: "ALARM",
    tone: "alarm",
    word: "报警",
    rows: [
      ["触发原因", "标签仍完整 / 裸手触管"],
      ["工位状态", "闭锁拦截"],
      ["处置动作", "转入返工区"],
      ["复位权限", "人工授权复位"],
    ],
  },
  blocked: {
    name: "反光或视野受阻",
    code: "BLOCKED",
    tone: "warn",
    word: "视野无效",
    rows: [
      ["清晰度", "低于门限"],
      ["高光反光", "超出上限"],
      ["安全策略", "默认闭锁"],
      ["现场动作", "微调料管角度"],
    ],
  },
}

const verdictKeys: VerdictKey[] = ["pass", "alarm", "blocked"]

export function ShowcasePage({ onEnterStation }: ShowcasePageProps) {
  const [tube, setTube] = useState<TubeKind>("yellow")
  const [verdict, setVerdict] = useState<VerdictKey>("pass")
  const verdictTabRefs = useRef<Partial<Record<VerdictKey, HTMLButtonElement | null>>>({})

  const current = verdictCases[verdict]

  const handleVerdictTabKeyDown = (e: React.KeyboardEvent<HTMLButtonElement>, currentKey: VerdictKey) => {
    const idx = verdictKeys.indexOf(currentKey)
    let nextIdx = -1
    if (e.key === "ArrowDown" || e.key === "ArrowRight") {
      nextIdx = (idx + 1) % verdictKeys.length
    } else if (e.key === "ArrowUp" || e.key === "ArrowLeft") {
      nextIdx = (idx - 1 + verdictKeys.length) % verdictKeys.length
    } else if (e.key === "Home") {
      nextIdx = 0
    } else if (e.key === "End") {
      nextIdx = verdictKeys.length - 1
    }
    if (nextIdx >= 0) {
      e.preventDefault()
      const nextKey = verdictKeys[nextIdx]
      setVerdict(nextKey)
      verdictTabRefs.current[nextKey]?.focus()
    }
  }

  return (
    <div className="sc">
      <header className="sc-header">
        <div className="sc-in sc-header-row">
          <a href="#top" className="sc-logo">
            <span className="sc-logo-mark" aria-hidden="true" />
            <span>晶界智检</span>
          </a>

          <nav className="sc-nav" aria-label="页面导航">
            <a href="#vision">检测实景</a>
            <a href="#logic">防呆逻辑</a>
            <a href="#deploy">工位部署</a>
          </nav>

          <button type="button" className="sc-btn" onClick={onEnterStation}>
            进入工位
          </button>
        </div>
      </header>

      <section id="top" className="sc-hero">
        <div className="sc-in sc-hero-grid">
          <div className="sc-hero-copy">
            <h1>IGBT 单管透明料管视觉防呆工位</h1>
            <div className="sc-hero-actions">
              <button type="button" className="sc-btn" onClick={onEnterStation}>
                实时监控
              </button>
              <a href="#vision" className="sc-btn ghost">
                检测实景
              </a>
            </div>
            <div className="sc-hero-specs">
              <div className="sc-spec-card">
                <b>720P · 25FPS</b>
                <span>实时视觉采集</span>
              </div>
              <div className="sc-spec-card">
                <b>1.20 s</b>
                <span>脱落确认门限</span>
              </div>
              <div className="sc-spec-card">
                <b>C++20</b>
                <span>本机原生引擎</span>
              </div>
            </div>
          </div>

          <div className="sc-photo-frame">
            <img src={stationHeroImg} alt="IGBT 单管透明料管视觉防呆工位实景" />
          </div>
        </div>
      </section>

      <section id="vision" className="sc-band">
        <div className="sc-in">
          <div className="sc-band-head">
            <h2>双色料管动态追踪</h2>
            <div className="sc-swatches" role="group" aria-label="料管颜色">
              <button
                type="button"
                className={`sc-swatch ${tube === "yellow" ? "on" : ""}`}
                aria-pressed={tube === "yellow"}
                onClick={() => setTube("yellow")}
              >
                <i style={{ background: "#d4a62a" }} aria-hidden="true" />
                黄管
              </button>
              <button
                type="button"
                className={`sc-swatch ${tube === "blue" ? "on" : ""}`}
                aria-pressed={tube === "blue"}
                onClick={() => setTube("blue")}
              >
                <i style={{ background: "#2b6cb0" }} aria-hidden="true" />
                蓝管
              </button>
            </div>
          </div>

          <div className="sc-photo-frame">
            <img
              key={tube}
              src={tube === "yellow" ? tubeYellowImg : tubeBlueImg}
              alt={tube === "yellow" ? "黄管检测实拍" : "蓝管检测实拍"}
            />
          </div>

          <div className="sc-facts">
            <div className="sc-fact">
              <b>五区隔离</b>
              <span>握持与标签分区</span>
            </div>
            <div className="sc-fact">
              <b>光学门控</b>
              <span>清晰度与反光校验</span>
            </div>
            <div className="sc-fact">
              <b>SHA256</b>
              <span>帧级防篡改归档</span>
            </div>
          </div>
        </div>
      </section>

      <section id="logic" className="sc-band sc-dark">
        <div className="sc-in">
          <div className="sc-band-head">
            <h2>默认闭锁状态机</h2>
          </div>

          <div className="sc-logic-grid">
            <div className="sc-state-list" role="tablist" aria-label="工况示例" aria-orientation="vertical">
              {verdictKeys.map((key) => {
                const item = verdictCases[key]
                const selected = verdict === key
                return (
                  <button
                    key={key}
                    ref={(el) => {
                      verdictTabRefs.current[key] = el
                    }}
                    id={`sc-verdict-tab-${key}`}
                    type="button"
                    role="tab"
                    tabIndex={selected ? 0 : -1}
                    aria-selected={selected}
                    aria-controls="sc-verdict-tabpanel"
                    className={`sc-state-btn ${selected ? "on" : ""}`}
                    onClick={() => setVerdict(key)}
                    onKeyDown={(e) => handleVerdictTabKeyDown(e, key)}
                  >
                    <b>{item.name}</b>
                    <span className={`sc-state-tag ${item.tone}`}>{item.code}</span>
                  </button>
                )
              })}
            </div>

            <div
              id="sc-verdict-tabpanel"
              className="sc-verdict-panel"
              role="tabpanel"
              aria-labelledby={`sc-verdict-tab-${verdict}`}
              tabIndex={0}
            >
              <div className="sc-verdict-top">
                <div className={`sc-verdict-word ${current.tone}`}>{current.word}</div>
                <span className={`sc-state-tag ${current.tone}`}>{current.code}</span>
              </div>

              <table className="sc-spec-table">
                <tbody>
                  {current.rows.map(([label, val]) => (
                    <tr key={label}>
                      <th>{label}</th>
                      <td>{val}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      <section id="deploy" className="sc-band">
        <div className="sc-in sc-deploy-grid">
          <div className="sc-deploy-col">
            <h2>三步接入产线</h2>

            <ol className="sc-steps">
              <li>
                <span>01</span>
                <b>五区视角标定</b>
              </li>
              <li>
                <span>02</span>
                <b>影子挂网验证</b>
              </li>
              <li>
                <span>03</span>
                <b>质量联锁上线</b>
              </li>
            </ol>

            <div>
              <button type="button" className="sc-btn" onClick={onEnterStation}>
                进入工位
              </button>
            </div>
          </div>

          <div className="sc-photo-frame">
            <img src={stationConceptImg} alt="工位结构方案图" />
          </div>
        </div>
      </section>

      <footer className="sc-footer">
        <div className="sc-in sc-footer-row">
          <span>晶界智检</span>
          <span>C++20 · 720P 25FPS · 127.0.0.1</span>
        </div>
      </footer>
    </div>
  )
}
