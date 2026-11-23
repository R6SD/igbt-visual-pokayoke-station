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
    summary: string
    code: string
    tone: "pass" | "alarm" | "warn"
    word: string
    rows: [string, string][]
  }
> = {
  pass: {
    name: "工艺动作到位，允许放行",
    summary: "双手白手套握持料管两端，防呆标签完成划除、遮挡或撕除满 1.2 秒。",
    code: "PASS",
    tone: "pass",
    word: "放行",
    rows: [
      ["双手握持区", "左端与右端均在位"],
      ["标签分类结论", "已撕除（C 类）"],
      ["连续无遮挡确认", "1.20 秒，已达放行门限"],
      ["光学视野状态", "清晰度与反光占比合格"],
      ["证据归档", "SHA256 独占写入本地库"],
    ],
  },
  alarm: {
    name: "标签未处理或操作异常，拦截报警",
    summary: "料管离开工位时原标签仍完整存在，或手部未按规范动作即移出。",
    code: "ALARM",
    tone: "alarm",
    word: "报警",
    rows: [
      ["双手握持区", "检测到提前松手移出"],
      ["标签分类结论", "原标签仍完整"],
      ["连续无遮挡确认", "0.00 秒，未触发放行"],
      ["处置指引", "将料管放入返工暂存区"],
      ["联锁策略", "影子模式告警 / 审批后硬线停机"],
    ],
  },
  blocked: {
    name: "管壁强反光或视野受阻，默认闭锁",
    summary: "镜头失焦、管壁镜面反光超限或标签背向镜头时，系统保持闭锁不放行。",
    code: "BLOCKED",
    tone: "warn",
    word: "视野无效",
    rows: [
      ["拉普拉斯清晰度", "低于设定下限"],
      ["高光反光面积", "超出安全上限"],
      ["安全策略", "默认闭锁，绝不据此输出放行"],
      ["操作提示", "微调料管角度，使标签正对镜头"],
      ["样本归类", "标记为未知挑战样本"],
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
            打开工位监控
          </button>
        </div>
      </header>

      {/* 第一屏：Hero，一屏只讲一件事 */}
      <section id="top" className="sc-hero">
        <div className="sc-in sc-hero-grid">
          <div className="sc-hero-copy">
            <h1>透明料管防呆动作，每一管都有据可查</h1>
            <p>
              面向站立式 IGBT 单管人工目检工位的本地机器视觉防呆系统。实时识别双手握持姿态与中央标签状态，未完成划除、遮挡或撕除确认的料管默认闭锁不放行。
            </p>
            <div className="sc-hero-actions">
              <button type="button" className="sc-btn lg" onClick={onEnterStation}>
                进入实时监控
              </button>
              <a href="#vision" className="sc-btn ghost lg">
                查看实拍检测画面
              </a>
            </div>
            <div className="sc-hero-specs">
              <div>
                <b>530 mm</b>
                <span>覆盖整管与两端握持区</span>
              </div>
              <div>
                <b>1.20 s</b>
                <span>标签脱落连续确认门限</span>
              </div>
              <div>
                <b>127.0.0.1</b>
                <span>本机环回与独占存证</span>
              </div>
            </div>
          </div>

          <div className="sc-photo-frame">
            <img
              src={stationHeroImg}
              alt="站立式 IGBT 单管透明料管人工目检视觉防呆工位实景"
            />
          </div>
        </div>
      </section>

      {/* 第二屏：真实检测实拍，色样切换真实改变画面 */}
      <section id="vision" className="sc-band">
        <div className="sc-in">
          <div className="sc-band-head">
            <div>
              <h2>同一套广角镜头，同时看清两端手套与中央标签</h2>
              <p style={{ marginTop: 12 }}>
                针对产线常见的黄色与蓝色防静电料管独立验证检测框，自动区分左右握持区与中央标签区域。
              </p>
            </div>
            <div className="sc-swatches" role="group" aria-label="料管规格切换">
              <button
                type="button"
                className={`sc-swatch ${tube === "yellow" ? "on" : ""}`}
                aria-pressed={tube === "yellow"}
                onClick={() => setTube("yellow")}
              >
                <i style={{ background: "#d4a62a" }} aria-hidden="true" />
                黄色料管
              </button>
              <button
                type="button"
                className={`sc-swatch ${tube === "blue" ? "on" : ""}`}
                aria-pressed={tube === "blue"}
                onClick={() => setTube("blue")}
              >
                <i style={{ background: "#2b6cb0" }} aria-hidden="true" />
                蓝色料管
              </button>
            </div>
          </div>

          <div className="sc-photo-frame">
            <img
              key={tube}
              src={tube === "yellow" ? tubeYellowImg : tubeBlueImg}
              alt={tube === "yellow" ? "黄色透明料管双手持管与检测框实拍参考" : "蓝色透明料管双手持管与检测框实拍参考"}
            />
          </div>

          <div className="sc-facts">
            <div className="sc-fact">
              <h3>左右握持区物理隔离</h3>
              <p>
                归一化 ROI 强制将左右白手套握持区与中央标签区分离，避免持管手势被误判为标签接触。
              </p>
            </div>
            <div className="sc-fact">
              <h3>光学质量前置校验</h3>
              <p>
                每一帧先核验拉普拉斯清晰度、平均灰度与高光反光面积占比，画面不合格直接拒判。
              </p>
            </div>
            <div className="sc-fact">
              <h3>原图与周期双向追溯</h3>
              <p>
                关键帧以 SHA256 指纹独占写入本地磁盘并绑定周期号与微秒级状态机日志，绝不覆写历史记录。
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 第三屏：唯一对比深色段，展示防呆判定逻辑 */}
      <section id="logic" className="sc-band sc-dark">
        <div className="sc-in">
          <div className="sc-band-head">
            <div>
              <h2>不确定就不放行，把默认安全写进状态机</h2>
            </div>
            <p>
              点击左侧三种典型现场工况，查看状态机在放行、报警与视野受阻时的真实判定依据。
            </p>
          </div>

          <div className="sc-logic-grid">
            <div className="sc-state-list" role="tablist" aria-label="现场工况示例" aria-orientation="vertical">
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
                    <small>{item.summary}</small>
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

      {/* 第四屏：有序部署流程 + 工位方案图 */}
      <section id="deploy" className="sc-band">
        <div className="sc-in sc-deploy-grid">
          <div>
            <h2 style={{ margin: 0, fontSize: "clamp(26px, 3vw, 38px)", fontWeight: 600, lineHeight: 1.24 }}>
              从影子观察起步，分三步接入现有工位
            </h2>

            <ol className="sc-steps">
              <li>
                <span>01</span>
                <div>
                  <b>架设相机并标定归一化区域</b>
                  <p>在参考帧上框选料管、标签与两端握持区，保存为不可覆写的版本化配置。</p>
                </div>
              </li>
              <li>
                <span>02</span>
                <div>
                  <b>影子模式零风险挂网验证</b>
                  <p>仅记录判定与证据帧，不干预产线节拍，也不驱动未经审批的机械联锁。</p>
                </div>
              </li>
              <li>
                <span>03</span>
                <div>
                  <b>验收后对接产线告警与追溯</b>
                  <p>确认准确率与节拍达标后，按产线规范接入声光提示、PLC 联锁或本地质量库。</p>
                </div>
              </li>
            </ol>

            <button type="button" className="sc-btn lg" onClick={onEnterStation}>
              打开工位监控
            </button>
          </div>

          <div className="sc-photo-frame">
            <img
              src={stationConceptImg}
              alt="IGBT 单管透明料管视觉防呆工位结构方案图"
            />
          </div>
        </div>
      </section>

      <footer className="sc-footer">
        <div className="sc-in sc-footer-row">
          <span>晶界智检，IGBT 单管透明料管视觉防呆系统</span>
          <span>本地环回节点 127.0.0.1，影子观察运行模式</span>
        </div>
      </footer>
    </div>
  )
}
