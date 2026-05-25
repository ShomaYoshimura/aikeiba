import { useState } from "react";

const COLORS = {
  bg: "#0a0e1a",
  surface: "#111827",
  surfaceHigh: "#1a2235",
  border: "#1e2d47",
  accent: "#e8b84b",
  accentDim: "#a07c28",
  accentGlow: "rgba(232,184,75,0.15)",
  blue: "#3b82f6",
  blueDim: "#1d4ed8",
  green: "#22c55e",
  red: "#ef4444",
  purple: "#a855f7",
  cyan: "#06b6d4",
  text: "#e2e8f0",
  textDim: "#94a3b8",
  textMuted: "#475569",
};

const layers = [
  {
    id: "L0",
    label: "Layer 0",
    title: "データ収集・前処理層",
    color: COLORS.textMuted,
    accentColor: "#64748b",
    icon: "⛁",
    summary: "全データの取得・クレンジング・特徴量エンジニアリング",
    modules: [
      {
        name: "公式データコネクタ",
        tech: "Python / JRA-VAN API",
        desc: "過去レース結果・ラップ・馬体重・オッズの全量取得",
        inputs: ["JRA-VAN", "netkeiba", "TARGET"],
        outputs: ["raw_races.parquet"],
      },
      {
        name: "スピード指数エンジン",
        tech: "Pandas / NumPy",
        desc: "馬場差・斤量補正済みスピード指数を全レースに付与",
        inputs: ["raw_races.parquet"],
        outputs: ["speed_index.parquet"],
      },
      {
        name: "血統ベクトル化",
        tech: "Word2Vec / 血統DB",
        desc: "父・母父・近親の血統を数値ベクトルに変換",
        inputs: ["pedigree_db"],
        outputs: ["pedigree_vectors.npy"],
      },
      {
        name: "特徴量ストア",
        tech: "Feast / DuckDB",
        desc: "全特徴量を時系列管理。未来リーク防止を保証",
        inputs: ["各種parquet"],
        outputs: ["feature_store"],
      },
    ],
  },
  {
    id: "L1",
    label: "Layer 1",
    title: "実力評価層",
    color: COLORS.blue,
    accentColor: COLORS.blue,
    icon: "◈",
    summary: "各馬の絶対実力・相対実力・長期トレンドを定量評価",
    modules: [
      {
        name: "ELOレーティングエンジン",
        tech: "Python（独自実装）",
        desc: "コース×距離×馬場別に独立したELOを管理。全頭リアルタイム更新",
        inputs: ["raw_races.parquet"],
        outputs: ["elo_ratings.db"],
        detail: "K係数はレースグレード×着差で動的調整。G1勝ちはK=64、条件戦はK=16",
      },
      {
        name: "スピード指数トレンド分析",
        tech: "statsmodels / Prophet",
        desc: "過去10走の指数トレンド・上昇/下降傾向・ピーク時期を検出",
        inputs: ["speed_index.parquet"],
        outputs: ["trend_scores"],
        detail: "Prophet で季節性（春秋）・長期トレンドを分解",
      },
      {
        name: "個別σ算出エンジン",
        tech: "NumPy",
        desc: "馬ごとの過去成績バラツキからモンテカルロ用の個別ノイズσを生成",
        inputs: ["speed_index.parquet"],
        outputs: ["individual_sigma"],
        detail: "経験値・初コース・距離経験補正を含む",
      },
    ],
  },
  {
    id: "L2",
    label: "Layer 2",
    title: "適性評価層",
    color: COLORS.cyan,
    accentColor: COLORS.cyan,
    icon: "◎",
    summary: "コース・距離・馬場・血統の適性を多軸で数値化",
    modules: [
      {
        name: "コース適性モデル",
        tech: "ロジスティック回帰",
        desc: "同コース・同距離・同回り・同馬場での過去成績から適性スコアを算出",
        inputs: ["feature_store", "elo_ratings.db"],
        outputs: ["course_fit_scores"],
        detail: "直線長・坂の位置・ストライド適性も特徴量として投入",
      },
      {
        name: "KNN類似レース検索",
        tech: "scikit-learn NearestNeighbors",
        desc: "今回のレース条件（距離・逃げ馬数・メンバー脚質構成）に類似した過去レースをK=15件抽出",
        inputs: ["feature_store"],
        outputs: ["similar_race_patterns"],
        detail: "類似レースの勝ち馬パターン・展開・ペースを参照情報として提供",
      },
      {
        name: "血統適性スコアラー",
        tech: "Random Forest",
        desc: "父・母父の組み合わせ×コース条件での過去成績から血統適性を算出",
        inputs: ["pedigree_vectors.npy", "feature_store"],
        outputs: ["pedigree_fit_scores"],
        detail: "産駒数が少ない場合はガウス過程で補間",
      },
      {
        name: "ガウス過程補間器",
        tech: "GPyTorch",
        desc: "データ不足馬（新馬・外国馬）の適性スコアを類似条件から補間。不確実性σも出力",
        inputs: ["course_fit_scores", "pedigree_fit_scores"],
        outputs: ["gp_imputed_scores", "uncertainty_sigma"],
        detail: "データ量が多い馬ほど信頼区間が狭くなる",
      },
    ],
  },
  {
    id: "L3",
    label: "Layer 3",
    title: "予測モデル層",
    color: COLORS.purple,
    accentColor: COLORS.purple,
    icon: "▣",
    summary: "全特徴量を統合した着順予測モデル群",
    modules: [
      {
        name: "ランキング学習モデル",
        tech: "LightGBM LambdaMART",
        desc: "全特徴量を統合して着順スコアを直接学習。過去3年G1/G2/G3データで訓練",
        inputs: ["L1全出力", "L2全出力", "feature_store"],
        outputs: ["ranking_scores"],
        detail: "NDCG@3を最適化。特徴量重要度でモデル解釈性を確保",
      },
      {
        name: "ハザードモデル",
        tech: "lifelines CoxPH",
        desc: "「他馬に抜かれるリスク」を時系列で推定。失速タイミングの予測",
        inputs: ["feature_store", "elo_ratings.db"],
        outputs: ["hazard_scores"],
        detail: "脚質×ペース×残り距離の交互作用項を含む",
      },
      {
        name: "ベイズ推定エンジン",
        tech: "PyMC",
        desc: "各馬の「真の実力」を事後分布として推定。当日情報で逐次更新",
        inputs: ["ranking_scores", "speed_index.parquet"],
        outputs: ["posterior_distributions"],
        detail: "MCMCサンプリング2000回。当日馬体重・馬場発表で事後更新",
      },
      {
        name: "エージェントベースシミュレーター",
        tech: "Mesa（Python ABS）",
        desc: "各馬をエージェントとして定義。位置取り→直線伸びの2段階プロセスを物理的に再現",
        inputs: ["ranking_scores", "similar_race_patterns"],
        outputs: ["abs_finish_orders"],
        detail: "カーブでの距離ロス・坂の影響・スタートダッシュを実装",
      },
    ],
  },
  {
    id: "L4",
    label: "Layer 4",
    title: "シミュレーション層",
    color: COLORS.accent,
    accentColor: COLORS.accent,
    icon: "⟳",
    summary: "全モデルを統合した50,000回シミュレーションと確率分布生成",
    modules: [
      {
        name: "動的ペース生成器",
        tech: "NumPy",
        desc: "逃げ馬数・先行馬質・枠順から1000m通過ペース分布を自動算出（静的シナリオ分類を排除）",
        inputs: ["feature_store"],
        outputs: ["pace_distribution"],
        detail: "連続変数として扱い、スロー〜ハイの全範囲をカバー",
      },
      {
        name: "共通ショック生成器",
        tech: "NumPy",
        desc: "馬場バイアス・内外バイアス・ペースショックを各試行で共通ノイズとして生成",
        inputs: ["pace_distribution"],
        outputs: ["common_shocks"],
        detail: "相関行列で各ショックの相関を管理",
      },
      {
        name: "モンテカルロエンジン",
        tech: "NumPy / Numba（JIT高速化）",
        desc: "t分布ノイズ＋個別σ＋共通ショックで50,000回試行。着順全分布を生成",
        inputs: ["posterior_distributions", "individual_sigma", "common_shocks"],
        outputs: ["finish_distributions"],
        detail: "Numba JITで高速化。50,000回を数秒で完了",
      },
      {
        name: "アンサンブル統合器",
        tech: "Stacking（LightGBM）",
        desc: "4モデル（ランキング学習・ハザード・ベイズ・ABS）の出力を重み付き統合。過去検証で重みを最適化",
        inputs: ["ranking_scores", "hazard_scores", "posterior_distributions", "abs_finish_orders"],
        outputs: ["ensemble_win_probs"],
        detail: "重み：ランキング学習35% / ベイズ25% / ABS25% / ハザード15%",
      },
    ],
  },
  {
    id: "L5",
    label: "Layer 5",
    title: "キャリブレーション・評価層",
    color: COLORS.green,
    accentColor: COLORS.green,
    icon: "⊕",
    summary: "予測確率の現実合わせと継続的精度検証",
    modules: [
      {
        name: "キャリブレーター",
        tech: "IsotonicRegression / Platt Scaling",
        desc: "過去G1/G2レースでの予測vs実績から補正関数を学習。シミュ30%=実際の30%を保証",
        inputs: ["ensemble_win_probs", "historical_actuals"],
        outputs: ["calibrated_probs"],
        detail: "等張回帰で単調性を保ちながら補正",
      },
      {
        name: "信頼区間生成器",
        tech: "Bootstrap / GPyTorch",
        desc: "各馬の勝率に95%信頼区間を付与。「確信を持てる予測」と「読みにくい馬」を区別",
        inputs: ["finish_distributions", "uncertainty_sigma"],
        outputs: ["confidence_intervals"],
      },
      {
        name: "継続評価モニター",
        tech: "MLflow / Evidently AI",
        desc: "ブライアスコア・ログロス・順位相関を全G1でトラッキング。モデルドリフトを検出",
        inputs: ["calibrated_probs", "actual_results"],
        outputs: ["model_performance_dashboard"],
        detail: "月次でキャリブレーション曲線を再チェック",
      },
    ],
  },
  {
    id: "L6",
    label: "Layer 6",
    title: "出力・インターフェース層",
    color: COLORS.red,
    accentColor: COLORS.red,
    icon: "▤",
    summary: "分析結果の多形式出力とインタラクティブ可視化",
    modules: [
      {
        name: "確率マトリクス出力",
        tech: "Pandas / Rich",
        desc: "全馬の1〜3着確率・複勝率・決着パターン頻度をターミナル/CSV出力",
        inputs: ["calibrated_probs", "confidence_intervals"],
        outputs: ["probability_matrix.csv"],
      },
      {
        name: "ペース感応度マップ",
        tech: "Matplotlib / Plotly",
        desc: "スロー〜ハイの各ペースで各馬の勝率変化をヒートマップ表示",
        inputs: ["finish_distributions"],
        outputs: ["pace_sensitivity_plot"],
      },
      {
        name: "モデル合意度レポート",
        tech: "Jinja2テンプレート",
        desc: "4モデルの予測一致度・モデル間分散を馬ごとにレポート化。確信度の指標",
        inputs: ["ranking_scores", "hazard_scores", "posterior_distributions", "abs_finish_orders"],
        outputs: ["consensus_report.md"],
      },
      {
        name: "Webダッシュボード",
        tech: "Streamlit / FastAPI",
        desc: "当日朝に実行して結果をブラウザで確認できるインタラクティブUI",
        inputs: ["全Layer出力"],
        outputs: ["Web UI"],
      },
    ],
  },
];

const dataFlows = [
  { from: "L0", to: "L1", label: "特徴量ストア" },
  { from: "L0", to: "L2", label: "特徴量ストア" },
  { from: "L1", to: "L3", label: "実力スコア" },
  { from: "L2", to: "L3", label: "適性スコア" },
  { from: "L3", to: "L4", label: "確率分布" },
  { from: "L4", to: "L5", label: "粗勝率" },
  { from: "L5", to: "L6", label: "補正済み確率" },
];

const techStack = [
  { category: "データ処理", items: ["Python 3.11", "Pandas", "DuckDB", "Polars"] },
  { category: "機械学習", items: ["LightGBM", "scikit-learn", "PyMC", "GPyTorch"] },
  { category: "シミュレーション", items: ["NumPy", "Numba (JIT)", "Mesa (ABS)", "lifelines"] },
  { category: "MLOps", items: ["MLflow", "Evidently AI", "Feast", "DVC"] },
  { category: "インフラ", items: ["FastAPI", "Streamlit", "Docker", "GitHub Actions"] },
  { category: "データソース", items: ["JRA-VAN", "netkeiba", "TARGET frontier", "JRA公式"] },
];

const kpis = [
  { label: "Top-3的中率", target: ">55%", baseline: "ランダム:16%", color: COLORS.green },
  { label: "ブライアスコア", target: "<0.12", baseline: "オッズ逆数:0.16", color: COLORS.blue },
  { label: "順位相関", target: ">0.45", baseline: "ランダム:0.0", color: COLORS.purple },
  { label: "モデル合意度", target: "4/4一致時的中率>65%", baseline: "—", color: COLORS.accent },
];

export default function App() {
  const [activeLayer, setActiveLayer] = useState(null);
  const [activeModule, setActiveModule] = useState(null);
  const [view, setView] = useState("architecture"); // architecture | tech | kpi | flow

  const selectedLayer = layers.find((l) => l.id === activeLayer);

  return (
    <div style={{
      background: COLORS.bg,
      minHeight: "100vh",
      fontFamily: "'JetBrains Mono', 'Fira Code', 'Courier New', monospace",
      color: COLORS.text,
      padding: "24px",
    }}>

      {/* ヘッダー */}
      <div style={{ marginBottom: "32px" }}>
        <div style={{
          display: "flex", alignItems: "baseline", gap: "16px", marginBottom: "8px",
        }}>
          <div style={{
            fontSize: "11px", letterSpacing: "4px", color: COLORS.accent,
            textTransform: "uppercase",
          }}>
            GRADE RACE PREDICTION ENGINE
          </div>
          <div style={{
            fontSize: "10px", color: COLORS.textMuted, letterSpacing: "1px",
          }}>
            v1.0 — ARCHITECTURE BLUEPRINT
          </div>
        </div>
        <h1 style={{
          fontSize: "clamp(18px, 3vw, 28px)", fontWeight: "700",
          color: COLORS.text, margin: 0, letterSpacing: "-0.5px",
        }}>
          重賞予想シミュレーター
          <span style={{ color: COLORS.accent }}> 完全アーキテクチャ</span>
        </h1>
        <p style={{ color: COLORS.textDim, fontSize: "13px", marginTop: "8px", lineHeight: "1.6" }}>
          ELO × KNN × ランキング学習 × ハザードモデル × ベイズ推定 × ABS × モンテカルロ の7手法統合
        </p>
      </div>

      {/* タブ */}
      <div style={{ display: "flex", gap: "4px", marginBottom: "24px", flexWrap: "wrap" }}>
        {[
          { id: "architecture", label: "🏗 レイヤー構成" },
          { id: "flow", label: "⟶ データフロー" },
          { id: "tech", label: "⚙ 技術スタック" },
          { id: "kpi", label: "📊 評価指標" },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setView(tab.id)}
            style={{
              padding: "8px 16px",
              background: view === tab.id ? COLORS.accent : COLORS.surfaceHigh,
              color: view === tab.id ? COLORS.bg : COLORS.textDim,
              border: `1px solid ${view === tab.id ? COLORS.accent : COLORS.border}`,
              borderRadius: "4px",
              cursor: "pointer",
              fontSize: "12px",
              fontFamily: "inherit",
              fontWeight: view === tab.id ? "700" : "400",
              transition: "all 0.15s",
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* ── アーキテクチャ ── */}
      {view === "architecture" && (
        <div style={{ display: "flex", gap: "20px", flexWrap: "wrap" }}>

          {/* レイヤーリスト */}
          <div style={{ flex: "0 0 300px", minWidth: "260px" }}>
            <div style={{
              fontSize: "10px", color: COLORS.textMuted,
              letterSpacing: "3px", marginBottom: "12px",
            }}>
              LAYERS — クリックで詳細展開
            </div>
            {layers.map((layer) => (
              <div
                key={layer.id}
                onClick={() => {
                  setActiveLayer(activeLayer === layer.id ? null : layer.id);
                  setActiveModule(null);
                }}
                style={{
                  padding: "12px 14px",
                  marginBottom: "6px",
                  background: activeLayer === layer.id ? COLORS.surfaceHigh : COLORS.surface,
                  border: `1px solid ${activeLayer === layer.id ? layer.accentColor : COLORS.border}`,
                  borderLeft: `3px solid ${layer.accentColor}`,
                  borderRadius: "4px",
                  cursor: "pointer",
                  transition: "all 0.15s",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                  <span style={{ color: layer.accentColor, fontSize: "16px" }}>{layer.icon}</span>
                  <div>
                    <div style={{
                      fontSize: "10px", color: layer.accentColor,
                      letterSpacing: "2px", marginBottom: "2px",
                    }}>
                      {layer.id} — {layer.label}
                    </div>
                    <div style={{ fontSize: "13px", fontWeight: "600" }}>
                      {layer.title}
                    </div>
                  </div>
                </div>
                <div style={{
                  fontSize: "11px", color: COLORS.textDim,
                  marginTop: "6px", lineHeight: "1.5",
                  paddingLeft: "26px",
                }}>
                  {layer.summary}
                </div>
                <div style={{
                  fontSize: "10px", color: COLORS.textMuted,
                  marginTop: "4px", paddingLeft: "26px",
                }}>
                  {layer.modules.length} modules
                </div>
              </div>
            ))}
          </div>

          {/* 詳細パネル */}
          <div style={{ flex: 1, minWidth: "280px" }}>
            {!selectedLayer && (
              <div style={{
                height: "200px", display: "flex", alignItems: "center",
                justifyContent: "center", color: COLORS.textMuted,
                border: `1px dashed ${COLORS.border}`, borderRadius: "6px",
                fontSize: "13px",
              }}>
                ← レイヤーを選択してください
              </div>
            )}

            {selectedLayer && (
              <div>
                <div style={{
                  padding: "16px",
                  background: COLORS.surfaceHigh,
                  border: `1px solid ${selectedLayer.accentColor}`,
                  borderRadius: "6px",
                  marginBottom: "16px",
                }}>
                  <div style={{
                    fontSize: "10px", color: selectedLayer.accentColor,
                    letterSpacing: "3px", marginBottom: "6px",
                  }}>
                    {selectedLayer.id} — {selectedLayer.label.toUpperCase()}
                  </div>
                  <div style={{ fontSize: "16px", fontWeight: "700", marginBottom: "8px" }}>
                    {selectedLayer.title}
                  </div>
                  <div style={{ fontSize: "12px", color: COLORS.textDim, lineHeight: "1.7" }}>
                    {selectedLayer.summary}
                  </div>
                </div>

                <div style={{
                  fontSize: "10px", color: COLORS.textMuted,
                  letterSpacing: "3px", marginBottom: "10px",
                }}>
                  MODULES
                </div>

                {selectedLayer.modules.map((mod, i) => (
                  <div
                    key={i}
                    onClick={() => setActiveModule(activeModule === i ? null : i)}
                    style={{
                      padding: "12px 14px",
                      marginBottom: "8px",
                      background: activeModule === i ? "#1a2235" : COLORS.surface,
                      border: `1px solid ${activeModule === i ? selectedLayer.accentColor : COLORS.border}`,
                      borderRadius: "4px",
                      cursor: "pointer",
                      transition: "all 0.15s",
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                      <div style={{ fontSize: "13px", fontWeight: "600", marginBottom: "4px" }}>
                        {mod.name}
                      </div>
                      <div style={{
                        fontSize: "10px",
                        background: COLORS.bg,
                        color: selectedLayer.accentColor,
                        border: `1px solid ${selectedLayer.accentColor}`,
                        padding: "2px 8px",
                        borderRadius: "3px",
                        whiteSpace: "nowrap",
                        marginLeft: "8px",
                      }}>
                        {mod.tech}
                      </div>
                    </div>
                    <div style={{ fontSize: "12px", color: COLORS.textDim, lineHeight: "1.6" }}>
                      {mod.desc}
                    </div>

                    {activeModule === i && (
                      <div style={{ marginTop: "10px", paddingTop: "10px", borderTop: `1px solid ${COLORS.border}` }}>
                        {mod.inputs && (
                          <div style={{ marginBottom: "6px" }}>
                            <span style={{ fontSize: "10px", color: COLORS.textMuted, letterSpacing: "2px" }}>INPUT  </span>
                            {mod.inputs.map((inp, j) => (
                              <span key={j} style={{
                                fontSize: "11px",
                                background: COLORS.bg,
                                color: COLORS.blue,
                                border: `1px solid ${COLORS.blueDim}`,
                                padding: "1px 6px",
                                borderRadius: "3px",
                                marginRight: "4px",
                              }}>{inp}</span>
                            ))}
                          </div>
                        )}
                        {mod.outputs && (
                          <div style={{ marginBottom: "6px" }}>
                            <span style={{ fontSize: "10px", color: COLORS.textMuted, letterSpacing: "2px" }}>OUTPUT </span>
                            {mod.outputs.map((out, j) => (
                              <span key={j} style={{
                                fontSize: "11px",
                                background: COLORS.bg,
                                color: COLORS.green,
                                border: `1px solid ${COLORS.green}`,
                                padding: "1px 6px",
                                borderRadius: "3px",
                                marginRight: "4px",
                              }}>{out}</span>
                            ))}
                          </div>
                        )}
                        {mod.detail && (
                          <div style={{
                            fontSize: "11px", color: COLORS.accent,
                            lineHeight: "1.6", marginTop: "4px",
                            padding: "8px",
                            background: COLORS.accentGlow,
                            borderRadius: "3px",
                          }}>
                            💡 {mod.detail}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── データフロー ── */}
      {view === "flow" && (
        <div>
          <div style={{
            fontSize: "10px", color: COLORS.textMuted,
            letterSpacing: "3px", marginBottom: "20px",
          }}>
            DATA FLOW — 7層のパイプライン
          </div>

          {/* パイプライン縦表示 */}
          <div style={{ maxWidth: "700px" }}>
            {layers.map((layer, idx) => (
              <div key={layer.id}>
                {/* レイヤーブロック */}
                <div style={{
                  padding: "16px 20px",
                  background: COLORS.surface,
                  border: `1px solid ${layer.accentColor}`,
                  borderLeft: `4px solid ${layer.accentColor}`,
                  borderRadius: "6px",
                  position: "relative",
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <div>
                      <div style={{
                        fontSize: "10px", color: layer.accentColor,
                        letterSpacing: "3px", marginBottom: "4px",
                      }}>
                        {layer.id} — {layer.label.toUpperCase()}
                      </div>
                      <div style={{ fontSize: "15px", fontWeight: "700" }}>
                        {layer.icon} {layer.title}
                      </div>
                    </div>
                    <div style={{
                      fontSize: "11px", color: COLORS.textMuted,
                      textAlign: "right",
                    }}>
                      {layer.modules.map((m) => (
                        <div key={m.name} style={{ marginBottom: "2px" }}>
                          · {m.name}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* 矢印 */}
                {idx < layers.length - 1 && (
                  <div style={{
                    display: "flex", flexDirection: "column",
                    alignItems: "center", padding: "4px 0",
                  }}>
                    <div style={{
                      width: "1px", height: "16px",
                      background: `linear-gradient(${layer.accentColor}, ${layers[idx+1].accentColor})`,
                    }} />
                    <div style={{
                      fontSize: "10px", color: COLORS.textMuted,
                      background: COLORS.bg,
                      padding: "2px 10px",
                      border: `1px solid ${COLORS.border}`,
                      borderRadius: "3px",
                      margin: "2px 0",
                    }}>
                      {dataFlows[idx]?.label}
                    </div>
                    <div style={{
                      width: "1px", height: "16px",
                      background: `linear-gradient(${layer.accentColor}, ${layers[idx+1].accentColor})`,
                    }} />
                    <div style={{ color: layers[idx+1].accentColor, fontSize: "14px" }}>▼</div>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* 補足 */}
          <div style={{
            marginTop: "32px",
            padding: "16px",
            background: COLORS.surface,
            border: `1px solid ${COLORS.border}`,
            borderRadius: "6px",
            maxWidth: "700px",
          }}>
            <div style={{
              fontSize: "10px", color: COLORS.accent,
              letterSpacing: "3px", marginBottom: "10px",
            }}>
              CROSS-LAYER DEPENDENCIES
            </div>
            {[
              ["L0 → L1/L2", "特徴量ストアが全レイヤーに共有される"],
              ["L2 → L4", "ガウス過程の不確実性σがモンテカルロの個別σに加算"],
              ["L3 → L4", "ベイズ事後分布がモンテカルロの初期分布として使われる"],
              ["L4 → L5", "50,000回の着順分布がキャリブレーターに入力される"],
              ["L5 → L5", "実レース結果でキャリブレーターを継続的に更新（フィードバックループ）"],
            ].map(([pair, desc], i) => (
              <div key={i} style={{
                display: "flex", gap: "12px",
                marginBottom: "8px", fontSize: "12px",
              }}>
                <div style={{
                  color: COLORS.accent, minWidth: "100px",
                  fontWeight: "600",
                }}>{pair}</div>
                <div style={{ color: COLORS.textDim }}>{desc}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── 技術スタック ── */}
      {view === "tech" && (
        <div>
          <div style={{
            fontSize: "10px", color: COLORS.textMuted,
            letterSpacing: "3px", marginBottom: "20px",
          }}>
            TECHNOLOGY STACK
          </div>
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
            gap: "16px",
          }}>
            {techStack.map((cat, i) => (
              <div key={i} style={{
                padding: "16px",
                background: COLORS.surface,
                border: `1px solid ${COLORS.border}`,
                borderRadius: "6px",
              }}>
                <div style={{
                  fontSize: "10px", color: COLORS.accent,
                  letterSpacing: "3px", marginBottom: "12px",
                }}>
                  {cat.category.toUpperCase()}
                </div>
                {cat.items.map((item, j) => (
                  <div key={j} style={{
                    display: "flex", alignItems: "center", gap: "8px",
                    marginBottom: "8px", fontSize: "13px",
                  }}>
                    <div style={{
                      width: "6px", height: "6px",
                      background: COLORS.accent,
                      borderRadius: "50%", flexShrink: 0,
                    }} />
                    <span>{item}</span>
                  </div>
                ))}
              </div>
            ))}
          </div>

          {/* ディレクトリ構成 */}
          <div style={{
            marginTop: "24px",
            padding: "20px",
            background: COLORS.surface,
            border: `1px solid ${COLORS.border}`,
            borderRadius: "6px",
          }}>
            <div style={{
              fontSize: "10px", color: COLORS.accent,
              letterSpacing: "3px", marginBottom: "14px",
            }}>
              PROJECT STRUCTURE
            </div>
            <pre style={{
              fontSize: "12px", color: COLORS.textDim,
              lineHeight: "1.8", margin: 0, overflowX: "auto",
            }}>{`keiba_simulator/
├── data/
│   ├── raw/                    # JRA-VANからの生データ
│   ├── processed/              # クレンジング済み
│   └── feature_store/          # Feast管理の特徴量
│
├── models/
│   ├── layer1_ability/
│   │   ├── elo_engine.py       # ELOレーティング
│   │   └── trend_analyzer.py   # Prophet トレンド
│   ├── layer2_fitness/
│   │   ├── course_model.py     # ロジスティック回帰
│   │   ├── knn_searcher.py     # KNN類似レース
│   │   ├── pedigree_scorer.py  # 血統スコア
│   │   └── gp_imputer.py       # ガウス過程補間
│   ├── layer3_prediction/
│   │   ├── ranking_lgbm.py     # LambdaMART
│   │   ├── hazard_model.py     # CoxPH
│   │   ├── bayesian_engine.py  # PyMC
│   │   └── abs_simulator.py    # Mesa ABS
│   └── layer4_simulation/
│       ├── pace_generator.py   # 動的ペース生成
│       ├── monte_carlo.py      # 50,000回MC
│       └── ensemble.py         # アンサンブル統合
│
├── calibration/
│   ├── calibrator.py           # IsotonicRegression
│   ├── confidence.py           # 信頼区間
│   └── monitor.py              # MLflow / Evidently
│
├── output/
│   ├── probability_matrix.py
│   ├── pace_heatmap.py
│   └── dashboard/              # Streamlit UI
│
├── backtest/
│   ├── historical_test.py      # 過去G1全レース検証
│   └── metrics.py              # ブライアスコア他
│
└── main.py                     # 本番実行エントリポイント`}</pre>
          </div>
        </div>
      )}

      {/* ── 評価指標 ── */}
      {view === "kpi" && (
        <div>
          <div style={{
            fontSize: "10px", color: COLORS.textMuted,
            letterSpacing: "3px", marginBottom: "20px",
          }}>
            EVALUATION METRICS — オッズ非依存の精度指標
          </div>

          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
            gap: "16px",
            marginBottom: "28px",
          }}>
            {kpis.map((kpi, i) => (
              <div key={i} style={{
                padding: "20px",
                background: COLORS.surface,
                border: `1px solid ${kpi.color}`,
                borderRadius: "6px",
              }}>
                <div style={{
                  fontSize: "11px", color: kpi.color,
                  letterSpacing: "2px", marginBottom: "8px",
                }}>
                  TARGET METRIC
                </div>
                <div style={{ fontSize: "15px", fontWeight: "700", marginBottom: "6px" }}>
                  {kpi.label}
                </div>
                <div style={{
                  fontSize: "22px", fontWeight: "700",
                  color: kpi.color, marginBottom: "6px",
                }}>
                  {kpi.target}
                </div>
                <div style={{ fontSize: "11px", color: COLORS.textMuted }}>
                  ベースライン: {kpi.baseline}
                </div>
              </div>
            ))}
          </div>

          {/* 評価フロー */}
          <div style={{
            padding: "20px",
            background: COLORS.surface,
            border: `1px solid ${COLORS.border}`,
            borderRadius: "6px",
            marginBottom: "20px",
          }}>
            <div style={{
              fontSize: "10px", color: COLORS.accent,
              letterSpacing: "3px", marginBottom: "14px",
            }}>
              BACKTEST STRATEGY
            </div>
            {[
              { step: "01", title: "データ分割", desc: "2021-2023年G1/G2を訓練データ、2024-2025年を検証データとして完全分離" },
              { step: "02", title: "時系列順守", desc: "未来リーク防止のため、常に「その時点で入手可能だったデータのみ」でシミュレーション実行" },
              { step: "03", title: "全手法バックテスト", desc: "7手法それぞれ単独 + アンサンブルの計8パターンで検証。どの手法がどの条件で効くかを確認" },
              { step: "04", title: "キャリブレーション曲線確認", desc: "シミュ勝率10%/20%/30%/40%の馬が実際に何%勝っているかを確認。乖離があれば補正関数を更新" },
              { step: "05", title: "継続モニタリング", desc: "本番運用後も全G1の結果をMLflowに蓄積。ブライアスコアが悪化したらアラート" },
            ].map((item, i) => (
              <div key={i} style={{
                display: "flex", gap: "16px",
                marginBottom: "14px", paddingBottom: "14px",
                borderBottom: i < 4 ? `1px solid ${COLORS.border}` : "none",
              }}>
                <div style={{
                  fontSize: "22px", fontWeight: "700",
                  color: COLORS.accentDim, minWidth: "32px",
                }}>
                  {item.step}
                </div>
                <div>
                  <div style={{ fontSize: "13px", fontWeight: "600", marginBottom: "4px" }}>
                    {item.title}
                  </div>
                  <div style={{ fontSize: "12px", color: COLORS.textDim, lineHeight: "1.6" }}>
                    {item.desc}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* 限界の明示 */}
          <div style={{
            padding: "16px",
            background: "rgba(239,68,68,0.05)",
            border: `1px solid rgba(239,68,68,0.3)`,
            borderRadius: "6px",
          }}>
            <div style={{
              fontSize: "10px", color: COLORS.red,
              letterSpacing: "3px", marginBottom: "10px",
            }}>
              MODEL LIMITATIONS — モデル化不可能な要素
            </div>
            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
              gap: "8px",
            }}>
              {[
                "ゲート出遅れ（確率的）",
                "パドックでの突発気配悪化",
                "騎手のコース取りミス",
                "競走中の不利（挟まれ）",
                "天候の急変",
                "レース中の落馬影響",
              ].map((item, i) => (
                <div key={i} style={{
                  fontSize: "11px", color: COLORS.textDim,
                  display: "flex", alignItems: "center", gap: "6px",
                }}>
                  <span style={{ color: COLORS.red }}>✕</span> {item}
                </div>
              ))}
            </div>
            <div style={{
              marginTop: "12px", fontSize: "12px",
              color: COLORS.textDim, lineHeight: "1.6",
            }}>
              これらを合算すると<span style={{ color: COLORS.red, fontWeight: "700" }}> 15〜25% </span>
              の予測不可能ノイズが残る。
              モデルは「残り75〜85%を最適化する道具」と割り切ること。
            </div>
          </div>
        </div>
      )}

      {/* フッター */}
      <div style={{
        marginTop: "48px",
        paddingTop: "16px",
        borderTop: `1px solid ${COLORS.border}`,
        display: "flex", justifyContent: "space-between",
        flexWrap: "wrap", gap: "8px",
      }}>
        <div style={{ fontSize: "10px", color: COLORS.textMuted, letterSpacing: "2px" }}>
          GRADE RACE PREDICTION ENGINE — ARCHITECTURE v1.0
        </div>
        <div style={{ fontSize: "10px", color: COLORS.textMuted }}>
          7 methods × 7 layers × 50,000 trials
        </div>
      </div>
    </div>
  );
}
