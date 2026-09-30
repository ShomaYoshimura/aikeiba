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

// 実装フェーズ。P3のモジュールはアブレーションでベースライン比の改善が確認できた場合のみ採用する
const PHASES = {
  P1: { label: "P1 · MVP", color: COLORS.green },
  P2: { label: "P2 · 拡張", color: COLORS.blue },
  P3: { label: "P3 · 効果検証後に採用", color: COLORS.purple },
};

const layers = [
  {
    id: "L0",
    label: "Layer 0",
    title: "データ収集・前処理層",
    color: COLORS.textMuted,
    accentColor: "#64748b",
    icon: "⛁",
    summary: "全データの取得・クレンジング・時点保証付き特徴量エンジニアリング",
    modules: [
      {
        name: "JV-Link取込ワーカー",
        tech: "Python / JV-Link（Windows COM）",
        phase: "P1",
        desc: "JRA-VAN Data LabのJV-LinkはWindows専用COMのため、取込のみWindows上で実行しparquetへ出力。以降の処理はLinux/Dockerで実行",
        inputs: ["JRA-VAN Data Lab", "TARGET frontier（CSVエクスポート）", "JRA公式"],
        outputs: ["raw_races.parquet"],
        detail: "netkeiba等のスクレイピングは利用規約を確認してから採否を判断。取得データはライセンス上リポジトリにコミットせず、DVCのリモートで管理",
      },
      {
        name: "スピード指数エンジン",
        tech: "Pandas / NumPy",
        phase: "P2",
        desc: "馬場差・斤量補正済みスピード指数を全レースに付与。馬場差はその開催日までのレースのみから推定",
        inputs: ["raw_races.parquet"],
        outputs: ["speed_index.parquet"],
      },
      {
        name: "血統ベクトル化",
        tech: "Word2Vec / 血統DB",
        phase: "P3",
        desc: "父・母父・近親の血統を数値ベクトルに変換",
        inputs: ["pedigree_db"],
        outputs: ["pedigree_vectors.npy"],
        detail: "産駒成績を使う埋め込みは、学習に使う成績を予測対象レース以前に限定して再学習する",
      },
      {
        name: "特徴量ストア",
        tech: "DuckDB（→ 必要ならFeast）",
        phase: "P1",
        desc: "全特徴量に available_at（入手可能時刻）を持たせて時系列管理。レース発走時刻より後の値はテストで検出して拒否",
        inputs: ["各種parquet"],
        outputs: ["feature_store"],
        detail: "当日情報（馬体重は発走約1時間前、最終オッズは締切後）は入手時刻を明示し、予測時点で未入手の値は使わない",
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
        phase: "P2",
        desc: "全体ELO＋条件別オフセット（コース・距離帯・馬場）の階層構造。条件別に完全分割するとデータが疎になるため共有部分を持たせる",
        inputs: ["raw_races.parquet"],
        outputs: ["elo_ratings.db"],
        detail: "K係数はレースグレード×着差で動的調整（初期値：G1=64、条件戦=16）。値はウォークフォワード検証で調整",
      },
      {
        name: "スピード指数トレンド分析",
        tech: "statsmodels",
        phase: "P2",
        desc: "直近のスピード指数の推移から成長・衰退トレンドを推定",
        inputs: ["speed_index.parquet"],
        outputs: ["trend_scores"],
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
        phase: "P2",
        desc: "同コース・同距離・同回り・同馬場での過去成績から適性スコアを算出",
        inputs: ["feature_store", "elo_ratings.db"],
        outputs: ["course_fit_scores"],
        detail: "直線長・坂の位置・ストライド適性も特徴量として投入",
      },
      {
        name: "KNN類似レース検索",
        tech: "scikit-learn NearestNeighbors",
        phase: "P2",
        desc: "今回のレース条件（距離・逃げ馬数・メンバー脚質構成）に類似した過去レースをK=15件抽出",
        inputs: ["feature_store"],
        outputs: ["similar_race_patterns"],
        detail: "検索対象は予測対象レースより前のレースに限定",
      },
      {
        name: "血統適性スコアラー",
        tech: "Random Forest",
        phase: "P3",
        desc: "父・母父の組み合わせ×コース条件での過去成績から血統適性を算出",
        inputs: ["pedigree_vectors.npy", "feature_store"],
        outputs: ["pedigree_fit_scores"],
        detail: "産駒数が少ない場合はガウス過程で補間",
      },
      {
        name: "ガウス過程補間器",
        tech: "GPyTorch",
        phase: "P3",
        desc: "データ不足馬（新馬・外国馬）の適性スコアを類似条件から補間。不確実性σも出力",
        inputs: ["course_fit_scores", "pedigree_fit_scores"],
        outputs: ["gp_imputed_scores", "uncertainty_sigma"],
        detail: "データ量が多い馬ほど信頼区間が狭くなる。uncertainty_sigma はモンテカルロの個別ノイズに使う",
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
    summary: "全特徴量を統合した着順予測モデル群。ベースライン1本から始め、効果のあるモデルだけを追加",
    modules: [
      {
        name: "ランキング学習モデル（ベースライン）",
        tech: "LightGBM LambdaMART",
        phase: "P1",
        desc: "全特徴量を統合して着順スコアを直接学習。重賞だけでは約60レース/年と少ないため、全クラスの全レースで学習し重賞に適用",
        inputs: ["L1全出力", "L2全出力", "feature_store"],
        outputs: ["ranking_scores"],
        detail: "NDCG@3を最適化。スコアはレース内softmax（温度は学習データで推定）で勝率に変換。以降の全モデルはこのベースライン比で評価",
      },
      {
        name: "ベイズ推定エンジン",
        tech: "PyMC",
        phase: "P2",
        desc: "各馬の「真の実力」を事後分布として推定。当日情報で逐次更新",
        inputs: ["ranking_scores", "speed_index.parquet"],
        outputs: ["posterior_distributions"],
        detail: "MCMCサンプリング2000回。当日馬体重・馬場発表で事後更新",
      },
      {
        name: "ハザードモデル",
        tech: "lifelines CoxPH",
        phase: "P3",
        desc: "「他馬に抜かれるリスク」を時系列で推定。失速タイミングの予測",
        inputs: ["feature_store", "elo_ratings.db"],
        outputs: ["hazard_scores"],
        detail: "脚質×ペース×残り距離の交互作用項を含む。ラップデータの粒度が足りるかを先に検証",
      },
      {
        name: "エージェントベースシミュレーター",
        tech: "Mesa（Python ABS）",
        phase: "P3",
        desc: "各馬をエージェントとして定義。位置取り→直線伸びの2段階プロセスを物理的に再現",
        inputs: ["ranking_scores", "similar_race_patterns"],
        outputs: ["abs_finish_orders"],
        detail: "実装コストが大きいため、アブレーションで改善が確認できた場合のみ採用",
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
    summary: "確率モデルに基づく50,000回シミュレーションと券種別確率の生成",
    modules: [
      {
        name: "確率モデル",
        tech: "Plackett-Luce / Harville",
        phase: "P1",
        desc: "スコアから勝率への変換方法を固定。各レースで勝率の合計を1に正規化し、馬連・馬単・三連複・三連単の確率を導出",
        inputs: ["ranking_scores"],
        outputs: ["win_probs", "exotic_probs"],
        detail: "Harville式（=Plackett-Luceの解析解）は人気馬の2・3着確率を過大評価しやすいことが知られているため、実績との乖離を検証し必要ならHenery/Stern型の補正を入れる",
      },
      {
        name: "動的ペース生成器",
        tech: "NumPy",
        phase: "P2",
        desc: "逃げ馬数・先行馬質・枠順から1000m通過ペース分布を自動算出（静的シナリオ分類を排除）",
        inputs: ["feature_store"],
        outputs: ["pace_distribution"],
        detail: "連続変数として扱い、スロー〜ハイの全範囲をカバー",
      },
      {
        name: "共通ショック生成器",
        tech: "NumPy",
        phase: "P2",
        desc: "馬場バイアス・内外バイアス・ペースショックを各試行で共通ノイズとして生成",
        inputs: ["pace_distribution"],
        outputs: ["common_shocks"],
        detail: "相関行列で各ショックの相関を管理",
      },
      {
        name: "モンテカルロエンジン",
        tech: "NumPy / Numba（JIT高速化）",
        phase: "P2",
        desc: "Plackett-Luce（Gumbelノイズ）を基本に、個別σ＋共通ショックを加えて50,000回試行。着順全分布を生成",
        inputs: ["win_probs", "posterior_distributions", "uncertainty_sigma", "common_shocks"],
        outputs: ["finish_distributions"],
        detail: "共通ショックなしの場合はPlackett-Luceの解析解と一致することをテストで確認",
      },
      {
        name: "アンサンブル統合器",
        tech: "Stacking（ロジスティック回帰）",
        phase: "P2",
        desc: "採用されたモデルの出力を統合。重みはウォークフォワード検証のout-of-fold予測で学習",
        inputs: ["ranking_scores", "posterior_distributions", "hazard_scores", "abs_finish_orders"],
        outputs: ["ensemble_win_probs"],
        detail: "過学習を避けるためメタモデルは低容量（正則化付きロジスティック回帰）にする。固定の重みは使わない",
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
    summary: "予測確率の現実合わせと、市場（オッズ）を基準とした継続的精度検証",
    modules: [
      {
        name: "市場ベンチマーク",
        tech: "Python",
        phase: "P1",
        desc: "オッズの逆数をレース内で正規化（控除率を除去）した市場確率と、ログロス・ブライアスコア・回収率を比較",
        inputs: ["ensemble_win_probs", "odds_at_prediction_time", "actual_results"],
        outputs: ["market_comparison_report"],
        detail: "比較に使うオッズは予測時点で入手可能なもの。最終オッズを使う場合は「締切直前予測」として区別する",
      },
      {
        name: "キャリブレーター",
        tech: "IsotonicRegression / Platt Scaling",
        phase: "P2",
        desc: "全レースのout-of-fold予測vs実績から補正関数を学習。シミュ30%=実際の30%を保証",
        inputs: ["ensemble_win_probs", "historical_actuals"],
        outputs: ["calibrated_probs"],
        detail: "補正後にレース内で再正規化。重賞だけで学習すると標本不足で過学習するため全レースを使う",
      },
      {
        name: "信頼区間生成器",
        tech: "Bootstrap / GPyTorch",
        phase: "P3",
        desc: "各馬の勝率に95%信頼区間を付与。「確信を持てる予測」と「読みにくい馬」を区別",
        inputs: ["finish_distributions", "uncertainty_sigma"],
        outputs: ["confidence_intervals"],
      },
      {
        name: "継続評価モニター",
        tech: "MLflow / Evidently AI",
        phase: "P2",
        desc: "ログロス・ブライアスコア・回収率を市場比でトラッキング。モデルドリフトを検出",
        inputs: ["calibrated_probs", "actual_results", "market_comparison_report"],
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
    summary: "分析結果の多形式出力・期待値判断とインタラクティブ可視化",
    modules: [
      {
        name: "確率マトリクス出力",
        tech: "Pandas / Rich",
        phase: "P1",
        desc: "全馬の1〜3着確率・複勝率・決着パターン頻度をターミナル/CSV出力",
        inputs: ["calibrated_probs", "exotic_probs"],
        outputs: ["probability_matrix.csv"],
      },
      {
        name: "期待値・購入判断",
        tech: "Python",
        phase: "P2",
        desc: "期待値（確率×オッズ）が閾値を超える買い目を抽出し、分数ケリーで賭け金比率を算出",
        inputs: ["calibrated_probs", "exotic_probs", "odds_at_prediction_time"],
        outputs: ["bet_recommendations.csv"],
        detail: "閾値とケリー係数はバックテストの回収率とドローダウンで決める",
      },
      {
        name: "ペース感応度マップ",
        tech: "Matplotlib / Plotly",
        phase: "P3",
        desc: "スロー〜ハイの各ペースで各馬の勝率変化をヒートマップ表示",
        inputs: ["finish_distributions"],
        outputs: ["pace_sensitivity_plot"],
      },
      {
        name: "モデル合意度レポート",
        tech: "Jinja2テンプレート",
        phase: "P3",
        desc: "採用モデル間の予測一致度・分散を馬ごとにレポート化。確信度の指標",
        inputs: ["ranking_scores", "posterior_distributions", "hazard_scores", "abs_finish_orders"],
        outputs: ["consensus_report.md"],
      },
      {
        name: "Webダッシュボード",
        tech: "Streamlit / FastAPI",
        phase: "P3",
        desc: "当日朝に実行して結果をブラウザで確認できるインタラクティブUI",
        inputs: ["全Layer出力"],
        outputs: ["Web UI"],
      },
    ],
  },
];

const dataFlows = [
  { from: "L0", to: "L1", label: "特徴量ストア" },
  { from: "L1", to: "L2", label: "実力スコア" },
  { from: "L2", to: "L3", label: "適性スコア・σ" },
  { from: "L3", to: "L4", label: "予測スコア" },
  { from: "L4", to: "L5", label: "勝率・着順分布" },
  { from: "L5", to: "L6", label: "補正済み確率" },
];

const techStack = [
  { category: "データ処理", items: ["Python 3.11", "Pandas", "DuckDB", "Polars"] },
  { category: "機械学習", items: ["LightGBM", "scikit-learn", "PyMC", "GPyTorch"] },
  { category: "シミュレーション", items: ["NumPy", "Numba (JIT)", "Mesa (ABS)", "lifelines"] },
  { category: "MLOps", items: ["MLflow", "Evidently AI", "DVC", "Feast（必要になれば）"] },
  { category: "開発基盤", items: ["uv", "ruff", "pytest", "GitHub Actions"] },
  { category: "インフラ", items: ["FastAPI", "Streamlit", "Docker", "Windows取込ワーカー"] },
  { category: "データソース", items: ["JRA-VAN Data Lab（JV-Link）", "TARGET frontier（CSV）", "JRA公式", "netkeiba（規約確認後）"] },
];

// 基準は「ランダム」ではなく市場（オッズの逆数をレース内で正規化した確率）
const kpis = [
  {
    label: "勝率ログロス",
    target: "市場未満",
    baseline: "市場確率のログロス",
    definition: "各レースの勝ち馬に付けた確率の −log の平均",
    color: COLORS.green,
  },
  {
    label: "勝率ブライアスコア",
    target: "市場未満",
    baseline: "市場確率のブライアスコア",
    definition: "各レースで Σ(予測勝率 − 勝ち0/1)² を計算し、レース平均",
    color: COLORS.blue,
  },
  {
    label: "回収率",
    target: ">100%",
    baseline: "全馬単勝均等買い（≈ 80%）",
    definition: "期待値（勝率×単勝オッズ）が閾値を超えた馬を単勝1単位ずつ購入した場合の払戻/投資",
    color: COLORS.accent,
  },
  {
    label: "本命の複勝率",
    target: "1番人気の複勝率以上",
    baseline: "1番人気の複勝率（一般に6割前後）",
    definition: "予測勝率1位の馬が3着以内に入ったレースの割合",
    color: COLORS.purple,
  },
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
            v1.1 — ARCHITECTURE BLUEPRINT
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
          LambdaMARTベースラインから始め、ELO × KNN × ベイズ推定 × ハザード × ABS × モンテカルロを効果検証しながら統合。評価基準は市場（オッズ）
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
            <div style={{ display: "flex", gap: "6px", flexWrap: "wrap", marginBottom: "12px" }}>
              {Object.entries(PHASES).map(([key, phase]) => (
                <span key={key} style={{
                  fontSize: "10px", color: phase.color,
                  border: `1px solid ${phase.color}`,
                  padding: "1px 6px", borderRadius: "3px",
                }}>{phase.label}</span>
              ))}
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
                        {mod.phase && (
                          <span style={{
                            fontSize: "10px", fontWeight: "400",
                            color: PHASES[mod.phase].color,
                            border: `1px solid ${PHASES[mod.phase].color}`,
                            padding: "0 6px", borderRadius: "3px",
                            marginLeft: "8px", whiteSpace: "nowrap",
                          }}>{PHASES[mod.phase].label}</span>
                        )}
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
              ["L0 → 全層", "特徴量ストアが全レイヤーに共有される。available_at で時点を保証"],
              ["L2 → L4", "ガウス過程の不確実性σ（uncertainty_sigma）がモンテカルロの個別σに加算"],
              ["L3 → L4", "ベイズ事後分布がモンテカルロの初期分布として使われる"],
              ["L4 → L5", "勝率と着順分布がキャリブレーター・市場ベンチマークに入力される"],
              ["L5 → L3", "市場比の評価結果でP3モデルの採否とアンサンブルを見直す（アブレーション）"],
              ["L5 → L5", "実レース結果でキャリブレーターを継続的に更新（フィードバックループ）"],
            ].map(([pair, desc], i) => (
              <div key={i} style={{
                display: "flex", gap: "12px",
                marginBottom: "8px", fontSize: "12px",
              }}>
                <div style={{
                  color: COLORS.accent, minWidth: "100px", flexShrink: 0,
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
            }}>{`aikeiba/
├── pyproject.toml              # uv / ruff / pytest 設定
├── src/aikeiba/
│   ├── schema.py               # 1行=1出走の列定義
│   ├── leakage.py              # 時点保証チェック
│   ├── features.py             # 過去レースのみを使う特徴量
│   ├── probability.py          # softmax正規化・Harville・Plackett-Luce
│   ├── metrics.py              # ログロス・ブライア・回収率
│   ├── validation.py           # ウォークフォワード分割
│   ├── baseline.py             # LightGBM LambdaMART ベースライン
│   ├── backtest.py             # 市場比バックテスト（CLI）
│   └── synthetic.py            # 実データなしで動かす合成データ
├── tests/                      # pytest
├── ingest/windows/             # JV-Link取込ワーカー（予定）
├── data/                       # gitignore・DVC管理（コミットしない）
└── docs/architecture/          # この設計図（Viteで表示）
`}</pre>
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
            EVALUATION METRICS — 市場（オッズ）を基準にした精度指標
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
                <div style={{ fontSize: "11px", color: COLORS.textDim, marginTop: "8px", lineHeight: "1.5" }}>
                  定義: {kpi.definition}
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
              { step: "01", title: "ウォークフォワード分割", desc: "年単位で「その年より前の全レースで学習 → その年を検証」を繰り返す。重賞は年約60レースしかないため、学習は全クラスの全レースで行い、評価は全レースと重賞サブセットの両方で報告" },
              { step: "02", title: "時系列順守", desc: "特徴量は available_at が発走時刻より前のもののみ。ハイパーパラメータ調整・キャリブレーション・スタッキングも学習期間内のデータだけで行う" },
              { step: "03", title: "市場との比較", desc: "全指標を市場確率（オッズの逆数をレース内正規化）と並べて報告。市場に勝てない改善は改善とみなさない" },
              { step: "04", title: "アブレーション", desc: "P1ベースラインに1手法ずつ追加し、ログロスが改善した手法だけを採用。7手法すべてを最初から作らない" },
              { step: "05", title: "キャリブレーション曲線確認", desc: "予測勝率10%/20%/30%/40%の帯で実際の勝率を確認。乖離があれば補正関数を更新" },
              { step: "06", title: "継続モニタリング", desc: "本番運用後も全レースの結果をMLflowに蓄積。市場比のログロスが悪化したらアラート" },
            ].map((item, i) => (
              <div key={i} style={{
                display: "flex", gap: "16px",
                marginBottom: "14px", paddingBottom: "14px",
                borderBottom: i < 5 ? `1px solid ${COLORS.border}` : "none",
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
          GRADE RACE PREDICTION ENGINE — ARCHITECTURE v1.1
        </div>
        <div style={{ fontSize: "10px", color: COLORS.textMuted }}>
          baseline first × market benchmark × walk-forward validation
        </div>
      </div>
    </div>
  );
}
