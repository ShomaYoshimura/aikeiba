# Prediction article (step 10)

Write the article in Japanese to `<dir>/article.md` (Markdown) from `<dir>/analysis.json`
and `<dir>/card.json`. Every number in it (laps, times, margins, odds, probabilities,
stakes) must come from those files or from a source listed in the card. If a value is not
there, write 「不明」; never estimate it.

## Principles

- **Conclusion first.** The marks and the core idea come right after the lead; the
  analysis follows as the evidence.
- **Numbers inside the prose**: 「前半46.0-後半46.6の平均ペース」, 「0.2秒差の4着」.
- **First person, assertive**: 「私の結論」「私が見るかぎり」. No hedging in every paragraph;
  uncertainty goes into the risk-disclosure section. Assertive does not mean invented: a
  claim without data behind it is left out.
- **A cause for every poor recent run, from the data only**: `trouble` (出遅れ・不利),
  going, distance, class, pace, layoff. Use 「敗因明確、参考外でいい」 only when the card
  records the cause. Without one, write 「敗因ははっきりしない」.
- **Concede, then answer**: 「〜への懸念があるのは承知している」, then the data that
  answers it.
- **Never drop a runner for one reason**: list at least two independent reasons
  (「ひとつ」「ふたつ」) and combine them. The jockey alone, or a single risk item
  alone, is never a reason.
- Odds are those in the card (state the stage: 前夜想定 / 当日). Quinella and trio odds are
  estimates; say so.

## Mapping from analysis.json

| Article part | Source |
| --- | --- |
| Marks | `marks` (◎○▲△☆×) |
| Market's blind spot, odds gaps | `market.ladder` (`gap_before`, `ratio_to_previous`, `cluster`), `market.runners` (`fundamental_rank` vs `popularity`, `ev`, `ev_lo`, `r9_overbet`) |
| Track and pace | `race` (going, cushion, moisture), `scenarios` (probability per A–E), card `same_day_races`, `course_trends`, `course_draw_stats` |
| Last three editions | `past_editions` |
| Each horse | card `past_runs` (laps are not in the card: use `time`, `last3f`, `margin`, `early_position`), `runners[].reasons` (factor names: translate into plain Japanese), `risk.notes` |
| Fancied runners left out | runners with `×`, or popular runners without a mark: their `risk.notes` (non-zero items), `ev`, `r9_overbet`, `fundamental_rank` |
| Tickets | `tickets` (label, numbers, stake, odds, `odds_estimated`), `ticket_summary` |
| Returns A/B/C | `return_scenarios` |
| Risk disclosure | `ticket_summary.p_total_loss`, unmarked runners, `caveats` |

## Template (keep this order)

```
# 【{レース名}{年}】本命は{N}番人気{馬名}｜{本命の核心をひと言で}

{年}/{月}/{日}（{曜}）{競馬場}{R}R｜{グレード} {レース名}｜{芝/ダート}{距離}m（{回り}）｜{頭数}頭立て｜発走{時刻}｜馬場：{状態}

## リード（200〜400字）
段落1: このレースへの思考プロセス（悩んだこと、出走表を眺めた感想）
段落2: ターニングポイント（何を見直したら答えが見えたか）
段落3: **今年の{レース名}は、{骨子}だ。** これが私の結論の骨子であり、そこに{期待値の根拠}が乗っている。

## 印
◎{番} {馬名}（{N}番人気）
○{番} {馬名}（{N}番人気）
▲{番} {馬名}（{N}番人気）
△{番} {馬名}（{N}番人気） …
なお、{馬名}（{N}番人気）と{馬名}（{N}番人気）は印から外す。理由は後述する。

## この記事で分かること
- 市場が見落としているもの / 本命の核心 / 消し根拠 / 買い目の設計思想 / 予算と想定リターン（4〜5項目）

## 馬場・展開を整理する（300〜500字）
前日・当日の観察（上がり時計など実測値）→ 馬場の結論を一文で断言 → ペースの結論（逃げ候補と頭数、シナリオ確率）→ レース質の一文

## 過去3年の{レース名}（past_editions がある場合）
年・勝ち馬・人気・脚質・前後半・ペース・馬場・1〜3着人気 の表と、今年に効く共通項

## なぜ「{戦略の骨子}」なのか（400〜600字）
メンバー全体の俯瞰 → 市場の見落とし（fundamental_rank と popularity の乖離、オッズ空白）→ では誰を狙うのか

## ◎本命 {番} {馬名}（600〜800字）
1. 評価の根幹レース（1〜2戦、時計・上がり・着差・位置取り）
2. 近走の敗因（凡走ごとに一文、データにある原因のみ）
3. 今回条件との合致点（①②③）
4. オッズ空白の可視化（必須）:

    {N}人気 {馬名}  {X.X}
    {N}人気 {馬名}  {X.X}  ←クラスター①
                    ↓×{比}の空白
    {N}人気 {本命}  {X.X}  ←◎
                    ↓×{比}の空白
    {N}人気 {馬名}  {X.X}  ←▲

「この空白は、市場が{本命}を{理由}として扱っている証拠だ。…能力評価{上位}の馬を{N}番人気{X.X}倍で買えるなら、単勝を厚めに投じる根拠は十分だ。」
勝率（90%区間）と期待値（区間下限での期待値も）を添える。

## ○対抗 {番} {馬名}（300〜400字）
実績の根幹、人気との妥当性。「ただ、{妙味面の理由}では本命の{馬名}に譲った。」

## ▲三番手 {番} {馬名}（300〜400字）
独自の武器。穴寄りなら「穴本命寄りの▲だ」。「{N}番人気{X.X}倍という値段は過小評価だ」で締める。

## △の{n}頭（各100〜200字）
### {馬名}（{N}番人気）
3着に来うる理由を1〜2段落。

## 消した想定上位人気——{馬名}と{馬名}（各400〜600字）
### {馬名}を切る根拠
ひとつ、{根拠1}。{データ・数値で200〜250字}
ふたつ、{根拠2}。{200〜250字}
（みっつ、…）
このN拍子を掛け合わせると、「{まとめ}」という、能力上位馬として最も買いにくいN拍子が揃ってしまう。{N}番人気{X.X}倍ならなおさら、ここは思い切って消す判断に至った。

## 馬券構成（{予算}円）（500〜700字）
設計の骨子（ひとつ、単勝に最も厚く… ふたつ、馬連は… みっつ、3連複は…）
買い目表（tickets から。券種・組み合わせ・金額・想定オッズ、推定オッズは「推定」と明記）と小計・合計
設計思想の補足（3〜5文）

## 想定リターン早見表
パターンA/B/C（return_scenarios）: 着順例、券種別の払戻、合計、投資額に対する倍率。
期待回収率と全滅確率（ticket_summary）も一行で。

## 切る馬が絡む決着には正直に開示しておく
印なし馬の名前と人気。「{馬名}が{条件}になった場合、私の馬券はほぼ全滅する」。
「ただ、それを承知のうえで、{戦略的根拠}という判断を下した。回収率を引き上げるための、計算されたリスクテイクだ。」
加えて: ハービル法の偏り、推定オッズ、不明データの多い馬、重みが事前値なら その旨。

## おわり（200〜300字）
骨子を別の言い方で繰り返す。「当てに行く競馬ではなく、長期的に回収率を上げる設計をする——それが今年の私の{レース名}だ。」
```

## Checklist before saving

- [ ] The title gives ◎'s popularity rank.
- [ ] All marks appear right after the lead.
- [ ] The ◎ section draws the odds ladder with its gaps (from `market.ladder`).
- [ ] Every poor recent run has a cause from the data, or is marked as unexplained.
- [ ] Each dropped fancied runner has at least two independent reasons (「ひとつ」「ふたつ」).
- [ ] The ticket table has stakes and odds, with estimated odds marked as estimates.
- [ ] Returns A/B/C are all given.
- [ ] The risk-disclosure section names the outcomes that lose everything.
- [ ] 「おわり」 restates the core idea in other words.
- [ ] Symmetry audit: the reasons used to drop a runner were checked for every runner
      (the risk sheet in `analysis.json` covers R1–R8; do not add ad-hoc negatives for one
      horse only), and counter-evidence was weighed.
- [ ] Every number traces to `analysis.json`, the card or a cited source; nothing guessed.

Without odds (before betting opens), say that EV, the ladder and estimated returns are not
available yet, and write the ◎ section and the tickets without them.
