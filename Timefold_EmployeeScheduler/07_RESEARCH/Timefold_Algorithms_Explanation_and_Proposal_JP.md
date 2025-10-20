# TIMEFOLD アルゴリズム — 解説と提案（従業員スケジューリング向け）

## 概要（SUMMARY）

Timefold は複数の最適化アプローチをサポートします：

- **構築ヒューリスティクス（Construction Heuristics; CH）**
    
- **局所探索／メタヒューリスティクス（Local Search / Metaheuristics）**
    
    - 遅延受理（**Late Acceptance; LA**）
        
    - タブー検索（**Tabu Search; TS**）
        
    - 焼きなまし法（**Simulated Annealing; SA**）
        
- **全探索（Exhaustive Search）**
    

これらは **フェーズ合成**（例：`CH → Local Search`）でき、**Acceptor / Forager / Move Selector** を調整してふるまいを最適化できます。

- ドキュメントの **Benchmarker** 例では **TS / SA / LA** の設定を並べて比較可能。
    
- **Move Selector**（Change / Swap / Pillar / Ruin & Recreate / リスト・チェーン系 など）は局所探索の「近傍操作」カタログ。
    
- 全探索（Brute Force / Branch & Bound）は極小規模の検証向けで、大規模には不向き。
    

> 参考（公式ドキュメント）：
> 
> - Optimization algorithms — overview: [https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/overview](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/overview?utm_source=chatgpt.com)
>     
> - Local Search: [https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/local-search](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/local-search?utm_source=chatgpt.com)
>     
> - Move Selector reference: [https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference?utm_source=chatgpt.com)
>     
> - Construction Heuristics: [https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/construction-heuristics](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/construction-heuristics?utm_source=chatgpt.com)
>     
> - Exhaustive Search: [https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/exhaustive-search](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/exhaustive-search?utm_source=chatgpt.com)
>     
> - Benchmarking and tweaking: [https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/benchmarking-and-tweaking](https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/benchmarking-and-tweaking?utm_source=chatgpt.com)
>     
> - Running the solver: [https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/running-the-solver](https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/running-the-solver?utm_source=chatgpt.com)
>     
> - Enterprise/default move selectors: [https://docs.timefold.ai/timefold-solver/latest/enterprise-edition/enterprise-edition](https://docs.timefold.ai/timefold-solver/latest/enterprise-edition/enterprise-edition?utm_source=chatgpt.com)
>     

---

## Timefold アルゴリズムの短い説明（EXPLANATION OF TIMEFOLD ALGORITHMS）

### A) 構築ヒューリスティクス（Construction Heuristics; CH）

**目的：** まず「完全な初期解」を高速に構築（最適とは限らない）。  
**代表例：** _First Fit / First Fit Decreasing_、_Cheapest / Regret Insertion_。  
**備考：** CH は自然に終了し、その後 **メタヒューリスティクス**（局所探索）で改善します。

### B) 局所探索（Local Search; メタヒューリスティクス）

**目的：** 小さな編集＝**Move** を繰り返し、スコアを改善。  
**構成：**

- **MoveSelector**（候補手の生成）
    
- **Acceptor**（受理基準。悪化手の受理も可）
    
- **Forager**（受理済みから採用手を選ぶ）
    

**代表的バリアント：**

- **遅延受理（Late Acceptance; LA）**：`L` ステップ前のスコアより良ければ受理（エイジング基準）。設定が簡単で堅牢なデフォルト。
    
- **タブー検索（Tabu Search; TS）**：直近の手や解を **tabu** として一時禁止。**Aspiration** でベスト更新なら禁制解除。強力な反復防止。
    
- **焼きなまし法（Simulated Annealing; SA）**：高温時は悪化手も確率受理し、大域的探索→**冷却**で厳格化・微調整。
    

### C) ムーブセレクタ（Move Selectors; 近傍）

- **Change**（一変数変更）、**Swap**（2 エンティティの値入替）
    
- **Pillar**（同一値グループの変更／入替）
    
- **Ruin & Recreate**（部分解体→CH で再構築）
    
- **リスト変数・チェーン変数** 向けの専用ムーブ（k-opt、SubList、SubChain など）  
    → これらを **union（混合）** して「細かい手」と「大きな手」を併用します。
    

### D) 全探索（Exhaustive Search：Brute Force / Branch & Bound）

全て（または枝刈り付き）を試す方式。**完了すれば最適保証**ですが、**実務規模では非現実的**。検証・小規模の正当性確認用。

### E) 実務的なフェーズ合成とベンチマーク

基本パターンは **CH で可行解 → LS（LA / TS / SA）で改善**。  
**Benchmarker** で **FFD + LA / FFD + TS / FFD + SA** のように横並び比較ができます。

---

## 提案：試行する実験

**（目的：可行化速度の向上 + ソフトスコア品質の向上）**

### コンテキスト（Context）

- **プロジェクト：** 2 パス従業員スケジューリング
    
    - **Pass 1**：ブロック設計（`start_day / heads / days`）
        
    - **Pass 2**：座席（Seat）／従業員アサイン
        
- **現在の基準：** LC のベースラインとして **Late Acceptance (LA)** を利用
    
- **評価候補：** **Tabu Search (TS)** と **Simulated Annealing (SA)**
    

### やりたいこと（What I want to try）

1. **Pass 2** で **LA → TS** へ置換（必要に応じ **Pass 1** でも TS を試す）
    
2. **Pass 1** で **LA → SA** へ置換（必要に応じ **Pass 2** でも SA を試す）
    
3. **CH は固定**（First Fit / FFD など）し、**LS のアルゴリズム & Move Selector の union** のみ変えて **純粋比較**（Benchmarker で計測）
    

### なぜ（LA との差・期待効果）

**TS vs LA**

- **TS** は短期記憶（tabu リスト）で最近の手を禁制、**LA** は「過去スコアとの差」基準。
    
- **TS** は反復（サイクリング）抑制に強く、**tabu size** で探索と活用のバランスを直感制御。制約の厳しい近傍でも安定しやすい。
    

**SA vs LA**

- **SA** は温度に応じて悪化手も確率受理、**LA** は固定ウィンドウの閾値判断。
    
- **SA** は序盤の広い探索→終盤の収束を自然に実現し、険しい「稜線」を越えやすい。
    

### 本プロジェクトでの使い所（How each is useful）

**Pass 1（ブロック設計）**

- **SA**：窓制約・自動導出 hours・日別 head 上限で地形がギザギザなとき、序盤の寛容さで別配置にジャンプしやすい。
    
- **TS**：容量／位相制約の壁付近で似た配置間を往復する場合、tabu で往復を遮断して安定化。
    

**Pass 2（従業員アサイン：manager-per-block 等のハード＋スキル/残業のソフト）**

- **TS**：割当問題と相性が良く、Swap の往復を抑えて安定可行へ収束しやすい。
    
- **SA**：ソフト指標が競合して同点が多いとき、確率受理で多様な組合せを試してから冷却で絞る。
    

### ムーブセレクタの推奨（Union）

- **コア：** `Change`（単一座席の付替え）、`Swap`（2 座席の入替）
    
- **拡大型：** `Pillar`（小さなクルー塊の移動／入替）、`Ruin & Recreate`（まれに使用して深い停滞から脱出）
    
- **方針：** 高コストムーブは **低頻度**、複数セレクタは **併用** が推奨
    

### 初期パラメータ（ベンチマーカーで要チューニング）

- **Tabu Search (TS)**：`entity/value tabu size = 7–11`（目安：1 ステップ当たり候補手数の √）、`Aspiration = ON`、`acceptedCountLimit ≈ 4–8`
    
- **Simulated Annealing (SA)**：**初期温度**＝典型的悪化幅 Δ に対し **開始時 50–70%** の悪化手を受理するレベル、**冷却率** `α ≈ 0.95–0.99`（N ステップごと）、**停止条件**＝温度下限 or 無改善時間
    
- **CH** は **First Fit / FFD** など **固定** にし、比較を純化
    

### 評価指標（Where to measure）

- **可行化率：** 各パスで `Hard = 0` 到達割合
    
- **初回可行までの時間：** `0-hard` 到達秒数
    
- **Medium / Soft スコア：** 中央値・ベスト（例：Pass 1＝stacking / phase-gap、Pass 2＝残業・スキルバランス・マネージャ要件など）
    
- **安定性：** 乱数シード間のばらつき
    
- **Benchmarker** で **(CH + LA) vs (CH + TS) vs (CH + SA)** を自動比較
    

### 具体レシピ（Conceptual wiring）

**Pass 1 × Simulated Annealing (SA)**

- **CH：** 既存の seeded blocks
    
- **LS：** SA + ムーブ混合
    
    - 例：`start_day ±1: 60%`、`heads ±1: 25%`、`days ±1: 15%`
        
    - 各種 **min/max** と **window** を尊重
        
- **冷却率：** `α ≈ 0.97`
    
- **初期温度：** 典型的 **Soft/Medium** の悪化幅から逆算（_Running the Solver_ の注意参照）
    

**Pass 2 × Tabu Search (TS)**

- **CH：** Greedy / First-fit 割当
    
- **LS：** TS + ムーブ混合
    
    - 例：同一 `(op_id, day)` 内 **Swap: 50%**、単一座席の **Change: 30%**、短い **chain swap: 20%**
        
    - **tabu size ≈ 7–11**、**Aspiration = ON**
        

### クイックガイド（いつどれを選ぶか）

- **Tabu Search (TS)：** 振動／反復が見える、記憶で安定前進させたい
    
- **Simulated Annealing (SA)：** 初期可行化が難しい、地形がスパイキー（序盤の広探索→終盤の収束）
    

---

## 参考リンク（OFFICIAL DOCS & TRUSTED SOURCES）

- Optimization algorithms — overview（アルゴリズム／フェーズ連結／構成例）  
    [https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/overview](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/overview?utm_source=chatgpt.com)
    
- Local Search（Acceptor／Forager と LA／TS／SA の考え方）  
    [https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/local-search](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/local-search?utm_source=chatgpt.com)
    
- Move Selector reference（Change／Swap／Pillar／Ruin & Recreate／リスト・チェーン系）  
    [https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference?utm_source=chatgpt.com)
    
- Construction Heuristics（First Fit／FFD など）  
    [https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/construction-heuristics](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/construction-heuristics?utm_source=chatgpt.com)
    
- Exhaustive Search（ブルートフォース／分枝限定法と設定例）  
    [https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/exhaustive-search](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/exhaustive-search?utm_source=chatgpt.com)
    
- Benchmarking and tweaking（TS／SA／LA 構成の比較）  
    [https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/benchmarking-and-tweaking](https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/benchmarking-and-tweaking?utm_source=chatgpt.com)
    
- Running the solver（焼きなまし法など time-gradient 系と終了条件の注意）  
    [https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/running-the-solver](https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/running-the-solver?utm_source=chatgpt.com)
    
- Enterprise/default move selectors（デフォルトや nearby selection の文脈）  
    [https://docs.timefold.ai/timefold-solver/latest/enterprise-edition/enterprise-edition](https://docs.timefold.ai/timefold-solver/latest/enterprise-edition/enterprise-edition?utm_source=chatgpt.com)