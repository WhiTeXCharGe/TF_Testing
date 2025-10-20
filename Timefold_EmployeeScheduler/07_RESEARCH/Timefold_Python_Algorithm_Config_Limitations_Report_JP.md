# Timefold（Python）: アルゴリズム／ムーブセレクタ設定は不可 ― 開発停止の公式見解＋再現証拠

## エグゼクティブサマリー

- **公式見解**：Timefold メンテナが **Python Solver（Beta）のアクティブ開発停止** を告知。JVM 版（Java/Kotlin）と同等の性能・機能に追いつくコストが大きい旨が明記されている。JVM 版に対する機能差（遅れ）が前提となる。 [GitHub](https://github.com/TimefoldAI/timefold-solver/discussions/1698?utm_source=chatgpt.com)
    
- **設定面**：**Local Search フェーズ／MoveSelector／Acceptor／Forager** の詳細な設定は **JVM 向け** ドキュメント（XML または Java の `*Config` API）で説明されているが、**Python パッケージには同等の設定入口が公開されていない**。 [docs.timefold.ai+1](https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/configuration?utm_source=chatgpt.com)
    
- **再現結果（Python 1.24.0b）**：`SolverConfig(...)` に `phase_list`（Local Search＋Tabu＋Union MoveSelectors）を渡すと **`TypeError: unexpected keyword argument 'phase_list'`** が発生。**アルゴリズム切替やムーブセレクタのカスタマイズを Python API から行えない** ことを実証する。
    
- **結論**：本プロジェクトが採用している Python 環境では、制約や終了条件の調整は可能だが、**Tabu Search／Simulated Annealing などへの切替や MoveSelector の組成・調整はできない**。これらを活用したい場合は **JVM スタック** を用いるべきである。 [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/local-search?utm_source=chatgpt.com)
    

---

## 背景とウェブ根拠

- **Python Solver の開発停止（公式）**  
    GitHub Discussions にて、Python Solver（Beta）のアクティブ開発停止がアナウンスされている。品質基準未達と、OSS の Java Solver に性能・機能で追随するための労力が大きいことが理由として挙げられている。 [GitHub](https://github.com/TimefoldAI/timefold-solver/discussions/1698?utm_source=chatgpt.com)
    
- **アルゴリズム切替は「設定で可能」だが文脈は JVM**  
    公式マニュアルは **設定を変えるだけでアルゴリズムを切替可能**、および **Benchmarker** による比較が容易と説明。これらの例や `*Config` クラスの言及は **JVM の構成** を前提にしている。 [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/configuration?utm_source=chatgpt.com)
    
- **Local Search の構成要素と設定（JVM ドキュメント）**  
    Local Search は **MoveSelector／Acceptor／Forager** の 3 要素でステップを決定し、フェーズ設定の例（構成の仕方）が示される。文脈は JVM の設定モデル。 [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/local-search?utm_source=chatgpt.com)
    
- **Move Selector カタログ（JVM ドキュメント）**  
    **Change／Swap／Pillar／Ruin & Recreate** などのムーブセレクタ実装が一覧化され、近傍の組み立て方が解説される。こちらも JVM 設定の枠組みで語られている。 [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference?utm_source=chatgpt.com)
    
- **他の参考（アルゴリズム概説・ベンチマーク・運用注意）**  
    アルゴリズム総覧（CH／メタヒューリスティクス／全探索）、ベンチマークの勘所、**SA のような time-gradient 系アルゴリズム** と終了条件の相性などが公式に記載されている。 [docs.timefold.ai+2docs.timefold.ai+2](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/overview?utm_source=chatgpt.com)
    

---

## 再現（Python 1.24.0b）：Local Search＋Tabu＋MoveSelectors の設定を試行 → 失敗

**意図**：`SolverConfig(...)` に **Local Search** を与え、**Tabu Search** と **Union MoveSelectors**（変数ごとの `Change` と `Swap`）を構成する。

`print("[EVIDENCE] Attempting Local Search + Tabu + move selectors (expected to FAIL on 1.24.0b).") cfg = SolverConfig(     solution_class=Pass1Plan,     entity_class_list=[BlockDecision],     score_director_factory_config=ScoreDirectorFactoryConfig(         constraint_provider_function=pass1_constraints     ),     # This key ('phase_list') + the LS section below is what should trigger the TypeError.     phase_list=[         {             "phaseType": "CONSTRUCTION_HEURISTIC",             "constructionHeuristicType": "FIRST_FIT"         },         {             "phaseType": "LOCAL_SEARCH",             "moveSelectorConfig": {                 "unionMoveSelectorConfig": {                     "moveSelectorConfigList": [                         { "changeMoveSelectorConfig": {                             "entityClass": BlockDecision, "variableName": "start_day"                         }},                         { "changeMoveSelectorConfig": {                             "entityClass": BlockDecision, "variableName": "heads"                         }},                         { "changeMoveSelectorConfig": {                             "entityClass": BlockDecision, "variableName": "days"                         }},                         { "swapMoveSelectorConfig": { "entityClass": BlockDecision } }                     ]                 }             },             "acceptorConfig": { "entityTabuSize": 7, "valueTabuSize": 7 },             "foragerConfig": { "acceptedCountLimit": 4 }         }     ],     termination_config=TerminationConfig(         spent_limit=Duration(seconds=10)     ) )`

**観測された例外（要旨）**：

`TypeError: SolverConfig.__init__() got an unexpected keyword argument 'phase_list'`

**解釈**：Python の `SolverConfig` には **`phase_list` 引数が存在せず**、その配下に置くはずの **Local Search／moveSelectorConfig／acceptorConfig／foragerConfig** も **公開 API 経由で指定できない**。JVM マニュアルに示される設定機能を Python で再現できないことの直接的な証拠となる。 [docs.timefold.ai+1](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/local-search?utm_source=chatgpt.com)

---

## 本プロジェクトへの含意

- **Python ではアルゴリズム切替・MoveSelector 組成は不可（現時点のパッケージ）**  
    **Tabu Search／Simulated Annealing** などへの切替や **MoveSelector の詳細設定** は、採用している Python API では実現できない。開発停止の公式見解とも整合的。 [GitHub](https://github.com/TimefoldAI/timefold-solver/discussions/1698?utm_source=chatgpt.com)
    
- **Python 側で扱えるレバー**  
    現行の **2 パス設計** を維持し、**制約設計の堅牢化**、**終了条件の調整**、および **外部オーケストレーション（例：Pass 1 の段階的な hours ティア・ループ）** を活用する。
    
- **アルゴリズム実験を実施する必要がある場合**  
    **JVM（Java/Kotlin）** へ移行し、`LocalSearchPhaseConfig`／`MoveSelectorConfig`／`AcceptorConfig`／`ForagerConfig` など **設定 API** を用いて構成・比較を行う。比較には **Benchmarker** を利用する。 [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/configuration?utm_source=chatgpt.com)
    

---

## 主要リファレンス

- Python Solver 開発停止の告知（GitHub Discussions） [GitHub](https://github.com/TimefoldAI/timefold-solver/discussions/1698?utm_source=chatgpt.com)
    
- アルゴリズム概説（CH／メタヒューリスティクス／全探索、フェーズ合成の文脈） [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/overview?utm_source=chatgpt.com)
    
- Local Search（MoveSelector／Acceptor／Forager、フェーズ設定の説明） [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/local-search?utm_source=chatgpt.com)
    
- Move Selector リファレンス（Change／Swap／Pillar／Ruin & Recreate など） [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference?utm_source=chatgpt.com)
    
- 設定ガイド（設定変更でアルゴリズム切替、Benchmarker による比較） [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/configuration?utm_source=chatgpt.com)
    
- ベンチマークと微調整（SA のような time-gradient 系の統計が時間設定に依存） [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/benchmarking-and-tweaking?utm_source=chatgpt.com)
    
- ソルバー運用（SA と終了条件の相性注意） [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/running-the-solver?utm_source=chatgpt.com)