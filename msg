課題リスト
SeatベースのComparatorの改善
　現在、候補人数が少ないSeatを優先するロジックにより、
　本来十分な人員がいるBlockでも、特定のSeatのみが優先的に割当される可能性がある。
　→ Block全体のバランスを考慮した優先度設計が必要。
SeatIndex優先＋Recommend制約の影響
　seatIndex優先およびpreferSeatPriorityByRecommendRangeにより、
　低IndexのSeatが固定化され、高Indexのみが変更対象になる傾向がある。
　→ 探索の多様性（バリエーション）が不足する問題がある。
Region Transit Gap制約の対象整理
　現在のロジックにおいて、
　- Fixed vs Fixed の比較は不要
　- Fixed vs Flexible はFlexible側で調整可能
　→ Fixed同士のチェックを除外することで、不要なHard違反を削減する。
Soft Continuous Region Stayの再導入
　現在OFFになっている制約について、
　Cutoff前の最終作業Region（履歴情報）を基に、
　同一Regionでの連続作業を優先するロジックを再検討する。
　→ 履歴データ（EmployeeFact）を活用した優先付け。
Shadow Variable + Incremental Scoreの検討
　現在のConstraint計算において、
　再計算コストが高い可能性がある。
　→ Shadow VariableおよびIncremental Score計算の導入により、
　パフォーマンス改善を検討する。
Weakest Fitアルゴリズムの導入検討
　Solverの探索戦略として、
　リソースが最も余っている場所ではなく、
　最も制約が厳しい箇所から割当する手法を検討する。
　→ 制約違反（特にHard）の早期回避を目的とする。
