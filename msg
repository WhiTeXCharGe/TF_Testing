1. ハード制約違反なしなら、全 assignment はいずれかの trip の区間に含まれるか

はい。FI/FO が書かれていない期間でも、作業があればその期間の trip を生成します。作業日はすべて、いずれかの trip の中に入ります。例外は、地域なしの misc_task です(3 を参照)。

2. Fixed の assignment しかない箇所も、trip の区間に含まれるか

はい。trip は最終的な assignment 全体から作ります。この中には Fixed の行と、plan 開始前の Fixed 履歴も含まれます。そのため、Fixed しかない箇所にも trip を生成します。

3. Fixed の misc_task しかない箇所も、trip の区間に含まれるか

はい。ただし、その misc_task に地域がある場合だけです。region が空、または r_other などの中立地域の場合は、地域なしの作業として扱うため、trip は生成されません。

4. plan_range の外にも trip は生成されるか

はい。trip は plan_range で切り取りません。plan 開始日の直後に作業があれば FI が start_date より前になり、plan 終了日の直前に作業があれば FO が end_date より後になります。plan 開始前の Fixed 履歴にも trip を生成します。古い履歴の trip を出力から外したい場合は、trip_history_days で日数を指定できます。

5. 移動日を確保できない場合、trip 生成結果はどうなるか

例として、r1 が 9/1、r2 が 9/2 にあるとします。
すると r1 の trip の FO は 9/2、r2 の trip の FI は 9/1 になり、重なってしまいます。
この場合、r2 の FI を 9/2 に直します。r1 の FO と r2 の FI が、同じ日になります。
同じ日に異なる地域の作業がある場合は、地域 ID が辞書順で小さい方の地域だけを採用します。

r_other を地域として扱わない理由

SU_OTHER では、同じ地域の作業が続く間に r_other の misc_task が入っていることが多くあります。r_other を地域として扱うと、その地域の滞在が分断され、trip が不要に分かれたり、移動日が余計に入ったりします。

r_other は特定の地域ではないので、「地域なし」として扱います。
