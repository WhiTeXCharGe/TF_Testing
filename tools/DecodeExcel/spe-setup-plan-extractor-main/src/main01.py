"""
Excelファイルから各作業員事のBarデータを抽出し、YAML形式で出力するスクリプト

## Notes

プログラム中の用語

データ1 = 各作業員のプロフィール的情報
データ2 = ガントチャート的情報
データ3 = 月日を取得するための情報
"""
from lib_bar_extractor import BarExtractorConfig, process_excel_files


def main():
    # 処理するExcelファイルのリスト
    excel_paths = [
        "C:/Users/EF070112/Documents/_data/20251210_SU_Others.xlsm",
        "C:/Users/EF070112/Documents/_data/20251211_SU_Others.xlsm",
        # "C:/Users/EF070112/Documents/_data/20251213_SU_Others.xlsm",
    ]
    
    # シート名の指定
    # 方法1: 全ファイルで同じシート名を使用
    sheet_names = "予定表_2025"
    
    # 方法2: 各ファイルごとに異なるシート名を指定
    # sheet_names = [
    #     "予定表_2025",
    #     "予定表_2025",
    #     "予定表_2025",
    # ]
    
    # Bar抽出の設定
    config = BarExtractorConfig(
        ignore_empty_text=True,
        ignore_no_equipment=True
    )

    # 複数のExcelファイルを処理
    process_excel_files(
        excel_paths=excel_paths,
        sheet_names=sheet_names,
        data1_columns=["A", "B", "C"],  # データ1で読み込む列
        data1_start_row=4,              # データ1の開始行
        data1_end_row=1000,             # データ1の終了行
        data2_top_left="J4",            # データ2の左上
        data2_bottom_right="RR1000",    # データ2の右下
        data3_start="J2",               # データ3（日付情報）の開始
        data3_end="RR2",                # データ3（日付情報）の終了
        output_dir=".",                  # 出力先ディレクトリ
        config = config
    )
    
    print("\n全ての処理が完了しました")


if __name__ == "__main__":
    main()
