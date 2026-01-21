"""
main01, main02, main03の処理を連続して実行するメインスクリプト

処理の流れ:
1. main01: ExcelファイルからBarデータを抽出してYAML形式で出力
2. main02: 2つのYAMLファイルの意味的な差分を取得
3. main03: 差分をExcelレポート形式で出力
"""
from pathlib import Path
from lib_bar_extractor import BarExtractorConfig, process_excel_files
from main02 import YAMLDiffGenerator
from main03 import SemanticDiffReporter


def step1_extract_bars_from_excel():
    """
    Step 1: ExcelファイルからBarデータを抽出してYAML形式で出力
    """
    print("=" * 80)
    print("Step 1: ExcelファイルからBarデータを抽出")
    print("=" * 80)
    
    # 処理するExcelファイルのリスト
    excel_paths = [
        "C:/Users/EF070112/Documents/_data/20251210_SU_Others.xlsm",
        "C:/Users/EF070112/Documents/_data/20251211_SU_Others.xlsm",
    ]
    
    # シート名
    sheet_names = "予定表_2025"
    
    # Bar抽出の設定
    config = BarExtractorConfig(
        ignore_empty_text=True,
        ignore_no_equipment=True
    )
    
    # process_excel_filesを使用して処理
    process_excel_files(
        excel_paths=excel_paths,
        sheet_names=sheet_names,
        data1_columns=["A", "B", "C"],
        data1_start_row=4,
        data1_end_row=1000,
        data2_top_left="J4",
        data2_bottom_right="RR1000",
        data3_start="J2",
        data3_end="RR2",
        output_dir=".",
        config=config
    )
    
    # 出力ファイルのリストを生成
    output_files = [f"{Path(path).stem}.yaml" for path in excel_paths]
    
    print("\nStep 1 完了")
    return output_files


def step2_generate_semantic_diff(yaml_files):
    """
    Step 2: 2つのYAMLファイルの意味的な差分を取得
    
    Args:
        yaml_files: YAMLファイルのリスト
    """
    print("\n" + "=" * 80)
    print("Step 2: YAMLファイルの意味的な差分を生成")
    print("=" * 80)
    
    if len(yaml_files) < 2:
        print("エラー: 差分を取得するには少なくとも2つのYAMLファイルが必要です")
        return None
    
    # 最初の2つのファイルを比較
    file1 = yaml_files[0]
    file2 = yaml_files[1]
    
    print("\n比較対象:")
    print(f"  比較元: {file1}")
    print(f"  比較先: {file2}")
    
    # ファイルの存在確認
    if not Path(file1).exists():
        print(f"エラー: ファイルが見つかりません: {file1}")
        return None
    
    if not Path(file2).exists():
        print(f"エラー: ファイルが見つかりません: {file2}")
        return None
    
    # 差分生成器を初期化
    diff_generator = YAMLDiffGenerator(file1, file2)
    
    # 意味的な差分をYAML形式で出力
    output_file = "diff_semantic.yaml"
    diff_generator.generate_semantic_diff(output_file)
    
    print(f"\nStep 2 完了: {output_file}")
    return output_file


def step3_generate_excel_report(diff_yaml_file):
    """
    Step 3: 差分をExcelレポート形式で出力
    
    Args:
        diff_yaml_file: 差分YAMLファイルのパス
    """
    print("\n" + "=" * 80)
    print("Step 3: Excelレポートを生成")
    print("=" * 80)
    
    if diff_yaml_file is None:
        print("エラー: 差分YAMLファイルが生成されていません")
        return None
    
    # ファイルの存在確認
    if not Path(diff_yaml_file).exists():
        print(f"エラー: ファイルが見つかりません: {diff_yaml_file}")
        return None
    
    # レポート生成器を初期化
    reporter = SemanticDiffReporter(diff_yaml_file)
    
    # Excelレポートを生成
    output_file = "diff_report.xlsx"
    reporter.generate_report(output_file)
    
    # サマリーを表示
    reporter.print_summary()
    
    print(f"\nStep 3 完了: {output_file}")
    return output_file


def main():
    """
    main01, main02, main03の処理を連続実行
    """
    print("処理を開始します...\n")
    
    try:
        # Step 1: ExcelファイルからBarデータを抽出
        yaml_files = step1_extract_bars_from_excel()
        
        # Step 2: YAMLファイルの意味的な差分を生成
        diff_yaml = step2_generate_semantic_diff(yaml_files)
        
        # Step 3: Excelレポートを生成
        report_file = step3_generate_excel_report(diff_yaml)
        
        # 完了メッセージ
        print("\n" + "=" * 80)
        print("全ての処理が完了しました！")
        print("=" * 80)
        print("\n生成されたファイル:")
        for yaml_file in yaml_files:
            if Path(yaml_file).exists():
                print(f"  - {yaml_file}")
        if diff_yaml and Path(diff_yaml).exists():
            print(f"  - {diff_yaml}")
        if report_file and Path(report_file).exists():
            print(f"  - {report_file}")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
