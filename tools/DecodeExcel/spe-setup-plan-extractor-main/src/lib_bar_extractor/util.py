"""ユーティリティ関数"""
from pathlib import Path
from typing import List, Union, Optional
from lib_bar_extractor.converter import ExcelToYAMLConverter
from lib_bar_extractor.bar_extractor import BarExtractorConfig


def process_excel_files(
    excel_paths: List[str],
    sheet_names: Union[str, List[str]],
    data1_columns: List[str],
    data1_start_row: int,
    data1_end_row: int,
    data2_top_left: str,
    data2_bottom_right: str,
    data3_start: str,
    data3_end: str,
    output_dir: str = ".",
    config: Optional[BarExtractorConfig] = None
):
    """
    複数のExcelファイルを処理してYAML形式で出力
    
    Args:
        excel_paths: Excelファイルのパスのリスト
        sheet_names: シート名（文字列の場合は全ファイルに適用、リストの場合は各ファイルに対応）
        data1_columns: データ1で読み込む列のリスト
        data1_start_row: データ1の開始行
        data1_end_row: データ1の終了行
        data2_top_left: データ2の左上セル
        data2_bottom_right: データ2の右下セル
        data3_start: データ3（日付情報）の開始セル
        data3_end: データ3（日付情報）の終了セル
        output_dir: 出力先ディレクトリ（デフォルトはカレントディレクトリ）
        config: Bar抽出の設定（Noneの場合はデフォルト設定を使用）

    Notes:
        configがNoneの場合は，基本的に全てのフィルターが有効になり，製番関連の情報のみが抽出されます．
        TODO:SCREEN フィルタリング機能自体は現在実装中
    """
    # sheet_namesの検証
    if isinstance(sheet_names, list):
        if len(sheet_names) != len(excel_paths):
            raise ValueError(
                f"sheet_namesがリストの場合、excel_pathsと同じ要素数が必要です。"
                f"excel_paths: {len(excel_paths)}, sheet_names: {len(sheet_names)}"
            )
    else:
        # 文字列の場合、全ファイルに同じシート名を使用
        sheet_names = [sheet_names] * len(excel_paths)
    
    # 出力ディレクトリの確認
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 各Excelファイルを処理
    for excel_path, sheet_name in zip(excel_paths, sheet_names):
        print(f"\n処理中: {excel_path}")
        print(f"  シート: {sheet_name}")
        
        # Excelファイル名から出力ファイル名を生成
        excel_filename = Path(excel_path).stem  # 拡張子なしのファイル名
        output_file = output_path / f"{excel_filename}.yaml"
        
        try:
            # コンバータを初期化
            converter = ExcelToYAMLConverter(excel_path, sheet_name)
            
            # 変換を実行
            converter.convert(
                data1_columns=data1_columns,
                data1_start_row=data1_start_row,
                data1_end_row=data1_end_row,
                data2_top_left=data2_top_left,
                data2_bottom_right=data2_bottom_right,
                data3_start=data3_start,
                data3_end=data3_end,
                output_path=str(output_file),
                config=config
            )
            
            print(f"  出力完了: {output_file}")
            
        except Exception as e:
            print(f"  エラー: {e}")
            continue
