"""メイン処理"""
from typing import List, Optional
from lib_bar_extractor.excel_reader import ExcelReader
from lib_bar_extractor.bar_extractor import BarExtractor, BarExtractorConfig
from lib_bar_extractor.yaml_exporter import YAMLExporter


class ExcelToYAMLConverter:
    """ExcelファイルをYAML形式に変換するクラス"""
    
    def __init__(self, excel_path: str, sheet_name: str):
        """
        Args:
            excel_path: Excelファイルのパス
            sheet_name: シート名
        """
        self.excel_path = excel_path
        self.sheet_name = sheet_name
    
    def convert(
        self,
        data1_columns: List[str],
        data1_start_row: int,
        data1_end_row: int,
        data2_top_left: str,
        data2_bottom_right: str,
        data3_start: str,
        data3_end: str,
        output_path: str,
        config: Optional[BarExtractorConfig] = None
    ):
        """
        Excelファイルを読み込み、YAML形式で出力
        
        Args:
            data1_columns: データ1で読み込む列のリスト (例: ["A", "B", "C"])
            data1_start_row: データ1の開始行番号
            data1_end_row: データ1の終了行番号
            data2_top_left: データ2の左上セル (例: "J46")
            data2_bottom_right: データ2の右下セル (例: "AC112")
            data3_start: データ3の開始セル (例: "J2")
            data3_end: データ3の終了セル (例: "AC2")
            output_path: 出力YAMLファイルのパス
            config: Bar抽出の設定（Noneの場合はデフォルト設定を使用）
        """
        # Excelファイルを読み込む
        reader = ExcelReader(self.excel_path, self.sheet_name)
        
        try:
            # データ2の列範囲を解析
            # from openpyxl.utils import column_index_from_string
            data2_start_col = ''.join(c for c in data2_top_left if c.isalpha())
            data2_end_col = ''.join(c for c in data2_bottom_right if c.isalpha())
            
            # データ3の列範囲を解析
            data3_start_col = ''.join(c for c in data3_start if c.isalpha())
            data3_end_col = ''.join(c for c in data3_end if c.isalpha())
            
            # 列範囲の検証
            if data3_start_col != data2_start_col or data3_end_col != data2_end_col:
                raise ValueError(
                    f"データ3の列範囲がデータ2と一致しません。"
                    f"データ2: {data2_start_col}~{data2_end_col}, "
                    f"データ3: {data3_start_col}~{data3_end_col}"
                )
            
            # データ1を読み込む
            print("データ1を読み込み中...")
            data1 = reader.read_data1(data1_columns, data1_start_row, data1_end_row)
            print(f"データ1: {len(data1)}行読み込み完了")
            
            # データ2を読み込む
            print("データ2を読み込み中...")
            data2 = reader.read_data2(data2_top_left, data2_bottom_right)
            print(f"データ2: {len(data2)}セル読み込み完了")
            
            # データ3を読み込む
            print("データ3（日付情報）を読み込み中...")
            data3 = reader.read_data3(data3_start, data3_end)
            print(f"データ3: {len(data3)}セル読み込み完了")
            
            # Barを抽出
            print("Barを抽出中...")
            extractor = BarExtractor(config=config)
            extractor.set_date_data(data3)
            bars = extractor.extract_bars(data2)
            print(f"抽出されたBar: {len(bars)}個")
            
            # 行ごとにグループ化してデータ1と合成
            print("データを合成中...")
            row_data_list = extractor.group_bars_by_row(bars, data1)
            print(f"合成完了: {len(row_data_list)}行")
            
            # YAMLファイルに出力
            print(f"YAMLファイルに出力中: {output_path}")
            exporter = YAMLExporter()
            exporter.export_to_yaml(row_data_list, output_path)
            print("完了")
            
        finally:
            reader.close()


if __name__ == "__main__":
    # 使用例
    # 実際の値は適宜変更してください
    converter = ExcelToYAMLConverter(
        excel_path="sample.xlsm",
        sheet_name="Sheet1"
    )
    
    converter.convert(
        data1_columns=["A", "B", "C"],  # データ1で読み込む列
        data1_start_row=46,              # データ1の開始行
        data1_end_row=112,               # データ1の終了行
        data2_top_left="J46",            # データ2の左上
        data2_bottom_right="AC112",      # データ2の右下
        data3_start="J1",               # データ3（日付情報）の開始
        data3_end="AC1",                # データ3（日付情報）の終了
        output_path="output.yaml"        # 出力ファイル名
    )
