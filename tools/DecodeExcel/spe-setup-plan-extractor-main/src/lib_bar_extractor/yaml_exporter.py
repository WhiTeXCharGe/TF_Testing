"""YAML出力機能"""
import yaml
from typing import List
from lib_bar_extractor.models import RowData


class YAMLExporter:
    """データをYAML形式でエクスポートするクラス"""
    
    @staticmethod
    def export_to_yaml(row_data_list: List[RowData], output_path: str):
        """
        行データのリストをYAMLファイルに出力
        
        Args:
            row_data_list: RowDataのリスト
            output_path: 出力ファイルパス
        """
        # 行番号でソート（決定的な出力を保証）
        sorted_row_data = sorted(row_data_list, key=lambda r: r.row_num)
        
        # 辞書形式に変換
        data = {
            'rows': [row_data.to_dict() for row_data in sorted_row_data]
        }
        
        # YAMLファイルに書き込み
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    
    @staticmethod
    def export_to_yaml_string(row_data_list: List[RowData]) -> str:
        """
        行データのリストをYAML文字列に変換
        
        Args:
            row_data_list: RowDataのリスト
        
        Returns:
            YAML文字列
        """
        # 行番号でソート（決定的な出力を保証）
        sorted_row_data = sorted(row_data_list, key=lambda r: r.row_num)
        
        # 辞書形式に変換
        data = {
            'rows': [row_data.to_dict() for row_data in sorted_row_data]
        }
        
        return yaml.dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False)
