"""データモデルの定義"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Bar:
    """連続するセルを表すBar"""
    start_pos: tuple[str, int]  # (列名, 行番号) 例: ("J", 46)
    end_pos: tuple[str, int]     # (列名, 行番号) 例: ("L", 46)
    texts: List[str]             # セル内の文字列のユニークなリスト
    color: Optional[str]         # セルの塗りつぶし色 (RGB形式)
    start_date: Optional[str] = None  # 開始日付 (例: "12/10")
    end_date: Optional[str] = None    # 終了日付 (例: "12/15")
    
    def to_dict(self):
        """辞書形式に変換"""
        # colorを文字列に変換（openpyxlオブジェクトの場合に対応）
        color_str = None
        if self.color is not None:
            color_str = str(self.color) if not isinstance(self.color, str) else self.color
        
        result = {
            'start': f"({self.start_pos[0]}, {self.start_pos[1]})",
            'end': f"({self.end_pos[0]}, {self.end_pos[1]})",
            'texts': self.texts,
            'color': color_str
        }
        
        # 日付情報を追加
        if self.start_date is not None:
            result['start_date'] = self.start_date
        if self.end_date is not None:
            result['end_date'] = self.end_date
        
        return result


@dataclass
class RowData:
    """行ごとのデータ"""
    row_num: int
    data1_values: List[str]  # データ1の値のリスト
    bars: List[Bar]          # その行に存在するBarのリスト（ソート済み）
    
    def to_dict(self):
        """
        辞書形式に変換
        
        Note: Python 3.7+では辞書は挿入順序を保持するため、
        キーの順序とbarsの順序は常に一定です。
        """
        return {
            'row': self.row_num,
            'data1': self.data1_values,
            'bars': [bar.to_dict() for bar in self.bars]
        }
