"""Barの抽出とグループ化処理"""
import re
from typing import List, Dict, Optional
from collections import defaultdict
from openpyxl.utils import column_index_from_string
from lib_bar_extractor.models import Bar, RowData
from dataclasses import dataclass

@dataclass
class BarExtractorConfig:
    ignore_empty_text: bool = True
    ignore_no_equipment: bool = True

class BarExtractor:
    """セルデータからBarを抽出するクラス"""
    
    def __init__(self, config: Optional[BarExtractorConfig] = None):
        """初期化
        
        Args:
            config: Bar抽出の設定（Noneの場合はデフォルト設定を使用）
        """
        self.data3 = None  # 日付情報 (年/月/日形式) {列名: 値}
        self.config = config if config is not None else BarExtractorConfig()
    
    def set_date_data(self, data3: Dict[str, str]):
        """
        日付情報を設定
        
        Args:
            data3: 日付情報 (年/月/日形式) {列名: 値}
        """
        self.data3 = data3
    
    def extract_bars(self, cell_data: List[dict]) -> List[Bar]:
        """
        セルデータからBarを抽出
        
        Args:
            cell_data: read_data2で取得したセルデータのリスト
        
        Returns:
            抽出されたBarのリスト
        """
        # 行ごとにセルをグループ化
        rows = defaultdict(list)
        for cell in cell_data:
            # 空白かつ塗りつぶしなしのセルは無視
            if not cell['value'] and not cell['fill_color']:
                continue
            rows[cell['row']].append(cell)
        
        # 各行でBarを抽出
        bars = []
        for row_num in sorted(rows.keys()):
            row_cells = sorted(rows[row_num], key=lambda c: column_index_from_string(c['col']))
            bars.extend(self._extract_bars_from_row(row_cells))
        
        # configに基づいてフィルタリング
        if self.config.ignore_empty_text:
            bars = [bar for bar in bars if bar.texts]
        if self.config.ignore_no_equipment:
            bars = [bar for bar in bars if self._should_include_bar(bar)]

        return bars
    
    def _should_include_bar(self, bar: Bar) -> bool:
        """
        Barをフィルタリングに含めるべきかを判定
        
        Args:
            bar: 判定対象のBar
        
        Returns:
            含めるべきならTrue
        """
        for text in bar.texts:
            # 「新規」で終わる
            if text.endswith("新規"):
                return True
            # 「PTC」が含まれる
            if "PTC" in text:
                return True
            # ハイフンと英数字が7文字以上連続
            if re.search(r'[a-zA-Z0-9-]{7,}', text):
                return True
        return False
    
    def _extract_bars_from_row(self, cells: List[dict]) -> List[Bar]:
        """
        1行のセルからBarを抽出
        
        同じ文字列が隣接している、または同じ塗りつぶし色が並んでいる場合、
        それらを1つのBarとみなす
        
        Args:
            cells: ソートされたセルのリスト
        
        Returns:
            その行のBarのリスト
        """
        if not cells:
            return []
        
        bars = []
        current_bar_cells = [cells[0]]
        
        for i in range(1, len(cells)):
            prev_cell = cells[i - 1]
            curr_cell = cells[i]
            
            # 列が連続しているかチェック
            prev_col_idx = column_index_from_string(prev_cell['col'])
            curr_col_idx = column_index_from_string(curr_cell['col'])
            
            # 連続している場合
            if curr_col_idx == prev_col_idx + 1:
                # 同じ文字列または同じ色の場合、同じBarに追加
                if self._should_merge(prev_cell, curr_cell):
                    current_bar_cells.append(curr_cell)
                else:
                    # 新しいBarの開始
                    bars.append(self._create_bar(current_bar_cells))
                    current_bar_cells = [curr_cell]
            else:
                # 列が連続していない場合、現在のBarを確定して新しいBarを開始
                bars.append(self._create_bar(current_bar_cells))
                current_bar_cells = [curr_cell]
        
        # 最後のBarを追加
        if current_bar_cells:
            bars.append(self._create_bar(current_bar_cells))
        
        return bars
    
    def _should_merge(self, cell1: dict, cell2: dict) -> bool:
        """
        2つのセルを同じBarとしてマージすべきかを判定
        
        Args:
            cell1: 1つ目のセル
            cell2: 2つ目のセル
        
        Returns:
            マージすべきならTrue
        """
        # 同じ文字列（空でない場合）
        if cell1['value'] and cell2['value'] and cell1['value'] == cell2['value']:
            return True
        
        # 同じ塗りつぶし色（Noneでない場合）
        if cell1['fill_color'] and cell2['fill_color'] and cell1['fill_color'] == cell2['fill_color']:
            return True
        
        return False
    
    def _create_bar(self, cells: List[dict]) -> Bar:
        """
        セルのリストからBarオブジェクトを生成
        
        Args:
            cells: セルのリスト
        
        Returns:
            Barオブジェクト
        """
        # ユニークな文字列を収集（空文字列を除く）
        # 決定的な出力のため、ソートする
        texts = sorted(list(set(cell['value'] for cell in cells if cell['value'])))
        
        # 色を決定（最初の非Noneの色を使用）
        color = None
        for cell in cells:
            if cell['fill_color']:
                color = cell['fill_color']
                break
        
        # 開始位置と終了位置
        start_cell = cells[0]
        end_cell = cells[-1]
        
        # 日付情報を取得
        start_date = None
        end_date = None
        if self.data3 is not None:
            start_col = start_cell['col']
            end_col = end_cell['col']
            
            # 開始日付を取得 (年/月/日形式)
            if start_col in self.data3:
                date_value = self.data3[start_col].strip()
                if date_value:
                    start_date = date_value
            
            # 終了日付を取得 (年/月/日形式)
            if end_col in self.data3:
                date_value = self.data3[end_col].strip()
                if date_value:
                    end_date = date_value
        
        return Bar(
            start_pos=(start_cell['col'], start_cell['row']),
            end_pos=(end_cell['col'], end_cell['row']),
            texts=texts,
            color=color,
            start_date=start_date,
            end_date=end_date
        )
    
    def group_bars_by_row(self, bars: List[Bar], data1: Dict[int, List[str]]) -> List[RowData]:
        """
        Barを行ごとにグループ化し、データ1と合成
        
        Args:
            bars: Barのリスト
            data1: read_data1で取得したデータ
        
        Returns:
            行ごとのデータのリスト
        """
        # 行ごとにBarをグループ化
        bars_by_row = defaultdict(list)
        for bar in bars:
            row_num = bar.start_pos[1]
            bars_by_row[row_num].append(bar)
        
        # 各行のBarを開始位置でソート（決定的な出力のため）
        for row_num in bars_by_row:
            bars_by_row[row_num].sort(key=lambda b: (column_index_from_string(b.start_pos[0]), b.start_pos[1]))
        
        # RowDataのリストを作成
        result = []
        for row_num in sorted(data1.keys()):
            row_data = RowData(
                row_num=row_num,
                data1_values=data1[row_num],
                bars=bars_by_row.get(row_num, [])
            )
            result.append(row_data)
        
        return result
