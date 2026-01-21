"""
2つのYAMLファイルから意味的な差分を取得して出力
"""
import yaml
from pathlib import Path
from typing import List, Dict, Any, Tuple


class YAMLDiffGenerator:
    """YAMLファイルの意味的な差分を生成するクラス"""
    
    def __init__(self, file1_path: str, file2_path: str):
        """
        Args:
            file1_path: 比較元のYAMLファイルパス
            file2_path: 比較先のYAMLファイルパス
        """
        self.file1_path = file1_path
        self.file2_path = file2_path
    
    def generate_semantic_diff(self, output_path: str):
        """
        意味を理解した構造化差分をYAML形式で生成
        
        hello.pyで出力されたYAML形式のデータを比較し、
        同じdata1を持つ行のBar単位で差分を取得
        
        Args:
            output_path: 出力ファイルパス
        """
        # YAMLファイルを読み込み（openpyxlオブジェクトが含まれている場合に対応）
        with open(self.file1_path, 'r', encoding='utf-8') as f:
            data1 = yaml.load(f, Loader=yaml.UnsafeLoader)
        
        with open(self.file2_path, 'r', encoding='utf-8') as f:
            data2 = yaml.load(f, Loader=yaml.UnsafeLoader)
        
        # 意味的な差分を計算
        semantic_diff = self._calculate_semantic_diff(data1, data2)
        
        # YAML形式で出力
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(semantic_diff, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        
        print(f"意味的な差分を {output_path} に出力しました")
    
    def _calculate_semantic_diff(self, data1: Dict, data2: Dict) -> Dict[str, Any]:
        """
        意味的な差分を計算
        
        Args:
            data1: 比較元のYAMLデータ
            data2: 比較先のYAMLデータ
        
        Returns:
            意味的な差分情報
        """
        rows1 = data1.get('rows', [])
        rows2 = data2.get('rows', [])
        
        # data1をキーとした辞書を作成（data1の値をタプルに変換してキーとする）
        rows1_dict = {tuple(row['data1']): row for row in rows1}
        rows2_dict = {tuple(row['data1']): row for row in rows2}
        
        changes = []
        
        # 全てのdata1キーを取得（ソートして決定的な順序を保証）
        all_data1_keys = sorted(set(rows1_dict.keys()) | set(rows2_dict.keys()))
        
        for data1_key in all_data1_keys:
            row1 = rows1_dict.get(data1_key)
            row2 = rows2_dict.get(data1_key)
            
            if row1 is None and row2 is not None:
                # data2にのみ存在（行が追加された）
                changes.extend(self._create_added_row_changes(list(data1_key), row2))
            elif row2 is None and row1 is not None:
                # data1にのみ存在（行が削除された）
                changes.extend(self._create_deleted_row_changes(list(data1_key), row1))
            else:
                # 両方に存在（Barを比較）
                assert row1 is not None and row2 is not None
                changes.extend(self._compare_bars(list(data1_key), row1.get('bars', []), row2.get('bars', [])))
        
        return {
            'metadata': {
                'file1': self.file1_path,
                'file2': self.file2_path,
                'total_changes': len(changes)
            },
            'changes': changes
        }
    
    def _create_added_row_changes(self, data1: List[str], row: Dict) -> List[Dict[str, Any]]:
        """
        追加された行の変更を作成
        
        Args:
            data1: data1の値
            row: 追加された行
        
        Returns:
            変更情報のリスト
        """
        changes = []
        for bar in row.get('bars', []):
            after_info = {
                'start': bar.get('start'),
                'end': bar.get('end')
            }
            # start_dateとend_dateが存在する場合は追加
            if bar.get('start_date') is not None:
                after_info['start_date'] = bar.get('start_date')
            if bar.get('end_date') is not None:
                after_info['end_date'] = bar.get('end_date')
            
            changes.append({
                'data1': data1,
                'texts': bar.get('texts', []),
                'color': bar.get('color'),
                'before': 'not_exists',
                'after': after_info
            })
        return changes
    
    def _create_deleted_row_changes(self, data1: List[str], row: Dict) -> List[Dict[str, Any]]:
        """
        削除された行の変更を作成
        
        Args:
            data1: data1の値
            row: 削除された行
        
        Returns:
            変更情報のリスト
        """
        changes = []
        for bar in row.get('bars', []):
            before_info = {
                'start': bar.get('start'),
                'end': bar.get('end')
            }
            # start_dateとend_dateが存在する場合は追加
            if bar.get('start_date') is not None:
                before_info['start_date'] = bar.get('start_date')
            if bar.get('end_date') is not None:
                before_info['end_date'] = bar.get('end_date')
            
            changes.append({
                'data1': data1,
                'texts': bar.get('texts', []),
                'color': bar.get('color'),
                'before': before_info,
                'after': 'deleted'
            })
        return changes
    
    def _compare_bars(self, data1: List[str], bars1: List[Dict], bars2: List[Dict]) -> List[Dict[str, Any]]:
        """
        同じdata1を持つ行のBarを比較
        
        Args:
            data1: data1の値
            bars1: 比較元のBarリスト
            bars2: 比較先のBarリスト
        
        Returns:
            変更情報のリスト
        """
        changes = []
        
        # Barの識別子を作成（texts + color）
        def bar_key(bar: Dict) -> Tuple:
            return (tuple(sorted(bar.get('texts', []))), bar.get('color'))
        
        # Barを辞書化
        bars1_dict = {bar_key(bar): bar for bar in bars1}
        bars2_dict = {bar_key(bar): bar for bar in bars2}
        
        # 全てのBarキーを取得（ソートして決定的な順序を保証）
        # Noneとstrを比較できるようにカスタムソートキーを使用
        def sort_key(key):
            texts_tuple, color = key
            # Noneは空文字列として扱う
            return (texts_tuple, color if color is not None else "")
        
        all_bar_keys = sorted(set(bars1_dict.keys()) | set(bars2_dict.keys()), key=sort_key)
        
        for key in all_bar_keys:
            bar1 = bars1_dict.get(key)
            bar2 = bars2_dict.get(key)
            
            texts = list(key[0])
            color = key[1]
            
            if bar1 is None and bar2 is not None:
                # bar2にのみ存在（Barが追加された）
                after_info = {
                    'start': bar2.get('start'),
                    'end': bar2.get('end')
                }
                # start_dateとend_dateが存在する場合は追加
                if bar2.get('start_date') is not None:
                    after_info['start_date'] = bar2.get('start_date')
                if bar2.get('end_date') is not None:
                    after_info['end_date'] = bar2.get('end_date')
                
                changes.append({
                    'data1': data1,
                    'texts': texts,
                    'color': color,
                    'before': 'not_exists',
                    'after': after_info
                })
            elif bar2 is None and bar1 is not None:
                # bar1にのみ存在（Barが削除された）
                before_info = {
                    'start': bar1.get('start'),
                    'end': bar1.get('end')
                }
                # start_dateとend_dateが存在する場合は追加
                if bar1.get('start_date') is not None:
                    before_info['start_date'] = bar1.get('start_date')
                if bar1.get('end_date') is not None:
                    before_info['end_date'] = bar1.get('end_date')
                
                changes.append({
                    'data1': data1,
                    'texts': texts,
                    'color': color,
                    'before': before_info,
                    'after': 'deleted'
                })
            else:
                assert bar1 is not None and bar2 is not None
                # 両方に存在（start/end/日付を比較）
                has_change = (
                    bar1.get('start') != bar2.get('start') or
                    bar1.get('end') != bar2.get('end') or
                    bar1.get('start_date') != bar2.get('start_date') or
                    bar1.get('end_date') != bar2.get('end_date')
                )
                
                if has_change:
                    # 変更がある場合
                    before_info = {
                        'start': bar1.get('start'),
                        'end': bar1.get('end')
                    }
                    if bar1.get('start_date') is not None:
                        before_info['start_date'] = bar1.get('start_date')
                    if bar1.get('end_date') is not None:
                        before_info['end_date'] = bar1.get('end_date')
                    
                    after_info = {
                        'start': bar2.get('start'),
                        'end': bar2.get('end')
                    }
                    if bar2.get('start_date') is not None:
                        after_info['start_date'] = bar2.get('start_date')
                    if bar2.get('end_date') is not None:
                        after_info['end_date'] = bar2.get('end_date')
                    
                    changes.append({
                        'data1': data1,
                        'texts': texts,
                        'color': color,
                        'before': before_info,
                        'after': after_info
                    })
                # 変更がない場合は差分なし（何も追加しない）
        
        return changes


def main():
    """
    使用例：2つのYAMLファイルの意味的な差分を取得
    """
    # 比較するYAMLファイルのパス
    file1 = "20251210_SU_Others.yaml"  # 比較元
    file2 = "20251211_SU_Others.yaml"  # 比較先
    
    # ファイルの存在確認
    if not Path(file1).exists():
        print(f"エラー: ファイルが見つかりません: {file1}")
        return
    
    if not Path(file2).exists():
        print(f"エラー: ファイルが見つかりません: {file2}")
        return
    
    # 差分生成器を初期化
    diff_generator = YAMLDiffGenerator(file1, file2)
    
    # 意味的な差分をYAML形式で出力
    diff_generator.generate_semantic_diff("diff_semantic.yaml")
    
    print("\n差分の生成が完了しました！")
    print("  - diff_semantic.yaml: Bar単位の意味的な差分")


if __name__ == "__main__":
    main()
