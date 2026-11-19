# 62クラスYOLO形式を1クラス検出用に変換する
#   入力: convert_labelme.py が出した yolo フォルダ
#         (各行: <クラスID 0-61> cx cy w h、classes.txt は62行)
#   出力: 各行の先頭クラスIDをすべて 0 に置き換えたものと、
#         classes.txt を1行(glyph)にしたもの
#   座標は一切変えない。検出器は「どの文字か」を捨てて
#   「文字がある場所」だけを学習する
# 使い方: python yolo_to_1class.py <入力yoloフォルダ> <出力フォルダ>
import sys
from pathlib import Path

def main():
  src = Path(sys.argv[1])
  dst = Path(sys.argv[2])
  dst.mkdir(parents=True, exist_ok=True)
  (dst / 'classes.txt').write_text('glyph\n', encoding='utf-8')
  n_files = 0
  n_lines = 0
  for txt_path in sorted(src.glob('*.txt')):
    if txt_path.name == 'classes.txt':
      continue
    out_lines = []
    for line in txt_path.read_text(encoding='utf-8').splitlines():
      if not line.strip():
        continue
      parts = line.split()
      out_lines.append('0 ' + ' '.join(parts[1:]))
    (dst / txt_path.name).write_text('\n'.join(out_lines) + '\n', encoding='utf-8')
    n_files += 1
    n_lines += len(out_lines)
  print(f"ファイル数: {n_files}, 箱の総数: {n_lines}")
  print("classes.txt: glyph (1クラス)")

if __name__ == '__main__':
  main()
