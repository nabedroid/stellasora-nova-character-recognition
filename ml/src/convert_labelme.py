# labelme形式のアノテーション(矩形 + 1文字ラベル)を2形式に変換する
#   1) YOLO検出形式: yolo/<画像名>.txt (class cx cy w h, 0-1に正規化)
#   2) 分類器用crop: crops/<upper_A等のフォルダ>/<画像名>_<連番>.png
# あわせて文字ごとの件数を集計して表示する
# 使い方: python convert_labelme.py <入力フォルダ> <出力フォルダ> [pad率=0.1]
#   入力フォルダには .json と対応する画像が同居している前提
import json
import sys
from pathlib import Path
from PIL import Image

# クラスIDは A-Z a-z 0-9 の固定順にする(どのファイルでも同じIDになるように)
CLASSES = [chr(c) for c in range(ord('A'), ord('Z') + 1)] + \
          [chr(c) for c in range(ord('a'), ord('z') + 1)] + \
          [chr(c) for c in range(ord('0'), ord('9') + 1)]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}


def label_dirname(label):
  # Windows はフォルダ名の大文字小文字を区別しないため、
  # A と a が同じフォルダに混ざらない名前を使う (upper_A / lower_a / digit_5)
  if label.isupper():
    return f'upper_{label}'
  if label.islower():
    return f'lower_{label}'
  return f'digit_{label}'


def load_items(json_path):
  # labelmeのjsonから矩形シェイプだけを (ラベル, x1, y1, x2, y2) で取り出す
  data = json.loads(json_path.read_text(encoding='utf-8'))
  items = []
  for s in data['shapes']:
    if s['shape_type'] != 'rectangle':
      continue
    (x1, y1), (x2, y2) = s['points']
    items.append({
      'label': s['label'],
      'x1': min(x1, x2), 'y1': min(y1, y2),
      'x2': max(x1, x2), 'y2': max(y1, y2),
    })
  return items, data['imageWidth'], data['imageHeight'], data['imagePath']


def main():
  src = Path(sys.argv[1])
  dst = Path(sys.argv[2])
  pad = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
  yolo_dir = dst / 'yolo'
  crop_dir = dst / 'crops'
  yolo_dir.mkdir(parents=True, exist_ok=True)
  crop_dir.mkdir(parents=True, exist_ok=True)
  (yolo_dir / 'classes.txt').write_text('\n'.join(CLASSES) + '\n', encoding='utf-8')

  counts = {}
  unknown = set()
  n_files = 0
  for json_path in sorted(src.glob('*.json')):
    items, w, h, image_name = load_items(json_path)
    img = Image.open(src / image_name)
    n_files += 1
    lines = []
    for i, it in enumerate(items):
      label = it['label']
      counts[label] = counts.get(label, 0) + 1
      # crop保存(周囲に pad 分の余白を付ける)
      pw = (it['x2'] - it['x1']) * pad
      ph = (it['y2'] - it['y1']) * pad
      box = (max(0, int(it['x1'] - pw)), max(0, int(it['y1'] - ph)),
             min(w, int(it['x2'] + pw)), min(h, int(it['y2'] + ph)))
      label_dir = crop_dir / label_dirname(label)
      label_dir.mkdir(exist_ok=True)
      img.crop(box).save(label_dir / f"{json_path.stem}_{i:02d}.png")
      # YOLO行(クラスIDが固定表にある場合のみ)
      if label in CLASS_ID:
        cx = (it['x1'] + it['x2']) / 2 / w
        cy = (it['y1'] + it['y2']) / 2 / h
        bw = (it['x2'] - it['x1']) / w
        bh = (it['y2'] - it['y1']) / h
        lines.append(f"{CLASS_ID[label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
      else:
        unknown.add(label)
    (yolo_dir / f"{json_path.stem}.txt").write_text('\n'.join(lines) + '\n', encoding='utf-8')

  total = sum(counts.values())
  print(f"ファイル数: {n_files}, シェイプ総数: {total}")
  print("文字ごとの件数:")
  for c in CLASSES:
    if c in counts:
      print(f"  {c}: {counts[c]}")
  missing = [c for c in CLASSES if c not in counts]
  print(f"データに無い文字({len(missing)}種): {' '.join(missing)}")
  if unknown:
    print(f"警告: A-Za-z0-9以外のラベル: {sorted(unknown)}")


if __name__ == '__main__':
  main()
