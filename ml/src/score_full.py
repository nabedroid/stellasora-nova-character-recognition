# predict_full.py が出力した predictions.csv を labelme の正解と照合し、
# 2段方式 (検出 + 分類) の end-to-end 精度を測る。
# 物差しは evaluate_yolo62.py と同一 (IoU>=0.5 の貪欲マッチング、既定は val 7ファイル)。
# 分類器が "?" と棄権した箱は、位置ヒット・文字不一致として数え、
# 位置ヒット中を 正解 / ? / 誤読 に分けて出す。
# 使い方:
#   python src/score_full.py --csv /workspace/data/predict_full_eval/predictions.csv \
#       --images /workspace/data/labelme/images
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from build_dataset import VAL_FILES

IOU_TH = 0.5


# 矩形同士の IoU
def iou(a, b):
  x1 = max(a[0], b[0])
  y1 = max(a[1], b[1])
  x2 = min(a[2], b[2])
  y2 = min(a[3], b[3])
  inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
  area_a = (a[2] - a[0]) * (a[3] - a[1])
  area_b = (b[2] - b[0]) * (b[3] - b[1])
  union = area_a + area_b - inter
  return inter / union if union > 0 else 0.0


# サイズ帯 (evaluate_detector と同じ境界)
def size_band(box):
  side = max(box[2] - box[0], box[3] - box[1])
  if side <= 16:
    return '小'
  if side <= 40:
    return '中'
  return '大'


# labelme JSON から正解 (ラベル, 箱) の一覧を読む
def load_gt(json_path):
  data = json.loads(json_path.read_text(encoding='utf-8'))
  gts = []
  for s in data['shapes']:
    xs = [p[0] for p in s['points']]
    ys = [p[1] for p in s['points']]
    gts.append((s['label'], (min(xs), min(ys), max(xs), max(ys))))
  return gts


# predictions.csv を画像ごとに (箱, 予測文字) の一覧へまとめる
def load_preds(csv_path):
  by_image = {}
  with open(csv_path, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
      box = (float(row['x1']), float(row['y1']),
             float(row['x2']), float(row['y2']))
      by_image.setdefault(Path(row['image']).stem, []).append((box, row['pred']))
  return by_image


# IoU の大きい順に貪欲マッチングし、(正解idx, 予測idx) の組を返す
def match(gts, preds):
  cand = []
  for gi, (_, gb) in enumerate(gts):
    for pi, (pb, _) in enumerate(preds):
      v = iou(gb, pb)
      if v >= IOU_TH:
        cand.append((v, gi, pi))
  cand.sort(reverse=True)
  used_g = set()
  used_p = set()
  pairs = []
  for _, gi, pi in cand:
    if gi in used_g or pi in used_p:
      continue
    used_g.add(gi)
    used_p.add(pi)
    pairs.append((gi, pi))
  return pairs


def rate(n, d):
  return f'{n}/{d} = {n / d:.4f}' if d else f'{n}/{d}'


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--csv', type=Path, required=True,
                      help='predict_full.py の predictions.csv')
  parser.add_argument('--images', type=Path,
                      default=Path('/workspace/data/labelme/images'),
                      help='labelme のフォルダ (json と画像)')
  parser.add_argument('--all', action='store_true',
                      help='val だけでなく全ファイルを対象にする')
  args = parser.parse_args()

  preds_by_image = load_preds(args.csv)
  jsons = sorted(args.images.glob('*.json'))
  if not args.all:
    jsons = [p for p in jsons if p.stem in VAL_FILES]
  print(f'対象: {"全" if args.all else "val"} {len(jsons)} ファイル')

  total = 0
  pos = 0
  exact = 0
  unknown = 0
  pred_total = 0
  band_total = Counter()
  band_exact = Counter()
  confuse = Counter()
  file_lines = []
  for jp in jsons:
    gts = load_gt(jp)
    preds = preds_by_image.get(jp.stem, [])
    pairs = match(gts, preds)
    total += len(gts)
    pred_total += len(preds)
    pos += len(pairs)
    f_exact = 0
    for gi, pi in pairs:
      label, gb = gts[gi]
      pred = preds[pi][1]
      if pred == label:
        exact += 1
        f_exact += 1
        band_exact[size_band(gb)] += 1
      elif pred == '?':
        unknown += 1
      else:
        confuse[(label, pred)] += 1
    for _, gb in gts:
      band_total[size_band(gb)] += 1
    file_lines.append(
      f'{jp.stem}: {f_exact}/{len(gts)} (位置 {len(pairs)}, 検出箱 {len(preds)})')

  wrong = pos - exact - unknown
  print(f'正解箱: {total}')
  print(f'位置再現率 (IoU>=0.5): {rate(pos, total)}')
  print(f'完全一致率 (位置+文字): {rate(exact, total)}')
  print(f'位置ヒット中の文字正答率: {rate(exact, pos)}')
  print(f'位置ヒット中の内訳: 正解 {exact}, ? {unknown}, 誤読 {wrong}')
  print(f'検出箱の総数: {pred_total} (正解に使われた {pos}, 余分 {pred_total - pos})')
  print('--- サイズ帯別 (完全一致) ---')
  print(f"小 (~16px): {rate(band_exact['小'], band_total['小'])}")
  print(f"中 (16-40px): {rate(band_exact['中'], band_total['中'])}")
  print(f"大 (40px~): {rate(band_exact['大'], band_total['大'])}")
  print('--- ファイル別 (完全一致 / 位置ヒット / 正解) ---')
  for line in file_lines:
    print(line)
  print('--- 混同ペア (正解 -> 予測, 位置ヒット分, ? を除く) ---')
  for (t, p), c in confuse.most_common(10):
    print(f'{t} -> {p}: {c}')


if __name__ == '__main__':
  main()
