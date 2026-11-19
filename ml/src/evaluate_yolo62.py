# 62クラス検出器の評価 (実験用): 位置と文字の両方が当たって初めて正解とする
#   あわせて「位置だけ」の再現率と、位置が当たった中での文字の正答率も出す。
#   完全一致率が、既存の2段パイプライン (検出 + 分類) と val 204件・同条件で
#   比べられる物差しになる
# 使い方: python src/evaluate_yolo62.py --weights <best.pt> --images <labelmeフォルダ>
#   --conf 0.25  検出の信頼度しきい値
#   --all        val だけでなく全ファイルで測る
import argparse
import json
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

from build_dataset import VAL_FILES
from convert_labelme import CLASSES
from evaluate_detector import iou, size_band


def load_gt62(json_path):
  # labelme の json から (ラベル, 箱) の列と画像名を返す
  data = json.loads(json_path.read_text(encoding='utf-8'))
  gt = []
  for s in data['shapes']:
    if s['shape_type'] != 'rectangle':
      continue
    (x1, y1), (x2, y2) = s['points']
    gt.append((s['label'],
               (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))))
  return gt, data['imagePath']


def match62(gt, preds, iou_thr=0.5):
  # gt は (label, box) の列、preds は (class_id, box, conf) の列
  # 正解箱ごとに最も IoU の高い未使用の検出箱を探し、位置が当たれば
  # その検出箱は使い切る。その上でクラスが一致するかを見る
  # 戻り値: (位置ヒット集合, 完全一致集合, 混同ペア列, 使われた検出箱数)
  used = set()
  pos_hits = set()
  exact_hits = set()
  confusions = []
  for gi, (label, g) in enumerate(gt):
    best, best_iou = -1, 0.0
    for i, (_, p, _) in enumerate(preds):
      if i in used:
        continue
      v = iou(g, p)
      if v > best_iou:
        best, best_iou = i, v
    if best_iou >= iou_thr:
      pos_hits.add(gi)
      used.add(best)
      pred_label = CLASSES[preds[best][0]]
      if pred_label == label:
        exact_hits.add(gi)
      else:
        confusions.append((label, pred_label))
  return pos_hits, exact_hits, confusions, len(used)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--weights', required=True)
  ap.add_argument('--images', required=True)
  ap.add_argument('--conf', type=float, default=0.25)
  ap.add_argument('--all', action='store_true')
  args = ap.parse_args()

  images = Path(args.images)
  model = YOLO(args.weights)
  targets = [p for p in sorted(images.glob('*.json'))
             if args.all or p.stem in VAL_FILES]

  total_gt = 0
  total_pos = 0
  total_exact = 0
  total_pred = 0
  total_used = 0
  band = {}
  per_file = []
  confusions = Counter()
  for json_path in targets:
    gt, image_name = load_gt62(json_path)
    result = model.predict(str(images / image_name), conf=args.conf,
                           verbose=False)[0]
    preds = []
    if result.boxes is not None:
      for xyxy, cls_id, conf in zip(result.boxes.xyxy.tolist(),
                                    result.boxes.cls.tolist(),
                                    result.boxes.conf.tolist()):
        preds.append((int(cls_id),
                      tuple(float(v) for v in xyxy), float(conf)))
    pos_hits, exact_hits, confs, used = match62(gt, preds)
    confusions.update(confs)
    total_gt += len(gt)
    total_pos += len(pos_hits)
    total_exact += len(exact_hits)
    total_pred += len(preds)
    total_used += used
    for gi, (_, g) in enumerate(gt):
      b = band.setdefault(size_band(g), [0, 0])
      b[1] += 1
      if gi in exact_hits:
        b[0] += 1
    per_file.append((json_path.stem, len(exact_hits), len(pos_hits),
                     len(gt), len(preds)))

  scope = '全ファイル' if args.all else 'val'
  print(f'対象: {scope} {len(targets)} ファイル')
  print(f'正解箱: {total_gt}')
  if total_gt:
    print(f'位置再現率 (IoU>=0.5): {total_pos}/{total_gt} '
          f'= {total_pos / total_gt:.4f}')
    print(f'完全一致率 (位置+文字): {total_exact}/{total_gt} '
          f'= {total_exact / total_gt:.4f}')
  if total_pos:
    print(f'位置ヒット中の文字正答率: {total_exact}/{total_pos} '
          f'= {total_exact / total_pos:.4f}')
  print(f'検出箱の総数: {total_pred} (正解に使われた {total_used}, '
        f'余分 {total_pred - total_used})')
  print('--- サイズ帯別 (完全一致) ---')
  for name in ('小 (~16px)', '中 (16-40px)', '大 (40px~)'):
    if name in band:
      hit, tot = band[name]
      print(f'{name}: {hit}/{tot} = {hit / tot:.4f}')
  print('--- ファイル別 (完全一致 / 位置ヒット / 正解) ---')
  for stem, exact, pos, tot, npred in per_file:
    print(f'{stem}: {exact}/{tot} (位置 {pos}, 検出箱 {npred})')
  if confusions:
    print('--- 混同ペア (正解 -> 予測, 位置ヒット分) ---')
    for (truth, pred), n in confusions.most_common(10):
      print(f'{truth} -> {pred}: {n}')


if __name__ == '__main__':
  main()
