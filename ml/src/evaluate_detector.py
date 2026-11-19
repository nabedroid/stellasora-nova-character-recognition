# 検出器の「位置の再現率」を測る (分類の正しさは見ない)
#   正解箱は labelme の json。検出箱と IoU >= 0.5 で当たれば検出成功とする
#   あわせて「正解箱の中心が検出箱の中」という緩い基準も出す (小さい箱は IoU が厳しいため)
#   余分な検出箱の数も数える (過検出は分類器側で落とす前提だが、量の把握のため)
# 使い方: python src/evaluate_detector.py --weights <best.pt> --images <labelmeフォルダ>
#   --conf 0.25  検出の信頼度しきい値 (再現率優先で低め)
#   --all        val だけでなく全ファイルで測る
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from build_dataset import VAL_FILES


def iou(a, b):
  # a, b は (x1, y1, x2, y2)。積集合 / 和集合 を返す
  ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
  ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
  inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
  area_a = (a[2] - a[0]) * (a[3] - a[1])
  area_b = (b[2] - b[0]) * (b[3] - b[1])
  union = area_a + area_b - inter
  return inter / union if union > 0 else 0.0


def match(gt_boxes, pred_boxes, iou_thr=0.5):
  # 正解箱ごとに最も IoU の高い未使用の検出箱を探す (当たればその検出箱は使い切り)
  # 戻り値: (IoUヒットした正解箱の集合, 中心ヒットした正解箱の集合, 使われた検出箱数)
  used = set()
  iou_hits = set()
  center_hits = set()
  for gi, g in enumerate(gt_boxes):
    best, best_iou = -1, 0.0
    for i, p in enumerate(pred_boxes):
      if i in used:
        continue
      v = iou(g, p)
      if v > best_iou:
        best, best_iou = i, v
    if best_iou >= iou_thr:
      iou_hits.add(gi)
      used.add(best)
    cx = (g[0] + g[2]) / 2
    cy = (g[1] + g[3]) / 2
    if any(p[0] <= cx <= p[2] and p[1] <= cy <= p[3] for p in pred_boxes):
      center_hits.add(gi)
  return iou_hits, center_hits, len(used)


def size_band(box):
  # 分類器の分析と同じ帯分け (最大辺で判定)
  s = max(box[2] - box[0], box[3] - box[1])
  if s < 16:
    return '小 (~16px)'
  if s < 40:
    return '中 (16-40px)'
  return '大 (40px~)'


def load_gt(json_path):
  data = json.loads(json_path.read_text(encoding='utf-8'))
  gt = []
  for s in data['shapes']:
    if s['shape_type'] != 'rectangle':
      continue
    (x1, y1), (x2, y2) = s['points']
    gt.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
  return gt, data['imagePath']


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
  total_iou = 0
  total_center = 0
  total_pred = 0
  total_used = 0
  band = {}
  per_file = []
  for json_path in targets:
    gt, image_name = load_gt(json_path)
    result = model.predict(str(images / image_name), conf=args.conf,
                           verbose=False)[0]
    preds = [tuple(float(v) for v in box)
             for box in result.boxes.xyxy.tolist()]
    iou_hits, center_hits, used = match(gt, preds)
    total_gt += len(gt)
    total_iou += len(iou_hits)
    total_center += len(center_hits)
    total_pred += len(preds)
    total_used += used
    for gi, g in enumerate(gt):
      b = band.setdefault(size_band(g), [0, 0])
      b[1] += 1
      if gi in iou_hits:
        b[0] += 1
    per_file.append((json_path.stem, len(iou_hits), len(gt), len(preds)))

  scope = '全ファイル' if args.all else 'val'
  print(f'対象: {scope} {len(targets)} ファイル')
  print(f'正解箱: {total_gt}')
  if total_gt:
    print(f'IoU>=0.5 再現率: {total_iou}/{total_gt} = {total_iou / total_gt:.4f}')
    print(f'中心ヒット再現率: {total_center}/{total_gt} = {total_center / total_gt:.4f}')
  print(f'検出箱の総数: {total_pred} (正解に使われた {total_used}, 余分 {total_pred - total_used})')
  print('--- サイズ帯別 (IoU>=0.5) ---')
  for name in ('小 (~16px)', '中 (16-40px)', '大 (40px~)'):
    if name in band:
      hit, tot = band[name]
      print(f'{name}: {hit}/{tot} = {hit / tot:.4f}')
  print('--- ファイル別 (IoU>=0.5) ---')
  for stem, hit, tot, npred in per_file:
    print(f'{stem}: {hit}/{tot} (検出箱 {npred})')


if __name__ == '__main__':
  main()
