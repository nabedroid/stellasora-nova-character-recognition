# 検出器 + 分類器を組み合わせた推論とプロット (完成形)
#   画像だけから「文字 + 座標 + 信頼度」を出す。labelme は不要
#   流れ: 検出器が箱を出す → 学習時と同じ条件で切り出して分類する
#   分類の信頼度が threshold 未満なら「?」。最終判断は人間が行う前提
# 使い方: python src/predict_full.py --detector <best.pt> --classifier <model.pt> --images <画像フォルダ or 画像1枚> --out <出力先>
#   --conf 0.25      検出の信頼度しきい値
#   --threshold 0.5  分類の信頼度がこれ未満なら「?」
#   --expand 0       切り出しの追加拡大率 (検出箱のズレ対策 A案。使う場合のみ指定)
# 出力: <出力先>/<画像名>_plot.png と predictions.csv
#   プロットは predict.py と同じ「箱 + 予測文字チップ」。色は2色のみで、
#   緑 = 文字が付いた, 橙 = ? (正解との一致は見ていないので「正解」の意味ではない)
import argparse
import csv
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

from predict import PAD, TO_TENSOR, load_font
from train import CLASSES, GlyphNet, pad_to_square

IMG_EXTS = {'.png', '.jpg', '.jpeg'}


def crop_glyph(img, x1, y1, x2, y2, pad):
  # 箱 + pad 率の余白で切り出す (pad は学習時の PAD + expand)
  w, h = img.size
  pw = (x2 - x1) * pad
  ph = (y2 - y1) * pad
  box = (max(0, int(x1 - pw)), max(0, int(y1 - ph)),
         min(w, int(x2 + pw)), min(h, int(y2 + ph)))
  return img.crop(box)


def classify_boxes(model, device, img, boxes, threshold, pad):
  # boxes は (x1, y1, x2, y2, det_conf) の列。分類結果の行リストを返す
  if not boxes:
    return []
  tensors = [TO_TENSOR(pad_to_square(crop_glyph(img, x1, y1, x2, y2, pad)))
             for x1, y1, x2, y2, _ in boxes]
  with torch.no_grad():
    probs = torch.softmax(model(torch.stack(tensors).to(device)), dim=1).cpu()
  confs, idxs = probs.max(1)
  rows = []
  for i, (x1, y1, x2, y2, det_conf) in enumerate(boxes):
    pred = CLASSES[idxs[i].item()]
    conf = float(confs[i].item())
    unknown = conf < threshold
    rows.append({'idx': i, 'pred': pred, 'conf': conf,
                 'det_conf': det_conf,
                 'status': 'unknown' if unknown else 'pred',
                 'disp': '?' if unknown else pred,
                 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
  return rows


def draw_plot(img, rows):
  # predict.py と同じ体裁: 2倍拡大した元画像に色付きの箱 + 予測文字チップ
  scale = 2
  out = img.convert('RGB').resize((img.width * scale, img.height * scale),
                                  Image.NEAREST)
  d = ImageDraw.Draw(out)
  font = load_font(20)
  colors = {'pred': '#00c800', 'unknown': '#ffa500'}
  for r in rows:
    x1, y1 = int(r['x1'] * scale), int(r['y1'] * scale)
    x2, y2 = int(r['x2'] * scale), int(r['y2'] * scale)
    color = colors[r['status']]
    d.rectangle([x1, y1, x2, y2], outline=color, width=3)
    text = r['disp']
    tb = d.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    cx = min(max(0, x1), out.width - tw - 4)
    cy = y1 - th - 5
    if cy < 0:
      cy = y2 + 3
    d.rectangle([cx, cy, cx + tw + 4, cy + th + 4], fill=color)
    d.text((cx + 2, cy + 1), text, font=font, fill='black')
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--detector', required=True)
  ap.add_argument('--classifier', required=True)
  ap.add_argument('--images', required=True)
  ap.add_argument('--out', required=True)
  ap.add_argument('--conf', type=float, default=0.25)
  ap.add_argument('--threshold', type=float, default=0.5)
  ap.add_argument('--expand', type=float, default=0.0)
  args = ap.parse_args()

  src = Path(args.images)
  if src.is_file():
    targets = [src]
  else:
    targets = sorted(p for p in src.iterdir()
                     if p.suffix.lower() in IMG_EXTS)
  out_dir = Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  classifier = GlyphNet().to(device)
  classifier.load_state_dict(torch.load(args.classifier,
                                        map_location=device))
  classifier.eval()
  detector = YOLO(args.detector)
  pad = PAD + args.expand

  all_rows = []
  for image_path in targets:
    img = Image.open(image_path).convert('RGB')
    result = detector.predict(str(image_path), conf=args.conf,
                              verbose=False)[0]
    boxes = []
    if result.boxes is not None:
      for xyxy, dconf in zip(result.boxes.xyxy.tolist(),
                             result.boxes.conf.tolist()):
        boxes.append((*[float(v) for v in xyxy], float(dconf)))
    # 読み順に近い並び (上→下, 左→右) にしてから分類する
    boxes.sort(key=lambda b: (b[1], b[0]))
    rows = classify_boxes(classifier, device, img, boxes,
                          args.threshold, pad)
    for r in rows:
      r['image'] = image_path.stem
    all_rows.extend(rows)
    draw_plot(img, rows).save(out_dir / f'{image_path.stem}_plot.png')
    n_pred = sum(1 for r in rows if r['status'] == 'pred')
    print(f'{image_path.stem}: 検出 {len(rows)} 件 '
          f'(文字 {n_pred}, ? {len(rows) - n_pred})')

  with open(out_dir / 'predictions.csv', 'w', newline='',
            encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['image', 'idx', 'pred', 'conf', 'det_conf', 'status',
                'x1', 'y1', 'x2', 'y2'])
    for r in all_rows:
      w.writerow([r['image'], r['idx'], r['pred'], f"{r['conf']:.4f}",
                  f"{r['det_conf']:.4f}", r['status'],
                  int(r['x1']), int(r['y1']), int(r['x2']), int(r['y2'])])


if __name__ == '__main__':
  main()
