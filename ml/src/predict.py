# 学習済み分類器で推論し、結果を元画像にプロットする (評価用)
# 使い方: python src/predict.py --model <model.pt> --json <labelmeのjsonまたはフォルダ> --out <出力先>
#   画像は json 内の imagePath から引く。フォルダ指定なら中の全 json を処理する
#   --threshold 0.5  信頼度がこれ未満の予測は「?」として出す
# 出力: <出力先>/<画像名>_plot.png と predictions.csv
#   プロットは「元画像に色付きの箱+予測文字」。信頼度はプロットに出さず csv のみに残す
#   色: 緑=正解, 赤=不一致, 橙=信頼度不足 (?)
import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from train import CLASSES, GlyphNet, pad_to_square

# 推論時の前処理 (train.py の GlyphDataset と揃える)
TO_TENSOR = transforms.Compose([
  transforms.Resize((48, 48)),
  transforms.ToTensor(),
  transforms.Normalize([0.5] * 3, [0.5] * 3),
])
PAD = 0.1  # crop の余白率 (convert_labelme.py と揃える)


def load_font(size):
  # ラベル描画用フォント。環境に無ければ Pillow のデフォルトで代替する
  for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
    if Path(p).exists():
      return ImageFont.truetype(p, size)
  try:
    return ImageFont.load_default(size=size)
  except TypeError:
    return ImageFont.load_default()


def crop_glyph(img, x1, y1, x2, y2):
  # 学習時と同じ余白付きで切り出す
  w, h = img.size
  pw = (x2 - x1) * PAD
  ph = (y2 - y1) * PAD
  box = (max(0, int(x1 - pw)), max(0, int(y1 - ph)),
         min(w, int(x2 + pw)), min(h, int(y2 + ph)))
  return img.crop(box)


def predict_image(model, device, img, shapes, threshold):
  # 1画像分の推論。戻り値は予測行のリスト (座標, 正解, 予測, 信頼度, 状態)
  if not shapes:
    return []
  tensors = []
  boxes = []
  for s in shapes:
    (ax, ay), (bx, by) = s['points']
    x1, x2 = min(ax, bx), max(ax, bx)
    y1, y2 = min(ay, by), max(ay, by)
    tensors.append(TO_TENSOR(pad_to_square(crop_glyph(img, x1, y1, x2, y2))))
    boxes.append((x1, y1, x2, y2, s['label']))
  with torch.no_grad():
    probs = torch.softmax(model(torch.stack(tensors).to(device)), dim=1).cpu()
  confs, idxs = probs.max(1)
  rows = []
  for i, (x1, y1, x2, y2, truth) in enumerate(boxes):
    pred = CLASSES[idxs[i].item()]
    conf = float(confs[i].item())
    if conf < threshold:
      status, disp = 'unknown', '?'
    elif pred == truth:
      status, disp = 'ok', pred
    else:
      status, disp = 'ng', pred
    rows.append({'idx': i, 'truth': truth, 'pred': pred, 'conf': conf,
                 'status': status, 'disp': disp,
                 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
  return rows


def draw_plot(img, rows):
  # 元画像 (2倍拡大) に色付きの箱を描き、箱の上に予測文字 (1文字) のチップを置く
  scale = 2
  out = img.convert('RGB').resize((img.width * scale, img.height * scale),
                                  Image.NEAREST)
  d = ImageDraw.Draw(out)
  font = load_font(20)
  colors = {'ok': '#00c800', 'ng': '#ff3c3c', 'unknown': '#ffa500'}
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
  ap.add_argument('--model', required=True)
  ap.add_argument('--json', required=True)
  ap.add_argument('--out', required=True)
  ap.add_argument('--threshold', type=float, default=0.5)
  args = ap.parse_args()

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = GlyphNet().to(device)
  model.load_state_dict(torch.load(args.model, map_location=device))
  model.eval()

  json_arg = Path(args.json)
  json_paths = sorted(json_arg.glob('*.json')) if json_arg.is_dir() else [json_arg]
  out_dir = Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)

  all_rows = []
  for json_path in json_paths:
    data = json.loads(json_path.read_text(encoding='utf-8'))
    img = Image.open(json_path.parent / data['imagePath']).convert('RGB')
    shapes = [s for s in data['shapes'] if s['shape_type'] == 'rectangle']
    rows = predict_image(model, device, img, shapes, args.threshold)
    for r in rows:
      r['image'] = json_path.stem
    all_rows.extend(rows)
    draw_plot(img, rows).save(out_dir / f'{json_path.stem}_plot.png')
    n = {k: sum(1 for r in rows if r['status'] == k)
         for k in ('ok', 'ng', 'unknown')}
    print(f"{json_path.stem}: {len(rows)}件 "
          f"(一致 {n['ok']}, 不一致 {n['ng']}, ? {n['unknown']})")

  with open(out_dir / 'predictions.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['image', 'idx', 'truth', 'pred', 'conf', 'status',
                'x1', 'y1', 'x2', 'y2'])
    for r in all_rows:
      w.writerow([r['image'], r['idx'], r['truth'], r['pred'],
                  f"{r['conf']:.4f}", r['status'],
                  r['x1'], r['y1'], r['x2'], r['y2']])
  tot = len(all_rows)
  if tot:
    ok = sum(1 for r in all_rows if r['status'] == 'ok')
    print(f"全体: {ok}/{tot} = {ok / tot:.4f} (? は不一致として数える)")


if __name__ == '__main__':
  main()
