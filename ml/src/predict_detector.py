# 検出器 (1クラス YOLO) の推論とプロット。画像だけから検出する (labelme 不要)
#   クラスは描かない (1クラスなので意味がない)。信頼度も数値では出さず、
#   枠の色だけで表す: しきい値ちょうど = 赤 → 中間 = 黄 → 1.0 = 緑
# 使い方: python src/predict_detector.py --weights <best.pt> --images <画像フォルダ or 画像1枚> --out <出力先>
#   --conf 0.25  検出の信頼度しきい値 (これ未満の箱は描かない)
import argparse
import colorsys
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

IMG_EXTS = {'.png', '.jpg', '.jpeg'}
SCALE = 2  # 小さい文字が見やすいよう、元画像を2倍に拡大してから描く


def conf_color(conf, conf_min):
  # 信頼度を枠の色に変換する (HSV の色相を赤 0 → 緑 1/3 へ動かす)
  t = (conf - conf_min) / (1.0 - conf_min) if conf_min < 1.0 else 1.0
  t = min(1.0, max(0.0, t))
  r, g, b = colorsys.hsv_to_rgb(t / 3.0, 1.0, 1.0)
  return (round(r * 255), round(g * 255), round(b * 255))


def plot(image_path, boxes, out_path, conf_min):
  # boxes は (x1, y1, x2, y2, conf) の列。拡大した画像に枠だけを描く
  img = Image.open(image_path).convert('RGB')
  img = img.resize((img.width * SCALE, img.height * SCALE), Image.NEAREST)
  draw = ImageDraw.Draw(img)
  for x1, y1, x2, y2, conf in boxes:
    draw.rectangle([x1 * SCALE, y1 * SCALE, x2 * SCALE, y2 * SCALE],
                   outline=conf_color(conf, conf_min), width=2)
  img.save(out_path)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--weights', required=True)
  ap.add_argument('--images', required=True)
  ap.add_argument('--out', required=True)
  ap.add_argument('--conf', type=float, default=0.25)
  args = ap.parse_args()

  src = Path(args.images)
  if src.is_file():
    targets = [src]
  else:
    targets = sorted(p for p in src.iterdir()
                     if p.suffix.lower() in IMG_EXTS)
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)

  model = YOLO(args.weights)
  for image_path in targets:
    result = model.predict(str(image_path), conf=args.conf, verbose=False)[0]
    boxes = []
    if result.boxes is not None:
      for xyxy, conf in zip(result.boxes.xyxy.tolist(),
                            result.boxes.conf.tolist()):
        boxes.append((*[float(v) for v in xyxy], float(conf)))
    plot(image_path, boxes, out / f'{image_path.stem}_plot.png', args.conf)
    print(f'{image_path.name}: 検出 {len(boxes)} 個')


if __name__ == '__main__':
  main()
