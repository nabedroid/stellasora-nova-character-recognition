# 検出器 (1クラス) の学習用データセットを YOLO 形式のレイアウトに組む
#   入力: labelme フォルダ (json + 画像) と、yolo_to_1class.py の出力フォルダ
#   出力: <out>/images/{train,val}, <out>/labels/{train,val}, <out>/data.yaml
#   分割は分類器と同じ VAL_FILES (ファイル単位)
# 使い方: python src/build_detector_dataset.py --images <labelmeフォルダ> --labels <1クラスYOLOフォルダ> --out <出力先>
import argparse
import json
import shutil
from pathlib import Path

from build_dataset import VAL_FILES


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--images', required=True)
  ap.add_argument('--labels', required=True)
  ap.add_argument('--out', required=True)
  args = ap.parse_args()

  images = Path(args.images)
  labels = Path(args.labels)
  out = Path(args.out)
  counts = {'train': [0, 0], 'val': [0, 0]}  # [画像数, 箱数]

  for json_path in sorted(images.glob('*.json')):
    data = json.loads(json_path.read_text(encoding='utf-8'))
    stem = json_path.stem
    label_path = labels / f'{stem}.txt'
    if not label_path.exists():
      print(f'警告: ラベルが無いので飛ばす: {stem}')
      continue
    split = 'val' if stem in VAL_FILES else 'train'
    img_dst = out / 'images' / split
    lab_dst = out / 'labels' / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lab_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(images / data['imagePath'], img_dst / data['imagePath'])
    shutil.copy(label_path, lab_dst / label_path.name)
    n_boxes = len([l for l in label_path.read_text(encoding='utf-8').splitlines()
                   if l.strip()])
    counts[split][0] += 1
    counts[split][1] += n_boxes

  yaml_text = (f"path: {out.resolve()}\n"
               "train: images/train\n"
               "val: images/val\n"
               "names:\n  0: glyph\n")
  (out / 'data.yaml').write_text(yaml_text, encoding='utf-8')
  for split in ('train', 'val'):
    print(f"{split}: 画像 {counts[split][0]} 枚, 箱 {counts[split][1]} 個")
  print('data.yaml を書いた:', out / 'data.yaml')


if __name__ == '__main__':
  main()
