# データセット構築: crop を元画像ファイル単位で train/test に分け、
# 実例の少ないクラスだけフォント合成で底上げする
# 使い方: python src/build_dataset.py --crops <cropフォルダ> --font <TTFパス> --out <出力先>
#   --floor 24        合成で底上げする下限 (実例がこれ未満のクラスだけ補う)
#   --no-synthetic    合成なし (stage1: 実データのみの物差し用)
import argparse
import random
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

CLASSES = [chr(c) for c in range(ord('A'), ord('Z') + 1)] + \
          [chr(c) for c in range(ord('a'), ord('z') + 1)] + \
          [chr(c) for c in range(ord('0'), ord('9') + 1)]

def label_dirname(label):
  # Windows はフォルダ名の大文字小文字を区別しないため、
  # A と a が同じフォルダに混ざらない名前を使う (upper_A / lower_a / digit_5)
  if label.isupper():
    return f'upper_{label}'
  if label.islower():
    return f'lower_{label}'
  return f'digit_{label}'


# テストに回す元画像。同じ画像のグリフが train/test の両方に入らないよう、
# グリフ単位ではなくファイル単位で分ける
TEST_FILES = ['outfit_1003', 'outfit_1005', 'outfit_2007', 'outfit_3005',
              'outfit_3012', 'outfit_3021', 'outfit_4018']

# 合成の配色 (文字色, 背景色)。コントラストのはっきりした組み合わせだけを使う
PALETTES = [('#111111', '#ffffff'), ('#ffffff', '#111111'),
            ('#14213d', '#f8f4e8'), ('#f8f4e8', '#14213d'),
            ('#3b0f2f', '#ffeef7'), ('#ffeef7', '#3b0f2f')]


def render_glyph(font_path, ch, rng):
  # フォントから1グリフを描く。字形を覚えさせる用途なので劣化は最小限に留める
  fg, bg = rng.choice(PALETTES)
  img = Image.new('RGB', (96, 96), bg)
  d = ImageDraw.Draw(img)
  font = ImageFont.truetype(font_path, rng.randint(58, 80))
  d.text((48, 48), ch, font=font, fill=fg, anchor='mm')
  img = img.rotate(rng.uniform(-6, 6), resample=Image.BILINEAR, fillcolor=bg)
  return img


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--crops', required=True)
  ap.add_argument('--font', required=True)
  ap.add_argument('--out', required=True)
  ap.add_argument('--floor', type=int, default=24)
  ap.add_argument('--no-synthetic', action='store_true')
  args = ap.parse_args()

  crops = Path(args.crops)
  out = Path(args.out)
  rng = random.Random(42)
  train_counts = {c: 0 for c in CLASSES}
  test_counts = {c: 0 for c in CLASSES}

  for label in CLASSES:
    src_dir = crops / label_dirname(label)
    if not src_dir.exists():
      continue
    for png in sorted(src_dir.glob('*.png')):
      stem = png.name.rsplit('_', 1)[0]
      split = 'test' if stem in TEST_FILES else 'train'
      dst = out / split / label_dirname(label)
      dst.mkdir(parents=True, exist_ok=True)
      shutil.copy(png, dst / png.name)
      if split == 'train':
        train_counts[label] += 1
      else:
        test_counts[label] += 1

  # 合成で底上げ (train の実例が floor 未満のクラスだけ)
  topped = {}
  if not args.no_synthetic:
    for label in CLASSES:
      need = args.floor - train_counts[label]
      if need <= 0:
        continue
      dst = out / 'train' / label_dirname(label)
      dst.mkdir(parents=True, exist_ok=True)
      for i in range(need):
        render_glyph(args.font, label, rng).save(dst / f'synth_{i:03d}.png')
      topped[label] = need

  print(f"train: {sum(train_counts.values())} (+合成 {sum(topped.values())}), "
        f"test: {sum(test_counts.values())}")
  print('底上げしたクラス:', topped)
  zero_test = [c for c in CLASSES if test_counts[c] == 0]
  print('テストに無いクラス:', ' '.join(zero_test))


if __name__ == '__main__':
  main()
