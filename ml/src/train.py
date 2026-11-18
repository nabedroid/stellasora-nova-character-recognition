# 共有グリフ分類器の学習 (62クラス)
# 使い方: python src/train.py --data <データセット> --stage <名前> [--epochs 60]
# 出力: runs/<stage>/model.pt と per_class.csv (クラス別正答率と混同の多いペア)
import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASSES = [chr(c) for c in range(ord('A'), ord('Z') + 1)] + \
          [chr(c) for c in range(ord('a'), ord('z') + 1)] + \
          [chr(c) for c in range(ord('0'), ord('9') + 1)]
IMG = 48


def label_dirname(label):
  # Windows はフォルダ名の大文字小文字を区別しないため、
  # A と a が同じフォルダに混ざらない名前を使う (upper_A / lower_a / digit_5)
  if label.isupper():
    return f'upper_{label}'
  if label.islower():
    return f'lower_{label}'
  return f'digit_{label}'


def pad_to_square(img):
  # 縦横比を保って正方形に寄せる (余白は画像の平均色で埋める)
  w, h = img.size
  side = max(w, h)
  avg = img.resize((1, 1)).getpixel((0, 0))
  canvas = Image.new('RGB', (side, side), avg)
  canvas.paste(img, ((side - w) // 2, (side - h) // 2))
  return canvas


class GlyphDataset(Dataset):
  def __init__(self, root, train=False):
    self.items = []
    root = Path(root)
    for ci, label in enumerate(CLASSES):
      label_dir = root / label_dirname(label)
      if not label_dir.exists():
        continue
      for png in sorted(label_dir.glob('*.png')):
        self.items.append((png, ci))
    self.train = train
    self.aug = transforms.Compose([
      transforms.RandomRotation(8),
      transforms.RandomAffine(0, scale=(0.9, 1.1)),
      transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ])
    self.to_tensor = transforms.Compose([
      transforms.Resize((IMG, IMG)),
      transforms.ToTensor(),
      transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

  def __len__(self):
    return len(self.items)

  def __getitem__(self, i):
    path, ci = self.items[i]
    img = Image.open(path).convert('RGB')
    img = pad_to_square(img)
    if self.train:
      img = self.aug(img)
    return self.to_tensor(img), ci


class GlyphNet(nn.Module):
  def __init__(self, n=62):
    super().__init__()

    def block(cin, cout):
      return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU())

    self.features = nn.Sequential(
      block(3, 32), nn.MaxPool2d(2),
      block(32, 64), nn.MaxPool2d(2),
      block(64, 128), nn.AdaptiveAvgPool2d(1))
    self.fc = nn.Linear(128, n)

  def forward(self, x):
    return self.fc(self.features(x).flatten(1))


def evaluate(model, loader, device):
  # クラス別の正解数/総数と混同行列を返す
  model.eval()
  correct = {c: 0 for c in CLASSES}
  total = {c: 0 for c in CLASSES}
  confusion = {}
  with torch.no_grad():
    for x, y in loader:
      pred = model(x.to(device)).argmax(1).cpu()
      for t, p in zip(y.tolist(), pred.tolist()):
        total[CLASSES[t]] += 1
        if t == p:
          correct[CLASSES[t]] += 1
        else:
          key = (CLASSES[t], CLASSES[p])
          confusion[key] = confusion.get(key, 0) + 1
  return correct, total, confusion


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--data', required=True)
  ap.add_argument('--stage', required=True)
  ap.add_argument('--epochs', type=int, default=60)
  ap.add_argument('--batch', type=int, default=64)
  ap.add_argument('--lr', type=float, default=1e-3)
  args = ap.parse_args()

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print('device:', device)
  data = Path(args.data)
  train_ds = GlyphDataset(data / 'train', train=True)
  val_ds = GlyphDataset(data / 'val')
  print(f'train: {len(train_ds)}, val: {len(val_ds)}')
  train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
  val_loader = DataLoader(val_ds, batch_size=args.batch)

  model = GlyphNet().to(device)
  opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
  loss_fn = nn.CrossEntropyLoss()

  for epoch in range(1, args.epochs + 1):
    model.train()
    sum_loss, sum_acc, n = 0.0, 0, 0
    for x, y in train_loader:
      x, y = x.to(device), y.to(device)
      opt.zero_grad()
      out = model(x)
      loss = loss_fn(out, y)
      loss.backward()
      opt.step()
      sum_loss += loss.item() * len(y)
      sum_acc += (out.argmax(1) == y).sum().item()
      n += len(y)
    sched.step()
    if epoch % 10 == 0 or epoch == 1:
      print(f'epoch {epoch}: loss {sum_loss / n:.4f}, train acc {sum_acc / n:.4f}')

  out_dir = Path('runs') / args.stage
  out_dir.mkdir(parents=True, exist_ok=True)
  torch.save(model.state_dict(), out_dir / 'model.pt')

  correct, total, confusion = evaluate(model, val_loader, device)
  tot = sum(total.values())
  cor = sum(correct.values())
  print(f'val acc: {cor}/{tot} = {cor / tot:.4f}' if tot else 'val: データなし')
  with open(out_dir / 'per_class.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['label', 'correct', 'total', 'acc'])
    for c in CLASSES:
      acc = correct[c] / total[c] if total[c] else ''
      w.writerow([c, correct[c], total[c], acc])
  top = sorted(confusion.items(), key=lambda kv: -kv[1])[:15]
  print('混同の多いペア (正解->予測: 件数):')
  for (t, p), cnt in top:
    print(f'  {t} -> {p}: {cnt}')


if __name__ == '__main__':
  main()
