# 1クラス検出器 (YOLO) の学習。ultralytics を使用する
# 使い方: python src/train_detector.py --data <data.yaml> --name detector
#   --model yolo11n.pt  ベースモデル (初回実行時に自動でダウンロードされる)
#   --epochs 100        エポック数
#   --imgsz 640         入力サイズ (元画像は 512 なので少し拡大)
# 出力: /workspace/runs/<name>/weights/best.pt (実際の保存先は終了時に表示する)
#   project は絶対パスで渡す。相対パスだと ultralytics が既定の runs/detect を
#   前に付けて runs/detect/runs/<name> のように入れ子になるため
import argparse
from pathlib import Path

from ultralytics import YOLO

# src の親 (= /workspace) の下の runs を保存先にする
RUNS_DIR = Path(__file__).resolve().parent.parent / 'runs'


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--data', required=True)
  ap.add_argument('--name', default='detector')
  ap.add_argument('--model', default='yolo11n.pt')
  ap.add_argument('--epochs', type=int, default=100)
  ap.add_argument('--imgsz', type=int, default=640)
  args = ap.parse_args()

  model = YOLO(args.model)
  # 拡張 (augmentation) は ultralytics 既定のうち文字と相性の悪いものを切る:
  #   fliplr  左右反転は字形が別物になるので切る (既定 0.5)
  #   mosaic  4枚合成で小さいグリフがさらに縮むので切る (既定 1.0)
  #   scale   拡縮の下限を浅くする (既定 0.5 = 0.5〜1.5倍 → 0.2 = 0.8〜1.2倍)
  #   degrees 斜め書きの実例があるので、回転を ±15° だけ入れる (既定 0.0)
  results = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                        fliplr=0.0, mosaic=0.0, scale=0.2, degrees=15.0,
                        project=str(RUNS_DIR), name=args.name)
  print('保存先:', results.save_dir)
  print('best.pt:', results.save_dir / 'weights' / 'best.pt')


if __name__ == '__main__':
  main()
