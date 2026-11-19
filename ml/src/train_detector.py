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
  results = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                        project=str(RUNS_DIR), name=args.name)
  print('保存先:', results.save_dir)
  print('best.pt:', results.save_dir / 'weights' / 'best.pt')


if __name__ == '__main__':
  main()
