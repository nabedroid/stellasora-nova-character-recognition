# ステラソラ文字OCR 学習までの手順

グリフ分類器（62クラス）の学習に到達するまでの手順。
設計前提：実データ主体、合成は実例の少ないクラスの底上げのみ。
val 分割は元画像ファイル単位（同じ画像のグリフを train/val に混ぜない）。

想定レイアウト（リポジトリ stellasora-nova-character-recognition）：

```
compose.yaml           Docker Compose 定義 (プロジェクトルートに集約)
data/
  labelme/images/      labelme JSON (PNG はローカルのみ、gitignore 済み)
  converted/           変換出力 (crops) ※ローカルのみ
  dataset_s1/          stage1 用データセット ※ローカルのみ
  dataset_s2/          stage2 用データセット ※ローカルのみ
  yolo_1class/         1クラス化した YOLO ラベル ※ローカルのみ
  detector/            検出用データセット ※ローカルのみ
  font/NovaModern.ttf  ブロック体フォント ※ローカルのみ
ml/
  Dockerfile
  src/ (convert_labelme.py, build_dataset.py, train.py)
  runs/                学習結果 ※ローカルのみ (per_class.csv は共有可)
```

## 1. 配置と .gitignore

フォントは woff2 を TTF に展開し、`data/font/NovaModern.ttf` に置く（ローカルのみ）。
`.gitignore` に以下を追加し、生成物とフォントが誤って上がらないようにする。

```
data/converted/
data/dataset_s1/
data/dataset_s2/
ml/runs/
*.ttf
*.woff2
```

## 2. Docker イメージのビルド

プロジェクトルートでビルドする。以降のコマンドはすべてプロジェクトルートで実行する。
学習は GPU 前提（RTX 2070 SUPER）。

```powershell
docker compose build
```

GPU がコンテナから見えることを確認する。True が返ればよい。

```powershell
docker compose run --rm ml python -c "import torch; print(torch.cuda.is_available())"
```

## 3. アノテーション変換

labelme JSON（とローカルの PNG）から、文字ごとの crop と件数表を作る。
入力フォルダには JSON と画像が同居していること。

```powershell
docker compose run --rm ml python src/convert_labelme.py /workspace/data/labelme/images /workspace/data/converted
```

出力の `crops/` が以降の入力になる。件数表で欠けている文字を確認する。
なお crop のフォルダ名は `upper_A` / `lower_a` / `digit_5` の形式にする
（Windows はフォルダ名の大文字小文字を区別しないため、A と a を素のラベル名で分けると同じフォルダに混ざってしまう）。

## 4. データセット構築 (stage1: 実データのみ)

crop を train/val にファイル単位で分割する。val に回す元画像は
build_dataset.py の VAL_FILES（outfit_1003, 1005, 2007, 3005, 3012, 3021, 4018、計204グリフ）に固定してある。
この val は開発中の評価用で、最終確認用の test は未アノテーションの画像を将来アノテーションして別途用意する。

```powershell
docker compose run --rm ml python src/build_dataset.py --crops /workspace/data/converted/crops --font /workspace/data/font/NovaModern.ttf --out /workspace/data/dataset_s1 --no-synthetic
```

## 5. データセット構築 (stage2: 合成で底上げ)

stage1 と同じ分割に、実例が 24 個未満のクラスだけフォント合成で底上げする。
合成はコントラストのはっきりした配色＋最小限の回転のみで、強い歪ませは入れない。

```powershell
docker compose run --rm ml python src/build_dataset.py --crops /workspace/data/converted/crops --font /workspace/data/font/NovaModern.ttf --out /workspace/data/dataset_s2
```

## 6. 学習 (stage1)

共有グリフ分類器（小規模 CNN、入力 48×48、62クラス）を実データのみで学習する。
これが物差しになる。

```powershell
docker compose run --rm ml python src/train.py --data /workspace/data/dataset_s1 --stage stage1
```

## 7. 学習 (stage2)

底上げ済みデータで同じ条件で学習する。

```powershell
docker compose run --rm ml python src/train.py --data /workspace/data/dataset_s2 --stage stage2
```

## 8. 結果の見方と判断基準

結果は `ml/runs/<stage>/` に出る（model.pt と per_class.csv）。

- 全体正答率だけでなく、**クラス別** に stage1 と stage2 を並べて比較する
- 合成で改善したクラス（特に j・x・数字・レア文字）と、悪化したクラスを分ける
- 悪化したクラスがあれば、そのクラスだけ合成を外す（floor の調整や対象クラスの見直し）
- per_class.csv と一緒に出る「混同の多いペア」上位を見て、実害のある混同かどうかを判断する

## 9. 推論とプロット (predict)

学習済みモデルで画像を推論し、プロット画像と predictions.csv を出す。

```powershell
docker compose run --rm ml python src/predict.py --model /workspace/runs/stage2/model.pt --json /workspace/data/labelme/images --out /workspace/runs/stage2/plots
```

- `--json` は json ファイル単体でも、フォルダ（中の全 json を処理）でもよい。画像は json の imagePath から引く
- プロットは「元画像に色付きの箱＋予測文字」。色は緑＝正解、赤＝不一致、橙＝信頼度不足。信頼度はプロットに出さず、predictions.csv にのみ残す
- 信頼度が `--threshold`（デフォルト 0.5）未満の予測は「?」として出す

## 10. 1クラス化 (yolo_to_1class)

検出器は「どの文字か」を捨てて「文字がある場所」だけを学習する。
convert_labelme.py が出した62クラスの YOLO ラベルを1クラスに変換する。
なお以降の手順では ultralytics を使うため、最初にイメージの再ビルドが要る。

```powershell
docker compose build
docker compose run --rm ml python src/yolo_to_1class.py /workspace/data/converted/yolo /workspace/data/yolo_1class
```

## 11. 検出用データセット構築

画像と1クラスのラベルを、分類器と同じ VAL_FILES で train/val に分けて
YOLO 学習用レイアウト（images/labels + data.yaml）に配置する。

```powershell
docker compose run --rm ml python src/build_detector_dataset.py --images /workspace/data/labelme/images --labels /workspace/data/yolo_1class --out /workspace/data/detector
```

## 12. 検出器の学習

```powershell
docker compose run --rm ml python src/train_detector.py --data /workspace/data/detector/data.yaml --name detector
```

- ベースモデル yolo11n.pt は初回実行時に自動でダウンロードされる
- 結果は runs/detector/weights/best.pt（実際の保存先は終了時に表示される）

## 13. 検出の評価 (位置の再現率)

分類の正しさではなく「正解箱の位置に検出箱が当たっているか」だけを測る。

```powershell
docker compose run --rm ml python src/evaluate_detector.py --weights /workspace/runs/detector/weights/best.pt --images /workspace/data/labelme/images
```

- デフォルトは val の7ファイルが対象（--all で全ファイル）
- IoU>=0.5 の再現率と、緩い基準（中心ヒット）の再現率、サイズ帯別・ファイル別内訳、余分な検出箱の数を出す
- 過検出は分類器側で落とす前提なので、まず見るのは再現率

## 14. 検出のプロット (predict_detector)

検出器だけで画像に枠を描く。labelme は不要なので、未アノテーションの画像にも使える。

```powershell
docker compose run --rm ml python src/predict_detector.py --weights /workspace/runs/detector/weights/best.pt --images /workspace/data/labelme/images --out /workspace/data/predict_detector
```

- クラスも信頼度の数値も描かない。信頼度は枠の色で表す（しきい値ちょうど = 赤 → 黄 → 1.0 = 緑のグラデーション）
- 画像は2倍に拡大して `<元名>_plot.png` を出力する
- `--images` は画像フォルダでも画像1枚でもよい。`--conf` のデフォルトは 0.25

## 15. 今回の範囲外（次フェーズ）

- predict への検出器の接続（画像だけで推論する形）
- reject の扱い（当面は分類器の信頼度に対する閾値で代用する想定）
- ONNX 化と実行環境への組み込み
