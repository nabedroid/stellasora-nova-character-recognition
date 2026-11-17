# stellasora-nova-character-recognition



# labelme のインストール
アノテーションツールとして`labelme`を使用する。
ローカル環境を壊したくないので、Embedded Python を使用する。

## Embedded Python のインストール
1. https://www.python.org/downloads/windows/ にアクセスする。
2. `Windows embeddable package (64-bit)` をダウンロードする。
3. ダウンロードしたファイルを解凍する。
4. 下記の構成となるようにファイルを配置する。
```
stellasora-nova-character-recognition/
└── python-3.13.15-embed-amd64/
```

## pip のインストール
1. python-3.13.15-embed-amd64/python313._pth をエディタで開く
2. `#import site` を `import site` に変更する（コメントアウトを解除）
```
python313.zip
.

# Uncomment to run site.main() automatically
import site
```
3. powershell を起動する
4. 下記のコマンドで pip をインストールする（）
```
curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

## labelme のインストール
1. powershell を起動する
2. 下記のコマンドで labelme をインストールする
```
./python -m pip install labelme
```

# labelme でアノテーションする
1. `./python-3.13.15-embed-amd64/Scripts/labelme.exe` を起動する
3. 下記のコマンドでアノテーションを開始する
```
./python -m labelme
```


# メモ

## labelme の起動コマンド
画像、アノテーション結果の入出力の初期値を設定するために、下記のコマンドで起動する。
```
f:\work\stellasora-nova-character-recognition\python-3.13.15-embed-amd64\Scripts\labelme.exe f:\work\stellasora-nova-character-recognition\data\labelme\images --labels a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,1,2,3,4,5,6,7,8,9,0
```


## `.labelmerc` の作成場所
`labelme` を初めて起動すると `C:\Users\ユーザー名\.labelmerc` に `.labelmerc` が作成される。
これは labelme の設定ファイル。
恐らく labelme を uninstall しても自動的に削除されないので、`labelme` 関連のファイルを削除する際は消し忘れに注意すること。


# 参考
- [labelme](https://github.com/wkentaro/labelme)
- [get-pip](https://github.com/pypa/get-pip)
- [embedded package](https://docs.python.org/ja/3.13/using/windows.html?utm_source=chatgpt.com#the-embeddable-package)