# セットアップ手順

## 事前準備（1回だけ）

**Node.js v18** をインストールする。  
→ https://nodejs.org/en/download/releases  
（v18.x を選ぶ。v20以上は不可）


## アプリを起動する

フォルダをターミナルで開いて、以下を順番に実行。

```
npm install --legacy-peer-deps
```
※ 数分かかる。エラーが出ても最後に `added ○ packages` と出ればOK。

```
npm run dev
```

ブラウザで開く → `http://localhost:5173`

