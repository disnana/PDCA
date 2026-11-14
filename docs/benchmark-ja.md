# PDCA v0.2.0-alpha 検証記録

日時：2026-11-14（JST）

## 測定方法

- Bright Data由来のAnker商品データ47件（JSON値）
- 基準は `tiktoken.get_encoding("o200k_base")`
- 元JSONは `json.dumps(..., separators=(",",":"), sort_keys=True, ensure_ascii=False)` で整形したものを測定
- 圧縮後は、**埋め込みProfile込み**のPDCA packet全体を同じ方法で測定
- 可逆性は `decode(encode(input, profile)) == input` に加え、実データ47件の復元一致を確認
- 実測結果：元39,343、v0.1（AIなし）30,709、v0.1+Luna 30,178、v0.1+Gemini 30,057、v0.1+Sol 29,440、v0.2（AIなし）27,996、v0.2+候補統合27,310トークン
- v0.2のAI候補統合は、それまでに各モデルで生成したProfile内の文字列候補を**まとめてPythonで再選別**。APIへの追加問い合わせは行っていない

## 再現方法（元のデータとProfileを持っている場合）

```powershell
pip install ".[tokenizer]"
pdca benchmark products.json
pdca optimize products.json -o merged.pdca.json --profile-out merged.profile.json --candidate-profiles optimized.pdca.luna.json optimized.pdca.sol.json optimized.pdca.gemini.json
pdca decode merged.pdca.json -o restored.json
```

数値はデータの内容、Profile候補、パッケージのバージョン、エンコーディングによって変わります。元JSONが公開・再配布可能か未確認のため、本リポジトリには商品一覧の原本を同梱していません。

## 改善の内訳（実装項目）

- 短いDefaultマーカー、欠損マーカー
- すべてのレコードで一定なフィールドをProfileへ移動
- 型番の重複部分をレコード内参照で圧縮
- 同一URL接頭辞の自動候補化
- 既存ProfileのAI提案辞書を統合し、総トークン数で評価

改善項目ごとの独立アブレーションは未計測。したがって個々の機能が何トークン削減したか、という内訳を本記録からは主張しません。

## 検証済み・未検証

実施：JSON値としての可逆性、実データ47件の復元照合、異常系・型変更・未知フィールド、既存packetのデコード、pytest。

未実施：OpenAI/Geminiでの圧縮後データの回答精度（理解度）、API通信の総費用とレイテンシ、Python 3.11-3.13のGitHub Actions実行結果（CIで確認）、異種・大規模データ群での平均圧縮率、セキュリティ監査、バイト同一の再現、第三者による再現実験。

## 解釈

この結果は「構造化JSONを特定の形式に可逆変換し、入力テキストの参考トークン数を削減できた」という**限定された実証**です。「LLMが読み取れる」「Geminiの課金が30%下がる」「あらゆるJSONが30%減る」という意味ではありません。