# Speech Transcriber - Development History

## 2025-01-25 18:30 JST - README強化・Notionインテグレーション権限付与の明記

### 送信プロンプト全文
修正依頼（README強化・Notionインテグレーションの権限付与を明記）：

背景：既存インテグレーションを流用した際、保存先ページに権限が付いておらず「Could not find page with ID …」が発生した。READMEに正しい手順を追記して、同種のトラブルを防ぎたい。

対応：README.md に下記「README追記用テキスト」を追加してください（文言微調整可、意味は等価に）。既存ファイルは上書きで構いません。処理は止まらないように。

あわせて prompts/history.md に、この修正依頼のログを追記してください（日時JST・件名・送信プロンプト全文・出力要約・次アクションを追記。追記であって上書き不可）。

README追記用テキスト（そのまま入れてOK）：

Notion Integration の設定
A. 新規にインテグレーションを作る場合

https://www.notion.so/my-integrations
 で New integration を作成し、NOTION_TOKEN を取得します。

Capabilities（権限）は「Read content」「Insert content」「Update content」を有効にしてください。

保存先となる親ページを開き、右上の「共有」または「…」メニューから Integration を接続（Add connections / Invite） します。

親ページURL末尾の32桁IDを .env に設定します（ハイフン有無どちらでも可）：
NOTION_PARENT_PAGE_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

.env 例：
NOTION_TOKEN=secret_xxxxxxxxxxxxxxxxx
NOTION_PARENT_PAGE_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

B. 既存インテグレーションを「使い回す」場合（ハマりどころ）

https://www.notion.so/profile/integrations/
 を開き、対象インテグレーションを選択します。

「アクセス」タブ → 右上の アクセス権限を編集 をクリックします。

検索欄で保存先の親ページを指定し、アクセスを更新（追加） します。

これでその親ページ配下に API で子ページを作成できるようになります。

.env は既存の NOTION_TOKEN をそのまま使い、NOTION_PARENT_PAGE_ID に親ページID（URL末尾32桁）を設定します。

よくあるエラー

Could not find page with ID: ...
→ ほぼ権限不足です。上記 A-2 または B-3 の「ページにインテグレーションを接続（アクセス付与）」が未実施です。

別ワークスペースのトークンを使っている
→ 親ページと同じワークスペースのインテグレーションを使用してください。

受け入れ基準：

README.md に新規作成ケース(A)と既存流用ケース(B)の両方が明記され、Could not find page with ID の再発を防げること。

prompts/history.md に本依頼のログが追記されていること（上書きではなく追記）。

### 出力概要
- README.md の Notion Integration セクションを大幅強化
- 新規作成ケース(A)と既存流用ケース(B)の詳細手順を追加
- 「Could not find page with ID」エラーの原因と対処法を明記
- prompts/history.md を新規作成し、開発履歴の記録を開始

### 主要ファイル
- README.md（73-110行目に詳細なNotion設定手順を追加）
- prompts/history.md（新規作成、本ログを記録）

### 注意点
- 既存インテグレーション流用時のアクセス権限追加手順が重要
- ワークスペース間違いによるエラーケースも言及

### 次アクション
- .env ファイルにNotion設定情報を入力
- 新規または既存インテグレーションの適切な権限設定を実施
- python main.py で動作確認実行

## 2025-01-25 19:15 JST - IMEとDiscordレイアウトの調整

### 送信プロンプト全文
修正依頼（IMEとDiscordレイアウトの調整）：

【背景】

保存確認プロンプトの y/n 入力で、日本語IMEが有効のまま 全角「ｙ／ｎ」 を入力→Enter→さらにEnter…の流れで、意図せず No 扱いで破棄 される事故が発生。

Discord への投稿で タイトル行とサマリー行の間に空行が1行 入ってしまい、詰めたい。

【対応方針】

保存確認の y/n 入力を ロバスト化。全角/半角や大小文字を正規化して判定。未判定は再入力を促す。

Discord 投稿の 余計な空行をなくす（タイトルの直下にサマリーを1行で続ける）。

【具体指示】

A. 保存確認の入力処理を改善

入力文字列に対して Unicode 正規化（NFKC）→ strip() → lower() を適用し、以下を同一視して判定する：

Yes 系：y, yes, ｙ, Ｙ, はい

No 系： n, no, ｎ, Ｎ, いいえ

上記どちらにも当てはまらない場合は 決してデフォルトで No にしない。
→ 「入力が認識できません。y/yes/はい または n/no/いいえ を入力してください。」と再度プロンプト。

事故防止のため、No を選んだときだけ二段階確認を入れてください：

例：「本当にこの録音を破棄しますか？ (y/n)」

ここでも同じ正規化ロジックを適用し、y/はい のときだけ破棄。n/いいえ は保存フローへ戻す。

既存の処理フロー（Enterで停止→保存 or 破棄→保存時はタイトル/概要入力→Discord/Notion送信）は維持。

B. Discord 投稿の空行を除去

Webhook の content を タイトル + 改行 + サマリー のみで構築し、余分な改行（先頭・末尾の \n、二重 \n\n）を入れない。

明示仕様：

サマリーあり："{title}\n{summary}"

サマリー空："{title}"（末尾改行なし）

送信前に title / summary は strip() で前後空白・改行を除去。

もし埋め込み（embeds）を使っている場合も、タイトル→説明の間に空行が入らないように構築。不要なら content 方式でOK。

既存の Discord 投稿の他仕様（リトライ、allowed_mentions など）は現状踏襲。

【受け入れ基準】

日本語IMEがオンで 全角「ｙ」「ｎ」 を入力しても、正しく Yes/No として認識される。

未判定文字（例：「ー」「yです」など）を入れても 勝手に No に落ちず、再入力を促す。

No 選択時のみ 二段階確認が出て、y/はい 確定で破棄、n/いいえ で保存フローに戻る。

Discordの投稿は タイトル行の直下にサマリー行が続き、空行は入らない。サマリー未入力時はタイトルのみが1行で投稿される。

既存の README / ドキュメントに、この入力仕様（全角対応＆No時の二段階確認）と Discord の改行ルールを追記。

prompts/history.md に本修正依頼のログを 追記（上書き禁止）。

備考：

文字正規化には Python の unicodedata.normalize("NFKC", s) を利用してください。実装は任意ですが、NFKC→strip→lower の順が望ましい。

既存ファイルは 上書きでOK。処理がダイアログで止まらないように進めてください。

### 出力概要
- main.py に Unicode 正規化による入力処理を実装
- 全角・半角文字の自動認識機能を追加（y/ｙ/yes/はい など）
- No選択時の二段階確認機能を実装
- discord_client.py の投稿形式を改善（余分な空行を除去）
- README.md に新しい入力仕様とDiscord投稿形式を明記

### 主要ファイル
- main.py（93-117行目に入力正規化機能、190-200行目に二段階確認フロー）
- discord_client.py（19-27行目にクリーンな投稿形式）
- README.md（125-144行目に詳細な操作手順と投稿形式）

### 注意点
- unicodedata.normalize("NFKC") による全角・半角統一
- 未判定入力時はデフォルトでNoにしない安全設計
- 二段階確認による誤操作防止

### 次アクション
- 日本語IME有効状態での動作確認
- 全角文字入力テスト（ｙ、ｎ、はい、いいえ）
- Discord投稿の空行確認とテスト