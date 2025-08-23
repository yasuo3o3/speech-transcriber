# Speech Transcriber

PC（Windows PowerShell）およびAndroid（Pydroid3）で動作する音声文字起こしツールです。音声を録音してOpenAI GPT-4で文字起こしし、DiscordとNotionに自動保存します。

## 機能

- 20秒チャンク + 2秒オーバーラップでの高品質音声録音
- OpenAI gpt-4o-transcribeを使用した高精度文字起こし
- 「。」ごとの自動改行で読みやすいテキスト整形
- Discordへのタイトル＋概要通知
- Notionへのタイトル＋概要＋本文全文保存
- エラー時の自動リトライ機能

## 必要な環境

- Python 3.8以上
- 音声録音可能な環境（マイク）
- インターネット接続

## セットアップ

### 1. 依存パッケージのインストール

\`\`\`bash
pip install -r requirements.txt
\`\`\`

#### Android（Pydroid3）の場合

Pydroid3アプリ内で以下のパッケージを個別にインストールしてください：

\`\`\`bash
pip install openai
pip install sounddevice
pip install numpy
pip install requests
pip install python-dotenv
pip install notion-client
\`\`\`

### 2. 環境変数の設定

\`.env.example\`を\`.env\`にコピーし、必要な値を設定してください：

\`\`\`bash
cp .env.example .env
\`\`\`

\`.env\`ファイルに以下の情報を入力：

\`\`\`env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Discord Webhook Configuration  
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your_webhook_url_here

# Notion API Configuration
NOTION_TOKEN=your_notion_integration_token_here
NOTION_PARENT_PAGE_ID=your_notion_parent_page_id_here
\`\`\`

#### 各APIキーの取得方法

**OpenAI API Key:**
1. [OpenAI Platform](https://platform.openai.com/) にログイン
2. API Keys セクションで新しいキーを作成

**Discord Webhook URL:**
1. Discordサーバーの設定 → 連携サービス → ウェブフック
2. 新しいウェブフックを作成し、URLをコピー

**Notion Integration の設定**

**A. 新規にインテグレーションを作る場合**

1. https://www.notion.so/my-integrations で New integration を作成し、NOTION_TOKEN を取得します。

2. Capabilities（権限）は「Read content」「Insert content」「Update content」を有効にしてください。

3. 保存先となる親ページを開き、右上の「共有」または「…」メニューから Integration を接続（Add connections / Invite） します。

4. 親ページURL末尾の32桁IDを .env に設定します（ハイフン有無どちらでも可）：
   NOTION_PARENT_PAGE_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

.env 例：
\`\`\`
NOTION_TOKEN=secret_xxxxxxxxxxxxxxxxx
NOTION_PARENT_PAGE_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
\`\`\`

**B. 既存インテグレーションを「使い回す」場合（ハマりどころ）**

1. https://www.notion.so/profile/integrations/ を開き、対象インテグレーションを選択します。

2. 「アクセス」タブ → 右上の アクセス権限を編集 をクリックします。

3. 検索欄で保存先の親ページを指定し、アクセスを更新（追加） します。

4. これでその親ページ配下に API で子ページを作成できるようになります。

5. .env は既存の NOTION_TOKEN をそのまま使い、NOTION_PARENT_PAGE_ID に親ページID（URL末尾32桁）を設定します。

**よくあるエラー**

- **Could not find page with ID: ...**
  → ほぼ権限不足です。上記 A-3 または B-3 の「ページにインテグレーションを接続（アクセス付与）」が未実施です。

- **別ワークスペースのトークンを使っている**
  → 親ページと同じワークスペースのインテグレーションを使用してください。

## 使用方法

### PC（Windows PowerShell）での実行

\`\`\`bash
python main.py
\`\`\`

### Android（Pydroid3）での実行

1. Pydroid3アプリでプロジェクトフォルダを開く
2. \`main.py\`を実行

## 操作手順

1. プログラムを実行すると音声録音が開始されます
2. **Enterキー**を押すと録音が停止します
3. 保存確認が表示されます：
   - \`y/yes/はい\` → 保存プロセスに進む
   - \`n/no/いいえ\` → 二段階確認後に録音を破棄
   - **全角文字（ｙ、ｎ）も自動認識します**
   - 判定不能な入力は再入力を促します
4. \`n\`を選択した場合：
   - 「本当に破棄しますか？」の追加確認が表示されます
   - \`y/はい\` で確定破棄、\`n/いいえ\` で保存フローに戻ります
5. 保存を選択した場合：
   - タイトルを入力
   - 概要を入力  
   - 自動的にDiscordとNotionに保存されます

### Discord投稿形式
- **サマリーあり**: タイトル行の直下にサマリー行
- **サマリーなし**: タイトル行のみ（余分な空行なし）

## ファイル構成

\`\`\`
speech-transcriber/
├── main.py              # メインアプリケーション
├── discord_client.py    # Discord連携
├── notion_api.py        # Notion連携
├── utils.py            # ユーティリティ関数
├── requirements.txt    # 依存パッケージ
├── .env.example       # 環境変数テンプレート
├── .env              # 環境変数（要設定）
├── .gitignore        # Git無視ファイル設定
├── LICENSE           # MITライセンス
└── README.md         # このファイル
\`\`\`

## トラブルシューティング

### よくあるエラーと対処法

**音声録音エラー:**
- マイクのアクセス許可を確認
- \`sounddevice\`パッケージの再インストール
- Windows: オーディオドライバーの更新

**API接続エラー:**
- インターネット接続を確認
- APIキーが正しく設定されているか確認
- APIの利用制限に達していないか確認

**Pydroid3での実行エラー:**
- 必要なパッケージが全てインストールされているか確認
- Pydroid3のアクセス許可（マイク）を確認
- メモリ不足の場合はアプリを再起動

**Discord送信失敗:**
- Webhook URLが正しいか確認
- Discordサーバーが利用可能か確認
- レート制限に達していないか確認

**Notion保存失敗:**
- Tokenが正しいか確認
- Parent Page IDが正しいか確認
- 統合がページに招待されているか確認

**Import エラー:**
- 自作ファイル名が公式SDKと衝突しないよう、notion_api.py を使用
- notion_client（公式SDK）との名前衝突を回避済み

### デバッグモード

エラーの詳細を確認したい場合は、コンソール出力を注意深く確認してください。各操作の成功/失敗が明確に表示されます。

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 開発者向け情報

### コード構成

- \`AudioRecorder\`: 音声録音を管理
- \`SpeechTranscriber\`: メイン制御クラス
- \`DiscordClient\`: Discord Webhook送信
- \`NotionClient\`: Notion API連携
- \`utils.py\`: テキスト整形とユーティリティ

### セキュリティ

- APIキーは必ず\`.env\`ファイルで管理
- \`.gitignore\`により秘密情報の誤コミットを防止
- 適切なエラーハンドリングでAPIキーの漏洩を防止
