# Speech Transcriber

PC（Windows PowerShell）およびAndroid（Pydroid3）で動作する音声文字起こしツールです。音声を録音してOpenAI GPT-4で文字起こしし、DiscordとNotionに自動保存します。

## 機能

- 20秒チャンク + 2秒オーバーラップでの高品質音声録音
- OpenAI gpt-4o-transcribeを使用した高精度文字起こし
- **録音停止時のテイル保存**: 最後の残りが20秒未満でも必ず保存
- **サンプル番号基準の正確なタイムライン管理**で音声欠落を防止
- **汎用テイル保存システム**: 録音が20秒未満でも必ず保存（TAIL_MIN_SEC=0.0で任意の残り音声を保存）
- 「。」ごとの自動改行で読みやすいテキスト整形
- **保存直前の二段補正処理**：
  1) ローカルフィルター：全角→半角正規化＋技術用語ホワイトリスト置換
  2) AI補正：gpt-4o-mini による境界限定補正（本文要約禁止）
- Discordへのタイトル＋概要通知
- Notionへのタイトル＋概要＋本文全文保存
- エラー時の自動リトライ機能

## 必要な環境

- Python 3.8以上（必須：型ヒント互換性のため）
- 音声録音可能な環境（マイク）
- インターネット接続

## セットアップ

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

#### Android（Pydroid3）の場合

Pydroid3アプリ内で以下のパッケージを個別にインストールしてください：

```bash
pip install openai
pip install sounddevice
pip install numpy
pip install requests
pip install python-dotenv
pip install notion-client
pip install scipy
```

### 2. 環境変数の設定

`.envSample`を`.env`にコピーし、必要な値を設定してください：

```bash
cp .envSample .env
```

`.env`ファイルに以下の情報を入力：

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Discord Webhook Configuration  
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your_webhook_url_here

# Notion API Configuration
NOTION_TOKEN=your_notion_integration_token_here
NOTION_PARENT_PAGE_ID=your_notion_parent_page_id_here

# Text Processing Configuration
NORMALIZE_TECH_TERMS=true
POSTPROCESS_TEST_MODE=false
POSTPROCESS_MODEL=gpt-4o-mini
AI_DEDUP_MODE=boundary_only
TAIL_MIN_SEC=1.0
```

#### 補正処理の設定

- **NORMALIZE_TECH_TERMS**: ローカルフィルターの有効/無効（`true`/`false`）
  - 全角英数記号を半角化、技術用語を統一（.env, README.md, Git/GitHub, Python など）
- **POSTPROCESS_MODEL**: 最終補正に使用するAIモデル（既定: `gpt-4o-mini`）
  - `gpt-4o-mini`: より素直で安全な補正（推奨）
  - `gpt-4o`: より高性能だが言い回しを動かしがち
- **POSTPROCESS_TEST_MODE**: AI補正のテストモード（`true`/`false`）
  - `false`: POSTPROCESS_MODEL で補正
  - `true`: gpt-4o-mini と gpt-4o 両方の結果を表示し比較、最終採用は mini
- **AI_DEDUP_MODE**: AI重複除去モード（`boundary_only`/`off`）
  - `boundary_only`: 境界の重複のみ除去、本文要約禁止（既定）
  - `off`: AI重複除去を完全無効化
- **TAIL_MIN_SEC**: 録音停止時の最小テイル保存秒数（既定: `0.0`）
  - Enter押下後の残り発話が指定秒数以上なら必ず保存
  - `0.0`設定により、任意の長さの録音（20秒未満含む）を確実に保存

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
```
NOTION_TOKEN=secret_xxxxxxxxxxxxxxxxx
NOTION_PARENT_PAGE_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

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

```bash
python main.py
```

### Windowsでワンボタン実行する方法

プロジェクトのルートディレクトリ（main.py がある場所）に `run_transcriber.bat` というファイルを作成し、以下の内容を貼り付けて保存します：

```batch
@echo off
REM ===== Speech Transcriber 実行バッチ =====
cd /d %~dp0
python main.py
pause
```

**各行の説明:**
- `cd /d %~dp0` : このバッチファイルが置いてある場所に移動します
- `python main.py` : プログラムを実行します  
- `pause` : 実行後にウィンドウが自動で閉じないようにします

`run_transcriber.bat` を **ダブルクリック**することで、Speech Transcriber が起動します。

**より便利にするには:**
- `run_transcriber.bat` の ショートカットを作成し、デスクトップなどに配置してください
- プロパティからアイコンを変更することで分かりやすくできます

### Android（Pydroid3）での実行

1. Pydroid3アプリでプロジェクトフォルダを開く
2. `main.py`を実行

## 操作手順

1. プログラムを実行すると音声録音が開始されます
2. **Enterキー**を押すと録音が停止します
3. **録音停止時の汎用テイル保存**: 録音時間に関係なく、任意の残り音声（1秒でも18秒でも）を必ず保存
4. 保存確認が表示されます：
   - 保存可否のプロンプトには **y** または **n** で回答してください（全角ｙ／ｎも可）
   - その他の入力は無効として再入力を促します
5. `n`を選択した場合：
   - **No を選んだ場合のみ破棄の再確認が表示されます**
   - 再確認で `y` なら確定破棄、`n` なら保存フローに戻ります
6. 保存を選択した場合：
   - タイトルを入力
   - 概要を入力  
   - 自動的にDiscordとNotionに保存されます

### Discord投稿形式
- **サマリーあり**: タイトル行の直下にサマリー行
- **サマリーなし**: タイトル行のみ（余分な空行なし）

## ファイル構成

```
speech-transcriber/
├── main.py              # メインアプリケーション
├── discord_client.py    # Discord連携
├── notion_api.py        # Notion連携
├── utils.py            # ユーティリティ関数（テキスト補正含む）
├── requirements.txt    # 依存パッケージ
├── .envSample         # 環境変数テンプレート
├── .env              # 環境変数（要設定）
├── .gitignore        # Git無視ファイル設定
├── LICENSE           # MITライセンス
├── README.md         # このファイル
└── prompts/
    └── history.md    # 開発履歴
```

## トラブルシューティング

### よくあるエラーと対処法

**音声録音エラー:**
- マイクのアクセス許可を確認
- `sounddevice`パッケージの再インストール
- Windows: オーディオドライバーの更新

**音声デバイス自動検出:**
- 起動時に利用可能なマイクデバイスを自動検出・優先順位選択
- サンプルレート（48k/44.1k/32k/16k）も自動フォールバック
- 48kHz/44.1kHz録音は16kHzへ自動ダウンサンプル

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

### 長文対応

長文でも自動的に段落分割・バッチ送信・必要に応じてページ分割されます。設定値（段落分割 1800 文字、1リクエスト 50 ブロック、1ページ 80,000 文字のソフト上限）は `notion_api.py` の定数で調整可能。

### プロンプト言語のカスタマイズ

プロンプトの言語を変更したい場合は `main.py` 冒頭の `MSG_*` 定数を書き換えるだけでよい。英語例はコメント（`EN: ...`）を参照。

### 非同期パイプライン（v2.0対応）

**v2.0** では音声欠落を防ぐため、録音とAPI呼び出しを完全に分離した非同期パイプライン設計に変更されました。

**パイプライン構成:**
1. **Ring Buffer Producer** (AudioRecorder) - 音声を軽量コールバックでリングバッファに蓄積
2. **Chunker Thread** (AudioChunker) - 20s/2s overlapの連続タイムライン管理、チャンク切り出し
3. **Consumer ThreadPool** - 文字起こしAPIを並列実行
4. **Real-time Printer** - start_index整列で順序保証、文末優先の表示制御
5. **Gap Detection** - 音声連続性の監視とログ出力
6. **汎用テイル処理** - 録音停止時の残り音声を必ず保存（録音時間に関係なく、1秒〜18秒の短い録音も確実に保存）

**新しい環境変数:**
```env
# 非同期処理設定
MAX_TRANSCRIBE_WORKERS=2          # 並列文字起こし数
TRANSCRIBE_QUEUE_MAX_SIZE=10      # キュー最大サイズ
RING_BUFFER_SECONDS=300          # リングバッファ保持秒数
TAIL_MIN_SEC=1.0                 # 最小テイル保存秒数
```

**ログ機能:**
- 各チャンクに capture_start/capture_end/api_start/api_end/printed_at を記録
- 連続チャンクの capture_start 差が (chunk_sec - overlap_sec) ±0.5s から外れたら WARNING ログ
- API レスポンス時間とキューサイズを監視
- テイルチャンク処理ログ: `Flushing tail chunk: samples=83200 sec=5.20 start=0 end=83200` （早期終了時は start=0 から全録音を保存）
- 早期終了検出ログ: `Displaying tail chunk N: [0-83200) = 5.20s` （録音時間<20秒の場合）

### デバッグモード

エラーの詳細を確認したい場合は、コンソール出力を注意深く確認してください。各操作の成功/失敗が明確に表示されます。v2.0では詳細なタイミングログが追加されています。

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 開発者向け情報

### コード構成

**v2.0 非同期パイプライン:**
- `AudioRecorder`: Ring Buffer型Producer（軽量コールバック録音）
- `AudioChunker`: チャンク切り出しThread（20s/2s overlap管理）
- `RealtimePrinter`: 順序保証とsentence-end優先の出力制御
- `SpeechTranscriber`: 非同期パイプライン制御とThreadPoolExecutor管理
- `DiscordClient`: Discord Webhook送信
- `NotionClient`: Notion API連携（長文対応）
- `utils.py`: テキスト整形、正規化、AI補正、重複除去

### セキュリティ

- APIキーは必ず`.env`ファイルで管理
- `.gitignore`により秘密情報の誤コミットを防止
- 適切なエラーハンドリングでAPIキーの漏洩を防止