# 歌唱変換プロトタイプ (Irodori 参照音声 → Seed-VC v1 歌唱モデル)

Irodori-TTS でターゲット話者の参照音声を作り(または既存 WAV を使い)、その WAV を
Seed-VC v1 の **F0 条件付き 44.1 kHz 歌唱モデル** の参照として、伴奏なしのボーカル WAV を変換する
Windows ローカル向けのプロトタイプです。Seed-VC v2 の話し声モデルは使いません。

検証結果は [SINGING_VALIDATION_JA.md](SINGING_VALIDATION_JA.md) を参照してください。

## 構成

| 役割 | 場所 | 備考 |
| --- | --- | --- |
| Irodori 環境 | `.venv` | 既存 `setup_windows.ps1` / `uv.lock` そのまま |
| Seed-VC 環境 | `.singing\venv` | Python 3.10, torch 2.4.0+cu121, `configs\singing\seedvc-requirements.lock.txt` |
| Seed-VC ソース | `.singing\seed-vc` | `Plachtaa/seed-vc` @ `51383efd921027683c89e5348211d93ff12ac2a8` |
| Seed-VC モデル | `.singing\seed-vc\checkpoints` | 下表の固定リビジョン、`.singing\seedvc_assets.json` に記録 |
| Irodori モデル | `model_data\huggingface` | 既存と同じ HF キャッシュ構成 |
| 出力 | `outputs\singing\<日時>_<名前>\` | Git 管理外 |

`.singing\`、`.venv`、`model_data\`、`outputs\` はすべて `.gitignore` 済みです。重み・音声・環境はコミットしません。

コード (`singing_vc\`):

- `pipeline.py` — 手順の実行と記録(標準ライブラリのみ)。各モデルは**別プロセスで順番に**実行するため、
  Irodori と Seed-VC が同時に GPU に載ることはなく、Torch/Transformers のバージョンも混ざりません。
- `seedvc_runner.py` — Seed-VC venv 内で upstream `inference.py` を実行。チェックポイントと config を常に明示し、
  config が F0 条件付き・44.1 kHz・BigVGAN `nvidia/bigvgan_v2_44khz_128band_512x` であることを確認してから読み込みます
  (話し声モデルへの暗黙の切り替えはしません)。出力先が入力と同じファイル(ハードリンク含む)なら推論前に拒否し、
  upstream の出力は実行ごとの一時フォルダに書かせるため、既存ファイルを削除しません。
- `audio_tool.py` — 入力の切り出し・コピーと、出力の客観チェック(サンプルレート、長さ、有限値、無音判定、クリップ率、pYIN による F0 比較)。
- `seedvc_assets.py` — Seed-VC 用モデルを固定リビジョンで取得し `refs/main` を固定(オフライン再実行用)。
- `seed_caches.py` — 既存ワークスペースの HF キャッシュをハードリンクで再利用(読み取りのみ)。
- `cli.py` / `webui.py` — CLI と Gradio UI。

固定している Seed-VC 関連モデル:

| リポジトリ | リビジョン | ファイル | ライセンス (HF 表記) |
| --- | --- | --- | --- |
| Plachta/Seed-VC | 257283f9f41585055e8f858fba4fd044e5caed6e | DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth | GPL-3.0 |
| lj1995/VoiceConversionWebUI | e6d0c1a17da07c33557852f9dfa2bd44cc75737d | rmvpe.pt | MIT |
| funasr/campplus | e4b6ede7ce16997aff4ae69fbca1f0175e2afede | campplus_cn_common.bin | Apache-2.0 |
| nvidia/bigvgan_v2_44khz_128band_512x | 95a9d1dcb12906c03edd938d77b9333d6ded7dfb | config.json, bigvgan_generator.pt | MIT |
| openai/whisper-small | 973afd24965f72e36ca33b3055d56a652f456b4d | config.json, model.safetensors, preprocessor_config.json | Apache-2.0 |

config は upstream の `configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml`
(HF 上のコピーとはモデルパラメータが一致し、学習用の `log_dir`/`pretrained_model` のみ異なる)を使います。
upstream `inference.py` の既定は `..._ft_ema_v2.pth` ですが、本プロトタイプは README で v1.0 歌唱モデルとして
掲載されている `..._ft_ema.pth` を既定にしています。別のチェックポイントは `--seedvc-checkpoint` で指定できます。

## セットアップ

```powershell
# 既存ワークスペースのモデルキャッシュを再利用する場合(推奨、ダウンロード量を大幅に削減)
.\setup_singing.ps1 -SeedFrom F:\workspace\Irodori-TTS_V4

# 再利用しない場合(必要なものをすべてダウンロード)
.\setup_singing.bat
```

処理内容: Seed-VC を固定コミットで clone → (任意)キャッシュのハードリンク → 既存 `setup_windows.ps1 -SkipModels`
で `.venv` を作成し v4.1-Small 一式を確認/取得 → `.singing\venv` をロックファイルで同期 → Seed-VC モデルを固定リビジョンで確認/取得。
再実行しても既存の環境・モデルは再利用され、取得済みならオフラインでも完了します。

`-SeedFrom` について: 指定したワークスペースの `model_data\huggingface\hub` から
v4.1-Small / Semantic-DACVAE / silentcipher / modernbert トークナイザ と whisper-small を
**ハードリンク**(同一ドライブ以外はコピー)します。元ファイルは読むだけで変更しません。1 MB 未満のメタデータはコピーします。
HF キャッシュのファイルはその場で書き換えられないため共有して問題ありませんが、
ハードリンク先のファイルを手で編集すると元側も変わる点に注意してください(削除は互いに影響しません)。

社内プロキシ等で TLS 検査がある環境では、既存の `local_env.py` が Windows 証明書ストアから
`model_data\certs\windows-ca.pem` を作成し、HF/Git/uv がそれを使います。

## 使い方 (CLI)

日本語を含む引数は PowerShell 版 `run_singing.ps1` を使ってください(`.bat` はコンソールのコードページで文字化けすることがあります)。
`.bat` で使う場合は `--tts-text-file` / `--tts-caption-file`(UTF-8)を使えます。

```powershell
# Irodori で参照音声を生成して変換(例の設定ファイルを既定値として使う)
.\run_singing.ps1 --options-json configs\singing\example_tts_reference.json `
  --source "D:\vocals\my song (vocal).wav" --source-start 15 --source-duration 20 --run-name test

# テキストとキャプションを直接指定
.\run_singing.ps1 --source vocal.wav --tts-text "こんにちは。今日は歌の練習をします。" `
  --tts-caption "落ち着いた、近い距離感の女性話者"

# 既存のターゲット音声 WAV を参照に使う(Irodori は起動しない)
.\run_singing.ps1 --source vocal.wav --reference-wav "D:\voices\target.wav" --semitone-shift -2

# 実行せずに設定と入力チェックだけ確認
.\run_singing.ps1 --source vocal.wav --reference-wav target.wav --dry-run
```

主なオプション(`--help` で一覧):

| オプション | 既定 | 説明 |
| --- | --- | --- |
| `--source` | 必須 | 伴奏を除いたボーカル WAV(初期スコープ。伴奏分離は行いません) |
| `--source-start` / `--source-duration` | 0 / 全体 | ソースの一部だけを変換 |
| `--reference-wav` / `--tts-text` | どちらか一方 | 既存参照 WAV、または Irodori で生成するテキスト |
| `--tts-caption` | なし | 声質・スタイルのキャプション |
| `--tts-checkpoint` / `--tts-hf-checkpoint` | v4.1-Small | ローカル checkpoint または HF リポジトリ |
| `--tts-lora-adapter` | なし | PEFT LoRA アダプタのフォルダ |
| `--tts-speaker-wav` | なし(`--no-ref`) | Irodori の話者参照 WAV |
| `--tts-seed` / `--tts-num-steps` / `--tts-seconds` | 0 / 既定 / 自動 | Irodori の生成設定 |
| `--tts-extra-arg` | なし | `infer.py` へのその他の引数(1 トークンずつ繰り返し指定) |
| `--f0-condition` | true | 歌唱モデルでは true 必須。false は F0 なしの config を明示した場合のみ |
| `--auto-f0-adjust` | false | ソースの音高をターゲットの音域へ自動移動 |
| `--semitone-shift` | 0 | 半音単位の移調 |
| `--length-adjust` | 1.0 | 長さの倍率 |
| `--diffusion-steps` / `--inference-cfg-rate` | 30 / 0.7 | upstream 推奨は歌唱で 30〜50 ステップ |
| `--fp16` | true | upstream 既定と同じ |
| `--seedvc-checkpoint` / `--seedvc-config` | 固定の v1 歌唱モデル | 差し替え用 |
| `--output-dir` / `--run-name` | `outputs\singing` | 出力先と実行名 |
| `--online` | オフライン | HF へのネットワークアクセスを許可(未取得モデルを使う場合) |

終了コード: 0 = 成功、1 = 入力エラーまたは実行失敗(失敗したステップのログ末尾を表示)、3 = 実行は完了したが自動チェックに不合格。

## 使い方 (Gradio UI)

```bat
run_singing_webui.bat
```

http://127.0.0.1:7862 を開きます。ソース歌声、参照の種類(Irodori 生成 / 既存 WAV)、Irodori・Seed-VC の設定を指定して「変換」。
ログ、参照音声、変換結果、チェック結果が表示されます。出力フォルダを既定 (`outputs\singing`) 以外にすると、
UI 上で音声を再生できない場合があります(ファイル自体は保存されます)。

## 出力

`outputs\singing\<日時>_<名前>\`:

- `source.wav`(切り出し後のソース)、`reference.wav`(参照音声)、`converted.wav`(変換結果、44.1 kHz)
- `metadata.json` — 状態、全オプション、各ステップの実行コマンドと所要時間、Irodori/Seed-VC の Git リビジョン、
  使用チェックポイント、Seed-VC レポート(GPU 名、ピーク VRAM、upstream 引数)、客観チェック結果
- `analysis.json` — 波形の数値と F0 比較、`seedvc_report.json`、`source_prepare.json`
- `summary_ja.md` — 日本語の要約
- `logs\*.log` — 各ステップの完全なログ

客観チェック(`checks`): 出力が 44.1 kHz、有限値のみ、無音でない(RMS > -60 dBFS)、長さがソース × length-adjust の ±5% 以内。
F0 比較は pYIN による音高の追従度で、**声の似ている度合いや歌唱品質を示すものではありません**。試聴での確認が必要です。
F0 はフレーム番号をそのまま対応させるため、length-adjust ≠ 1 では比較せず `not_comparable_reason` を記録します。
auto-f0-adjust=true では指定半音との cent 誤差を出しません(相関と有声 IoU のみ)。発音・立ち上がりのタイミングは F0 指標では分かりません。

## 制限事項

- 入力は伴奏を除いたボーカルのみを想定しています。伴奏分離は含みません。
- Seed-VC は参照音声の先頭 25 秒のみを使います(upstream の仕様)。Irodori 参照は 5〜15 秒程度が目安です。
- 長いソースは upstream が 30 秒単位で分割・クロスフェードします。
- Irodori v4.1-Small の参照音声は SilentCipher の透かし入りで出力されます(既存 Irodori の挙動のまま)。
- 既定ではオフライン(`HF_HUB_OFFLINE=1`)で実行します。未取得のモデルを指定した場合は `--online` が必要です。

## ライセンス

- Seed-VC のコードと `Plachta/Seed-VC` の重みは **GPL-3.0** です。本リポジトリには Seed-VC のコードや重みを同梱せず、
  セットアップ時に取得した別ディレクトリの upstream を別プロセスとして起動します。
  Seed-VC を含む形で再配布する場合は GPL-3.0 の条件に従ってください。
- RMVPE (MIT)、CAMPPlus (Apache-2.0)、BigVGAN (MIT)、Whisper (Apache-2.0) はそれぞれのライセンスに従います。
- Irodori-TTS のモデルは各モデルカードのライセンスに従います。
- 変換に使う歌声・参照音声の権利(著作権、肖像・声の利用許諾など)は利用者が確認してください。
