# Irodori-TTS BAT Launchers

このファイルは、Irodori-TTS 直下にある各 `.bat` が何を起動するのかをまとめたものです。

## 一覧

| bat | 起動/実行するもの | 用途 | URL/ポート |
| --- | --- | --- | --- |
| `setup.bat` | `uv sync` | `.venv` 作成・依存関係インストール。WebUIやAPIは起動しません。 | なし |
| `run.bat` | 起動メニュー | Reference Audio WebUI / VoiceDesign WebUI を選択して起動します。 | 選択先による |
| `run_webui_reference_audio.bat` | `gradio_app.py` | 参照音声を使う通常Irodori-TTS WebUIを起動します。 | `http://localhost:7860` |
| `run_webui_voice_design.bat` | `gradio_app_voicedesign.py` | captionで声質・話し方を指定するVoiceDesign WebUIを起動します。 | `http://localhost:7861` |
| `run_irodori_wrapper_server.bat` | `irodori_wrapper_server.py` | SBV2互換のバックエンドAPIサーバーを起動します。GUIバッチTTSなどからHTTPで使う用途です。 | `http://127.0.0.1:6000` |

## 共通の保存先

各batは、起動時にIrodori-TTSのプロジェクトディレクトリへ移動してから実行します。そのため、別ディレクトリからbatを呼び出しても、モデルや出力先は基本的にこのIrodori-TTSディレクトリ内に収まります。

| ディレクトリ | 内容 |
| --- | --- |
| `.venv/` | Python仮想環境 |
| `model_data/` | HuggingFaceモデル、codec、tokenizer、各種キャッシュ |
| `outputs/` | 生成音声、学習checkpoint、ログ |
| `data/` | manifest、latentなどの学習用データ |

## `setup.bat`

依存関係をインストールするための初期セットアップ用batです。

主な処理:

- `uv` がPATH上にあるか確認します。
- `.venv` をIrodori-TTS直下に作成/更新します。
- `model_data/`, `outputs/`, `data/` などの保存先を作成します。
- `uv sync` を実行して依存パッケージを同期します。

WebUIやAPIサーバーは起動しません。最初に1回、または依存関係を更新したいときに使います。

## `run.bat`

WebUIを選択して起動するためのメニューbatです。

表示される選択肢:

- `1`: Reference Audio WebUI
- `2`: VoiceDesign WebUI

これはランチャーなので、バックエンドAPI用の `irodori_wrapper_server.py` は起動しません。

## `run_webui_reference_audio.bat`

参照音声を使う通常Irodori-TTSのGradio WebUIを起動します。

内部で実行するコマンド:

```bat
uv run python gradio_app.py --server-name 0.0.0.0 --server-port 7860
```

用途:

- ブラウザ上でIrodori-TTSを試す
- 参照音声を指定して声質や話し方を寄せる
- `Aratako/Irodori-TTS-500M-v2` 系のReference Audioモデルを使う

アクセス先:

```text
http://localhost:7860
```

## `run_webui_voice_design.bat`

VoiceDesign用のGradio WebUIを起動します。Reference Audioモデルとは違い、参照音声ではなくcaptionで声質や話し方を指定します。

内部で実行するコマンド:

```bat
uv run python gradio_app_voicedesign.py --server-name 0.0.0.0 --server-port 7861
```

用途:

- ブラウザ上でVoiceDesignモデルを試す
- 「落ち着いた女性の声」「近い距離感」などのcaptionでスタイルを指定する
- `Aratako/Irodori-TTS-500M-v2-VoiceDesign` 系のモデルを使う

アクセス先:

```text
http://localhost:7861
```

## `run_irodori_wrapper_server.bat`

SBV2互換のバックエンドAPIサーバーを起動します。ブラウザ操作用ではなく、`gui_batch_tts.py` のような外部ツールからHTTPで叩くためのものです。

内部で実行するコマンド:

```bat
uv run python irodori_wrapper_server.py %*
```

既定のURL:

```text
http://127.0.0.1:6000
```

主なAPI:

- `GET /models/info`
- `POST /models/refresh`
- `GET` / `POST /voice`
- `GET /status`
- `POST /jobs/cancel`
- `POST /jobs/resume`
- `GET /jobs/status`
- `POST /shutdown`

参照音声を1つ指定する例:

```bat
run_irodori_wrapper_server.bat --reference-audio path\to\voice.wav
```

普通・怒・哀・楽の参照音声を自動選択する例:

```bat
run_irodori_wrapper_server.bat --reference-audio-dir path\to\refs
```

参照音声ディレクトリ例:

```text
refs/
  normal.wav
  angry.wav
  sad.wav
  happy.wav
```

または:

```text
refs/
  普通.wav
  怒.wav
  哀.wav
  楽.wav
```

複数サーバーを起動する場合は、ポートを分けます。

```bat
run_irodori_wrapper_server.bat --port 6000 --reference-audio-dir refs_a
run_irodori_wrapper_server.bat --port 6001 --reference-audio-dir refs_b
```

同じポートで複数起動はできません。また、複数プロセスを起動するとモデルもプロセスごとに読み込まれるため、VRAM/RAM使用量も増えます。

## どれを使うべきか

| やりたいこと | 使うbat |
| --- | --- |
| 初期セットアップしたい | `setup.bat` |
| 手動でWebUIを選んで起動したい | `run.bat` |
| 参照音声つきでブラウザから試したい | `run_webui_reference_audio.bat` |
| caption指定でブラウザから試したい | `run_webui_voice_design.bat` |
| `gui_batch_tts.py` など外部ツールから音声生成したい | `run_irodori_wrapper_server.bat` |

`gui_batch_tts.py` 連携で使う場合は、基本的に `run_irodori_wrapper_server.bat` を使います。WebUI系batはブラウザ操作向けであり、SBV2互換バックエンドAPIとして使うものではありません。
