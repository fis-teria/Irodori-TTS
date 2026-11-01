# Irodori-TTS V4 / V4.1 Windows セットアップ

このパッケージは、Irodori-TTSのソース、モデル選択画面、Windows用セットアップ・起動ファイルをまとめたものです。モデル重みや仮想環境は含まれません。初回セットアップ時に選んだモデルと依存パッケージを取得します。

## 必要なもの

- Windows x64
- Python 3.10以上（`python` コマンドをPATHから実行できること）
- Git for Windows（`git` コマンドをPATHから実行できること）
- インターネット接続
- CUDA版を使う場合は対応するNVIDIA GPUとドライバー

Pythonは [python.org](https://www.python.org/downloads/windows/)、Gitは [Git for Windows](https://gitforwindows.org/) から用意してください。GPUを使わない場合はCPU版セットアップを選べます。

## 初回セットアップ

1. ZIPをすべて展開し、中の `Irodori-TTS_V4` フォルダを開きます。
2. `setup.bat` を実行します。
3. 画面で **V4.1 Small・V4 Large・V4.1 Anime** を選択します。複数選択できます。
4. 「セットアップ開始」を押し、処理が完了するまで待ちます。
5. `run.bat` からWebUIを起動します。

初期選択はSmall、実行環境の既定はCUDA 12.8です。導入済みのファイルは再利用します。選択をすべて外すと開始ボタンは無効になります。「キャンセル」や画面を閉じる操作では、セットアップは始まりません。

| モデル | WebUIのチェックポイント欄に指定するID | 重みの容量 |
|---|---|---:|
| V4.1 Small | `Aratako/Irodori-TTS-v4.1-Small` | 約3.06GB |
| V4 Large | `Aratako/Irodori-TTS-v4-Large` | 約13.15GB |
| V4.1 Anime | `phasefield-audio/Irodori-TTS-v4.1-Anime` | 約3.06GB |

容量は通常精度版の重みのみの値です。このほかにPyTorch、コーデック、共有版FFmpeg、キャッシュなどの保存領域が必要です。Smallだけでも参照音声によるクローンとVoiceDesignの両方に対応します。WebUIの初期モデルはSmallで、モデルの追加ダウンロードだけでは初期選択は変わりません。

LargeはRTX 4080で **Model Device＝CUDA / Model Precision＝BF16 / Codec Device＝CPU** の40ステップ生成を確認しています。必要なメモリは文章や同時生成数でも変わります。

## 起動ファイル

| ファイル | 用途 |
|---|---|
| `run.bat` | WebUIの選択メニュー |
| `run_webui_reference_audio.bat` | 参照音声WebUI（http://127.0.0.1:7860） |
| `run_webui_voice_design.bat` | VoiceDesign WebUI（http://127.0.0.1:7861） |
| `run_python.bat` | この環境でCLIや学習スクリプトを実行 |

## 保存先

次の保存先は、**このパッケージを展開したフォルダの中**に作られます。

| 対象 | 保存先 |
|---|---|
| Python仮想環境 | `.venv` |
| モデル・トークナイザー・コーデック・透かしモデル | `model_data\huggingface\hub` |
| モデルの取得履歴・リビジョン | `model_data\installed_models.json` |
| 追加取得するPython | `model_data\python` |
| uv・共有版FFmpeg | `.tools` |
| uv・pip・コンパイル・一時キャッシュ | `.cache` |
| 学習データ | `data` |
| 生成音声・学習出力 | `outputs` |
| WebUIの音声 | `outputs\gradio_reference_audio` / `outputs\gradio_voicedesign` |

保存先の設定は、このアプリのプロセス内で行います。ユーザーやシステムの環境変数は変更しません。Windows CA証明書はセットアップ先のPCから作成します。

## コマンドからのセットアップ・実行

展開した `Irodori-TTS_V4` フォルダでPowerShellを開いて実行します。

```powershell
.\setup.bat -Backend cpu                 # CPU版。モデル選択画面が開きます
.\setup.bat -SkipModels                  # 環境だけを同期します
.\setup.bat -ModelRepo Aratako/Irodori-TTS-v4-Large
.\setup.bat -ModelRepo phasefield-audio/Irodori-TTS-v4.1-Anime
.\run_python.bat scripts\check_environment.py
.\run_python.bat train.py --help
```

`-ModelRepo` または `-SkipModels` を指定した場合はモデル選択画面を開きません。モデルを複数選んでも、共通のコーデックと透かしモデルはまとめて取得します。

推論例:

```powershell
.\run_python.bat infer.py --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small --text "音声合成のテストです。" --no-ref --model-precision bf16 --output-wav outputs\sample.wav
```

この例はCUDA向けです。CPU版では `--model-device cpu --codec-device cpu --model-precision fp32` を指定してください。セットアップ済み環境の起動には `run_python.bat` を使うと、選んだPyTorch環境の再同期を避けられます。

## この配布版の変更

- モデルを複数選択できる日本語セットアップ画面。
- モデル・仮想環境・キャッシュ・出力を展開先フォルダ内に保存。
- SHA256を検証して専用uvと共有版FFmpegを取得。
- Windowsの大容量モデル読み込みで `safetensors` 0.8.0の `pread` を使用。
- Windowsの大容量・非量子化・事前学習エンコーダ付きモデルでは、推論用モデルの重複確保を抑え、指定精度へ変換してからGPUに転送。
- RF/LoRA学習・再開時に、checkpoint同梱のtokenizerと埋め込み本文エンコーダー設定を利用。モデル識別用の元のrepo・revisionは維持。
- 学習前にキャプションの有無を確認し、キャプションがないバッチのトークン化・エンコーダー計算を省略。
- 完全に凍結された事前学習済み本文・キャプションエンコーダーの特徴量計算を `no_grad` にし、不要な入力勾配と再計算を省略。学習対象のprojector・LoRA・duration predictorの勾配は維持。
- V4系の本文をバッチ内の最長トークン数まで動的padding。切り詰め上限・有効トークン・duration特徴量の基準は維持。
- mask付きSDPAを **Efficient Attention → cuDNN → Math** の優先順に変更。対応しない形状・dtypeにはfallbackを維持。
- 参照なしバッチで、凍結・非確率的な話者エンコーダーの計算を省略。
- 凍結・非確率的な本文backboneの特徴をプロセス内で再利用。学習するprojector・LoRA・duration predictorの計算は継続。
- 最適化の回帰テストを `tests` に同梱。

これらは次の学習・推論プロセス起動から適用されます。既存データやcheckpointの変換は不要です。数値丸めや確率的処理により、学習重み・生成音声の完全一致は保証しません。変更の適用範囲と比較結果は `docs\OPTIMIZATION_NOTES.md` を参照してください。DiTの勾配チェックポイント設定は変更していません。

セットアップは同梱の `uv.lock` を `--frozen` で使用します。単一話者LoRAの学習データや学習済みアダプターは含まれません。学習条件については `README.md` と `docs\parameters.md` を参照してください。

## 20261026-r1の学習開始処理

通常のRF/LoRA学習とcheckpointからの再開で、ベースcheckpointに同梱されたtokenizerと埋め込みtext encoder configを利用します。モデルの互換性比較に使うrepo・revisionは元の値を保ち、読み込み用のローカルパスだけを分けて扱います。同梱tokenizerがない場合はrepoからの読み込みへ戻ります。MeanFlowではteacher側の素材を優先します。

この変更の回帰テスト7件を、既存の最適化テストとともに `tests` へ収録しています。ZIPにはモデルやtokenizerの実データ、学習データ、学習済みLoRAを含めません。後述の処理時間・音声評価は2026-10-25に行った最適化比較の結果です。

## 話者エンコーダー省略・本文特徴キャッシュ

適用条件を満たす学習では、次の設定が既定で有効です。既存のYAMLに項目がなくても、次の学習起動から適用されます。

```yaml
train:
  skip_absent_speaker_encoder: true
  cache_frozen_text_features: true
  text_feature_cache_max_mb: 256
```

参照あり・なしが混在するバッチや話者エンコーダーを学習する設定では、話者側は通常計算を使います。本文backboneの学習や確率的dropout、`torch.compile`、MeanFlowなど再利用に適さない条件では、本文キャッシュを使いません。キャッシュは上限256MiBのプロセス内LRUで、checkpointには保存せず、学習起動ごとに作り直します。

既存の学習コマンドに `--no-skip-absent-speaker-encoder` と `--no-cache-frozen-text-features` を追加すると、個別に従来計算へ戻せます。上限は `--text-feature-cache-max-mb`（MiB）で変更でき、`0` でもキャッシュを無効にできます。DiTの設定は独立しており、速度優先は `--no-gradient-checkpointing`、VRAM節約は `--gradient-checkpointing` で切り替えます。

V4.1 Small・RTX 4080・BF16・batch32/grad accum1・参照なし83件の各2,048step比較では、話者エンコーダー省略で平均6.55%、本文キャッシュ追加でさらに19.21%、合計24.51%短縮しました。Efficient Attention優先・DiT checkpointing OFFを共通にした結果です。音声の自動評価では明確な悪化は検出されませんでしたが、RF lossは約1.4%上昇しており、重みや聴感の同等性を保証する結果ではありません。Large/Animeへの一般化は未検証です。条件・数値・制限は `docs\OPTIMIZATION_NOTES.md` に記載しています。

## 出典とライセンス

元のコードは [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) です。コードのMITライセンスと著作権表示は同梱の `LICENSE` に保存しています。このZIPに外部モデル重みや外部ツールの実行ファイルは含まれません。取得するモデル・ツールにはそれぞれの配布元の利用条件が適用されるため、各配布元の説明も確認してください。

各ファイルのSHA256は `RELEASE_MANIFEST.json`、ZIP全体のSHA256はZIPと一緒に配布する `.zip.sha256` に記載されています。
