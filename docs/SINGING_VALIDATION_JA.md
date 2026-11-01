# 歌唱変換プロトタイプ 検証記録

実施日: 2026-11-01 / 環境: Windows 10 Pro, NVIDIA GeForce RTX 4080 16GB, PowerShell。
音声と詳細な記録 (`metadata.json`, `analysis.json`, `summary_ja.md`, ログ) は Git 管理外の `outputs\singing\` にあります。

## 使用したもの

- Irodori-TTS: このチェックアウト、`Aratako/Irodori-TTS-v4.1-Small` @ `2b28324d`、codec `Semantic-DACVAE-Japanese-32dim` @ `47376ee2`、`.venv`(既存 `uv.lock`)
- Seed-VC: `Plachtaa/seed-vc` @ `51383efd`、`Plachta/Seed-VC` @ `257283f9` の
  `DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth`(SHA256 `883d19be…163f9`)、
  `configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml`、`.singing\venv`(torch 2.4.0+cu121, transformers 4.46.3)
- ソース歌声: 利用者が指定したボーカル分離済み WAV(44.1 kHz ステレオ, 183.2 秒)。読み取りのみで、コピーと出力は Git 管理外。
- 参照音声: Irodori v4.1-Small、キャプション「落ち着いた、近い距離感の女性話者」、`--no-ref`、seed 0 で生成(8.12 秒, 48 kHz)。
- Seed-VC 設定(全実行共通、+2 半音の実行以外は既定値): f0-condition=true, auto-f0-adjust=false, semitone=0,
  length-adjust=1.0, diffusion-steps=30, inference-cfg-rate=0.7, fp16=true, オフライン (`HF_HUB_OFFLINE=1`)

## 実行結果(すべて実 GPU 実行)

| 実行 | 経路 | ソース | 出力 | 長さ比 | RMS / peak / クリップ率 | F0 中央値誤差 | ±50 cent | オクターブ誤り | F0 相関 | 有声 IoU | Seed-VC 時間 / ピーク VRAM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | CLI, Irodori 生成参照 | 15–35 秒 (20 秒) | 44.1 kHz, 19.99 秒 | 0.9996 | -14.0 dBFS / 0.99 / 0 | 10 cent | 92.1% | 0.0% | 0.994 | 0.89 | 31.6 秒 / 2.7 GB |
| B | CLI, 既存 WAV 参照, 空白・日本語を含むパス | 全体 183.2 秒 | 44.1 kHz, 183.22 秒 | 0.9999 | -12.4 dBFS / 1.00 / 0.002% | 10 cent | 90.9% | 1.8% | 0.953 | 0.88 | 35.7 秒 / 3.1 GB |
| C | Gradio UI (gradio_client 経由), 既存 WAV 参照 | 15–25 秒 | 44.1 kHz, 10.00 秒 | 0.9996 | -14.2 dBFS / 0.95 / 0 | 10 cent | 88.5% | 0.0% | 0.989 | 0.93 | 14.1 秒 / 2.4 GB |
| D | CLI, semitone +2 | 15–25 秒 | 44.1 kHz, 10.00 秒 | 0.9996 | -12.7 dBFS / 1.00 / 0 | 10 cent(+200 cent 補正後) | 90.8% | 0.0% | 0.995 | 0.87 | 15.4 秒 / 2.4 GB |

- 全実行で自動チェック(44.1 kHz、有限値、非無音、長さ ±5%)に合格。
- D: F0 中央値がソース 327.6 Hz → 出力 367.7 Hz(+2 半音の理論値 367.7 Hz と一致)。
- A の Irodori 参照生成は 34 秒(モデル読み込み込み)。Irodori と Seed-VC は別プロセスで順に実行。
- F0 は librosa pYIN(16 kHz, 10 ms)でソースと出力をフレーム位置そのままで比較した値。示すのは**この対応付けのもとでの音高の一致**だけです。
  時間方向のずれが無いことの証明にはならず、発音・立ち上がりのタイミングや長い区間でのずれ(ドリフト)は別の確認と試聴が必要です。
- Seed-VC 時間はモデル読み込みを含み、upstream 表示の RTF は A で 0.21。

## その他に確認したこと

- 単体テスト `tests/test_singing_pipeline.py`(26 件, unittest, GPU 不要)合格: 参照指定の排他、存在しないファイル、F0 条件なし指定の拒否
  (話し声モデルへの暗黙の切り替えなし)、空白・日本語パスが 1 引数のまま渡ること、歌唱既定値、子プロセスの失敗/起動不可、
  JSON 設定ファイル(JSON の offline / f0_report は `--online` / `--no-f0-report` 指定時のみ上書き)、UTF-8 テキストファイル、
  Seed-VC config の検査(upstream の歌唱 preset、44.1 kHz BigVGAN 以外や不正な vocoder 設定の拒否)、
  出力先と入力の衝突(同一パス・ハードリンク)を書き込み前に拒否し入力のバイト列が不変であること、
  実行ごとの一時フォルダで既存の `seedvc_raw\*.wav` などを残すこと、壊れた/想定外のレポートでも metadata に failed が残ること、
  F0 比較(length-adjust ≠ 1 と auto-f0-adjust では比較不可の理由を記録、平坦な F0 でも NaN を出さない)。
- `run_singing.bat` 経由で存在しない入力を指定すると、モデルを読み込まずに理由を列挙して終了コード 1。
- `setup_singing.ps1 -SeedFrom <既存ワークスペース>` を構築済みの状態で再実行し、終了コード 0(環境・モデルを再利用、Seed-VC 資産はオフライン確認で完了)。
- 実行時は既定でオフライン(`HF_HUB_OFFLINE=1`)。上記 A–D はすべてオフラインで実行。
- `python -m compileall singing_vc` 成功。
- 出力処理の修正後に短いスモーク 1 回(A の `source.wav` 先頭 5 秒 + A の `reference.wav`、既存 WAV 参照、CLI):
  succeeded、44.1 kHz・4.99 秒、Seed-VC 19.4 秒 / ピーク VRAM 2.3 GB。一時フォルダ `seedvc_raw_*` は移動後に削除され、
  入力 2 ファイルの SHA256 は実行前後で一致。
  (最初の試行は Git Bash のパイプへ出力したため、進捗表示の文字を cp932 で書けず親プロセスが停止し、metadata に failed が記録されました。
  最終修正で CLI の標準出力・標準エラーに文字の置換処理を追加しました。cp932 で表現できない進捗表示は置換し、変換を継続します。
  cp932 ストリームで進捗を表示する回帰テストも合格しています。完全な UTF-8 ログは実行フォルダの `logs` に保存されます。)

## 未検証・注意点

- **試聴による評価は未実施です。** 声がターゲットに似ているか、歌唱として自然か、ノイズやアーティファクトの有無は判断していません。
  上記の数値は技術的な動作(サンプルレート、長さ、非無音、音高の追従)だけを示します。
- 参照音声が Irodori の合成音声であることによる品質への影響は未評価です。
- 出力は upstream のまま正規化しないため、B では 0.002% のサンプルが ±1.0 付近に達しました(クリップ率として記録)。
- まっさらな環境からの初回セットアップ(Seed-VC の clone から)と、ネットワーク無しでのセットアップ再実行は未検証です
  (この検証では Seed-VC の clone と資産取得を個別手順で実施し、その後セットアップスクリプトの再実行で確認)。
- 伴奏入りの音源、LoRA/ローカル checkpoint/話者参照 WAV を使った Irodori 生成は、コマンド組み立てのテストのみで実モデル実行は未実施。
- 長いソースの F0 分析(pYIN)は CPU で遅く、183 秒で約 170 秒かかりました。`--no-f0-report` で省略できます。
