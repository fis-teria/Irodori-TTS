# Windows配布版の学習最適化（2026-10-25）

この配布版には以下の最適化を含みます。学習データ・モデル重み・学習済みLoRAは同梱していません。既存データやcheckpointの変換は不要で、新しいプロセスから適用されます。

## キャプションがないバッチ

学習開始時に、学習・検証データそれぞれのキャプション件数を表示します。キャプションがないバッチではトークン化とキャプションエンコーダー計算を省略します。キャプションがあるバッチの条件付けは維持し、混在データにも対応します。

## 凍結された事前学習済みエンコーダー

本文・キャプション共通のbackboneが完全に凍結されている場合、特徴量計算を `no_grad` で実行し、そのbackboneの勾配チェックポイントと入力勾配フックを無効にします。学習対象projector・LoRA・duration predictorの勾配は維持します。backbone自身を学習する設定では通常の勾配計算を使います。

起動ログは `Frozen pretrained text/caption backbone: no_grad forward`。DiT側の勾配チェックポイント設定は独立して維持します。

## 本文の動的padding

V4系の `text_encoder_type=pretrained` では、バッチ内で不要な本文末尾paddingをCPU側で取り除いてからGPUへ転送します。BOS・有効トークン・切り詰め上限・duration特徴量の正規化基準は維持し、空入力でも最低1列を残します。キャプションのpadding設定は従来どおりです。

起動ログは `Dynamic text padding enabled: batch-longest right-padding`。従来の固定長に戻す場合はCLIの `--no-dynamic-text-padding`、またはYAMLの `train.dynamic_text_padding: false` を使います。本文長の変化により `compile_model` の再コンパイルやGPUカーネルの選択が変わるため、計算量の減少がそのまま速度向上を保証するものではありません。

## Efficient Attention優先

mask付きSDPAは `Efficient Attention → cuDNN → Math` の優先順で実行します。本文・音声長が変わる学習時に観測されたcuDNN実行プラン構築の待ちを抑えるためです。Efficientが扱えない形状・dtypeのfallback、対応GPUでのFA3専用経路、maskなしSDPA、CPU経路は維持します。

RTX 4080 / PyTorch 2.10.0+cu128 / V4.1 Small / BF16 / batch32・grad accum1で、各1,024stepの比較を行いました。暖機後の平均は約3.2%短縮しましたが、後半512stepはほぼ同じでした。大きな初回待ちの減少が主な効果で、継続的な2倍速化を示す結果ではありません。

各方式で学習したLoRAから、学習に含まれない8文章×3生成seed・計24組の音声も比較しました。共通のFP32推論条件で、学習元との話者cosine平均は旧0.9087／新0.9114、UTMOS自然さ推定平均は旧2.878／新2.821でした。UTMOS差の95%区間は−0.135〜+0.017です。明確な話者指標の悪化は検出されませんでしたが、自然さには小さな低下傾向があり、人の聴感上の同等性を証明する結果ではありません。学習seedは各1回で、Large／Animeへの一般化も未確認です。

## DiTの勾配チェックポイント

この配布更新ではDiTの設定を変更しません。再計算を減らして速度を上げる場合は、その分だけVRAM使用量が増えます。既存のYAML設定は次の意味です。

CLIでは `--no-gradient-checkpointing` で無効、`--gradient-checkpointing` で有効にできます。本文・話者エンコーダーの最適化とは独立した設定です。

| `train.gradient_checkpointing` | `train.gradient_checkpointing_store_every` | 動作 |
|---|---:|---|
| `true` | `0` | 全DiTブロックでcheckpointを使う |
| `true` | `2` | 2ブロックごとに1個の活性値を保持し、半分だけcheckpointを使う |
| `false` | 任意 | DiTのcheckpointを使わない |

`store_every` は保存ファイルの頻度ではなく、逆伝播のために活性値を保持するブロック間隔です。値が大きいほど再計算が少なくなる設定ではありません。`store_every: 1` は全ブロックの活性値を保持します。

## 回帰テスト

セットアップ済みの環境で実行します。

```powershell
.\run_python.bat -m unittest discover -s tests -v
```

キャプションなし・混在・あり、凍結backboneと学習対象の勾配、動的padding、AttentionのCUDA backend・出力・勾配・LoRA更新・再読み込みを確認します。CUDA専用テストはGPUがない環境ではスキップされます。テストに含まれる小規模モデルの検証は、任意の長時間学習における音質保証ではありません。

## 無参照時の話者エンコーダー省略

`train.skip_absent_speaker_encoder` は既定で `true` です。CPU側でバッチ全件が参照なしと確認でき、話者エンコーダーとnormが凍結され、確率的dropoutがない場合に、話者特徴の計算を省略します。Attentionへ渡すcontext長は維持します。参照あり・なしの混在バッチ、話者エンコーダーを学習する設定、Speaker Inversion、直接指定の話者埋め込みは従来の経路を使います。

従来計算へ戻す場合はCLIの `--no-skip-absent-speaker-encoder`、またはYAMLの `train.skip_absent_speaker_encoder: false` を使います。モデル構造やcheckpoint形式を変更せず、次回の学習起動から適用されます。

## 凍結本文特徴のキャッシュ

`train.cache_frozen_text_features` は既定で `true`、`train.text_feature_cache_max_mb` は既定で `256`（MiB）です。凍結・非確率的な事前学習済み本文backboneの出力だけを、初出時に元のdtypeで保存して再利用します。本文token IDsとmaskからCPUでキーを作り、末尾paddingを除いて保存し、利用時のバッチ長へ再paddingします。projector・norm・LoRA・duration predictorと条件dropoutは毎回計算します。キャプション特徴はこのキャッシュの対象外です。

キャッシュはプロセス内・backboneインスタンスごとのLRUで、特徴テンソル本体の上限が256MiBです。checkpointには保存せず、training/eval・dtype・device・inference modeを区別します。重み再読み込み、device/dtype変更、unfreeze時は破棄します。backbone自身を学習する設定、backboneの確率的dropout、CPUキーと異なるmask、`torch.compile`、MeanFlowでは通常の計算を使います。本文の種類が多く上限を超えるデータでは、再計算が増えて短縮効果が小さくなります。

起動ログは `Frozen text feature cache enabled`、終了時はhit/miss・使用量を含む `Frozen text feature cache stats` です。無効化する場合はCLIの `--no-cache-frozen-text-features`、またはYAMLの `train.cache_frozen_text_features: false` を使います。上限はCLIの `--text-feature-cache-max-mb` でも変更でき、`0` で無効になります。UIから生成した従来の設定も、項目が未指定なら既定値を使います。

別のバッチ形状で計算した特徴の再利用には、BF16/TF32の丸め差が残ります。学習重みのビット一致や任意のデータ・モデルでの音質同等性を保証する機能ではありません。データやcheckpointの変換は不要です。

### ④→③の比較結果

V4.1 Small / RTX 4080 / BF16 / TF32 / batch32・grad accum1 / 参照・キャプションなし83件で、各方式を同じ初期重み・学習seed0から2,048step学習しました。Efficient Attention優先、DiT checkpointing OFF、本文動的padding、CUDA prefetchを共通で有効にしました。

| 方式 | 平均 秒/step | 前の段階から短縮 | PyTorch確保ピーク |
|---|---:|---:|---:|
| 変更前 | 0.452600 | — | 9.156GiB |
| ④ 話者エンコーダー省略 | 0.422955 | 6.55% | 9.159GiB |
| ④＋③ 本文特徴キャッシュ | 0.341687 | 19.21% | 9.160GiB |

暖機64stepと検証・保存直後の待ちを除く1,977stepの平均です。④＋③は変更前から24.51%短縮し、後半1,024stepでも26.04%短縮しました。バッチ内容・順序・形状と64stepごとのCPU/CUDA乱数状態は一致しました。③の特徴キャッシュは約1.48MiB、上限による破棄は0件でした。同一プロセスで方式を交互に切り替えるforward＋backwardの補助測定でも、両機能による短縮を確認しました。

同じ評価経路での83件のRF MSEは、変更前0.207406、④0.207402、④＋③0.210217でした。③追加で約1.4%上昇しています。学習重みは完全一致していません。④の同一重み比較は出力・lossが一致し、GPU勾配の微差は同方式の反復でも確認しました。③では別のバッチ形状の特徴を再利用するとBF16の潜在出力に約1.51%差が出ましたが、TF32を切ったFP32では約0.000117%へ縮小しました。観測差の主因はGPU丸め差と考えられます。

各方式から未学習8文章×3生成seedの音声24本、計72本を共通のFP32・40step推論で生成しました。学習元12音声とのWavLM話者cosine平均は変更前0.944361／④0.946281／④＋③0.945545、UTMOS自然さ推定は3.230307／3.160183／3.234670、CERは1.540%／1.849%／1.984%でした。各比較で話者指標・UTMOS・CERの対応差の95%bootstrap区間は0を含み、今回の範囲では明確な悪化は検出されませんでした。CERには漢字・かなの表記差も含まれます。

学習seedは各方式1回です。生成seedは学習の反復ではなく、UTMOS・WavLMによる自動評価も人による聴感判定を代替しません。Large/Animeや別データへの一般化は未検証です。ソースsnapshot、各stepの計測値、比較用LoRA、音声、A/Bページは開発ワークスペースの `outputs/conditioning_optimizations/20261025-r1` に保存しており、配布ZIPには含めません。
