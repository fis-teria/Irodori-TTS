# Irodori-TTS Training Model Types

このメモは、Irodori-TTS の学習方法がそれぞれどの種類のモデルを作るのかを整理したものです。

## 結論

Irodori-TTS で実用上分けて考えるべき学習系統は、主に次の3つです。

| 系統 | 主なconfig | 作るモデル | 推論時の使い方 |
| --- | --- | --- | --- |
| Reference Audioモデル | `configs/train_500m_v2.yaml` | テキスト + 参照音声で声質/話し方を条件付けするモデル | `infer.py --ref-wav ...` / `gradio_app.py` / `irodori_wrapper_server.py` |
| VoiceDesignモデル | `configs/train_500m_v2_voice_design.yaml` | テキスト + captionで声質/話し方を指定するモデル | `infer.py --caption ... --no-ref` / `gradio_app_voicedesign.py` |
| LoRA微調整モデル | `configs/train_500m_v2_lora.yaml` または `configs/train_500m_v2_voice_design_lora.yaml` | 既存モデルに対する軽量adapter、または変換後の統合済みsafetensors | ベースモデル系/VoiceDesign系のどちらをLoRA化したかに従う |

`torchrun` によるMulti-GPU学習は、モデル種別ではなく実行方法の違いです。作られるモデルの種類は、指定したconfigに従います。

## 1. Reference Audioモデル

主なconfig:

```bash
configs/train_500m_v2.yaml
```

この系統は、Irodori-TTS の通常版です。モデルは次の条件を使って音声latentを生成するように学習します。

- 読み上げるテキスト
- 参照音声から作ったDACVAE latent

学習後のモデルは、推論時に `--ref-wav` または `--ref-latent` を渡すことで、その参照音声の声質や話し方に寄せて合成します。現在追加している `irodori_wrapper_server.py` の voice_reference_audio 的な使い方はこの系統です。

`speaker_id` 付きmanifestを使う場合、このモデルは「同じspeaker_idの別発話」を参照音声latentとしてサンプリングしやすくなります。これにより、モデルは「同一話者の別音声を参照してターゲット音声を生成する」訓練を受けます。

## 2. VoiceDesignモデル

主なconfig:

```bash
configs/train_500m_v2_voice_design.yaml
```

この系統は、参照音声ではなくcaptionで声質や話し方を指定するモデルです。configで `use_caption_condition: true` になり、コード上では speaker/reference branch が無効化されます。

学習データには、音声とテキストに加えてcaptionを含めます。

```json
{"text": "こんにちは", "caption": "落ち着いた女性の声で、近い距離感", "latent_path": "data/latents/00001.pt"}
```

推論時は参照音声ではなく `--caption` でスタイルを指定します。`speaker_id` はmanifestに含めてもよいですが、VoiceDesign系では参照音声条件としては使われません。READMEにある通り、VoiceDesign path は `text + caption` を学習するものです。

## 3. LoRA微調整モデル

主なconfig:

```bash
configs/train_500m_v2_lora.yaml
configs/train_500m_v2_voice_design_lora.yaml
```

LoRAは、既存の公開weightや手元のcheckpointをベースにして、一部の線形層などにadapterを追加して微調整する方法です。フルモデル全体を更新するより軽く、少量データで既存モデルの性質を変えたい場合に向いています。

Reference AudioモデルをLoRA化する場合は、推論時も参照音声を使うモデルになります。VoiceDesignモデルをLoRA化する場合は、推論時もcaptionを使うモデルになります。

LoRA学習中のcheckpointはadapter-onlyのディレクトリとして保存されます。`convert_checkpoint_to_safetensors.py` で変換すると、ベースモデルにLoRAをmergeした推論用 `.safetensors` として扱えます。

## `speaker_id` 付きmanifestの意味

READMEの次の記述:

```text
To include `speaker_id` in the manifest (for speaker-conditioned training):
```

これは、SBV2のように「speaker_idをモデル内の話者番号として学習し、推論時にspeaker_idを選ぶ」という意味ではありません。

Irodori-TTS の `speaker_id` は、主に学習時の参照音声ペア作成に使われます。`irodori_tts/dataset.py` では、同じ `speaker_id` を持つサンプル群から別発話を選び、そのlatentを `ref_latent` として渡します。つまり、モデルに「この参照音声と同じ話者/スタイルで、別のテキストを読む」能力を学ばせるためのグループ情報です。

同じ `speaker_id` のサンプルが複数ある場合:

- ターゲット音声: 今回生成対象として学習する音声latent
- 参照音声: 同じspeaker_idの別サンプルの音声latent
- 目的: 参照音声から話者性/スタイルを抽出して、ターゲット音声を生成する

同じ `speaker_id` の別サンプルがない場合、参照latentはターゲット自身になりやすく、zero-shot参照音声モデルとしての訓練効果は弱くなります。Reference Audioモデルを作るなら、同じ話者につき複数発話を用意するのが重要です。

## SBV2のようなモデル学習との差異

SBV2系の典型的な多話者学習では、`speaker_id` はモデル内の話者埋め込み、つまり `spk2id` のような固定スロットに対応します。推論時は `speaker_id` や `speaker_name` を選ぶことで、学習済み話者の声を出します。基本的には、その話者が学習データに含まれていることが前提です。

一方、Irodori-TTS の Reference Audioモデルでは、`speaker_id` は推論時に選ぶ固定話者スロットではありません。推論時に必要なのは、speaker番号ではなく参照音声です。モデルは参照音声から条件を取り、学習済みspeaker_id一覧から声を選ぶのではなく、入力された参照音声に寄せて生成します。

| 観点 | SBV2系の典型 | Irodori-TTS Reference Audio |
| --- | --- | --- |
| `speaker_id` の役割 | モデル内の話者ID/話者embedding | 学習時に同一話者の参照音声を選ぶためのグループID |
| 推論時の指定 | `speaker_id` / `speaker_name` / `style_id` | `ref-wav` / `ref-latent` |
| 未学習話者への対応 | 基本的には弱い | 参照音声によるzero-shot寄せが主目的 |
| 学習で増えるもの | 話者embeddingや話者ID対応 | 参照音声条件付け能力 |
| GUI側の話者リスト | 学習済みspeakerを列挙する設計になりやすい | wrapper側で参照音声ファイルをspeaker風に見せる必要がある |

## 今回のwrapperとの関係

`irodori_wrapper_server.py` は、SBV2互換の `/models/info` や `/voice` を提供するため、参照音声ファイルをSBV2のspeaker/styleのように見せています。ただし内部的には、Irodori-TTSのReference Audioモデルへ `text + reference_audio` を渡しているだけです。

そのため、`normal.wav`, `angry.wav`, `sad.wav`, `happy.wav` のような参照音声を用意して自動切替する設計は、SBV2の複数speaker/style学習というより、「複数の参照音声をリクエストごとに選ぶ」設計です。
