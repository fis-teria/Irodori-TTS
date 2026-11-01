"""Minimal Gradio UI for the singing prototype (runs in the Irodori venv)."""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from singing_vc.pipeline import DEFAULT_OUTPUT_ROOT, PipelineError, SingingOptions, run_pipeline

DEFAULT_TEXT = "こんにちは。今日は少しだけ、新しい歌の練習をしてみようと思います。よろしくお願いします。"
DEFAULT_CAPTION = "落ち着いた、近い距離感の女性話者"


def _blank(value):
    return None if value is None or str(value).strip() == "" else value


def convert(mode, source, source_start, source_duration, reference, text, caption, checkpoint, lora,
            speaker, seed, steps, cfg_rate, f0_condition, auto_f0, semitone, length_adjust, fp16, output_dir):
    """Generator: streams the log, then yields reference/converted audio and metadata."""
    use_tts = mode == "Irodori で生成"
    options = SingingOptions(
        source_wav=source or "",
        reference_wav=None if use_tts else _blank(reference),
        tts_text=_blank(text) if use_tts else None,
        tts_caption=_blank(caption),
        tts_checkpoint=_blank(checkpoint),
        tts_lora_adapter=_blank(lora),
        tts_speaker_wav=_blank(speaker),
        tts_seed=None if seed is None or seed < 0 else int(seed),
        source_start=float(source_start or 0),
        source_duration=float(source_duration) if source_duration else None,
        diffusion_steps=int(steps), inference_cfg_rate=float(cfg_rate),
        f0_condition=bool(f0_condition), auto_f0_adjust=bool(auto_f0),
        semitone_shift=int(semitone), length_adjust=float(length_adjust), fp16=bool(fp16),
        output_dir=_blank(output_dir) or str(DEFAULT_OUTPUT_ROOT),
    )
    lines: list[str] = []
    events: queue.Queue = queue.Queue()

    def work():
        try:
            events.put(("ok", run_pipeline(options, log=lambda line: events.put(("log", line)))))
        except PipelineError as exc:
            events.put(("error", str(exc)))
        except Exception as exc:  # surface unexpected failures in the UI instead of a stack trace
            events.put(("error", f"{type(exc).__name__}: {exc}"))

    threading.Thread(target=work, daemon=True).start()
    while True:
        kind, payload = events.get()
        if kind == "log":
            lines.append(payload)
            yield None, None, "\n".join(lines[-200:]), ""
            continue
        if kind == "error":
            lines.append(f"ERROR: {payload}")
            yield None, None, "\n".join(lines[-200:]), ""
            return
        run_dir = Path(payload["run_dir"])
        summary = {k: payload.get(k) for k in ("status", "checks", "run_dir")}
        yield (str(run_dir / "reference.wav"), str(run_dir / "converted.wav"),
               "\n".join(lines[-200:]), json.dumps(summary, indent=2, ensure_ascii=False))
        return


def build():
    import gradio as gr

    with gr.Blocks(title="Irodori → Seed-VC 歌唱変換 (prototype)") as demo:
        gr.Markdown("## Irodori 参照音声 → Seed-VC v1 歌唱変換 (44.1 kHz, F0 条件付き)\n"
                    "ソースは伴奏を除いたボーカルのみの WAV を指定してください。モデルは順番に別プロセスで実行されます。")
        with gr.Row():
            with gr.Column():
                source = gr.Audio(label="ソース歌声 (ボーカルのみ)", type="filepath")
                with gr.Row():
                    source_start = gr.Number(label="開始秒", value=0)
                    source_duration = gr.Number(label="長さ秒 (空=全部)", value=None)
                mode = gr.Radio(["Irodori で生成", "既存 WAV"], value="Irodori で生成", label="ターゲット参照音声")
                reference = gr.Audio(label="既存の参照 WAV", type="filepath")
                text = gr.Textbox(label="Irodori 読み上げテキスト", value=DEFAULT_TEXT, lines=2)
                caption = gr.Textbox(label="キャプション (声の指定)", value=DEFAULT_CAPTION)
                with gr.Accordion("Irodori 詳細", open=False):
                    checkpoint = gr.Textbox(label="ローカル checkpoint パス (空=v4.1-Small)")
                    lora = gr.Textbox(label="LoRA アダプタ フォルダ")
                    speaker = gr.Textbox(label="話者参照 WAV パス (空=--no-ref)")
                    seed = gr.Number(label="seed (-1=ランダム)", value=0, precision=0)
            with gr.Column():
                with gr.Accordion("Seed-VC 設定", open=True):
                    steps = gr.Slider(1, 100, value=30, step=1, label="diffusion steps")
                    cfg_rate = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="inference cfg rate")
                    f0_condition = gr.Checkbox(value=True, label="f0 condition (歌唱モデルでは必須)")
                    auto_f0 = gr.Checkbox(value=False, label="auto f0 adjust")
                    semitone = gr.Slider(-24, 24, value=0, step=1, label="semitone shift")
                    length_adjust = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="length adjust")
                    fp16 = gr.Checkbox(value=True, label="fp16")
                output_dir = gr.Textbox(label="出力フォルダ", value=str(DEFAULT_OUTPUT_ROOT))
                run = gr.Button("変換", variant="primary")
                ref_out = gr.Audio(label="参照音声", type="filepath")
                converted = gr.Audio(label="変換結果", type="filepath")
                result = gr.Code(label="結果", language="json")
                log = gr.Textbox(label="ログ", lines=12, max_lines=30)
        run.click(convert, [mode, source, source_start, source_duration, reference, text, caption, checkpoint,
                            lora, speaker, seed, steps, cfg_rate, f0_condition, auto_f0, semitone,
                            length_adjust, fp16, output_dir], [ref_out, converted, log, result])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7862)
    args = parser.parse_args()
    build().queue().launch(server_name=args.server_name, server_port=args.server_port,
                           allowed_paths=[str(DEFAULT_OUTPUT_ROOT)])


if __name__ == "__main__":
    main()
