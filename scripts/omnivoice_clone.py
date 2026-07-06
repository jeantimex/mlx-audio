#!/usr/bin/env python3
"""
OmniVoice voice cloning with proper preprocessing workflow.

OmniVoice requires a specific workflow for voice cloning:
1. Preprocess reference audio (silence removal, RMS normalization)
2. Transcribe the PREPROCESSED audio (not the original)
3. Generate with both ref_tokens and ref_text

Usage:
    # Basic voice cloning
    python scripts/omnivoice_clone.py --ref_audio reference.wav \
        --text "Hello, this is my cloned voice!" --language english

    # Chinese with expression tags
    python scripts/omnivoice_clone.py --ref_audio yangmi.wav \
        --text "这真是太好笑了 [laughter] 我简直不敢相信。" --language chinese

    # Save preprocessed audio and transcript for reuse
    python scripts/omnivoice_clone.py --ref_audio reference.wav \
        --text "Hello!" --language english --save-preprocessed

    # Reuse previously saved preprocessed files
    python scripts/omnivoice_clone.py --ref_audio reference.preprocessed.wav \
        --ref_text "$(cat reference.preprocessed.txt)" \
        --text "Another sentence!" --language english --skip-preprocess
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="OmniVoice voice cloning with proper preprocessing"
    )
    parser.add_argument(
        "--ref_audio", "-r", required=True, help="Path to reference audio file"
    )
    parser.add_argument(
        "--ref_text",
        default=None,
        help="Transcript of reference audio (if already transcribed)",
    )
    parser.add_argument(
        "--text", "-t", required=True, help="Text to synthesize"
    )
    parser.add_argument(
        "--language",
        "-l",
        default="english",
        help="Language (english, chinese, french, etc.)",
    )
    parser.add_argument(
        "--output", "-o", default="output.wav", help="Output audio file"
    )
    parser.add_argument(
        "--play", "-p", action="store_true", help="Play audio after generation"
    )
    parser.add_argument(
        "--model",
        default="mlx-community/OmniVoice-bf16",
        help="OmniVoice model to use",
    )
    parser.add_argument(
        "--stt_model",
        default="mlx-community/Qwen3-ASR-0.6B-8bit",
        help="STT model for transcription",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing (use if ref_audio is already preprocessed)",
    )
    parser.add_argument(
        "--save-preprocessed",
        action="store_true",
        help="Save preprocessed audio and transcript for reuse",
    )
    parser.add_argument(
        "--num_steps", type=int, default=32, help="Number of generation steps"
    )
    parser.add_argument(
        "--duration", type=float, default=None, help="Target duration in seconds"
    )
    parser.add_argument(
        "--expression-tags",
        action="store_true",
        help="Auto-add expression tags to text",
    )
    parser.add_argument(
        "--expression-provider",
        choices=["simple", "anthropic", "mlx"],
        default="simple",
        help="Provider for expression tag insertion",
    )
    args = parser.parse_args()

    ref_audio_path = Path(args.ref_audio)
    if not ref_audio_path.exists():
        print(f"Error: Reference audio not found: {ref_audio_path}")
        sys.exit(1)

    # Load TTS model
    print(f"Loading TTS model: {args.model}")
    from mlx_audio.tts.utils import load_model as load_tts

    tts = load_tts(args.model)
    tokenizer = tts.audio_tokenizer

    if tokenizer is None:
        print("Error: Model does not have audio_tokenizer. Voice cloning not supported.")
        sys.exit(1)

    # Step 1: Preprocess reference audio (or load pre-tokenized)
    from mlx_audio.tts.models.omnivoice.utils import create_voice_clone_prompt
    from mlx_audio.audio_io import write as audio_write

    if args.skip_preprocess and args.ref_text:
        # Load audio directly without preprocessing
        print("Loading reference audio (skipping preprocessing)...")
        from mlx_audio.utils import load_audio

        ref_wav = load_audio(str(ref_audio_path), sample_rate=24000)
        ref_wav_mx = mx.array(ref_wav)
        if ref_wav_mx.ndim == 1:
            ref_wav_mx = ref_wav_mx[None, :, None]
        ref_tokens = tokenizer.encode(ref_wav_mx)[0]
        mx.eval(ref_tokens)
        ref_text = args.ref_text
    else:
        # Preprocess reference audio
        print("Preprocessing reference audio (silence removal, normalization)...")
        ref_tokens = create_voice_clone_prompt(
            str(ref_audio_path), tokenizer=tokenizer
        )
        mx.eval(ref_tokens)

        # Decode to audio for transcription
        print("Decoding preprocessed audio...")
        preprocessed = np.array(tokenizer.decode(ref_tokens).astype(mx.float32))

        # Save preprocessed if requested
        if args.save_preprocessed:
            preprocessed_path = ref_audio_path.with_suffix(".preprocessed.wav")
            audio_write(str(preprocessed_path), preprocessed, 24000)
            print(f"Saved preprocessed audio: {preprocessed_path}")

        # Transcribe if ref_text not provided
        if args.ref_text:
            ref_text = args.ref_text
            print(f"Using provided ref_text: {ref_text[:50]}...")
        else:
            # Save to temp file for transcription
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            audio_write(tmp.name, preprocessed, 24000)
            tmp.close()

            # Transcribe
            print(f"Transcribing with {args.stt_model}...")
            from mlx_audio.stt.utils import load_model as load_stt

            stt = load_stt(args.stt_model)

            # Map language for ASR
            asr_language = args.language.capitalize()
            if args.language.lower() == "chinese":
                asr_language = "Chinese"

            result = stt.generate(tmp.name, language=asr_language)
            ref_text = result.text
            os.unlink(tmp.name)

            print(f"Transcribed: {ref_text}")

            # Save transcript if requested
            if args.save_preprocessed:
                transcript_path = ref_audio_path.with_suffix(".preprocessed.txt")
                transcript_path.write_text(ref_text)
                print(f"Saved transcript: {transcript_path}")

            # Free STT model memory
            del stt
            mx.clear_cache()

    # Process expression tags if enabled
    text = args.text
    if args.expression_tags:
        from mlx_audio.tts.expression_tags import add_expression_tags

        original_text = text
        text = add_expression_tags(text, args.model, provider=args.expression_provider)
        print(f"\n\033[94mOriginal text:\033[0m {original_text}")
        if text != original_text:
            print(f"\033[92mWith expression tags:\033[0m {text}")
        else:
            print(f"\033[93mNo expression tags added\033[0m (no patterns matched)")

    # Step 2: Generate speech
    print(f"\nGenerating speech...")
    print(f"  Language: {args.language}")
    if not args.expression_tags:
        print(f"  Text: {text}")

    gen_kwargs = {
        "text": text,
        "language": args.language,
        "ref_tokens": ref_tokens,
        "ref_text": ref_text,
        "num_steps": args.num_steps,
    }
    if args.duration:
        gen_kwargs["duration_s"] = args.duration

    results = list(tts.generate(**gen_kwargs))

    if not results or results[0].audio is None or len(results[0].audio) == 0:
        print("Error: No audio generated")
        sys.exit(1)

    # Save output
    audio = np.array(results[0].audio)
    sample_rate = results[0].sample_rate
    audio_write(args.output, audio, sample_rate)
    print(f"\nSaved: {args.output}")
    print(f"Duration: {len(audio) / sample_rate:.2f}s")

    # Play if requested
    if args.play:
        print("Playing audio...")
        from mlx_audio.tts.audio_player import AudioPlayer

        player = AudioPlayer(sample_rate=sample_rate)
        player.queue_audio(mx.array(audio))
        player.wait_for_drain()
        player.stop()

    # Print reuse instructions
    if args.save_preprocessed:
        print("\n" + "=" * 60)
        print("To reuse this voice (faster, no preprocessing):")
        print("=" * 60)
        preprocessed_path = ref_audio_path.with_suffix(".preprocessed.wav")
        transcript_path = ref_audio_path.with_suffix(".preprocessed.txt")
        print(f"""
python scripts/omnivoice_clone.py \\
  --ref_audio {preprocessed_path} \\
  --ref_text "$(cat {transcript_path})" \\
  --text "Your new text here" \\
  --language {args.language} \\
  --skip-preprocess \\
  --play
""")


if __name__ == "__main__":
    main()
