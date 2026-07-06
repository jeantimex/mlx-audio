#!/usr/bin/env python3
"""
Chatterbox-Turbo TTS with voice profile support.

Generate speech using a pre-computed voice profile or reference audio.

Usage:
    # With a voice profile (fast, no audio processing needed)
    python scripts/chatterbox_tts.py --profile my_voice.profile --text "Hello world!"

    # With reference audio (slower, processes audio each time)
    python scripts/chatterbox_tts.py --ref_audio reference.wav --text "Hello world!"

    # Play audio immediately
    python scripts/chatterbox_tts.py --profile my_voice.profile --text "Hello!" --play

    # Save to file
    python scripts/chatterbox_tts.py --profile my_voice.profile --text "Hello!" -o output.wav
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Chatterbox-Turbo TTS with voice profile support"
    )
    parser.add_argument(
        "--text", "-t", required=True, help="Text to synthesize"
    )
    parser.add_argument(
        "--profile", "-p", help="Path to voice profile (.profile file)"
    )
    parser.add_argument(
        "--ref_audio", "-r", help="Path to reference audio (alternative to --profile)"
    )
    parser.add_argument(
        "--output", "-o", default="output.wav", help="Output audio file"
    )
    parser.add_argument(
        "--play", action="store_true", help="Play audio immediately"
    )
    parser.add_argument(
        "--model",
        default="mlx-community/chatterbox-turbo-fp16",
        help="Model to use",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95, help="Top-p sampling"
    )
    parser.add_argument(
        "--top_k", type=int, default=1000, help="Top-k sampling"
    )
    args = parser.parse_args()

    if not args.profile and not args.ref_audio:
        print("Error: Must provide either --profile or --ref_audio")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    from mlx_audio.tts.utils import load_model

    model = load_model(args.model)

    # Load voice profile or prepare from reference audio
    if args.profile:
        profile_path = Path(args.profile)
        if not profile_path.exists():
            print(f"Error: Profile not found: {profile_path}")
            sys.exit(1)

        print(f"Loading voice profile: {profile_path}")
        from mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo import Conditionals

        model._conds = Conditionals.load(profile_path)
    else:
        ref_audio_path = Path(args.ref_audio)
        if not ref_audio_path.exists():
            print(f"Error: Reference audio not found: {ref_audio_path}")
            sys.exit(1)

        print(f"Processing reference audio: {ref_audio_path}")
        model.prepare_conditionals(
            ref_audio=str(ref_audio_path),
            exaggeration=0.0,
            norm_loudness=True,
        )

    print(f"Generating speech: \"{args.text}\"")

    # Setup audio player if needed
    player = None
    if args.play:
        from mlx_audio.tts.audio_player import AudioPlayer

        player = AudioPlayer(sample_rate=model.sample_rate)

    # Generate speech
    audio_chunks = []
    for result in model.generate(
        text=args.text,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    ):
        audio_chunks.append(result.audio)
        if player:
            player.queue_audio(result.audio)

        print(f"  Generated {result.audio.shape[0]} samples, RTF: {result.real_time_factor:.2f}x")

    # Wait for playback to finish
    if player:
        player.wait_for_drain()
        player.stop()

    # Save audio
    if audio_chunks:
        import mlx.core as mx
        from mlx_audio.audio_io import write as audio_write

        audio = mx.concatenate(audio_chunks, axis=0) if len(audio_chunks) > 1 else audio_chunks[0]
        audio_write(args.output, audio, model.sample_rate)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
