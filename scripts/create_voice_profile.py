#!/usr/bin/env python3
"""
Create a voice profile from reference audio for Chatterbox or Chatterbox-Turbo.

Pre-computes speaker embeddings and conditionals so you don't need to
process the reference audio every time you generate speech.

Usage:
    # Create a voice profile for Chatterbox-Turbo
    python scripts/create_voice_profile.py reference.wav -o my_voice.profile

    # Create a voice profile for regular Chatterbox
    python scripts/create_voice_profile.py reference.wav --model mlx-community/chatterbox-fp16

    # Use the profile for TTS (see example at the end)

Requirements:
    - Reference audio must be >5 seconds (Turbo) or >6 seconds (regular)
    - Clear speech with minimal background noise

Note:
    Voice profiles are only supported for Chatterbox and Chatterbox-Turbo models.
    For OmniVoice, use --ref_audio and --ref_text directly with mlx_audio.tts.generate.
"""

import argparse
import sys
from pathlib import Path

SUPPORTED_MODELS = [
    "chatterbox-turbo",
    "chatterbox-fp16",
    "chatterbox",
]


def is_supported_model(model_name: str) -> bool:
    """Check if model supports voice profiles."""
    model_lower = model_name.lower()
    return any(supported in model_lower for supported in SUPPORTED_MODELS)


def main():
    parser = argparse.ArgumentParser(
        description="Create a voice profile from reference audio for Chatterbox models"
    )
    parser.add_argument(
        "ref_audio",
        help="Path to reference audio file (must be >5 seconds)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output profile filename (default: <input_name>.profile)",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/chatterbox-turbo-fp16",
        help="Model to use (default: mlx-community/chatterbox-turbo-fp16)",
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="Emotion exaggeration factor 0-1 (default: 0.5, only for regular Chatterbox)",
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Test the profile by generating a sample",
    )
    parser.add_argument(
        "--test-text",
        default="Hello! This is a test of my voice profile.",
        help="Text to use for testing",
    )
    args = parser.parse_args()

    ref_audio_path = Path(args.ref_audio)
    if not ref_audio_path.exists():
        print(f"Error: Reference audio not found: {ref_audio_path}")
        sys.exit(1)

    # Check if model supports voice profiles
    if not is_supported_model(args.model):
        print(f"Error: Voice profiles are only supported for Chatterbox models.")
        print(f"Model '{args.model}' does not support pre-computed voice profiles.")
        print()
        print("For OmniVoice, use --ref_audio and --ref_text directly:")
        print()
        print("  python -m mlx_audio.tts.generate \\")
        print("    --model mlx-community/OmniVoice-bf16 \\")
        print(f"    --ref_audio {ref_audio_path} \\")
        print("    --ref_text \"Transcript of the reference audio\" \\")
        print("    --text \"Your text here\" \\")
        print("    --lang_code chinese \\")
        print("    --play")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = ref_audio_path.with_suffix(".profile")

    # Detect model type
    is_turbo = "turbo" in args.model.lower()

    print(f"Loading model: {args.model}")
    from mlx_audio.tts.utils import load_model

    model = load_model(args.model)

    print(f"Processing reference audio: {ref_audio_path}")
    print("  Extracting speaker embeddings...")
    print("  Computing T3 conditionals...")
    print("  Computing S3Gen conditionals...")

    # Prepare conditionals from reference audio
    if is_turbo:
        model.prepare_conditionals(
            ref_audio=str(ref_audio_path),
            exaggeration=0.0,
            norm_loudness=True,
        )
    else:
        # Regular Chatterbox
        model._conds = model.prepare_conditionals(
            ref_wav=str(ref_audio_path),
            ref_sr=model.sample_rate,
            exaggeration=args.exaggeration,
        )

    # Save the profile
    model._conds.save(output_path)
    print(f"\nVoice profile saved: {output_path}")

    # Test if requested
    if args.test:
        print(f"\nTesting profile with: \"{args.test_text}\"")

        # Load the appropriate Conditionals class
        if is_turbo:
            from mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo import Conditionals
        else:
            from mlx_audio.tts.models.chatterbox.chatterbox import Conditionals

        # Reload to verify it works
        model._conds = Conditionals.load(output_path)

        for result in model.generate(
            text=args.test_text,
            temperature=0.8,
        ):
            from mlx_audio.audio_io import write as audio_write

            test_output = output_path.with_suffix(".test.wav")
            audio_write(str(test_output), result.audio, result.sample_rate)
            print(f"Test audio saved: {test_output}")

    # Print usage instructions
    print("\n" + "=" * 60)
    print("Usage with the voice profile:")
    print("=" * 60)
    print(f"""
# Command line
python -m mlx_audio.tts.generate \\
  --model {args.model} \\
  --profile {output_path} \\
  --text "Your text here" \\
  --play
""")


if __name__ == "__main__":
    main()
