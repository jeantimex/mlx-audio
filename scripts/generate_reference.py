#!/usr/bin/env python3
"""
Generate a reference audio clip for voice cloning with transcription.

Supports local audio files (mp3/wav) or YouTube URLs. Optionally removes
background noise and generates reference text using faster-whisper.

Usage:
    # From local file
    python scripts/generate_reference.py audio.mp3 --start 0:30 --end 0:40

    # From YouTube
    python scripts/generate_reference.py "https://www.youtube.com/watch?v=VIDEO_ID" --start 1:30 --end 1:45

    # With voice isolation and transcription
    python scripts/generate_reference.py audio.mp3 --start 0:10 --end 0:20 --isolate-voice --transcribe

    # Custom output
    python scripts/generate_reference.py audio.mp3 -s 0:10 -e 0:25 -o my_reference.wav

Requirements:
    pip install faster-whisper yt-dlp
    brew install ffmpeg
    pipx install 'audio-separator[cpu]'  # for --isolate-voice
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple


def parse_timestamp(ts: str) -> Optional[float]:
    """Parse timestamp string (e.g., '1:30' or '90') to seconds."""
    if ts is None:
        return None
    parts = ts.split(":")
    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    else:
        raise ValueError(f"Invalid timestamp format: {ts}")


def is_youtube_url(source: str) -> bool:
    """Check if the source is a YouTube URL."""
    return any(
        domain in source.lower()
        for domain in ["youtube.com", "youtu.be", "youtube-nocookie.com"]
    )


def check_dependencies(need_yt_dlp: bool = False, need_separator: bool = False):
    """Check that required tools are installed."""
    # Check ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("Error: ffmpeg not found. Install it with:")
        print("  brew install ffmpeg")
        sys.exit(1)

    # Check yt-dlp if needed
    if need_yt_dlp:
        try:
            subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        except FileNotFoundError:
            print("Error: yt-dlp not found. Install it with:")
            print("  brew install yt-dlp")
            print("  # or: pip install yt-dlp")
            sys.exit(1)

    # Check audio-separator if needed
    if need_separator:
        audio_sep_cmd = find_audio_separator()
        if not audio_sep_cmd:
            print("Error: audio-separator not found. Install it with:")
            print("  pipx install 'audio-separator[cpu]'")
            sys.exit(1)


def find_audio_separator() -> Optional[str]:
    """Find audio-separator command."""
    for cmd_path in [
        Path.home() / ".local/pipx/venvs/audio-separator/bin/audio-separator",
        Path.home() / ".local/bin/audio-separator",
        "audio-separator",
    ]:
        result = subprocess.run([str(cmd_path), "--help"], capture_output=True)
        if result.returncode == 0:
            return str(cmd_path)
    return None


def download_youtube_audio(url: str, output_path: Path) -> Path:
    """Download audio from YouTube."""
    print(f"Downloading audio from: {url}")

    temp_pattern = "_temp_yt_audio"
    download_cmd = [
        "yt-dlp",
        "-x",
        "--audio-quality",
        "0",
        "--no-playlist",
        "-o",
        f"{temp_pattern}.%(ext)s",
        url,
    ]

    result = subprocess.run(download_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"yt-dlp error:\n{result.stderr}")
        sys.exit(1)

    # Find downloaded file
    temp_files = list(Path(".").glob(f"{temp_pattern}.*"))
    if not temp_files:
        print("Error: Downloaded file not found")
        sys.exit(1)

    temp_audio = temp_files[0]
    print(f"Downloaded: {temp_audio}")
    return temp_audio


def extract_clip(
    input_path: Path,
    output_path: Path,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    sample_rate: int = 24000,
) -> None:
    """Extract a clip from audio file and convert to proper format."""
    ffmpeg_cmd = ["ffmpeg", "-y", "-i", str(input_path)]

    if start_sec is not None:
        ffmpeg_cmd.extend(["-ss", str(start_sec)])
    if end_sec is not None:
        if start_sec is not None:
            duration = end_sec - start_sec
            ffmpeg_cmd.extend(["-t", str(duration)])
        else:
            ffmpeg_cmd.extend(["-to", str(end_sec)])

    # Convert to mono at specified sample rate
    ffmpeg_cmd.extend(["-ac", "1", "-ar", str(sample_rate), str(output_path)])

    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error:\n{result.stderr}")
        sys.exit(1)


def get_duration(filepath: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(filepath),
        ],
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def isolate_voice(input_path: Path, output_path: Path) -> bool:
    """Use audio-separator to separate vocals from background."""
    audio_sep_cmd = find_audio_separator()
    if not audio_sep_cmd:
        return False

    print("Isolating voice (this may take a minute)...")

    output_dir = Path("_separated")
    output_dir.mkdir(exist_ok=True)

    try:
        # Stage 1: Separate vocals from instrumental
        print("  Stage 1: Separating vocals from instrumental...")
        result = subprocess.run(
            [
                audio_sep_cmd,
                str(input_path),
                "-m",
                "UVR-MDX-NET-Voc_FT.onnx",
                "--output_dir",
                str(output_dir),
                "--output_format",
                "WAV",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Stage 1 error: {result.stderr}")
            return False

        # Find vocals file
        vocals_file = None
        for pattern in ["*Vocals*", "*vocal*"]:
            for f in output_dir.glob(pattern):
                vocals_file = f
                break
            if vocals_file:
                break

        if not vocals_file or not vocals_file.exists():
            print("Could not find vocals in output")
            return False

        # Stage 2: De-reverb
        print("  Stage 2: Removing reverb...")
        result = subprocess.run(
            [
                audio_sep_cmd,
                str(vocals_file),
                "-m",
                "Reverb_HQ_By_FoxJoy.onnx",
                "--output_dir",
                str(output_dir),
                "--output_format",
                "WAV",
            ],
            capture_output=True,
            text=True,
        )

        # Find dry vocals or use stage 1 output
        dry_vocals = vocals_file
        if result.returncode == 0:
            for f in output_dir.glob("*No Reverb*"):
                dry_vocals = f
                break

        # Convert to proper format
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(dry_vocals),
                "-ac",
                "1",
                "-ar",
                "24000",
                str(output_path),
            ],
            capture_output=True,
            check=True,
        )

        return True

    except Exception as e:
        print(f"Voice isolation error: {e}")
        return False

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def transcribe_audio(audio_path: Path, model_size: str = "base") -> str:
    """Transcribe audio using faster-whisper."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("Error: faster-whisper not installed. Install it with:")
        print("  pip install faster-whisper")
        sys.exit(1)

    print(f"Transcribing audio with faster-whisper ({model_size})...")

    model = WhisperModel(model_size, device="auto", compute_type="auto")
    segments, info = model.transcribe(str(audio_path), beam_size=5)

    text_parts = []
    for segment in segments:
        text_parts.append(segment.text.strip())

    transcript = " ".join(text_parts)
    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

    return transcript


def main():
    parser = argparse.ArgumentParser(
        description="Generate reference audio for voice cloning with transcription"
    )
    parser.add_argument(
        "source", help="Audio file path (mp3/wav) or YouTube URL"
    )
    parser.add_argument(
        "--start", "-s", help="Start time (e.g., 1:30 or 90 seconds)"
    )
    parser.add_argument(
        "--end", "-e", help="End time (e.g., 1:45 or 105 seconds)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="reference.wav",
        help="Output audio filename (default: reference.wav)",
    )
    parser.add_argument(
        "--isolate-voice",
        "-i",
        action="store_true",
        help="Remove background music/noise and isolate voice",
    )
    parser.add_argument(
        "--transcribe",
        "-t",
        action="store_true",
        help="Transcribe audio using faster-whisper",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size for transcription (default: base)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Output sample rate in Hz (default: 24000)",
    )
    args = parser.parse_args()

    source = args.source
    output_path = Path(args.output)
    is_youtube = is_youtube_url(source)

    # Check dependencies
    check_dependencies(
        need_yt_dlp=is_youtube,
        need_separator=args.isolate_voice,
    )

    # Get audio file
    temp_files = []
    if is_youtube:
        temp_audio = download_youtube_audio(source, output_path)
        temp_files.append(temp_audio)
        input_audio = temp_audio
    else:
        input_audio = Path(source)
        if not input_audio.exists():
            print(f"Error: File not found: {input_audio}")
            sys.exit(1)

    # Parse timestamps
    start_sec = parse_timestamp(args.start)
    end_sec = parse_timestamp(args.end)

    if start_sec is not None or end_sec is not None:
        print(f"Extracting clip: {args.start or '0:00'} to {args.end or 'end'}")

    # Extract/convert clip
    if args.isolate_voice:
        # First extract to temp file, then isolate
        temp_clip = Path("_temp_clip.wav")
        temp_files.append(temp_clip)
        extract_clip(input_audio, temp_clip, start_sec, end_sec, args.sample_rate)

        if not isolate_voice(temp_clip, output_path):
            print("Voice isolation failed, using original audio")
            shutil.copy(temp_clip, output_path)
        else:
            print("Voice isolated successfully!")
    else:
        extract_clip(input_audio, output_path, start_sec, end_sec, args.sample_rate)

    # Cleanup temp files
    for temp_file in temp_files:
        if temp_file.exists():
            temp_file.unlink()

    # Get duration and validate
    duration = get_duration(output_path)
    print(f"\nSaved: {output_path}")
    print(f"Duration: {duration:.1f} seconds")
    print(f"Sample rate: {args.sample_rate} Hz")

    if duration < 5:
        print("\n⚠️  Warning: Audio is shorter than 5 seconds.")
        print("   Chatterbox-Turbo requires >5 seconds for voice cloning.")
    elif duration < 6:
        print("\n⚠️  Warning: Audio is short. Consider a longer clip for best results.")
    elif duration > 15:
        print("\nNote: Only first 15 sec used by Turbo, first 6 sec by standard Chatterbox.")
    else:
        print("\n✓ Duration is good for voice cloning!")

    # Transcribe if requested
    transcript = None
    if args.transcribe:
        transcript = transcribe_audio(output_path, args.whisper_model)
        print(f"\nTranscript:\n  \"{transcript}\"")

        # Save transcript to file
        transcript_path = output_path.with_suffix(".txt")
        transcript_path.write_text(transcript)
        print(f"Saved transcript: {transcript_path}")

    # Print usage example
    print("\n" + "=" * 60)
    print("Ready for voice cloning! Example usage:")
    print("=" * 60)

    if transcript:
        print(f"""
python -m mlx_audio.tts.generate \\
  --model mlx-community/chatterbox-turbo-fp16 \\
  --ref_audio {output_path} \\
  --ref_text "{transcript[:80]}{'...' if len(transcript) > 80 else ''}" \\
  --text "Your text to synthesize here." \\
  --play
""")
    else:
        print(f"""
python -m mlx_audio.tts.generate \\
  --model mlx-community/chatterbox-turbo-fp16 \\
  --ref_audio {output_path} \\
  --text "Your text to synthesize here." \\
  --play
""")


if __name__ == "__main__":
    main()
