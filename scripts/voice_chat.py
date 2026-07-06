#!/usr/bin/env python3
"""
Real-time voice chat with LLM and TTS.

Uses:
- Gemma 3 as the LLM
- Fish Audio S2 Pro for TTS with voice cloning
- Whisper for STT

Usage:
    # Basic chat (no voice cloning)
    python scripts/voice_chat.py

    # With voice cloning
    python scripts/voice_chat.py --ref_audio reference.wav --ref_text "Reference transcript"

    # Custom models
    python scripts/voice_chat.py --llm_model mlx-community/gemma-3-12b-it-4bit
"""

import argparse
import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sd


def record_audio(sample_rate: int = 16000, silence_threshold: float = 0.01,
                 silence_duration: float = 1.5, max_duration: float = 30.0) -> np.ndarray:
    """Record audio from microphone until silence is detected."""
    print("\n🎤 Listening... (speak now, pause to finish)")

    chunk_duration = 0.1  # 100ms chunks
    chunk_samples = int(sample_rate * chunk_duration)
    max_samples = int(sample_rate * max_duration)
    silence_samples = int(sample_rate * silence_duration)

    audio_chunks = []
    silent_chunks = 0
    has_speech = False

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:
        while True:
            chunk, _ = stream.read(chunk_samples)
            audio_chunks.append(chunk.copy())

            # Calculate RMS energy
            rms = np.sqrt(np.mean(chunk ** 2))

            if rms > silence_threshold:
                has_speech = True
                silent_chunks = 0
            else:
                silent_chunks += 1

            # Stop if silence detected after speech
            if has_speech and silent_chunks * chunk_samples >= silence_samples:
                break

            # Stop if max duration reached
            if len(audio_chunks) * chunk_samples >= max_samples:
                print("⏱️  Max duration reached")
                break

    audio = np.concatenate(audio_chunks, axis=0).flatten()
    duration = len(audio) / sample_rate
    print(f"✅ Recorded {duration:.1f}s of audio")

    return audio


def transcribe_audio(audio: np.ndarray, sample_rate: int, stt_model, language: str = "auto") -> str:
    """Transcribe audio using STT model."""
    print("📝 Transcribing...")

    # Save to temp file for STT
    from mlx_audio.audio_io import write as audio_write

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name

    audio_write(temp_path, audio, sample_rate)

    try:
        # Try with language parameter
        try:
            if language and language != "auto":
                result = stt_model.generate(temp_path, language=language)
            else:
                result = stt_model.generate(temp_path)
        except TypeError:
            # Fallback if language param not supported
            result = stt_model.generate(temp_path)

        text = result.text.strip() if hasattr(result, 'text') else str(result).strip()
        print(f"   You said: \"{text}\"")
        return text
    finally:
        Path(temp_path).unlink(missing_ok=True)


def generate_response(user_input: str, conversation: list, llm_model, llm_tokenizer,
                     system_prompt: str) -> str:
    """Generate LLM response."""
    from mlx_lm import generate

    print("🤔 Thinking...")

    # Build conversation
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation)
    messages.append({"role": "user", "content": user_input})

    # Format prompt
    prompt = llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Generate response
    response = generate(
        llm_model, llm_tokenizer,
        prompt=prompt,
        max_tokens=256,
        verbose=False,
    )

    response = response.strip()
    print(f"   Assistant: \"{response}\"")

    return response


def speak_response(text: str, tts_model, ref_audio=None, ref_text=None,
                   expression_tags: bool = True) -> None:
    """Convert text to speech and play it."""
    import mlx.core as mx
    from mlx_audio.tts.audio_player import AudioPlayer

    # Add expression tags if enabled
    if expression_tags:
        from mlx_audio.tts.expression_tags import add_expression_tags
        text = add_expression_tags(text, "fish-audio-s2-pro", "simple")

    print(f"🔊 Speaking...")
    if expression_tags:
        print(f"   (with tags: {text})")

    # Generate speech
    gen_kwargs = {"text": text}
    if ref_audio is not None:
        gen_kwargs["ref_audio"] = ref_audio
    if ref_text is not None:
        gen_kwargs["ref_text"] = ref_text

    results = list(tts_model.generate(**gen_kwargs))

    if not results or results[0].audio is None:
        print("❌ Failed to generate speech")
        return

    # Play audio
    audio = results[0].audio
    sample_rate = results[0].sample_rate

    player = AudioPlayer(sample_rate=sample_rate)
    player.queue_audio(mx.array(audio) if not isinstance(audio, mx.array) else audio)
    player.wait_for_drain()
    player.stop()


def main():
    parser = argparse.ArgumentParser(description="Real-time voice chat with LLM and TTS")
    parser.add_argument(
        "--llm_model",
        default="mlx-community/gemma-3-4b-it-4bit",
        help="LLM model for conversation",
    )
    parser.add_argument(
        "--tts_model",
        default="mlx-community/fish-audio-s2-pro",
        help="TTS model for speech synthesis",
    )
    parser.add_argument(
        "--stt_model",
        default="mlx-community/whisper-large-v3-turbo-asr-fp16",
        help="STT model for transcription",
    )
    parser.add_argument(
        "--ref_audio",
        help="Reference audio for voice cloning",
    )
    parser.add_argument(
        "--ref_text",
        help="Transcript of reference audio",
    )
    parser.add_argument(
        "--system_prompt",
        default="You are a helpful, friendly assistant. Keep responses concise and conversational.",
        help="System prompt for the LLM",
    )
    parser.add_argument(
        "--no-expression-tags",
        action="store_true",
        help="Disable automatic expression tags",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help="Language for STT (auto, en, zh, etc.)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🎙️  Voice Chat")
    print("=" * 60)
    print(f"LLM: {args.llm_model}")
    print(f"TTS: {args.tts_model}")
    print(f"STT: {args.stt_model}")
    if args.ref_audio:
        print(f"Voice: {args.ref_audio}")
    print("=" * 60)

    # Load models
    print("\n⏳ Loading models...")

    print("   Loading LLM...")
    from mlx_lm import load as load_llm
    llm_model, llm_tokenizer = load_llm(args.llm_model)

    print("   Loading TTS...")
    from mlx_audio.tts.utils import load_model as load_tts
    tts_model = load_tts(args.tts_model)

    print("   Loading STT...")
    from mlx_audio.stt.utils import load_model as load_stt
    stt_model = load_stt(args.stt_model)

    # Load reference audio if provided
    ref_audio = None
    ref_text = args.ref_text
    if args.ref_audio:
        print(f"   Loading reference audio: {args.ref_audio}")
        from mlx_audio.utils import load_audio
        ref_audio = load_audio(args.ref_audio, sample_rate=44100)

        # Transcribe reference if not provided
        if not ref_text:
            print("   Transcribing reference audio...")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            from mlx_audio.audio_io import write as audio_write
            audio_write(temp_path, ref_audio, 44100)
            try:
                if args.language and args.language != "auto":
                    result = stt_model.generate(temp_path, language=args.language)
                else:
                    result = stt_model.generate(temp_path)
            except TypeError:
                result = stt_model.generate(temp_path)
            ref_text = result.text.strip() if hasattr(result, 'text') else str(result).strip()
            Path(temp_path).unlink(missing_ok=True)
            print(f"   Reference text: \"{ref_text}\"")

    print("\n✅ Models loaded! Starting chat...")
    print("   Press Ctrl+C to exit\n")

    # Conversation history
    conversation = []

    try:
        while True:
            # Record user speech
            audio = record_audio()

            if len(audio) < 1600:  # Less than 0.1s
                print("⚠️  No speech detected, try again")
                continue

            # Transcribe
            user_text = transcribe_audio(audio, 16000, stt_model, args.language)

            if not user_text or user_text.lower() in ["", "."]:
                print("⚠️  Could not understand, try again")
                continue

            # Check for exit commands
            if user_text.lower() in ["exit", "quit", "bye", "goodbye", "再见", "退出"]:
                print("\n👋 Goodbye!")
                speak_response(
                    "Goodbye! It was nice chatting with you.",
                    tts_model, ref_audio, ref_text,
                    expression_tags=not args.no_expression_tags
                )
                break

            # Generate response
            response = generate_response(
                user_text, conversation, llm_model, llm_tokenizer,
                args.system_prompt
            )

            # Update conversation history
            conversation.append({"role": "user", "content": user_text})
            conversation.append({"role": "assistant", "content": response})

            # Keep conversation manageable
            if len(conversation) > 20:
                conversation = conversation[-20:]

            # Speak response
            speak_response(
                response, tts_model, ref_audio, ref_text,
                expression_tags=not args.no_expression_tags
            )

    except KeyboardInterrupt:
        print("\n\n👋 Chat ended by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
