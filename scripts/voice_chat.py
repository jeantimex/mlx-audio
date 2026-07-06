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

    # Fast mode (smaller models, streaming)
    python scripts/voice_chat.py --fast

    # Custom models
    python scripts/voice_chat.py --llm_model mlx-community/gemma-3-12b-it-4bit
"""

import argparse
import re
import sys
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


def record_and_transcribe_streaming(stt_model, sample_rate: int = 16000,
                                    silence_threshold: float = 0.01,
                                    silence_duration: float = 1.5,
                                    max_duration: float = 30.0) -> str:
    """Record audio and transcribe in real-time using Voxtral streaming."""
    import threading
    from mlx_audio.stt.models.voxtral_realtime.streaming import StreamingAudioSource

    print("\n🎤 Listening... (speak now, pause to finish)")

    chunk_duration = 0.1  # 100ms chunks
    chunk_samples = int(sample_rate * chunk_duration)
    max_samples = int(sample_rate * max_duration)
    silence_samples = int(sample_rate * silence_duration)

    source = StreamingAudioSource()
    transcription_result = []
    transcription_done = threading.Event()

    def transcribe_thread():
        """Background thread to process transcription."""
        try:
            for delta in stt_model.generate_streaming(source):
                if delta:
                    transcription_result.append(delta)
                    print(f"\r   Hearing: {' '.join(transcription_result)}", end="", flush=True)
        except Exception as e:
            print(f"\n⚠️  Transcription error: {e}")
        finally:
            transcription_done.set()

    # Start transcription in background
    t = threading.Thread(target=transcribe_thread, daemon=True)
    t.start()

    # Record audio and feed to transcriber
    silent_chunks = 0
    has_speech = False
    total_samples = 0

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:
        while True:
            chunk, _ = stream.read(chunk_samples)
            total_samples += chunk_samples

            # Feed to streaming transcriber
            source.append(chunk.flatten())

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
            if total_samples >= max_samples:
                print("\n⏱️  Max duration reached")
                break

    # Signal end of audio
    source.close()

    # Wait for transcription to complete (with timeout)
    transcription_done.wait(timeout=5.0)
    print()  # New line after transcription

    text = ''.join(transcription_result).strip()
    duration = total_samples / sample_rate
    print(f"✅ Recorded {duration:.1f}s | You said: \"{text}\"")

    return text


def transcribe_audio(audio: np.ndarray, sample_rate: int, stt_model, language: str = "auto") -> str:
    """Transcribe audio using STT model."""
    print("📝 Transcribing...")

    # Save to temp file for STT
    from mlx_audio.audio_io import write as audio_write

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name

    audio_write(temp_path, audio, sample_rate)

    try:
        # Try different parameter combinations for different models
        result = None

        # Try with language parameter (Qwen3-ASR style)
        if language and language != "auto":
            try:
                # Capitalize language for some models
                lang = language.capitalize() if len(language) == 2 else language
                result = stt_model.generate(temp_path, language=lang)
            except (TypeError, ValueError):
                pass

        # Try without language parameter
        if result is None:
            try:
                result = stt_model.generate(temp_path)
            except Exception:
                pass

        # Try with audio path only
        if result is None:
            result = stt_model.generate(temp_path)

        text = result.text.strip() if hasattr(result, 'text') else str(result).strip()
        print(f"   You said: \"{text}\"")
        return text
    finally:
        Path(temp_path).unlink(missing_ok=True)


EXPRESSION_TAGS_PROMPT = """
You can use expression tags to make speech more natural. Available tags:
[laughing], [chuckle], [sigh], [excited], [angry], [sad], [whisper], [surprised], [pause], [emphasis]

Use them naturally, like: "That's amazing! [laughing] I love it!" or "[sigh] That's unfortunate."
Don't overuse - only add where it genuinely improves expression.
"""

EXPRESSION_TAGS_PROMPT_ZH = """
你可以使用表情标签让语音更自然。可用标签：
[laughing] 笑声, [sigh] 叹气, [excited] 兴奋, [angry] 生气, [sad] 难过, [whisper] 低语, [surprised] 惊讶, [pause] 停顿, [cute] 撒娇

自然使用，如："太好了！[laughing] 我太开心了！" 或 "[sigh] 真可惜。"
不要过度使用，只在真正需要时添加。
"""


def generate_response(user_input: str, conversation: list, llm_model, llm_tokenizer,
                     system_prompt: str, expression_mode: str = "simple",
                     language: str = "auto") -> str:
    """Generate LLM response."""
    from mlx_lm import generate

    print("🤔 Thinking...")

    # Add expression tags instruction to system prompt if using LLM mode
    full_system_prompt = system_prompt
    if expression_mode == "llm":
        if language == "zh":
            full_system_prompt += "\n" + EXPRESSION_TAGS_PROMPT_ZH
        else:
            full_system_prompt += "\n" + EXPRESSION_TAGS_PROMPT

    # Build conversation
    messages = [{"role": "system", "content": full_system_prompt}]
    messages.extend(conversation)
    messages.append({"role": "user", "content": user_input})

    # Format prompt
    prompt = llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Generate response with shorter max_tokens for conversational responses
    response = generate(
        llm_model, llm_tokenizer,
        prompt=prompt,
        max_tokens=128,  # Shorter for conversational responses
        verbose=False,
    )

    # Clean up response - remove special tokens and garbage
    response = response.strip()

    # Remove common end tokens
    for end_token in ["<end_of_turn>", "</s>", "<|endoftext|>", "<|im_end|>", "\n\n\n"]:
        if end_token in response:
            response = response.split(end_token)[0]

    # Remove repeated newlines
    import re
    response = re.sub(r'\n{2,}', '\n', response)

    # Remove any non-printable characters or garbage
    response = ''.join(char for char in response if char.isprintable() or char in '\n ')

    # Remove repeated expression tags
    response = re.sub(r'(\[[\w]+\]\s*){3,}', '', response)

    response = response.strip()

    # If response is empty after cleanup, provide a fallback
    if not response:
        response = "I'm here!" if language != "zh" else "我在这里！"

    print(f"   Assistant: \"{response}\"")

    return response


def speak_response(text: str, tts_model, ref_audio=None, ref_text=None,
                   expression_mode: str = "simple", stream: bool = False) -> None:
    """Convert text to speech and play it."""
    import mlx.core as mx
    from mlx_audio.tts.audio_player import AudioPlayer

    # Add expression tags if using simple mode (rule-based)
    # LLM mode already has tags in text, none mode skips tags
    if expression_mode == "simple":
        from mlx_audio.tts.expression_tags import add_expression_tags
        text = add_expression_tags(text, "fish-audio-s2-pro", "simple")

    print(f"🔊 Speaking{'...' if not stream else ' (streaming)...'}")
    if expression_mode != "none" and "[" in text:
        print(f"   (with tags: {text})")

    # Generate speech
    gen_kwargs = {"text": text}
    if ref_audio is not None:
        gen_kwargs["ref_audio"] = ref_audio
    if ref_text is not None:
        gen_kwargs["ref_text"] = ref_text

    # Get sample rate from model
    sample_rate = getattr(tts_model, 'sample_rate', 44100)
    player = AudioPlayer(sample_rate=sample_rate)

    if stream:
        # Streaming mode: play chunks as they're generated
        gen_kwargs["chunk_length"] = 100  # Smaller chunks for faster first audio
        has_audio = False
        for result in tts_model.generate(**gen_kwargs):
            if result.audio is not None and len(result.audio) > 0:
                has_audio = True
                audio = result.audio
                if hasattr(result, 'sample_rate'):
                    sample_rate = result.sample_rate
                player.queue_audio(mx.array(audio) if not isinstance(audio, mx.array) else audio)
        if not has_audio:
            print("❌ Failed to generate speech")
            player.stop()
            return
    else:
        # Non-streaming: generate all then play
        results = list(tts_model.generate(**gen_kwargs))
        if not results or results[0].audio is None:
            print("❌ Failed to generate speech")
            player.stop()
            return
        audio = results[0].audio
        if hasattr(results[0], 'sample_rate'):
            sample_rate = results[0].sample_rate
        player.queue_audio(mx.array(audio) if not isinstance(audio, mx.array) else audio)

    player.wait_for_drain()
    player.stop()


def main():
    parser = argparse.ArgumentParser(description="Real-time voice chat with LLM and TTS")
    parser.add_argument(
        "--llm_model",
        default=None,
        help="LLM model for conversation (default: gemma-3-4b-it-4bit, fast: gemma-3-1b-it-4bit)",
    )
    parser.add_argument(
        "--tts_model",
        default="mlx-community/fish-audio-s2-pro",
        help="TTS model for speech synthesis",
    )
    parser.add_argument(
        "--stt_model",
        default=None,
        help="STT model (default: whisper-large-v3-turbo-asr-fp16, fast: whisper-small)",
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
        default=None,
        help="System prompt for the LLM (auto-generated based on language if not provided)",
    )
    parser.add_argument(
        "--no-expression-tags",
        action="store_true",
        help="Disable automatic expression tags",
    )
    parser.add_argument(
        "--expression-mode",
        choices=["simple", "llm", "none"],
        default="simple",
        help="Expression tag mode: simple (rule-based), llm (LLM generates tags), none",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help="Language for STT (auto, en, zh, etc.)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use faster (smaller) models for lower latency",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream TTS output (play while generating)",
    )
    args = parser.parse_args()

    # Set model defaults based on fast mode and language
    # Note: Voxtral Realtime is fast but struggles with Chinese
    # Qwen3-ASR is better for Chinese/multilingual
    is_chinese = args.language.lower() in ["zh", "chinese", "cn", "mandarin"]

    if args.fast:
        if args.llm_model is None:
            args.llm_model = "mlx-community/gemma-3-1b-it-4bit"
        if args.stt_model is None:
            if is_chinese:
                # Use Qwen3-ASR for Chinese (Voxtral struggles with Chinese)
                args.stt_model = "mlx-community/Qwen3-ASR-0.6B-8bit"
            else:
                # Voxtral Realtime for streaming STT (fastest for English)
                args.stt_model = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"
    else:
        if args.llm_model is None:
            args.llm_model = "mlx-community/gemma-3-4b-it-4bit"
        if args.stt_model is None:
            args.stt_model = "mlx-community/Qwen3-ASR-0.6B-8bit"

    # Check if STT model supports streaming (Voxtral)
    use_streaming_stt = "voxtral" in args.stt_model.lower() or "realtime" in args.stt_model.lower()

    # Set default system prompt based on language
    if args.system_prompt is None:
        if is_chinese:
            args.system_prompt = "你是一个友好的中文助手。请用简短的中文回答，1-2句话即可。"
        else:
            args.system_prompt = "You are a helpful, friendly assistant. Keep responses concise and conversational. Respond in 1-2 sentences."

    # Determine expression mode
    expr_mode = "none" if args.no_expression_tags else args.expression_mode

    print("=" * 60)
    print("🎙️  Voice Chat" + (" ⚡ FAST MODE" if args.fast else ""))
    print("=" * 60)
    print(f"LLM: {args.llm_model}")
    print(f"TTS: {args.tts_model}" + (" (streaming)" if args.stream else ""))
    print(f"STT: {args.stt_model}" + (" (realtime)" if use_streaming_stt else ""))
    print(f"Expression: {expr_mode}")
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

            result = None
            # Try with language parameter
            if args.language and args.language != "auto":
                try:
                    lang = args.language.capitalize() if len(args.language) == 2 else args.language
                    result = stt_model.generate(temp_path, language=lang)
                except (TypeError, ValueError):
                    pass

            # Fallback without language
            if result is None:
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
            # Record and transcribe
            if use_streaming_stt:
                # Streaming STT: transcribe while recording (fastest)
                user_text = record_and_transcribe_streaming(stt_model)
            else:
                # Traditional: record first, then transcribe
                audio = record_audio()

                if len(audio) < 1600:  # Less than 0.1s
                    print("⚠️  No speech detected, try again")
                    continue

                user_text = transcribe_audio(audio, 16000, stt_model, args.language)

            if not user_text or user_text.lower() in ["", "."]:
                print("⚠️  Could not understand, try again")
                continue

            # Determine expression mode
            expr_mode = "none" if args.no_expression_tags else args.expression_mode

            # Check for exit commands
            if user_text.lower() in ["exit", "quit", "bye", "goodbye", "再见", "退出"]:
                print("\n👋 Goodbye!")
                speak_response(
                    "Goodbye! It was nice chatting with you.",
                    tts_model, ref_audio, ref_text,
                    expression_mode=expr_mode,
                    stream=args.stream
                )
                break

            # Generate response
            response = generate_response(
                user_text, conversation, llm_model, llm_tokenizer,
                args.system_prompt,
                expression_mode=expr_mode,
                language=args.language
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
                expression_mode=expr_mode,
                stream=args.stream
            )

    except KeyboardInterrupt:
        print("\n\n👋 Chat ended by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
