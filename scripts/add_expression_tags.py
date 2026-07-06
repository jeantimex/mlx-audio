#!/usr/bin/env python3
"""
Add expression/style tags to text using an LLM.

This script analyzes text and adds appropriate expression tags for TTS models
like Chatterbox-Turbo or OmniVoice.

Usage:
    # Using local MLX model
    python scripts/add_expression_tags.py --text "That's hilarious! I can't believe you did that."

    # Using Anthropic API (requires ANTHROPIC_API_KEY)
    python scripts/add_expression_tags.py --text "I'm so frustrated with this situation." --provider anthropic

    # For Chinese text with OmniVoice tags
    python scripts/add_expression_tags.py --text "这太好笑了！我简直不敢相信。" --model-type omnivoice

    # Pipe to TTS
    python scripts/add_expression_tags.py --text "Hello! That's amazing!" | xargs -I {} \
      python scripts/chatterbox_tts.py --profile voice.profile --text "{}" --play
"""

import argparse
import sys

# Tag sets for different models
CHATTERBOX_TURBO_TAGS = {
    "sounds": ["[laugh]", "[chuckle]", "[sigh]", "[gasp]", "[cough]", "[groan]", "[sniff]", "[shush]", "[clear throat]"],
    "styles": ["[happy]", "[angry]", "[dramatic]", "[sarcastic]", "[whispering]", "[crying]", "[fear]", "[surprised]", "[narration]"],
}

OMNIVOICE_TAGS = {
    "sounds": ["[laughter]", "[sigh]"],
    "emotions": ["[confirmation-en]", "[question-en]", "[question-ah]", "[question-oh]", "[question-ei]", "[question-yi]",
                 "[surprise-ah]", "[surprise-oh]", "[surprise-wa]", "[surprise-yo]", "[dissatisfaction-hnn]"],
}

FISH_SPEECH_TAGS = {
    "sounds": ["[laughing]", "[chuckle]", "[sigh]", "[inhale]", "[exhale]", "[panting]", "[clearing throat]", "[tsk]"],
    "styles": ["[excited]", "[angry]", "[sad]", "[whisper]", "[shouting]", "[loud]", "[low voice]", "[singing]",
               "[surprised]", "[shocked]", "[delight]", "[pause]", "[emphasis]", "[laughing tone]", "[excited tone]"],
}


def get_system_prompt(model_type: str) -> str:
    """Get the system prompt for tag insertion."""
    if model_type == "omnivoice":
        tags = OMNIVOICE_TAGS
        tag_list = ", ".join(tags["sounds"] + tags["emotions"])
    elif model_type == "fish-speech":
        tags = FISH_SPEECH_TAGS
        tag_list = ", ".join(tags["sounds"] + tags["styles"])
    else:  # chatterbox-turbo
        tags = CHATTERBOX_TURBO_TAGS
        tag_list = ", ".join(tags["sounds"] + tags["styles"])

    return f"""You are a text preprocessor for expressive text-to-speech synthesis.

Your task is to analyze the input text and insert appropriate expression tags to make the speech more natural and expressive.

Available tags: {tag_list}

Rules:
1. Insert tags where natural pauses or expressions would occur
2. Don't overuse tags - only add where it genuinely improves expression
3. Place tags BEFORE or AFTER the relevant phrase, not in the middle of words
4. For laughter/sighs, place after the triggering phrase
5. For style tags like [happy] or [angry], place at the start of the emotional section
6. Keep the original text intact - only add tags
7. Return ONLY the modified text, no explanations

Examples:
- "That's so funny!" → "That's so funny! [laugh]"
- "I can't believe this happened..." → "[sigh] I can't believe this happened..."
- "I'M SO ANGRY RIGHT NOW!" → "[angry] I'm so angry right now!"
- "Shh, be quiet" → "[whispering] Shh, be quiet [shush]"
- "Oh my god, what is that?!" → "[surprised] Oh my god, what is that?! [gasp]"
"""


def add_tags_with_anthropic(text: str, model_type: str) -> str:
    """Use Anthropic API to add expression tags."""
    import os
    try:
        import anthropic
    except ImportError:
        print("Error: anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=get_system_prompt(model_type),
        messages=[
            {"role": "user", "content": text}
        ]
    )

    return response.content[0].text.strip()


def add_tags_with_mlx(text: str, model_type: str, model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit") -> str:
    """Use local MLX model to add expression tags."""
    try:
        from mlx_lm import load, generate
    except ImportError:
        print("Error: mlx-lm not installed. Run: pip install mlx-lm")
        sys.exit(1)

    print(f"Loading model: {model_name}", file=sys.stderr)
    model, tokenizer = load(model_name)

    system_prompt = get_system_prompt(model_type)

    # Format as chat
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=512,
        verbose=False,
    )

    # Clean up response
    result = response.strip()
    # Remove any markdown formatting
    if result.startswith("```"):
        lines = result.split("\n")
        result = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    return result


def add_tags_simple(text: str, model_type: str) -> str:
    """Simple rule-based tag insertion (no LLM needed)."""
    from mlx_audio.tts.expression_tags import add_expression_tags

    # Map model_type to a model name that the module can detect
    model_name_map = {
        "chatterbox-turbo": "chatterbox-turbo",
        "omnivoice": "omnivoice",
        "fish-speech": "fish-audio-s2-pro",
    }
    model_name = model_name_map.get(model_type, model_type)

    return add_expression_tags(text, model_name, provider="simple")


def main():
    parser = argparse.ArgumentParser(
        description="Add expression tags to text for expressive TTS"
    )
    parser.add_argument(
        "--text", "-t", required=True, help="Text to process"
    )
    parser.add_argument(
        "--model-type", "-m",
        choices=["chatterbox-turbo", "omnivoice", "fish-speech"],
        default="chatterbox-turbo",
        help="TTS model type (determines available tags)",
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["anthropic", "mlx", "simple"],
        default="simple",
        help="LLM provider: anthropic (API), mlx (local), simple (rule-based)",
    )
    parser.add_argument(
        "--llm-model",
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        help="MLX model to use (only for --provider mlx)",
    )
    args = parser.parse_args()

    if args.provider == "anthropic":
        result = add_tags_with_anthropic(args.text, args.model_type)
    elif args.provider == "mlx":
        result = add_tags_with_mlx(args.text, args.model_type, args.llm_model)
    else:
        result = add_tags_simple(args.text, args.model_type)

    print(result)


if __name__ == "__main__":
    main()
