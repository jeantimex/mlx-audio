"""
Expression tag processing for TTS models.

Different models use different tag formats:
- Chatterbox-Turbo: [laugh], [chuckle], [sigh], [happy], [angry], etc.
- OmniVoice: [laughter], [sigh], [surprise-ah], [question-oh], etc.

This module provides:
1. Tag conversion between model formats
2. LLM-based tag insertion
3. Rule-based tag insertion
"""

import re
from typing import Optional

# Canonical tags (model-agnostic) -> Model-specific tags
TAG_MAPPINGS = {
    "chatterbox-turbo": {
        # Sounds
        "laugh": "[laugh]",
        "laughter": "[laugh]",
        "chuckle": "[chuckle]",
        "sigh": "[sigh]",
        "gasp": "[gasp]",
        "cough": "[cough]",
        "groan": "[groan]",
        "sniff": "[sniff]",
        "shush": "[shush]",
        "clear_throat": "[clear throat]",
        # Styles
        "happy": "[happy]",
        "angry": "[angry]",
        "dramatic": "[dramatic]",
        "sarcastic": "[sarcastic]",
        "whisper": "[whispering]",
        "whispering": "[whispering]",
        "crying": "[crying]",
        "fear": "[fear]",
        "surprised": "[surprised]",
        "narration": "[narration]",
        # Question/surprise (map to style)
        "question": "[surprised]",
        "surprise": "[surprised]",
    },
    "omnivoice": {
        # Sounds
        "laugh": "[laughter]",
        "laughter": "[laughter]",
        "chuckle": "[laughter]",
        "sigh": "[sigh]",
        # Questions
        "question": "[question-ah]",
        "question_ah": "[question-ah]",
        "question_oh": "[question-oh]",
        "question_ei": "[question-ei]",
        "question_yi": "[question-yi]",
        "question_en": "[question-en]",
        # Surprise
        "surprise": "[surprise-ah]",
        "surprised": "[surprise-ah]",
        "surprise_ah": "[surprise-ah]",
        "surprise_oh": "[surprise-oh]",
        "surprise_wa": "[surprise-wa]",
        "surprise_yo": "[surprise-yo]",
        "gasp": "[surprise-ah]",
        # Other
        "confirmation": "[confirmation-en]",
        "dissatisfaction": "[dissatisfaction-hnn]",
        # Map emotions to closest OmniVoice equivalents
        "happy": "[laughter]",  # Map happiness to laughter
        "angry": "[dissatisfaction-hnn]",  # Map anger to dissatisfaction
        # Unsupported in OmniVoice (remove)
        "dramatic": "",
        "sarcastic": "",
        "whisper": "",
        "whispering": "",
        "crying": "[sigh]",  # Map crying to sigh
        "fear": "[surprise-ah]",  # Map fear to surprise
        "narration": "",
        "cough": "",
        "groan": "[sigh]",
        "sniff": "[sigh]",
        "shush": "",
        "clear_throat": "",
    },
    "chatterbox": {
        # Regular Chatterbox doesn't support expression tags
        # Remove all tags
    },
}

# Models that support expression tags
SUPPORTED_MODELS = ["chatterbox-turbo", "omnivoice"]


def detect_model_type(model_name: str) -> Optional[str]:
    """Detect model type from model name/path."""
    model_lower = model_name.lower()

    if "chatterbox-turbo" in model_lower or "chatterbox_turbo" in model_lower:
        return "chatterbox-turbo"
    elif "omnivoice" in model_lower:
        return "omnivoice"
    elif "chatterbox" in model_lower:
        return "chatterbox"

    return None


def supports_expression_tags(model_name: str) -> bool:
    """Check if a model supports expression tags."""
    model_type = detect_model_type(model_name)
    return model_type in SUPPORTED_MODELS


def convert_tags(text: str, target_model: str) -> str:
    """
    Convert expression tags in text to the target model's format.

    Args:
        text: Text with expression tags (any format)
        target_model: Target model name/path

    Returns:
        Text with tags converted to target model's format
    """
    model_type = detect_model_type(target_model)

    if model_type is None or model_type not in TAG_MAPPINGS:
        # Unknown model, remove all tags
        return re.sub(r'\[[^\]]+\]', '', text)

    mapping = TAG_MAPPINGS[model_type]

    # Find all tags in text
    def replace_tag(match):
        tag_content = match.group(1).lower().replace(" ", "_").replace("-", "_")

        # Direct lookup
        if tag_content in mapping:
            return mapping[tag_content]

        # Try without brackets that might be in the mapping value
        for canonical, model_tag in mapping.items():
            if canonical == tag_content:
                return model_tag

        # Check if it's already a valid tag for this model
        tag_with_brackets = f"[{match.group(1)}]"
        if tag_with_brackets in mapping.values():
            return tag_with_brackets

        # Unknown tag, remove it
        return ""

    result = re.sub(r'\[([^\]]+)\]', replace_tag, text)

    # Clean up extra spaces
    result = re.sub(r'\s+', ' ', result).strip()

    return result


def add_expression_tags(
    text: str,
    target_model: str,
    provider: str = "simple",
    llm_model: Optional[str] = None,
) -> str:
    """
    Add expression tags to text for a specific TTS model.

    Args:
        text: Input text without tags
        target_model: Target TTS model name/path
        provider: "simple" (rule-based), "anthropic", or "mlx"
        llm_model: LLM model for mlx provider

    Returns:
        Text with appropriate expression tags for the model
    """
    model_type = detect_model_type(target_model)

    if model_type is None or model_type not in SUPPORTED_MODELS:
        # Model doesn't support tags, return as-is
        return text

    if provider == "simple":
        tagged_text = _add_tags_simple(text)
    elif provider == "anthropic":
        tagged_text = _add_tags_anthropic(text, model_type)
    elif provider == "mlx":
        tagged_text = _add_tags_mlx(text, model_type, llm_model)
    else:
        tagged_text = text

    # Convert to target model's tag format
    return convert_tags(tagged_text, target_model)


def _add_tags_simple(text: str) -> str:
    """Simple rule-based tag insertion using canonical tags."""
    result = text

    # === English patterns ===

    # Laughing indicators
    result = re.sub(
        r'(haha|hehe|lol|lmao|rofl|😂|🤣)',
        r'\1 [laugh]',
        result,
        flags=re.IGNORECASE
    )
    result = re.sub(
        r'([!?])\s*(that\'s (so )?funny|hilarious|too funny)',
        r'\1 \2 [laugh]',
        result,
        flags=re.IGNORECASE
    )

    # Sighing indicators
    result = re.sub(
        r'\b(sigh|ugh)\b',
        r'[sigh] \1',
        result,
        flags=re.IGNORECASE
    )
    result = re.sub(
        r'(unfortunately|sadly|i\'m (so )?tired)',
        r'[sigh] \1',
        result,
        flags=re.IGNORECASE
    )

    # Surprise indicators
    result = re.sub(
        r'\b(oh my god|omg|wow|no way)\b',
        r'[surprise] \1 [gasp]',
        result,
        flags=re.IGNORECASE
    )
    result = re.sub(
        r'(what\?!+|really\?!+)',
        r'[surprise] \1',
        result,
        flags=re.IGNORECASE
    )

    # Question indicators
    result = re.sub(
        r'(huh\?|eh\?|what\?(?![!]))',
        r'[question] \1',
        result,
        flags=re.IGNORECASE
    )

    # Anger indicators (ALL CAPS with exclamation)
    result = re.sub(
        r'\b([A-Z]{4,}!+)',
        r'[angry] \1',
        result
    )

    # Whisper indicators
    result = re.sub(
        r'\b(shh+|psst|quietly)\b',
        r'[whisper] \1',
        result,
        flags=re.IGNORECASE
    )

    # Happy indicators
    result = re.sub(
        r'\b(yay|hooray|wonderful|amazing|fantastic)\b',
        r'[happy] \1',
        result,
        flags=re.IGNORECASE
    )

    # === Chinese patterns ===

    # Laughing indicators (哈哈, 嘻嘻, 呵呵, 笑死)
    result = re.sub(
        r'(哈哈+|嘻嘻+|呵呵+|笑死了?|太好笑了)',
        r'\1 [laugh]',
        result
    )

    # Happy indicators (开心, 高兴, 太棒了, 太好了, 真好, 太爽了)
    # Use word boundary-like pattern to avoid partial matches
    result = re.sub(
        r'(真?他妈的?开心|真?开心|好开心|很开心|超开心|太开心了?|好高兴|很高兴|真高兴|太高兴了?|太棒了|太好了|真好|太爽了|好爽|爽死了?|耶+)',
        r'[happy] \1',
        result
    )

    # Sighing indicators (唉, 哎, 算了, 无奈, 累)
    result = re.sub(
        r'(唉+|哎+|算了|无奈|好累|真累|太累了|累死了|心累)',
        r'[sigh] \1',
        result
    )

    # Surprise indicators (哇, 天哪, 我的天, 卧槽, 靠, 厉害, 神奇)
    result = re.sub(
        r'(哇+塞?|天哪|我的天|天啊|我靠|卧槽|牛逼|太厉害了?|不可思议|很神奇|太神奇了?|真神奇|难以置信|不敢相信)',
        r'[surprise] \1',
        result
    )

    # Anger indicators (气死, 烦死, 讨厌, 该死)
    result = re.sub(
        r'(气死.{0,2}了?|烦死.{0,2}了?|讨厌|该死|真烦|好烦|太烦了|受不了|忍不了)',
        r'[angry] \1',
        result
    )

    # Question indicators (啊?, 吗?, 呢?, 什么)
    result = re.sub(
        r'(真的吗|是吗|什么\?|啥\?|为什么|怎么回事)',
        r'[question] \1',
        result
    )

    # Dissatisfaction (哼, 切, 无语)
    result = re.sub(
        r'(哼+|切+|无语|服了|醉了)',
        r'[dissatisfaction] \1',
        result
    )

    # Clean up multiple spaces
    result = re.sub(r'\s+', ' ', result).strip()

    return result


def _get_llm_system_prompt(model_type: str) -> str:
    """Get system prompt for LLM tag insertion."""
    if model_type == "omnivoice":
        tags = "[laughter], [sigh], [question-ah], [question-oh], [surprise-ah], [surprise-oh], [dissatisfaction-hnn]"
    else:
        tags = "[laugh], [chuckle], [sigh], [gasp], [happy], [angry], [surprised], [whispering], [dramatic], [sarcastic]"

    return f"""You add expression tags to text for text-to-speech synthesis.

Available tags: {tags}

Rules:
1. Add tags where natural expressions would occur
2. Don't overuse - only where it genuinely improves expression
3. Place sound tags ([laugh], [sigh]) AFTER the triggering phrase
4. Place style tags ([happy], [angry]) BEFORE the emotional section
5. Return ONLY the modified text, no explanations

Examples:
- "That's so funny!" → "That's so funny! [laugh]"
- "I can't believe this..." → "[sigh] I can't believe this..."
- "I'M SO ANGRY!" → "[angry] I'm so angry!"
"""


def _add_tags_anthropic(text: str, model_type: str) -> str:
    """Use Anthropic API to add expression tags."""
    import os
    try:
        import anthropic
    except ImportError:
        return _add_tags_simple(text)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _add_tags_simple(text)

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=_get_llm_system_prompt(model_type),
        messages=[{"role": "user", "content": text}]
    )

    return response.content[0].text.strip()


def _add_tags_mlx(text: str, model_type: str, model_name: Optional[str] = None) -> str:
    """Use local MLX model to add expression tags."""
    try:
        from mlx_lm import load, generate
    except ImportError:
        return _add_tags_simple(text)

    if model_name is None:
        model_name = "mlx-community/Llama-3.2-3B-Instruct-4bit"

    model, tokenizer = load(model_name)

    messages = [
        {"role": "system", "content": _get_llm_system_prompt(model_type)},
        {"role": "user", "content": text}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    response = generate(model, tokenizer, prompt=prompt, max_tokens=512, verbose=False)

    return response.strip()
