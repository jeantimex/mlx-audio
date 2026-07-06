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

import random
import re
from typing import Optional

# Canonical tags (model-agnostic) -> Model-specific tags
TAG_MAPPINGS = {
    "fish-speech": {
        # Sounds
        "laugh": "[laughing]",
        "laughter": "[laughing]",
        "chuckle": "[chuckle]",
        "sigh": "[sigh]",
        "gasp": "[inhale]",
        "cough": "[clearing throat]",
        "groan": "[moaning]",
        "sniff": "[inhale]",
        "shush": "[whisper]",
        "clear_throat": "[clearing throat]",
        # Styles/Emotions
        "happy": "[excited]",
        "angry": "[angry]",
        "dramatic": "[emphasis]",
        "sarcastic": "[laughing tone]",
        "whisper": "[whisper]",
        "whispering": "[whisper]",
        "crying": "[sad]",
        "fear": "[shocked]",
        "surprised": "[surprised]",
        "narration": "[low voice]",
        # Questions/Surprise
        "question": "[surprised]",
        "surprise": "[surprised]",
        # Fish Speech specific
        "pause": "[pause]",
        "emphasis": "[emphasis]",
        "excited": "[excited]",
        "singing": "[singing]",
        "shouting": "[shouting]",
        "loud": "[loud]",
        "exhale": "[exhale]",
        "inhale": "[inhale]",
        "panting": "[panting]",
        "delight": "[delight]",
        "dissatisfaction": "[angry]",
    },
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
SUPPORTED_MODELS = ["chatterbox-turbo", "omnivoice", "fish-speech"]


def detect_model_type(model_name: str) -> Optional[str]:
    """Detect model type from model name/path."""
    model_lower = model_name.lower()

    if "chatterbox-turbo" in model_lower or "chatterbox_turbo" in model_lower:
        return "chatterbox-turbo"
    elif "omnivoice" in model_lower:
        return "omnivoice"
    elif "chatterbox" in model_lower:
        return "chatterbox"
    elif "fish" in model_lower and ("speech" in model_lower or "audio" in model_lower or "s2" in model_lower):
        return "fish-speech"

    return None


def supports_expression_tags(model_name: str) -> bool:
    """Check if a model supports expression tags."""
    model_type = detect_model_type(model_name)
    return model_type in SUPPORTED_MODELS


def convert_tags(text: str, target_model: str, preserve_unknown: bool = True) -> str:
    """
    Convert expression tags in text to the target model's format.

    Args:
        text: Text with expression tags (any format)
        target_model: Target model name/path
        preserve_unknown: If True, keep unknown tags as-is instead of removing them

    Returns:
        Text with tags converted to target model's format
    """
    model_type = detect_model_type(target_model)

    if model_type is None or model_type not in TAG_MAPPINGS:
        # Unknown model - preserve all tags if preserve_unknown, else remove
        if preserve_unknown:
            return text
        return re.sub(r'\[[^\]]+\]', '', text)

    mapping = TAG_MAPPINGS[model_type]

    # Build set of valid tags for this model (for quick lookup)
    valid_tags = set(mapping.values()) - {""}  # Exclude empty mappings

    # Find all tags in text
    def replace_tag(match):
        original_tag = f"[{match.group(1)}]"
        tag_content = match.group(1).lower().replace(" ", "_").replace("-", "_")

        # Check if it's already a valid tag for this model (preserve as-is)
        if original_tag in valid_tags:
            return original_tag

        # Check case-insensitive match against valid tags
        for valid_tag in valid_tags:
            if valid_tag.lower() == original_tag.lower():
                return valid_tag

        # Direct lookup in mapping
        if tag_content in mapping:
            mapped = mapping[tag_content]
            return mapped if mapped else (original_tag if preserve_unknown else "")

        # Try without brackets that might be in the mapping value
        for canonical, model_tag in mapping.items():
            if canonical == tag_content:
                return model_tag if model_tag else (original_tag if preserve_unknown else "")

        # Unknown tag - preserve or remove based on flag
        return original_tag if preserve_unknown else ""

    result = re.sub(r'\[([^\]]+)\]', replace_tag, text)

    # Clean up extra spaces
    result = re.sub(r'\s+', ' ', result).strip()

    return result


def add_expression_tags(
    text: str,
    target_model: str,
    provider: str = "simple",
    llm_model: Optional[str] = None,
    temperature: float = 1.0,
) -> str:
    """
    Add expression tags to text for a specific TTS model.

    Args:
        text: Input text without tags
        target_model: Target TTS model name/path
        provider: "simple" (rule-based), "anthropic", or "mlx"
        llm_model: LLM model for mlx provider
        temperature: Controls how many tags to add (0.0 = none, 1.0 = all matches).
                    Default 0.5 adds tags to ~50% of matches.

    Returns:
        Text with appropriate expression tags for the model
    """
    model_type = detect_model_type(target_model)

    if model_type is None or model_type not in SUPPORTED_MODELS:
        # Model doesn't support tags, return as-is
        return text

    # Temperature 0 means no tags
    if temperature <= 0:
        return text

    if provider == "simple":
        tagged_text = _add_tags_simple(text, temperature=temperature)
    elif provider == "anthropic":
        tagged_text = _add_tags_anthropic(text, model_type, temperature=temperature)
    elif provider == "mlx":
        tagged_text = _add_tags_mlx(text, model_type, llm_model, temperature=temperature)
    else:
        tagged_text = text

    # Convert to target model's tag format
    return convert_tags(tagged_text, target_model)


def _add_tags_simple(text: str, temperature: float = 1.0) -> str:
    """Simple rule-based tag insertion using canonical tags.

    Args:
        text: Input text
        temperature: Controls tag density (0.0 = none, 1.0 = all matches)
    """
    result = text

    def prob_sub(pattern, replacement, text, flags=0):
        """Apply regex substitution with probability based on temperature."""
        def replacer(match):
            # Use temperature as probability threshold
            if random.random() < temperature:
                # Expand the replacement pattern
                return match.expand(replacement)
            return match.group(0)  # Keep original
        return re.sub(pattern, replacer, text, flags=flags)

    # === English patterns ===

    # Laughing sounds - REPLACE with tag (not append)
    # haha, hehe, lol, lmao, rofl, emojis → [laugh]
    result = prob_sub(
        r'(ha\s*ha+|he\s*he+|heh+|lol+|lmao|rofl|😂|🤣|🤭)',
        r'[laugh]',
        result,
        flags=re.IGNORECASE
    )
    # Descriptive laughter - keep text, add tag
    result = prob_sub(
        r'(that\'s (so )?funny|hilarious|too funny)',
        r'\1 [laugh]',
        result,
        flags=re.IGNORECASE
    )

    # Sighing sounds - REPLACE with tag
    result = prob_sub(
        r'\b(sigh+|ugh+)\b',
        r'[sigh]',
        result,
        flags=re.IGNORECASE
    )
    # Descriptive sighing - keep text, add tag
    result = prob_sub(
        r'(unfortunately|sadly|i\'m (so )?tired)',
        r'[sigh] \1',
        result,
        flags=re.IGNORECASE
    )

    # Surprise exclamations - keep text, add tag
    result = prob_sub(
        r'\b(omg|wow+|whoa+|oh my god|no way|unbelievable)\b',
        r'[surprise] \1',
        result,
        flags=re.IGNORECASE
    )
    result = prob_sub(
        r'(what\?!+|really\?!+)',
        r'[surprise] \1',
        result,
        flags=re.IGNORECASE
    )

    # Question sounds - REPLACE with tag
    result = prob_sub(
        r'\b(huh|hmm+|eh)\?',
        r'[question]',
        result,
        flags=re.IGNORECASE
    )

    # Anger indicators (ALL CAPS with exclamation)
    result = prob_sub(
        r'\b([A-Z]{4,}!+)',
        r'[angry] \1',
        result
    )

    # Whisper sounds - REPLACE with tag
    result = prob_sub(
        r'\b(shh+|psst+)\b',
        r'[whisper]',
        result,
        flags=re.IGNORECASE
    )
    # Descriptive whisper - keep text, add tag
    result = prob_sub(
        r'\b(quietly|softly)\b',
        r'[whisper] \1',
        result,
        flags=re.IGNORECASE
    )

    # Happy sounds - REPLACE with tag
    result = prob_sub(
        r'\b(yay+|woohoo+|woo+)\b',
        r'[happy]',
        result,
        flags=re.IGNORECASE
    )
    # Descriptive happy - keep text, add tag
    result = prob_sub(
        r'\b(wonderful|amazing|fantastic|awesome)\b',
        r'[happy] \1',
        result,
        flags=re.IGNORECASE
    )

    # === Chinese patterns ===
    # Onomatopoeia are REPLACED with tags, descriptive expressions get tags added

    # Laughing sounds - REPLACE with tag
    # 哈哈/嘻嘻/呵呵/噗 = laughter sounds → [laugh]
    result = prob_sub(
        r'(哈哈+|嘻嘻+|呵呵+|噗+|hhh+)',
        r'[laugh]',
        result
    )
    # Descriptive laughter - keep text, add tag
    result = prob_sub(
        r'(笑死了?|太好笑了)',
        r'\1 [laugh]',
        result
    )

    # Happy sounds - REPLACE with tag
    result = prob_sub(
        r'(耶+|哇+塞|哇+哦+)',
        r'[happy]',
        result
    )
    # Descriptive happy - keep text, add tag
    result = prob_sub(
        r'(真?他妈的?开心|真?开心|好开心|很开心|超开心|太开心了?|好高兴|很高兴|真高兴|太高兴了?|太棒了|太好了|真好|太爽了|好爽|爽死了?)',
        r'[happy] \1',
        result
    )

    # Sighing sounds - REPLACE with tag
    result = prob_sub(
        r'(唉+|哎+|呃+)',
        r'[sigh]',
        result
    )
    # Descriptive sighing - keep text, add tag
    result = prob_sub(
        r'(算了|无奈|好累|真累|太累了|累死了|心累)',
        r'[sigh] \1',
        result
    )

    # Surprise sounds - REPLACE with tag
    result = prob_sub(
        r'(哇+(?!塞|哦)|我靠|卧槽|我去|天啊*|我滴妈)',
        r'[surprise]',
        result
    )
    # Descriptive surprise - keep text, add tag
    result = prob_sub(
        r'(天哪|我的天|太厉害了?|不可思议|很神奇|太神奇了?|真神奇|难以置信|不敢相信|牛逼)',
        r'[surprise] \1',
        result
    )

    # Anger sounds - REPLACE with tag
    result = prob_sub(
        r'(靠+|草+|妈的)',
        r'[angry]',
        result
    )
    # Descriptive anger - keep text, add tag
    result = prob_sub(
        r'(气死.{0,2}了?|烦死.{0,2}了?|讨厌|该死|真烦|好烦|太烦了|受不了|忍不了)',
        r'[angry] \1',
        result
    )

    # Question sounds - REPLACE with tag
    result = prob_sub(
        r'(啊\?|嗯\?|哈\?)',
        r'[question]',
        result
    )
    # Descriptive question - keep text, add tag
    result = prob_sub(
        r'(真的吗|是吗|什么\?|啥\?|为什么|怎么回事)',
        r'[question] \1',
        result
    )

    # Dissatisfaction sounds - REPLACE with tag
    result = prob_sub(
        r'(哼+|切+|tsk+)',
        r'[dissatisfaction]',
        result
    )
    # Descriptive dissatisfaction - keep text, add tag
    result = prob_sub(
        r'(无语|服了|醉了)',
        r'[dissatisfaction] \1',
        result
    )

    # Clean up multiple spaces
    result = re.sub(r'\s+', ' ', result).strip()

    return result


def _get_llm_system_prompt(model_type: str, temperature: float = 1.0) -> str:
    """Get system prompt for LLM tag insertion."""
    if model_type == "omnivoice":
        tags = "[laughter], [sigh], [question-ah], [question-oh], [surprise-ah], [surprise-oh], [dissatisfaction-hnn]"
    elif model_type == "fish-speech":
        tags = "[laughing], [chuckle], [sigh], [excited], [angry], [sad], [whisper], [shouting], [surprised], [pause], [emphasis], [singing], [inhale], [exhale], [delight]"
    else:
        tags = "[laugh], [chuckle], [sigh], [gasp], [happy], [angry], [surprised], [whispering], [dramatic], [sarcastic]"

    # Adjust density instruction based on temperature
    if temperature <= 0.3:
        density = "Be very conservative - only add tags for the most obvious and strong expressions."
    elif temperature <= 0.6:
        density = "Be moderate - add tags where expressions are clearly present, but don't overdo it."
    else:
        density = "Be generous - add tags wherever an expression could naturally occur."

    return f"""You add expression tags to text for text-to-speech synthesis.

Available tags: {tags}

Rules:
1. Add tags where natural expressions would occur
2. {density}
3. Place sound tags ([laugh], [sigh]) AFTER the triggering phrase
4. Place style tags ([happy], [angry]) BEFORE the emotional section
5. Preserve any existing tags in the text - do not remove or modify them
6. Return ONLY the modified text, no explanations

Examples:
- "That's so funny!" → "That's so funny! [laugh]"
- "I can't believe this..." → "[sigh] I can't believe this..."
- "I'M SO ANGRY!" → "[angry] I'm so angry!"
"""


def _add_tags_anthropic(text: str, model_type: str, temperature: float = 1.0) -> str:
    """Use Anthropic API to add expression tags."""
    import os
    try:
        import anthropic
    except ImportError:
        return _add_tags_simple(text, temperature=temperature)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _add_tags_simple(text, temperature=temperature)

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=_get_llm_system_prompt(model_type, temperature=temperature),
        messages=[{"role": "user", "content": text}]
    )

    return response.content[0].text.strip()


def _add_tags_mlx(text: str, model_type: str, model_name: Optional[str] = None, temperature: float = 1.0) -> str:
    """Use local MLX model to add expression tags."""
    try:
        from mlx_lm import load, generate
    except ImportError:
        return _add_tags_simple(text, temperature=temperature)

    if model_name is None:
        model_name = "mlx-community/Llama-3.2-3B-Instruct-4bit"

    model, tokenizer = load(model_name)

    messages = [
        {"role": "system", "content": _get_llm_system_prompt(model_type, temperature=temperature)},
        {"role": "user", "content": text}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    response = generate(model, tokenizer, prompt=prompt, max_tokens=512, verbose=False)

    return response.strip()
