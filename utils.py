import os
import re
import unicodedata
from datetime import datetime, timezone, timedelta
from openai import OpenAI

def format_text_with_periods(text: str) -> str:
    """Format text by adding line breaks after periods for better readability"""
    if not text:
        return text
    
    # Replace periods followed by space with period + newline
    formatted = re.sub(r'。\s*', '。\n', text)
    
    # Clean up multiple consecutive newlines
    formatted = re.sub(r'\n\s*\n', '\n\n', formatted)
    
    # Remove leading/trailing whitespace
    formatted = formatted.strip()
    
    return formatted

def get_japanese_datetime() -> str:
    """Get current datetime in Japanese timezone (JST)"""
    jst = timezone(timedelta(hours=9))
    now = datetime.now(jst)
    return now.strftime("%Y-%m-%d %H:%M")

def normalize_tech_terms(text: str) -> str:
    """
    Normalize text with two-stage process:
    1) Unicode NFKC normalization (full-width → half-width)
    2) Whitelist-based tech term replacement
    """
    if not text:
        return text
    
    # Stage A: Unicode normalization (full-width → half-width)
    normalized = unicodedata.normalize("NFKC", text)
    
    # Stage B: Whitelist-based tech term replacement
    tech_terms = {
        ".env": ".env",
        "readme.md": "README.md",
        "README.md": "README.md",
        "history.md": "history.md",
        "git": "Git",
        "github": "GitHub",
        "python": "Python",
        "notion": "Notion",
        "discord": "Discord",
        "openai": "OpenAI",
        "gpt": "GPT",
        "api": "API",
        "json": "JSON",
        "html": "HTML",
        "css": "CSS",
        "javascript": "JavaScript",
        "typescript": "TypeScript",
        "react": "React",
        "vue": "Vue",
        "nodejs": "Node.js",
        "npm": "npm",
        "yarn": "yarn",
        "docker": "Docker",
        "kubernetes": "Kubernetes",
        "aws": "AWS",
        "gcp": "GCP",
        "azure": "Azure"
    }
    
    result = normalized
    for original, replacement in tech_terms.items():
        # Case-insensitive replacement, preserving word boundaries
        pattern = r'\b' + re.escape(original) + r'\b'
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    return result

def postprocess_with_ai(text: str, context_hint: str = "", test_mode: bool = False) -> str:
    """
    Post-process text using AI for light correction of punctuation and typos.
    
    Args:
        text: Input text to correct
        context_hint: Context information (title + summary)
        test_mode: If True, returns both gpt-4o-mini and gpt-4o results
        
    Returns:
        Corrected text, or formatted comparison in test mode
    """
    if not text.strip():
        return text
    
    system_prompt = """入力テキストを壊さず、句読点・誤字を軽く整える。内容は変えない。専門用語やタグ（例: .env, README.md）はそのまま保持。自然な日本語に軽く整形するだけで、大幅な変更は不要。"""
    
    user_prompt = f"以下のテキストを軽く補正してください：\n\n{text}"
    if context_hint.strip():
        user_prompt = f"コンテキスト: {context_hint}\n\n{user_prompt}"
    
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def get_ai_correction(model: str) -> str:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"AI correction error with {model}: {e}")
            return text
    
    if test_mode:
        mini_result = get_ai_correction("gpt-4o-mini")
        gpt4o_result = get_ai_correction("gpt-4o")
        
        return f"""=== gpt-4o-mini 補正結果 ===
{mini_result}

=== gpt-4o 補正結果 ===
{gpt4o_result}"""
    else:
        return get_ai_correction("gpt-4o")

def format_realtime_text(text: str) -> str:
    """Format text for real-time display with sentence-end priority"""
    if not text.strip():
        return text
    
    # Check for sentence endings
    sentence_endings = ['。', '！', '？', '.', '!', '?']
    
    # Find the last sentence ending
    last_ending_pos = -1
    for ending in sentence_endings:
        pos = text.rfind(ending)
        if pos > last_ending_pos:
            last_ending_pos = pos
    
    # If sentence ending found, return up to that point (including the ending)
    if last_ending_pos >= 0:
        return text[:last_ending_pos + 1]
    
    # Otherwise, return the entire text (chunked display)
    return text

def remove_chunk_duplicates(chunks: list) -> str:
    """Remove duplicates when combining chunks and return combined text"""
    if not chunks:
        return ""
    
    if len(chunks) == 1:
        return chunks[0]
    
    combined = chunks[0]
    
    # Local duplicate removal - exact match at boundaries
    for i in range(1, len(chunks)):
        current_chunk = chunks[i]
        
        # Find overlapping text between end of combined and start of current
        max_overlap = min(len(combined), len(current_chunk))
        overlap_found = 0
        
        for overlap_len in range(max_overlap, 0, -1):
            if combined[-overlap_len:] == current_chunk[:overlap_len]:
                overlap_found = overlap_len
                break
        
        # Add current chunk without the overlapping part
        combined += current_chunk[overlap_found:]
    
    return combined

def refine_duplicates_with_ai(text: str) -> str:
    """Use GPT-4o to refine text and remove semantic duplicates"""
    if not text.strip():
        return text
    
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    system_prompt = """音声認識テキストの重複や表記違いを自然に整理してください。
- 同じ意味の単語や文が連続している場合は、より自然な表現に統合
- 音声認識特有の繰り返しや言い直しを削除
- 文脈を保ちながら読みやすく整形
- 内容は変更せず、重複のみ整理"""
    
    user_prompt = f"以下のテキストの重複を整理してください：\n\n{text}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"AI duplicate removal failed: {e}")
        return text