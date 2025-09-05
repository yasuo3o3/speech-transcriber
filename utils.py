import os
import re
import unicodedata
from datetime import datetime, timezone, timedelta
from typing import Tuple
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

def postprocess_with_ai(text: str, context_hint: str = "", test_mode: bool = False, model: str = "gpt-4o-mini") -> str:
    """
    Post-process text using AI for boundary-only correction (non-summarizing).
    
    Args:
        text: Input text to correct
        context_hint: Context information (title + summary)
        test_mode: If True, returns both gpt-4o-mini and gpt-4o results
        model: Model to use for production (default: gpt-4o-mini)
        
    Returns:
        Corrected text, or formatted comparison in test mode
    """
    if not text.strip():
        return text
    
    # Boundary-only correction prompt (anti-summarization)
    system_prompt = """以下のテキストを、意味の要約・圧縮・言い換えを一切行わずに軽く整形してください。
タスク：
- 句読点・誤字の軽微な修正のみ。
- 専門用語やタグ（例: .env, README.md）はそのまま保持。
- 数列や箇条書き、反復表現の短文化は禁止。
- 非重複部分の削除、語順変更、文意の改変は一切禁止。
出力は、入力本文と同じ内容・分量を維持したまま、軽微な整形のみ行ったテキストとする。
返答は本文のみ（前置き/説明/囲みテキスト不要）。"""
    
    user_prompt = f"以下のテキストを軽く補正してください：\n\n{text}"
    if context_hint.strip():
        user_prompt = f"コンテキスト: {context_hint}\n\n{user_prompt}"
    
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def get_ai_correction(correction_model: str) -> str:
        try:
            response = client.chat.completions.create(
                model=correction_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.05,  # Lower temperature for consistency
                max_tokens=4000
            )
            result = response.choices[0].message.content.strip()
            
            # Safety check: revert if too much change
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, text, result).ratio()
            length_change = abs(len(result) - len(text)) / len(text) if len(text) > 0 else 0
            
            if similarity < 0.95 or length_change > 0.1:
                print(f"AI postprocess reverted (similarity={similarity:.3f}, delta={length_change:.3f})")
                return text
            
            return result
        except Exception as e:
            print(f"AI correction error with {correction_model}: {e}")
            return text
    
    if test_mode:
        mini_result = get_ai_correction("gpt-4o-mini")
        gpt4o_result = get_ai_correction("gpt-4o")
        
        # In test mode, show both but adopt mini for final output
        print(f"""=== gpt-4o-mini 補正結果 ===
{mini_result}

=== gpt-4o 補正結果 ===
{gpt4o_result}""")
        print("✓ Test mode: Showing both results, adopting gpt-4o-mini")
        return mini_result
    else:
        return get_ai_correction(model)

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

def _calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate LCS-based similarity ratio between two texts"""
    if not text1 or not text2:
        return 0.0
    
    # Simple LCS approximation using difflib
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, text1, text2)
    return matcher.ratio()

def _should_skip_ai_dedup(text: str) -> Tuple[bool, str]:
    """Determine if AI deduplication should be skipped based on text patterns"""
    if not text.strip():
        return False, "empty_text"
    
    # Count digits and total characters
    digit_count = sum(1 for c in text if c.isdigit())
    total_chars = len(text.strip())
    digit_ratio = digit_count / total_chars if total_chars > 0 else 0
    
    # Skip if high digit ratio (likely numeric sequences)
    if digit_ratio > 0.3:
        return True, f"high_digit_ratio_{digit_ratio:.2f}"
    
    # Check for repetitive patterns (simple heuristic)
    words = text.split()
    if len(words) > 10:
        unique_words = set(words)
        unique_ratio = len(unique_words) / len(words)
        if unique_ratio < 0.3:  # Low unique word ratio
            return True, f"low_unique_ratio_{unique_ratio:.2f}"
    
    return False, "pass"

def refine_duplicates_with_ai(text: str, mode: str = "boundary_only") -> str:
    """Use GPT-4o to refine text and remove boundary duplicates only (safe mode)"""
    if not text.strip():
        return text
    
    # Check mode setting
    if mode == "off":
        print("AI dedup skipped (mode=off)")
        return text
    
    # Auto-bypass check
    should_skip, reason = _should_skip_ai_dedup(text)
    if should_skip:
        print(f"AI dedup skipped (pattern heuristic: {reason})")
        return text
    
    original_length = len(text)
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Strict boundary-only prompt
    system_prompt = """以下のテキストを、意味の要約・圧縮・言い換えを一切行わずに処理してください。
タスクは1点のみ：
- チャンク結合時に生じる「末尾と先頭の重複」だけを取り除く（完全一致または2〜20文字程度の軽微な表記差のみ）。
禁止事項：
- 数列や箇条書き、反復表現の短文化（例：「1,2,3, …」を「連続した数字」と要約）は禁止。
- 非重複部分の削除、語順変更、文意の改変は一切禁止。
- 改行は原文の文末記号（。！？）を尊重し、過剰に詰めない。
出力は、入力本文と同じ内容を維持したまま、境界重複部分だけを取り除いたテキストとする。
返答は本文のみ（前置き/説明/囲みテキスト不要）。"""
    
    user_prompt = f"以下のテキストの境界重複のみを取り除いてください：\n\n{text}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.05,  # Lower temperature for consistency
            max_tokens=4000
        )
        
        ai_result = response.choices[0].message.content.strip()
        
        # Safety check: measure changes
        result_length = len(ai_result)
        length_change_ratio = abs(result_length - original_length) / original_length if original_length > 0 else 0
        similarity_ratio = _calculate_text_similarity(text, ai_result)
        
        # Revert if changes are too significant
        if similarity_ratio < 0.97 or length_change_ratio > 0.05:
            print(f"AI dedup reverted (delta={length_change_ratio:.3f}, lcs={similarity_ratio:.3f})")
            return text
        
        print(f"Applied AI boundary dedup (chars: {original_length} → {result_length})")
        return ai_result
        
    except Exception as e:
        print(f"AI duplicate removal failed: {e}")
        return text