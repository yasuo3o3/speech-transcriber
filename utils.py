import re
from datetime import datetime, timezone, timedelta

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