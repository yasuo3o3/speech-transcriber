import os
import time
from typing import Optional

import requests

class DiscordClient:
    def __init__(self):
        self.webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        self.max_retries = 3
        self.retry_delay = 1
        
    def send_message(self, title: str, summary: str) -> bool:
        """Send message to Discord via webhook with retry logic"""
        if not self.webhook_url:
            print("Error: DISCORD_WEBHOOK_URL not configured")
            return False
        
        # Strip whitespace and newlines from title and summary
        clean_title = title.strip()
        clean_summary = summary.strip()
        
        # Build message content without extra blank lines
        if clean_summary:
            message_content = f"**{clean_title}**\n{clean_summary}"
        else:
            message_content = f"**{clean_title}**"
        
        payload = {
            "content": message_content,
            "username": "Speech Transcriber"
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.webhook_url,
                    json=payload,
                    timeout=10
                )
                
                if response.status_code == 204:
                    return True
                elif response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', self.retry_delay))
                    print(f"Rate limited. Retrying after {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                else:
                    print(f"Discord webhook failed with status {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                print(f"Discord request error (attempt {attempt + 1}): {e}")
                
            if attempt < self.max_retries - 1:
                print(f"Retrying Discord webhook in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
                self.retry_delay *= 2
        
        print(f"Failed to send to Discord after {self.max_retries} attempts")
        return False