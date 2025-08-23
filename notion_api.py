import os
import time
from datetime import datetime
from typing import Optional

from notion_client import Client as NotionAPIClient

from utils import get_japanese_datetime

class NotionClient:
    def __init__(self):
        self.token = os.getenv('NOTION_TOKEN')
        self.parent_page_id = os.getenv('NOTION_PARENT_PAGE_ID')
        self.max_retries = 3
        self.retry_delay = 1
        
        if self.token:
            self.client = NotionAPIClient(auth=self.token)
        else:
            self.client = None
    
    def create_page(self, title: str, summary: str, full_text: str) -> bool:
        """Create a new page in Notion with the transcription"""
        if not self.client:
            print("Error: NOTION_TOKEN not configured")
            return False
            
        if not self.parent_page_id:
            print("Error: NOTION_PARENT_PAGE_ID not configured")
            return False
        
        timestamp = get_japanese_datetime()
        page_title = f"{timestamp} - {title}"
        
        page_content = self._build_page_content(summary, full_text)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.pages.create(
                    parent={"page_id": self.parent_page_id},
                    properties={
                        "title": {
                            "title": [
                                {
                                    "type": "text",
                                    "text": {
                                        "content": page_title
                                    }
                                }
                            ]
                        }
                    },
                    children=page_content
                )
                
                if response and response.get('id'):
                    return True
                else:
                    print(f"Unexpected Notion response: {response}")
                    
            except Exception as e:
                print(f"Notion API error (attempt {attempt + 1}): {e}")
                
                if "rate_limited" in str(e).lower():
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                
            if attempt < self.max_retries - 1:
                print(f"Retrying Notion API in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
                self.retry_delay *= 2
        
        print(f"Failed to create Notion page after {self.max_retries} attempts")
        return False
    
    def _build_page_content(self, summary: str, full_text: str) -> list:
        """Build the content structure for the Notion page"""
        content = []
        
        # Add summary section
        content.extend([
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": "概要"
                            }
                        }
                    ]
                }
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": summary
                            }
                        }
                    ]
                }
            },
            {
                "object": "block",
                "type": "divider",
                "divider": {}
            }
        ])
        
        # Add full text section
        content.extend([
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": "全文"
                            }
                        }
                    ]
                }
            }
        ])
        
        # Split full text by lines and add as paragraphs
        lines = full_text.split('\n')
        for line in lines:
            if line.strip():
                content.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {
                                    "content": line.strip()
                                }
                            }
                        ]
                    }
                })
        
        return content