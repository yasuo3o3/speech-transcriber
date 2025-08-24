import os
import time
from datetime import datetime
from typing import Optional

from notion_client import Client as NotionAPIClient

from utils import get_japanese_datetime

# Long text handling constants
PARAGRAPH_CHUNK_LIMIT = 1800
BATCH_SIZE = 50  
PAGE_SOFT_LIMIT_CHARS = 80000

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
    
    def chunk_paragraphs_for_notion(self, text: str, max_len: int = PARAGRAPH_CHUNK_LIMIT) -> list[str]:
        """Split text into chunks that fit within Notion's paragraph limits"""
        if not text.strip():
            return []
        
        lines = text.split('\n')
        chunks = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # If line is within limit, add as-is
            if len(line) <= max_len:
                chunks.append(line)
            else:
                # Split long line into chunks
                while len(line) > max_len:
                    chunks.append(line[:max_len])
                    line = line[max_len:]
                if line:
                    chunks.append(line)
        
        return chunks
    
    def make_paragraph_block(self, text: str) -> dict:
        """Create a paragraph block for Notion"""
        return {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {
                            "content": text
                        }
                    }
                ]
            }
        }
    
    def append_children_batched(self, blocks: list[dict], parent_id: str, batch_size: int = BATCH_SIZE) -> bool:
        """Append children blocks in batches with retry logic"""
        total_blocks = len(blocks)
        processed_blocks = 0
        
        for i in range(0, total_blocks, batch_size):
            batch = blocks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_blocks + batch_size - 1) // batch_size
            
            print(f"Sending batch {batch_num}/{total_batches} ({len(batch)} blocks)...")
            
            for attempt in range(self.max_retries):
                try:
                    response = self.client.blocks.children.append(
                        block_id=parent_id,
                        children=batch
                    )
                    
                    if response:
                        processed_blocks += len(batch)
                        break
                        
                except Exception as e:
                    print(f"Batch {batch_num} error (attempt {attempt + 1}): {e}")
                    
                    if "rate_limited" in str(e).lower() or "429" in str(e):
                        wait_time = self.retry_delay * (2 ** attempt)
                        print(f"Rate limited. Waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    
                if attempt < self.max_retries - 1:
                    print(f"Retrying batch {batch_num} in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    self.retry_delay *= 2
            else:
                print(f"Failed to send batch {batch_num} after {self.max_retries} attempts")
                print(f"Successfully processed {processed_blocks}/{total_blocks} blocks before failure")
                return False
        
        print(f"✓ Successfully sent all {processed_blocks} blocks")
        return True
    
    def create_or_split_page_for_long_content(self, title: str, header_blocks: list[dict], body_blocks: list[dict], parent_page_id: str) -> bool:
        """Create page(s) for content, splitting if necessary"""
        # Calculate total body text length
        total_body_chars = sum(
            len(block.get("paragraph", {}).get("rich_text", [{}])[0].get("text", {}).get("content", ""))
            for block in body_blocks
            if block.get("type") == "paragraph"
        )
        
        pages_created = 0
        
        # If content fits in one page
        if total_body_chars <= PAGE_SOFT_LIMIT_CHARS:
            page_title = f"{get_japanese_datetime()} - {title}"
            all_blocks = header_blocks + body_blocks
            
            if self._create_single_page(page_title, all_blocks, parent_page_id):
                pages_created = 1
                print(f"✓ Created single page with {len(all_blocks)} blocks ({total_body_chars} chars)")
                return True
            else:
                return False
        
        # Split into multiple pages
        print(f"Long content detected ({total_body_chars} chars). Splitting into multiple pages...")
        
        current_chars = 0
        current_body_blocks = []
        page_num = 0
        
        for block in body_blocks:
            block_chars = len(block.get("paragraph", {}).get("rich_text", [{}])[0].get("text", {}).get("content", ""))
            
            # If adding this block would exceed limit, create current page
            if current_chars + block_chars > PAGE_SOFT_LIMIT_CHARS and current_body_blocks:
                page_num += 1
                suffix = f"（続き{page_num}）" if page_num > 1 else ""
                page_title = f"{get_japanese_datetime()} - {title}{suffix}"
                all_blocks = header_blocks + current_body_blocks
                
                if self._create_single_page(page_title, all_blocks, parent_page_id):
                    pages_created += 1
                    print(f"✓ Created page {page_num} with {len(all_blocks)} blocks ({current_chars} chars)")
                else:
                    print(f"✗ Failed to create page {page_num}")
                    return False
                
                current_body_blocks = []
                current_chars = 0
            
            current_body_blocks.append(block)
            current_chars += block_chars
        
        # Create final page if there are remaining blocks
        if current_body_blocks:
            page_num += 1
            suffix = f"（続き{page_num}）" if page_num > 1 else ""
            page_title = f"{get_japanese_datetime()} - {title}{suffix}"
            all_blocks = header_blocks + current_body_blocks
            
            if self._create_single_page(page_title, all_blocks, parent_page_id):
                pages_created += 1
                print(f"✓ Created final page {page_num} with {len(all_blocks)} blocks ({current_chars} chars)")
            else:
                print(f"✗ Failed to create final page {page_num}")
                return False
        
        print(f"✓ Created {pages_created} pages total")
        return True
    
    def _create_single_page(self, page_title: str, all_blocks: list[dict], parent_page_id: str) -> bool:
        """Create a single page with all blocks"""
        for attempt in range(self.max_retries):
            try:
                # Create page first
                response = self.client.pages.create(
                    parent={"page_id": parent_page_id},
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
                    }
                )
                
                if not response or not response.get('id'):
                    print(f"Failed to create page: {response}")
                    continue
                
                page_id = response['id']
                
                # Add blocks in batches
                return self.append_children_batched(all_blocks, page_id)
                
            except Exception as e:
                print(f"Page creation error (attempt {attempt + 1}): {e}")
                
                if "rate_limited" in str(e).lower():
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                
            if attempt < self.max_retries - 1:
                print(f"Retrying page creation in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
                self.retry_delay *= 2
        
        return False
    
    def create_page(self, title: str, summary: str, full_text: str) -> bool:
        """Create a new page in Notion with proper block structure: H1(title) -> H2(概要) -> P(summary) -> H2(全文) -> P(text)"""
        if not self.client:
            print("Error: NOTION_TOKEN not configured")
            return False
            
        if not self.parent_page_id:
            print("Error: NOTION_PARENT_PAGE_ID not configured")
            return False
        
        # Build structured header blocks with proper separation
        header_blocks = self._build_structured_header_blocks(summary)
        
        # Process full text into paragraph blocks with chunking
        text_chunks = self.chunk_paragraphs_for_notion(full_text)
        body_blocks = [self.make_paragraph_block(chunk) for chunk in text_chunks]
        
        print(f"Processing {len(text_chunks)} text chunks into {len(body_blocks)} blocks")
        
        # Create page(s) with automatic splitting if needed
        return self.create_or_split_page_for_long_content(
            title, header_blocks, body_blocks, self.parent_page_id
        )
    
    def _build_structured_header_blocks(self, summary: str) -> list[dict]:
        """Build properly structured header blocks with clear separation"""
        blocks = []
        
        # H2: 概要 (as separate block)
        blocks.append({
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
        })
        
        # Paragraph: 概要本文 (as separate block)
        if summary and summary.strip():
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": summary.strip()
                            }
                        }
                    ]
                }
            })
        
        # H2: 全文 (as separate block)
        blocks.append({
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
        })
        
        return blocks
    
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