import os
import sys
import threading
import time
import unicodedata
import wave
from typing import Optional

import numpy as np
import sounddevice as sd
from openai import OpenAI
from dotenv import load_dotenv

from discord_client import DiscordClient
from notion_api import NotionClient
from utils import format_text_with_periods, get_japanese_datetime

load_dotenv()

# User prompt messages
MSG_SAVE_CONFIRM = "この録音を保存しますか？ (y=保存 / n=破棄): "  # EN: "Do you want to save this recording? (y=save / n=discard): "
MSG_DISCARD_CONFIRM = "本当にこの録音を破棄しますか？ (y=破棄 / n=保存に戻る): "  # EN: "Are you sure you want to discard this recording? (y=discard / n=go back to save): "
MSG_ENTER_TITLE = "タイトルを入力してください: "  # EN: "Enter title: "
MSG_ENTER_SUMMARY = "概要を入力してください（空でも可）: "  # EN: "Enter summary (optional): "
MSG_INPUT_INVALID = "入力が認識できません。y/yes/はい または n/no/いいえ を入力してください。"

class AudioRecorder:
    def __init__(self):
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_duration = 20
        self.overlap_duration = 2
        self.is_recording = False
        self.audio_data = []
        self.stop_event = threading.Event()
        
    def record_audio(self):
        """Record audio in chunks with overlap"""
        chunk_frames = int(self.sample_rate * self.chunk_duration)
        overlap_frames = int(self.sample_rate * self.overlap_duration)
        
        def callback(indata, frames, time, status):
            if status:
                print(f"Audio callback status: {status}")
            if not self.stop_event.is_set():
                self.audio_data.extend(indata.copy())
        
        print("Recording started. Press Enter to stop...")
        print("=" * 50)
        
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=callback,
            dtype=np.int16
        ):
            while not self.stop_event.is_set():
                time.sleep(0.1)
    
    def start_recording(self):
        """Start recording in a separate thread"""
        self.is_recording = True
        self.audio_data = []
        self.stop_event.clear()
        
        recording_thread = threading.Thread(target=self.record_audio)
        recording_thread.daemon = True
        recording_thread.start()
        
        return recording_thread
    
    def stop_recording(self):
        """Stop recording"""
        self.stop_event.set()
        self.is_recording = False
        print("\nRecording stopped.")
    
    def save_audio_to_file(self, filename: str):
        """Save recorded audio to WAV file"""
        if not self.audio_data:
            return False
            
        audio_array = np.array(self.audio_data, dtype=np.int16)
        
        with wave.open(filename, 'w') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_array.tobytes())
        
        return True

class SpeechTranscriber:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.discord_client = DiscordClient()
        self.notion_client = NotionClient()
        self.recorder = AudioRecorder()
    
    def normalize_input(self, user_input: str) -> str:
        """Normalize user input using Unicode NFKC, strip, and lower"""
        return unicodedata.normalize("NFKC", user_input).strip().lower()
    
    def get_yes_no_input(self, prompt: str) -> bool:
        """Get robust yes/no input with Unicode normalization"""
        while True:
            print(prompt, end="")
            user_input = input()
            normalized = self.normalize_input(user_input)
            
            # Yes patterns
            if normalized in ['y', 'yes', 'はい']:
                return True
            
            # No patterns  
            if normalized in ['n', 'no', 'いいえ']:
                return False
            
            # Unrecognized input
            print(MSG_INPUT_INVALID)
    
    def confirm_discard(self) -> bool:
        """Two-stage confirmation for discarding recording"""
        return self.get_yes_no_input(MSG_DISCARD_CONFIRM)
        
    def transcribe_audio(self, audio_file_path: str) -> Optional[str]:
        """Transcribe audio using OpenAI gpt-4o-transcribe"""
        try:
            with open(audio_file_path, 'rb') as audio_file:
                response = self.openai_client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=audio_file,
                    language="ja"
                )
                
            transcription = response.text
            if transcription:
                formatted_text = format_text_with_periods(transcription)
                return formatted_text
            else:
                print("Warning: Empty transcription received")
                return None
                
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None
    
    def process_transcription(self, text: str, title: str, summary: str):
        """Process transcription by sending to Discord and Notion"""
        success_count = 0
        
        # Send to Discord
        if self.discord_client.send_message(title, summary):
            print("✓ Successfully sent to Discord")
            success_count += 1
        else:
            print("✗ Failed to send to Discord")
        
        # Save to Notion
        if self.notion_client.create_page(title, summary, text):
            print("✓ Successfully saved to Notion")
            success_count += 1
        else:
            print("✗ Failed to save to Notion")
        
        return success_count == 2
    
    def run(self):
        """Main application loop"""
        print("Speech Transcriber")
        print("=" * 50)
        
        # Check environment variables
        required_vars = ['OPENAI_API_KEY', 'DISCORD_WEBHOOK_URL', 'NOTION_TOKEN', 'NOTION_PARENT_PAGE_ID']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"Error: Missing environment variables: {', '.join(missing_vars)}")
            print("Please check your .env file.")
            return
        
        try:
            # Start recording
            recording_thread = self.recorder.start_recording()
            
            # Wait for Enter key to stop recording
            input()
            
            # Stop recording
            self.recorder.stop_recording()
            recording_thread.join(timeout=2)
            
            if not self.recorder.audio_data:
                print("No audio data recorded.")
                return
            
            # Ask user if they want to save
            want_save = self.get_yes_no_input(f"\n{MSG_SAVE_CONFIRM}")
            
            if not want_save:
                # Two-stage confirmation for discard
                if self.confirm_discard():
                    print("Recording discarded.")
                    return
                else:
                    print("保存フローに戻ります。")
                    want_save = True
            
            # Save audio to temporary file
            temp_audio_file = "temp_recording.wav"
            if not self.recorder.save_audio_to_file(temp_audio_file):
                print("Error: Failed to save audio file")
                return
            
            print("Transcribing audio...")
            
            # Transcribe audio
            transcription = self.transcribe_audio(temp_audio_file)
            
            # Clean up temporary file
            try:
                os.remove(temp_audio_file)
            except:
                pass
            
            if not transcription:
                print("Failed to transcribe audio.")
                return
            
            print("\nTranscription:")
            print("-" * 30)
            print(transcription)
            print("-" * 30)
            
            # Get title and summary from user
            print(f"\n{MSG_ENTER_TITLE}", end="")
            title = input().strip()
            if not title:
                print("タイトルは必須です。")  # EN: "Title is required."
                return
            
            print(f"{MSG_ENTER_SUMMARY}", end="")
            summary = input().strip()
            
            # Process and save
            print("\nSaving...")
            success = self.process_transcription(transcription, title, summary)
            
            if success:
                print("✓ All operations completed successfully!")
            else:
                print("⚠ Some operations failed. Check the logs above.")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            self.recorder.stop_recording()
        except Exception as e:
            print(f"Error: {e}")
            self.recorder.stop_recording()

if __name__ == "__main__":
    app = SpeechTranscriber()
    app.run()