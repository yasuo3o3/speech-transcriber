import os
import sys
import threading
import time
import unicodedata
import wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Optional, Dict, List
import logging

import numpy as np
import sounddevice as sd
from openai import OpenAI
from dotenv import load_dotenv

from discord_client import DiscordClient
from notion_api import NotionClient
from utils import format_text_with_periods, get_japanese_datetime, normalize_tech_terms, postprocess_with_ai, format_realtime_text, remove_chunk_duplicates, refine_duplicates_with_ai

load_dotenv()

# Configuration flags
NORMALIZE_TECH_TERMS = os.getenv('NORMALIZE_TECH_TERMS', 'true').lower() == 'true'
POSTPROCESS_TEST_MODE = os.getenv('POSTPROCESS_TEST_MODE', 'false').lower() == 'true'

# Pipeline configuration
MAX_TRANSCRIBE_WORKERS = int(os.getenv('MAX_TRANSCRIBE_WORKERS', '2'))
TRANSCRIBE_QUEUE_MAX_SIZE = int(os.getenv('TRANSCRIBE_QUEUE_MAX_SIZE', '10'))
RING_BUFFER_SECONDS = int(os.getenv('RING_BUFFER_SECONDS', '300'))  # 5 minutes

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('SpeechTranscriber')

@dataclass
class AudioChunk:
    """Audio chunk with timing information"""
    start_ts: float
    end_ts: float
    audio_data: np.ndarray
    chunk_id: int
    capture_start: float
    capture_end: float

@dataclass
class TranscriptionResult:
    """Transcription result with timing and latency information"""
    chunk_id: int
    start_ts: float
    end_ts: float
    text: str
    api_start: float
    api_end: float
    api_latency_ms: int
    printed_at: Optional[float] = None

# User prompt messages
MSG_SAVE_CONFIRM = "今回の文字起こしを保存しますか？ (y=保存 / n=破棄): "  # EN: "Do you want to save this recording? (y=save / n=discard): "
MSG_DISCARD_CONFIRM = "破棄するのですね？ (y=破棄 / n=保存): "  # EN: "Are you sure you want to discard this recording? (y=discard / n=go back to save): "
MSG_ENTER_TITLE = "タイトルを入力してください: "  # EN: "Enter title: "
MSG_ENTER_SUMMARY = "概要を入力してください（空でも可）: "  # EN: "Enter summary (optional): "
MSG_INPUT_INVALID = "入力が認識できません。[y]または[n]を入力してください。"

class AudioRecorder:
    """Ring Buffer-based audio recorder (Producer)"""
    
    def __init__(self):
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_duration = 20
        self.overlap_duration = 2
        self.is_recording = False
        self.stop_event = threading.Event()
        
        # Ring buffer for audio samples - stores (timestamp, audio_data) tuples
        buffer_size = int(self.sample_rate * RING_BUFFER_SECONDS)
        self.ring_buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()
        self.recording_start_time = None
        
        logger.info(f"AudioRecorder initialized: {self.sample_rate}Hz, {buffer_size} samples buffer")
        
    def record_audio(self):
        """Record audio into ring buffer (lightweight producer)"""
        
        def callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio callback status: {status}")
            
            if not self.stop_event.is_set():
                # Only copy and timestamp - no heavy processing in callback
                current_time = time.inputBufferAdcTime
                audio_copy = indata.copy().flatten()  # Flatten for easier handling
                
                with self.buffer_lock:
                    # Store each sample with precise timestamp
                    for i, sample in enumerate(audio_copy):
                        sample_timestamp = current_time + (i / self.sample_rate)
                        self.ring_buffer.append((sample_timestamp, sample))
        
        print("Recording started. Press Enter to stop...")
        print("Real-time transcription will appear below:")
        print("=" * 50)
        
        self.recording_start_time = time.time()
        logger.info(f"Audio recording started at {self.recording_start_time}")
        
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=callback,
            dtype=np.int16
        ):
            while not self.stop_event.is_set():
                time.sleep(0.1)
        
        logger.info("Audio recording stopped")
    
    def start_recording(self):
        """Start recording in a separate thread"""
        self.is_recording = True
        self.ring_buffer.clear()
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
        logger.info("Recording stop signal sent")
    
    def extract_chunk(self, start_ts: float, end_ts: float) -> Optional[np.ndarray]:
        """Extract audio chunk from ring buffer by timestamp"""
        with self.buffer_lock:
            if not self.ring_buffer:
                return None
            
            chunk_samples = []
            for timestamp, sample in self.ring_buffer:
                if start_ts <= timestamp <= end_ts:
                    chunk_samples.append(sample)
            
            if chunk_samples:
                return np.array(chunk_samples, dtype=np.int16)
            return None
    
    def get_recording_duration(self) -> float:
        """Get current recording duration"""
        if self.recording_start_time is None:
            return 0.0
        return time.time() - self.recording_start_time

class AudioChunker:
    """Chunker Thread - extracts 20s/2s overlap chunks from ring buffer"""
    
    def __init__(self, recorder: AudioRecorder, transcribe_queue: Queue):
        self.recorder = recorder
        self.transcribe_queue = transcribe_queue
        self.chunk_duration = 20
        self.overlap_duration = 2
        self.chunk_interval = self.chunk_duration - self.overlap_duration  # 18s
        self.chunk_id_counter = 0
        self.is_running = False
        self.stop_event = threading.Event()
        
    def start(self):
        """Start chunker thread"""
        self.is_running = True
        self.stop_event.clear()
        self.chunk_id_counter = 0
        
        chunker_thread = threading.Thread(target=self._chunker_loop)
        chunker_thread.daemon = True
        chunker_thread.start()
        logger.info("AudioChunker started")
        return chunker_thread
    
    def stop(self):
        """Stop chunker thread"""
        self.stop_event.set()
        self.is_running = False
        logger.info("AudioChunker stop signal sent")
    
    def _chunker_loop(self):
        """Main chunker loop - time-based window sliding"""
        next_chunk_time = self.chunk_duration  # First chunk at 20s
        
        while not self.stop_event.is_set():
            current_duration = self.recorder.get_recording_duration()
            
            if current_duration >= next_chunk_time:
                capture_start = time.time()
                
                # Calculate chunk time window
                chunk_end_ts = self.recorder.recording_start_time + next_chunk_time
                chunk_start_ts = chunk_end_ts - self.chunk_duration
                
                # Extract audio data from ring buffer
                audio_data = self.recorder.extract_chunk(chunk_start_ts, chunk_end_ts)
                
                if audio_data is not None and len(audio_data) > 0:
                    self.chunk_id_counter += 1
                    capture_end = time.time()
                    
                    chunk = AudioChunk(
                        start_ts=chunk_start_ts,
                        end_ts=chunk_end_ts, 
                        audio_data=audio_data,
                        chunk_id=self.chunk_id_counter,
                        capture_start=capture_start,
                        capture_end=capture_end
                    )
                    
                    # Check queue size and warn if getting full
                    if self.transcribe_queue.qsize() >= TRANSCRIBE_QUEUE_MAX_SIZE:
                        logger.warning(f"Transcribe queue near capacity: {self.transcribe_queue.qsize()}/{TRANSCRIBE_QUEUE_MAX_SIZE}")
                    
                    try:
                        self.transcribe_queue.put(chunk, timeout=1.0)
                        logger.debug(f"Chunk {self.chunk_id_counter} queued: {chunk_start_ts:.1f}-{chunk_end_ts:.1f}s, {len(audio_data)} samples")
                        
                        # Check for audio gaps
                        if self.chunk_id_counter > 1:
                            expected_interval = self.chunk_interval
                            actual_interval = chunk_start_ts - (self.recorder.recording_start_time + (self.chunk_id_counter - 2) * self.chunk_interval)
                            if abs(actual_interval - expected_interval) > 0.5:
                                logger.warning(f"Audio timeline gap detected: expected {expected_interval}s, got {actual_interval:.1f}s")
                                
                    except:
                        logger.error(f"Failed to queue chunk {self.chunk_id_counter} - queue full, dropping")
                        
                else:
                    logger.warning(f"No audio data for chunk at {next_chunk_time}s")
                
                # Move to next chunk time
                next_chunk_time += self.chunk_interval
            else:
                time.sleep(0.5)  # Check every 500ms

class SpeechTranscriber:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.discord_client = DiscordClient()
        self.notion_client = NotionClient()
        self.recorder = AudioRecorder()
        
        # Async pipeline components
        self.transcribe_queue = Queue(maxsize=TRANSCRIBE_QUEUE_MAX_SIZE)
        self.result_queue = Queue()
        self.chunker = AudioChunker(self.recorder, self.transcribe_queue)
        self.executor = ThreadPoolExecutor(max_workers=MAX_TRANSCRIBE_WORKERS)
        self.printer = RealtimePrinter(self.result_queue)
        self.transcription_results: List[TranscriptionResult] = []
    
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
    
    def transcribe_chunk_async(self, chunk: AudioChunk) -> TranscriptionResult:
        """Transcribe audio chunk asynchronously (Consumer)"""
        api_start = time.time()
        
        try:
            logger.debug(f"Starting transcription for chunk {chunk.chunk_id}")
            
            # Save chunk to temporary WAV file
            temp_filename = f"temp_chunk_{chunk.chunk_id}.wav"
            with wave.open(temp_filename, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(chunk.audio_data.tobytes())
            
            # Transcribe the chunk
            with open(temp_filename, 'rb') as audio_file:
                response = self.openai_client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=audio_file,
                    language="ja"
                )
            
            # Clean up temp file
            try:
                os.remove(temp_filename)
            except:
                pass
            
            api_end = time.time()
            api_latency_ms = int((api_end - api_start) * 1000)
            
            transcription_text = response.text.strip() if response.text else ""
            
            result = TranscriptionResult(
                chunk_id=chunk.chunk_id,
                start_ts=chunk.start_ts,
                end_ts=chunk.end_ts,
                text=transcription_text,
                api_start=api_start,
                api_end=api_end,
                api_latency_ms=api_latency_ms
            )
            
            logger.info(f"Chunk {chunk.chunk_id} transcribed: {api_latency_ms}ms, {len(transcription_text)} chars")
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed for chunk {chunk.chunk_id}: {e}")
            api_end = time.time()
            api_latency_ms = int((api_end - api_start) * 1000)
            
            return TranscriptionResult(
                chunk_id=chunk.chunk_id,
                start_ts=chunk.start_ts,
                end_ts=chunk.end_ts,
                text="",
                api_start=api_start,
                api_end=api_end,
                api_latency_ms=api_latency_ms
            )
    
    def start_transcription_workers(self):
        """Start transcription worker pool"""
        logger.info(f"Starting {MAX_TRANSCRIBE_WORKERS} transcription workers")
        
        def worker():
            while True:
                try:
                    chunk = self.transcribe_queue.get(timeout=1.0)
                    result = self.transcribe_chunk_async(chunk)
                    self.result_queue.put(result)
                    self.transcribe_queue.task_done()
                except Empty:
                    continue
                except Exception as e:
                    logger.error(f"Transcription worker error: {e}")
        
        # Start worker threads
        for i in range(MAX_TRANSCRIBE_WORKERS):
            worker_thread = threading.Thread(target=worker, name=f"TranscribeWorker-{i}")
            worker_thread.daemon = True
            worker_thread.start()

class RealtimePrinter:
    """Real-time output printer with start_ts ordering and sentence-end priority"""
    
    def __init__(self, result_queue: Queue):
        self.result_queue = result_queue
        self.pending_results = []  # Buffer for out-of-order results
        self.next_expected_chunk_id = 1
        self.sentence_buffer = ""  # Buffer for incomplete sentences
        self.is_running = False
        self.stop_event = threading.Event()
        
    def start(self):
        """Start printer thread"""
        self.is_running = True
        self.stop_event.clear()
        self.next_expected_chunk_id = 1
        self.pending_results = []
        self.sentence_buffer = ""
        
        printer_thread = threading.Thread(target=self._printer_loop, name="RealtimePrinter")
        printer_thread.daemon = True
        printer_thread.start()
        logger.info("RealtimePrinter started")
        return printer_thread
    
    def stop(self):
        """Stop printer thread"""
        self.stop_event.set()
        self.is_running = False
        
        # Flush any remaining buffer
        if self.sentence_buffer.strip():
            print(f"\n[Final] {self.sentence_buffer}")
        
        logger.info("RealtimePrinter stopped")
    
    def _printer_loop(self):
        """Main printer loop - processes results in order"""
        while not self.stop_event.is_set():
            try:
                # Get new result
                result = self.result_queue.get(timeout=1.0)
                result.printed_at = time.time()
                self.pending_results.append(result)
                
                # Sort pending results by chunk_id
                self.pending_results.sort(key=lambda r: r.chunk_id)
                
                # Process results in order
                self._process_ordered_results()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"RealtimePrinter error: {e}")
    
    def _process_ordered_results(self):
        """Process results in correct order"""
        processed = []
        
        for result in self.pending_results:
            if result.chunk_id == self.next_expected_chunk_id:
                self._print_result(result)
                processed.append(result)
                self.next_expected_chunk_id += 1
            else:
                # Out of order - keep for later
                break
        
        # Remove processed results
        for result in processed:
            self.pending_results.remove(result)
    
    def _print_result(self, result: TranscriptionResult):
        """Print single result with sentence-end priority"""
        if not result.text.strip():
            logger.debug(f"Chunk {result.chunk_id}: empty transcription")
            return
        
        # Add to sentence buffer
        self.sentence_buffer += result.text
        
        # Check for sentence endings
        sentence_endings = ['。', '！', '？', '.', '!', '?', '\n']
        
        # Find the last sentence ending
        last_ending_pos = -1
        for ending in sentence_endings:
            pos = self.sentence_buffer.rfind(ending)
            if pos > last_ending_pos:
                last_ending_pos = pos
        
        if last_ending_pos >= 0:
            # Found sentence ending - print up to that point
            complete_text = self.sentence_buffer[:last_ending_pos + 1]
            remaining_text = self.sentence_buffer[last_ending_pos + 1:]
            
            print(f"\n=== chunk {result.chunk_id} [{result.start_ts:.1f}s~{result.end_ts:.1f}s] ===")
            print(complete_text)
            
            self.sentence_buffer = remaining_text
        else:
            # No sentence ending - print immediately (avoid delay)
            print(f"\n=== chunk {result.chunk_id} [{result.start_ts:.1f}s~{result.end_ts:.1f}s] ===")
            print(result.text)
            self.sentence_buffer = ""  # Clear buffer
        
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
    
    def apply_text_corrections(self, text: str, title: str, summary: str, from_chunks: bool = False) -> str:
        """Apply comprehensive text correction: duplicate removal + local normalization + AI correction"""
        corrected_text = text
        
        # Stage 0: Remove duplicates if combining from chunks
        if from_chunks:
            try:
                corrected_text = refine_duplicates_with_ai(corrected_text)
                print("✓ Applied AI duplicate removal")
            except Exception as e:
                print(f"⚠ AI duplicate removal failed: {e}")
                print("Proceeding without AI duplicate removal")
        
        # Stage 1: Local normalization (if enabled)
        if NORMALIZE_TECH_TERMS:
            corrected_text = normalize_tech_terms(corrected_text)
            print("✓ Applied local tech terms normalization")
        
        # Stage 2: AI post-processing (if OpenAI API is available)
        try:
            context_hint = f"{title} {summary}".strip()
            corrected_text = postprocess_with_ai(corrected_text, context_hint=context_hint, test_mode=POSTPROCESS_TEST_MODE)
            if POSTPROCESS_TEST_MODE:
                print("✓ Applied AI post-processing (test mode - showing both results)")
            else:
                print("✓ Applied AI post-processing with gpt-4o")
        except Exception as e:
            print(f"⚠ AI post-processing failed: {e}")
            print("Proceeding with local normalization only")
        
        return corrected_text
    
    def process_transcription(self, text: str, title: str, summary: str, from_chunks: bool = False):
        """Process transcription by sending to Discord and Notion"""
        # Apply comprehensive text correction before saving
        corrected_text = self.apply_text_corrections(text, title, summary, from_chunks=from_chunks)
        
        success_count = 0
        
        # Send to Discord
        if self.discord_client.send_message(title, summary):
            print("✓ Successfully sent to Discord")
            success_count += 1
        else:
            print("✗ Failed to send to Discord")
        
        # Save to Notion with corrected text
        if self.notion_client.create_page(title, summary, corrected_text):
            print("✓ Successfully saved to Notion")
            success_count += 1
        else:
            print("✗ Failed to save to Notion")
        
        return success_count == 2
    
    def collect_final_results(self) -> List[TranscriptionResult]:
        """Collect all transcription results and wait for completion"""
        logger.info("Collecting final transcription results...")
        
        # Wait for remaining results with timeout
        timeout_start = time.time()
        timeout_duration = 30  # 30 seconds timeout
        
        while (time.time() - timeout_start) < timeout_duration:
            try:
                result = self.result_queue.get(timeout=1.0)
                result.printed_at = time.time()
                self.transcription_results.append(result)
            except Empty:
                # Check if there are any pending chunks
                if self.transcribe_queue.empty():
                    break
        
        # Sort by start timestamp for final processing
        self.transcription_results.sort(key=lambda r: r.start_ts)
        logger.info(f"Collected {len(self.transcription_results)} transcription results")
        return self.transcription_results
    
    def combine_and_process_results(self, results: List[TranscriptionResult]) -> str:
        """Combine results and apply final processing"""
        if not results:
            return ""
        
        # Extract text from results in timestamp order
        text_chunks = [r.text for r in results if r.text.strip()]
        
        if not text_chunks:
            return ""
        
        # Stage 1: Local duplicate removal
        combined_text = remove_chunk_duplicates(text_chunks)
        logger.info(f"Combined {len(text_chunks)} chunks into {len(combined_text)} characters")
        
        # Stage 2: AI duplicate refinement
        try:
            refined_text = refine_duplicates_with_ai(combined_text)
            print("✓ Applied AI duplicate removal")
        except Exception as e:
            logger.warning(f"AI duplicate removal failed: {e}")
            refined_text = combined_text
        
        # Stage 3: Format for readability
        final_text = format_text_with_periods(refined_text)
        return final_text

    def run(self):
        """Main application loop with async pipeline"""
        print("Speech Transcriber - Async Pipeline Version")
        print("=" * 50)
        
        # Check environment variables
        required_vars = ['OPENAI_API_KEY', 'DISCORD_WEBHOOK_URL', 'NOTION_TOKEN', 'NOTION_PARENT_PAGE_ID']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"Error: Missing environment variables: {', '.join(missing_vars)}")
            print("Please check your .env file.")
            return
        
        try:
            # Initialize pipeline
            self.transcription_results = []
            logger.info("Starting async transcription pipeline")
            
            # Start pipeline components
            self.start_transcription_workers()
            self.printer.start()
            recording_thread = self.recorder.start_recording()
            self.chunker.start()
            
            logger.info("All pipeline components started")
            
            # Wait for Enter key to stop recording
            input()
            
            # Stop pipeline components in reverse order
            print("\nStopping recording pipeline...")
            self.chunker.stop()
            self.recorder.stop_recording()
            recording_thread.join(timeout=5)
            
            # Collect final results
            final_results = self.collect_final_results()
            self.printer.stop()
            
            logger.info("Pipeline stopped")
            
            if not final_results:
                print("No transcriptions available.")
                return
            
            # Ask user if they want to save
            want_save = self.get_yes_no_input(f"\n{MSG_SAVE_CONFIRM}")
            
            if not want_save:
                if self.confirm_discard():
                    print("Recording discarded.")
                    return
                else:
                    print("保存フローに戻ります。")
                    want_save = True
            
            # Process final transcription
            print("\nProcessing final transcription...")
            final_text = self.combine_and_process_results(final_results)
            
            if not final_text.strip():
                print("No valid transcription content.")
                return
            
            print("\nFinal transcription:")
            print("-" * 30)
            print(final_text)
            print("-" * 30)
            
            # Get title and summary from user
            print(f"\n{MSG_ENTER_TITLE}", end="")
            title = input().strip()
            if not title:
                print("タイトルは必須です。")
                return
            
            print(f"{MSG_ENTER_SUMMARY}", end="")
            summary = input().strip()
            
            # Apply final corrections and save
            print("\nApplying final corrections and saving...")
            success = self.process_transcription(final_text, title, summary, from_chunks=True)
            
            if success:
                print("✓ All operations completed successfully!")
            else:
                print("⚠ Some operations failed. Check the logs above.")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            self._cleanup_pipeline()
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            print(f"Error: {e}")
            self._cleanup_pipeline()
    
    def _cleanup_pipeline(self):
        """Clean up pipeline components"""
        try:
            self.chunker.stop()
            self.recorder.stop_recording()
            self.printer.stop()
            logger.info("Pipeline cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

if __name__ == "__main__":
    app = SpeechTranscriber()
    app.run()