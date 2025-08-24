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
from typing import Optional, Dict, List, Tuple
import logging

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
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
    """Audio chunk with sample index and timing information"""
    start_index: int
    end_index: int
    audio_data: np.ndarray
    chunk_id: int
    capture_start: float
    capture_end: float
    
    @property
    def start_ts(self) -> float:
        """Convert start index to timestamp for display"""
        return self.start_index / 16000
    
    @property
    def end_ts(self) -> float:
        """Convert end index to timestamp for display"""
        return self.end_index / 16000

@dataclass
class TranscriptionResult:
    """Transcription result with sample index and latency information"""
    chunk_id: int
    start_index: int
    end_index: int
    text: str
    api_start: float
    api_end: float
    api_latency_ms: int
    printed_at: Optional[float] = None
    
    @property
    def start_ts(self) -> float:
        """Convert start index to timestamp for display"""
        return self.start_index / 16000
    
    @property
    def end_ts(self) -> float:
        """Convert end index to timestamp for display"""
        return self.end_index / 16000

# User prompt messages
MSG_SAVE_CONFIRM = "今回の文字起こしを保存しますか？ (y=保存 / n=破棄): "  # EN: "Do you want to save this recording? (y=save / n=discard): "
MSG_DISCARD_CONFIRM = "破棄するのですね？ (y=破棄 / n=保存): "  # EN: "Are you sure you want to discard this recording? (y=discard / n=go back to save): "
MSG_ENTER_TITLE = "タイトルを入力してください: "  # EN: "Enter title: "
MSG_ENTER_SUMMARY = "概要を入力してください（空でも可）: "  # EN: "Enter summary (optional): "
MSG_INPUT_INVALID = "入力が認識できません。[y]または[n]を入力してください。"

class AudioRecorder:
    """Ring Buffer-based audio recorder with auto device detection and downsampling (Producer)"""
    
    def __init__(self):
        # Target output settings (internal processing)
        self.target_sample_rate = 16000
        self.target_channels = 1
        self.chunk_duration = 20
        self.overlap_duration = 2
        self.is_recording = False
        self.stop_event = threading.Event()
        
        # Device and recording settings (auto-detected)
        self.device_id = None
        self.device_name = None
        self.input_sample_rate = None
        self.input_channels = None
        self.dtype = np.float32
        
        # Callback monitoring
        self.callback_count = 0
        self.received_samples_total = 0
        self.next_sample_index = 0  # Next sample index to assign
        self.last_sample_report_time = 0
        self.downsample_logged = False
        
        # Ring buffer for audio samples - stores (start_index, ndarray) tuples
        buffer_size = int(self.target_sample_rate * RING_BUFFER_SECONDS)
        self.ring_buffer = deque(maxlen=buffer_size // 1024)  # Store chunks, not individual samples
        self.buffer_lock = threading.Lock()
        self.recording_start_time = None
        
        # Auto-detect and configure audio device
        self._auto_configure_audio()
        
        logger.info(f"AudioRecorder initialized: target {self.target_sample_rate}Hz/{self.target_channels}ch, buffer capacity {buffer_size} samples")
        
    def _auto_configure_audio(self):
        """Auto-detect optimal input device and sample rate with fallback"""
        logger.info("Auto-configuring audio device...")
        
        # Get all input devices
        try:
            devices = sd.query_devices()
            input_devices = [(i, dev) for i, dev in enumerate(devices) 
                           if dev['max_input_channels'] > 0]
            
            if not input_devices:
                raise Exception("No input devices found")
            
            logger.debug(f"Found {len(input_devices)} input devices")
            
        except Exception as e:
            logger.error(f"Failed to query audio devices: {e}")
            raise
        
        # Priority order for sample rates
        sample_rates = [48000, 44100, 32000, 16000]
        
        # Priority order for devices (external mic > built-in > others)
        def device_priority(device_info):
            name = device_info[1]['name'].lower()
            if 'microphone' in name or 'mic' in name:
                return 0  # Highest priority
            elif 'built-in' in name or 'internal' in name:
                return 1  # Medium priority
            else:
                return 2  # Lower priority
        
        input_devices.sort(key=device_priority)
        
        # Try each device with each sample rate
        for device_idx, device_info in input_devices:
            device_name = device_info['name']
            max_channels = int(device_info['max_input_channels'])
            
            for sample_rate in sample_rates:
                for channels in [min(2, max_channels), 1]:  # Try stereo first, then mono
                    try:
                        # Test if we can open the stream
                        test_stream = sd.InputStream(
                            device=device_idx,
                            samplerate=sample_rate,
                            channels=channels,
                            dtype=self.dtype
                        )
                        test_stream.close()
                        
                        # Success! Use these settings
                        self.device_id = device_idx
                        self.device_name = device_name
                        self.input_sample_rate = sample_rate
                        self.input_channels = channels
                        
                        logger.info(f"Selected device: '{device_name}' (ID: {device_idx})")
                        logger.info(f"Input settings: {sample_rate}Hz, {channels} channels, {self.dtype}")
                        
                        if sample_rate != self.target_sample_rate:
                            logger.info(f"Will downsample from {sample_rate}Hz to {self.target_sample_rate}Hz")
                        if channels > self.target_channels:
                            logger.info(f"Will convert from {channels} channels to {self.target_channels} channel(s)")
                        
                        return
                        
                    except Exception as e:
                        logger.debug(f"Device '{device_name}' failed at {sample_rate}Hz/{channels}ch: {e}")
                        continue
        
        # If we get here, no device worked
        device_list = [f"'{dev[1]['name']}' (ID: {dev[0]})" for dev in input_devices[:5]]
        raise Exception(
            f"Failed to configure any audio device. Tried devices: {', '.join(device_list)} "
            f"with sample rates: {sample_rates}"
        )
    
    def _convert_to_target_format(self, audio_data: np.ndarray) -> np.ndarray:
        """Convert input audio to target format (16kHz mono int16)"""
        # Convert from float32 to appropriate range if needed
        if audio_data.dtype == np.float32:
            # Assume float32 is in [-1, 1] range
            audio_data = audio_data.astype(np.float32)
        
        # Handle multi-channel to mono conversion
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            # Convert stereo/multi-channel to mono by averaging
            audio_data = np.mean(audio_data, axis=1)
        elif len(audio_data.shape) > 1:
            # Single channel but 2D array
            audio_data = audio_data.flatten()
        
        # Downsample if needed
        if self.input_sample_rate != self.target_sample_rate:
            # Calculate rational downsampling factors
            gcd = np.gcd(self.target_sample_rate, self.input_sample_rate)
            up_factor = self.target_sample_rate // gcd
            down_factor = self.input_sample_rate // gcd
            
            # Perform resampling
            audio_data = resample_poly(audio_data, up_factor, down_factor)
            
            # Log conversion info once
            if not self.downsample_logged:
                logger.debug(f"Downsampling: {self.input_sample_rate}Hz -> {self.target_sample_rate}Hz (factors: {up_factor}/{down_factor})")
                logger.debug(f"Audio shape conversion: {len(audio_data)} samples after resampling")
                self.downsample_logged = True
        
        # Convert to int16 for final output
        if audio_data.dtype != np.int16:
            # Scale float to int16 range
            if audio_data.dtype in [np.float32, np.float64]:
                audio_data = (audio_data * 32767).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)
        
        return audio_data
    
    def record_audio(self):
        """Record audio into ring buffer with monitoring (lightweight producer)"""
        
        def callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio callback status: {status}")
            
            if self.stop_event.is_set():
                return
            
            self.callback_count += 1
            
            # Log callback details for first few calls
            if self.callback_count <= 3:
                logger.debug(f"Callback #{self.callback_count}: indata.shape={indata.shape}, dtype={indata.dtype}, frames={frames}")
            
            try:
                # Convert audio to target format (downsampling, mono conversion, etc.)
                current_time = time.inputBufferAdcTime
                processed_audio = self._convert_to_target_format(indata.copy())
                
                # Update sample count for monitoring
                self.received_samples_total += len(processed_audio)
                
                # Report sample statistics every second
                current_report_time = int(current_time)
                if current_report_time > self.last_sample_report_time:
                    logger.info(f"Received samples: {self.received_samples_total} (callback #{self.callback_count})")
                    self.last_sample_report_time = current_report_time
                
                # Store in ring buffer with sample indices
                with self.buffer_lock:
                    chunk_start_index = self.next_sample_index
                    self.ring_buffer.append((chunk_start_index, processed_audio.copy()))
                    self.next_sample_index += len(processed_audio)
                        
            except Exception as e:
                logger.error(f"Callback processing error: {e}")
        
        print("Recording started. Press Enter to stop...")
        print("Real-time transcription will appear below:")
        print("=" * 50)
        
        self.recording_start_time = time.time()
        logger.info(f"Audio recording started at {self.recording_start_time}")
        
        try:
            with sd.InputStream(
                device=self.device_id,
                samplerate=self.input_sample_rate,
                channels=self.input_channels,
                callback=callback,
                dtype=self.dtype
            ):
                while not self.stop_event.is_set():
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Recording stream error: {e}")
            raise
        
        logger.info(f"Audio recording stopped. Total samples received: {self.received_samples_total}")
    
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
    
    def extract_chunk_by_index(self, start_index: int, end_index: int) -> Optional[np.ndarray]:
        """Extract audio chunk from ring buffer by sample index range"""
        with self.buffer_lock:
            if not self.ring_buffer:
                logger.warning(f"Ring buffer is empty when extracting chunk [{start_index}-{end_index})")
                return None
            
            # Find all chunks that overlap with the requested range
            chunk_samples = []
            collected_samples = 0
            
            for chunk_start_idx, chunk_data in self.ring_buffer:
                chunk_end_idx = chunk_start_idx + len(chunk_data)
                
                # Check if chunk overlaps with requested range
                if chunk_end_idx <= start_index or chunk_start_idx >= end_index:
                    continue  # No overlap
                
                # Calculate overlap range
                overlap_start = max(start_index, chunk_start_idx)
                overlap_end = min(end_index, chunk_end_idx)
                
                # Extract overlapping portion
                chunk_offset_start = overlap_start - chunk_start_idx
                chunk_offset_end = overlap_end - chunk_start_idx
                
                overlapping_data = chunk_data[chunk_offset_start:chunk_offset_end]
                chunk_samples.append(overlapping_data)
                collected_samples += len(overlapping_data)
            
            if chunk_samples:
                result = np.concatenate(chunk_samples) if len(chunk_samples) > 1 else chunk_samples[0]
                logger.debug(f"Extracted {len(result)} samples for index range [{start_index}-{end_index})")
                return result.astype(np.int16)
            else:
                logger.warning(f"No samples found in ring buffer for index range [{start_index}-{end_index})")
                return None
    
    def get_recording_duration(self) -> float:
        """Get current recording duration based on sample count"""
        with self.buffer_lock:
            return self.next_sample_index / self.target_sample_rate
    
    def get_current_sample_index(self) -> int:
        """Get current sample index (total samples received)"""
        with self.buffer_lock:
            return self.next_sample_index

class AudioChunker:
    """Chunker Thread - extracts 20s/2s overlap chunks from ring buffer using sample indices"""
    
    def __init__(self, recorder: AudioRecorder, transcribe_queue: Queue):
        self.recorder = recorder
        self.transcribe_queue = transcribe_queue
        self.chunk_duration = 20  # seconds
        self.overlap_duration = 2  # seconds
        self.chunk_interval = self.chunk_duration - self.overlap_duration  # 18s
        
        # Convert to sample counts
        self.chunk_len_samples = int(self.chunk_duration * recorder.target_sample_rate)  # 320,000 samples
        self.overlap_samples = int(self.overlap_duration * recorder.target_sample_rate)  # 32,000 samples
        self.chunk_interval_samples = self.chunk_len_samples - self.overlap_samples  # 288,000 samples
        
        self.chunk_id_counter = 0
        self.next_window_start_index = 0
        self.is_running = False
        self.stop_event = threading.Event()
        self.debug_logged = False
        
    def start(self):
        """Start chunker thread"""
        self.is_running = True
        self.stop_event.clear()
        self.chunk_id_counter = 0
        self.next_window_start_index = 0
        self.debug_logged = False
        
        chunker_thread = threading.Thread(target=self._chunker_loop)
        chunker_thread.daemon = True
        chunker_thread.start()
        logger.info(f"AudioChunker started: {self.chunk_len_samples} samples/chunk, {self.chunk_interval_samples} samples interval")
        return chunker_thread
    
    def stop(self):
        """Stop chunker thread"""
        self.stop_event.set()
        self.is_running = False
        logger.info("AudioChunker stop signal sent")
    
    def _chunker_loop(self):
        """Main chunker loop - sample index-based window sliding"""
        
        while not self.stop_event.is_set():
            current_sample_index = self.recorder.get_current_sample_index()
            window_end_index = self.next_window_start_index + self.chunk_len_samples
            
            # Check if we have enough samples for the next chunk
            if current_sample_index >= window_end_index:
                capture_start = time.time()
                
                # Extract audio data from ring buffer by sample index
                audio_data = self.recorder.extract_chunk_by_index(
                    self.next_window_start_index, 
                    window_end_index
                )
                
                if audio_data is not None and len(audio_data) > 0:
                    self.chunk_id_counter += 1
                    capture_end = time.time()
                    
                    # Debug logging for first few chunks
                    if not self.debug_logged and self.chunk_id_counter <= 2:
                        logger.debug(f"Chunker requested [{self.next_window_start_index}-{window_end_index}), got {len(audio_data)} samples")
                        if self.chunk_id_counter == 2:
                            self.debug_logged = True
                    
                    chunk = AudioChunk(
                        start_index=self.next_window_start_index,
                        end_index=window_end_index,
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
                        logger.debug(f"Chunk {self.chunk_id_counter} queued: [{self.next_window_start_index}-{window_end_index}) = {chunk.start_ts:.1f}-{chunk.end_ts:.1f}s, {len(audio_data)} samples")
                        
                    except:
                        logger.error(f"Failed to queue chunk {self.chunk_id_counter} - queue full, dropping")
                    
                    # Move to next window position (with overlap)
                    self.next_window_start_index += self.chunk_interval_samples
                        
                else:
                    logger.warning(f"No audio data for chunk at sample index [{self.next_window_start_index}-{window_end_index})")
                    # Still advance the window to avoid getting stuck
                    self.next_window_start_index += self.chunk_interval_samples
            else:
                # Wait for more samples
                time.sleep(0.5)

class SpeechTranscriber:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.discord_client = DiscordClient()
        self.notion_client = NotionClient()
        self.recorder = AudioRecorder()
        
        # Async pipeline components
        self.transcribe_queue = Queue(maxsize=TRANSCRIBE_QUEUE_MAX_SIZE)
        self.realtime_queue = Queue()  # For RealtimePrinter
        self.results_store = []  # Thread-safe storage for final collection
        self.results_store_lock = threading.Lock()
        self.stored_results_count = 0
        
        self.chunker = AudioChunker(self.recorder, self.transcribe_queue)
        self.executor = ThreadPoolExecutor(max_workers=MAX_TRANSCRIBE_WORKERS)
        self.printer = RealtimePrinter(self.realtime_queue)
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
                wf.setframerate(16000)  # Always use 16kHz for transcription API
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
                start_index=chunk.start_index,
                end_index=chunk.end_index,
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
                start_index=chunk.start_index,
                end_index=chunk.end_index,
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
                    
                    # Fan-out: Send result to both realtime printer and results store
                    self.realtime_queue.put(result)  # For RealtimePrinter
                    
                    # Store for final collection
                    with self.results_store_lock:
                        self.results_store.append(result)
                        self.stored_results_count += 1
                        if self.stored_results_count <= 5 or self.stored_results_count % 5 == 0:
                            logger.info(f"Stored results: {self.stored_results_count} chunks")
                    
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
    """Real-time output printer with start_index ordering and sentence-end priority"""
    
    def __init__(self, result_queue: Queue):
        self.result_queue = result_queue
        self.pending_results = []  # Buffer for out-of-order results
        self.next_expected_chunk_id = 1
        self.sentence_buffer = ""  # Buffer for incomplete sentences
        self.is_running = False
        self.stop_event = threading.Event()
        self.printed_count = 0  # Track printed chunks
        self.printed_count_lock = threading.Lock()
        
    def start(self):
        """Start printer thread"""
        self.is_running = True
        self.stop_event.clear()
        self.next_expected_chunk_id = 1
        self.pending_results = []
        self.sentence_buffer = ""
        self.printed_count = 0
        
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
    
    def get_printed_count(self) -> int:
        """Get the number of chunks that were printed"""
        with self.printed_count_lock:
            return self.printed_count
    
    def _print_result(self, result: TranscriptionResult):
        """Print single result with sentence-end priority"""
        if not result.text.strip():
            logger.debug(f"Chunk {result.chunk_id}: empty transcription")
            # Still count empty results as printed
            with self.printed_count_lock:
                self.printed_count += 1
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
        
        # Increment printed count
        with self.printed_count_lock:
            self.printed_count += 1



# Add methods to SpeechTranscriber class dynamically
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
        """Collect all transcription results from results_store"""
        logger.info("Collecting final transcription results...")
        
        # Wait for any remaining in-flight transcriptions with timeout
        timeout_start = time.time()
        timeout_duration = 10  # 10 seconds timeout
        
        while (time.time() - timeout_start) < timeout_duration:
            if self.transcribe_queue.empty():
                break
            time.sleep(0.5)
        
        # Collect from results_store (thread-safe)
        with self.results_store_lock:
            final_results = self.results_store.copy()
            stored_count = len(final_results)
        
        # Sort by start index for final processing
        final_results.sort(key=lambda r: r.start_index)
        
        # Log collection summary
        printed_count = self.printer.get_printed_count() if hasattr(self.printer, 'get_printed_count') else 'unknown'
        logger.info(f"Final collect: {stored_count} chunks stored, {printed_count} chunks printed")
        
        if stored_count != printed_count and printed_count != 'unknown':
            logger.warning(f"Mismatch: stored={stored_count}, printed={printed_count}")
        
        if stored_count == 0:
            logger.warning("No transcription results collected from results_store")
        
        self.transcription_results = final_results
        return final_results
    
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


# Dynamically assign methods to SpeechTranscriber class
SpeechTranscriber.transcribe_audio = transcribe_audio
SpeechTranscriber.apply_text_corrections = apply_text_corrections
SpeechTranscriber.process_transcription = process_transcription
SpeechTranscriber.collect_final_results = collect_final_results
SpeechTranscriber.combine_and_process_results = combine_and_process_results
SpeechTranscriber.run = run
SpeechTranscriber._cleanup_pipeline = _cleanup_pipeline

if __name__ == "__main__":
    app = SpeechTranscriber()
    app.run()