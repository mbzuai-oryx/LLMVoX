import sys
import threading
import time
import logging
import base64
import numpy as np
import librosa
import queue
import requests
import cv2
import os
import tempfile
import argparse
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                            QWidget, QLabel, QTextEdit, QFrame, QSlider, QHBoxLayout,
                            QTabWidget, QLineEdit)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap
from pyaudio import PyAudio, paFloat32

# Import your existing modules and functions
from speech_recognition import Recognizer, Microphone, AudioData
# Import functions from demo_duplex_full.py
from client.endpoints import  voicechat, tts_stream, vlmschat, multimodalchat

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create temp directory for images if it doesn't exist
TEMP_DIR = os.path.join(tempfile.gettempdir(), "voicechat_app")
os.makedirs(TEMP_DIR, exist_ok=True)

def int2float(sound):
    """
    Convert int16 numpy array to float32
    """
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()
    return sound

class WebcamCapture(QThread):
    """Thread for capturing webcam frames"""
    frame_ready = pyqtSignal(QImage)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.capture = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
    def run(self):
        self.running = True
        self.capture = cv2.VideoCapture(0)
        
        if not self.capture.isOpened():
            logger.error("Failed to open webcam")
            return
            
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                # Convert to Qt format for display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.frame_ready.emit(qt_image)
            
            time.sleep(0.03)  # ~30 FPS
    
    def stop(self):
        self.running = False
        if self.capture:
            self.capture.release()
    
    def get_current_frame(self):
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None
    
    def save_frame(self, filename):
        with self.frame_lock:
            if self.current_frame is not None:
                cv2.imwrite(filename, self.current_frame)
                return True
        return False

class MicrophoneListener(QThread):
    """Thread for listening to the microphone"""
    audio_captured = pyqtSignal(bytes, float)  # Added energy level
    status_update = pyqtSignal(str)
    energy_level = pyqtSignal(float)  # Signal to update energy level display
    
    def __init__(self, energy_threshold=0.001):
        super().__init__()
        self.running = False
        self.recognizer = Recognizer()
        self.recognizer.energy_threshold = 500  # Default for speech_recognition
        self.recognizer.non_speaking_duration = 0.1
        self.recognizer.pause_threshold = 0.1
        self.energy_threshold = energy_threshold  # Custom energy threshold
        self.processing = False  # Flag to prevent processing during TTS playback
        
    def run(self):
        self.running = True
        microphone = Microphone(sample_rate=16000)
        
        with microphone as source:
            self.status_update.emit("Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source)
            self.status_update.emit("Listening...")
            
            while self.running:
                if self.processing:
                    time.sleep(0.1)  # Wait a bit if currently processing
                    continue
                    
                try:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    raw_data = audio.get_raw_data()
                    
                    # Calculate energy level
                    audio_int16 = np.frombuffer(raw_data, np.int16)
                    audio_float32 = int2float(audio_int16)
                    energy = np.sum(audio_float32 ** 2) / len(audio_float32)
                    
                    # Emit the energy level for display
                    self.energy_level.emit(energy)
                    
                    # Only process if energy is above threshold
                    if energy > self.energy_threshold:
                        logger.info(f"Audio energy {energy:.6f} is above threshold {self.energy_threshold:.6f}, processing audio.")
                        audio_data_base64 = base64.b64encode(raw_data)
                        self.audio_captured.emit(audio_data_base64, energy)
                        self.status_update.emit("Processing audio...")
                    else:
                        logger.info(f"Audio energy {energy:.6f} is below threshold {self.energy_threshold:.6f}, skipping audio.")
                        self.status_update.emit("Energy too low, skipping...")
                        
                except Exception as e:
                    if "listening timed out" not in str(e).lower():
                        logger.error(f"Error capturing audio: {e}")
                        self.status_update.emit(f"Error: {str(e)}")
    
    def stop(self):
        self.running = False
    
    def set_processing(self, is_processing):
        """Set whether the system is currently processing/playing audio"""
        self.processing = is_processing
    
    def set_energy_threshold(self, threshold):
        """Update the energy threshold"""
        self.energy_threshold = threshold
        logger.info(f"Energy threshold set to {threshold:.6f}")

class AudioProcessingThread(QThread):
    """Thread for processing audio in the background"""
    finished_signal = pyqtSignal(str)
    processing_signal = pyqtSignal(bool)
    
    def __init__(self, audio_data, server_ip, server_port):
        super().__init__()
        self.audio_data = audio_data
        self.server_ip = server_ip
        self.server_port = server_port
        
    def run(self):
        try:
            # Signal that processing has started
            self.processing_signal.emit(True)
            
            # Process the audio
            voicechat(self.audio_data, server_ip=self.server_ip, server_port=self.server_port)
            
            # Give a little time buffer after TTS finishes
            time.sleep(0.5)
            
            self.finished_signal.emit("Processing completed")
        except Exception as e:
            logger.error(f"Error in audio processing thread: {e}")
            self.finished_signal.emit(f"Error: {str(e)}")
        finally:
            # Signal that processing has ended
            self.processing_signal.emit(False)

class VisualSpeechProcessingThread(QThread):
    """Thread for processing visual speech (voice + image)"""
    finished_signal = pyqtSignal(str)
    processing_signal = pyqtSignal(bool)
    
    def __init__(self, audio_data, image_path, server_ip, server_port):
        super().__init__()
        self.audio_data = audio_data
        self.image_path = image_path
        self.server_ip = server_ip
        self.server_port = server_port
        
    def run(self):
        try:
            # Signal that processing has started
            self.processing_signal.emit(True)
            
            # Process with vlmschat (voice + image)
            logger.info(f"Processing visual speech with image: {self.image_path}")
            vlmschat(self.audio_data, self.image_path, server_ip=self.server_ip, server_port=self.server_port)
            
            # Give a little time buffer after processing finishes
            time.sleep(0.5)
            
            self.finished_signal.emit("Visual speech processing completed")
        except Exception as e:
            logger.error(f"Error in visual speech processing thread: {e}")
            self.finished_signal.emit(f"Error: {str(e)}")
        finally:
            # Signal that processing has ended
            self.processing_signal.emit(False)

class TextProcessingThread(QThread):
    """Thread for processing text input"""
    started_signal = pyqtSignal()
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    
    def __init__(self, text, server_ip, server_port):
        super().__init__()
        self.text = text
        self.server_ip = server_ip
        self.server_port = server_port
        
    def run(self):
        try:
            self.started_signal.emit()
            logger.info(f"Processing text: {self.text[:50]}...")
            
            # Send the text to TTS
            tts_stream(self.text, server_ip=self.server_ip, server_port=self.server_port)
            
            self.finished_signal.emit()
        except Exception as e:
            logger.error(f"Error in text processing thread: {e}")
            self.error_signal.emit(str(e))

class EnhancedVoiceChatUI(QMainWindow):
    def __init__(self, server_ip, server_port):
        super().__init__()
        
        # Store server information
        self.server_ip = server_ip
        self.server_port = server_port
        
        # Display server info in window title
        self.setWindowTitle(f"Multimodal Voice Chat - Server: {server_ip}:{server_port}")
        self.setGeometry(100, 100, 1000, 700)
        
        # Main widget and tab structure
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Server info display
        self.server_info_label = QLabel(f"Connected to server: {server_ip}:{server_port}")
        self.server_info_label.setAlignment(Qt.AlignCenter)
        self.server_info_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        self.main_layout.addWidget(self.server_info_label)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.voice_tab = QWidget()
        self.text_tab = QWidget()
        self.visual_tab = QWidget()
        
        self.tabs.addTab(self.voice_tab, "Voice Chat")
        self.tabs.addTab(self.text_tab, "Text Chat")
        self.tabs.addTab(self.visual_tab, "Visual Speech")
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        self.main_layout.addWidget(self.tabs)
        
        # Set up the voice chat tab
        self.setup_voice_tab()
        
        # Set up the text chat tab
        self.setup_text_tab()
        
        # Set up the visual speech tab
        self.setup_visual_tab()
        
        # Initialize state
        self.listening = False
        self.mic_thread = None
        self.webcam_thread = None
        self.energy_threshold = 0.001
        self.current_tab = "voice"
        self.current_image_path = None
        
        # Apply styling
        self.apply_styling()
    
    def setup_voice_tab(self):
        """Set up the voice chat tab UI"""
        voice_layout = QVBoxLayout(self.voice_tab)
        
        # Status display
        self.status_frame = QFrame()
        self.status_frame.setFrameShape(QFrame.StyledPanel)
        status_layout = QVBoxLayout(self.status_frame)
        
        self.status_label = QLabel("Status: Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Arial", 12))
        status_layout.addWidget(self.status_label)
        
        # Energy level indicator
        self.energy_label = QLabel("Energy Level: 0.0000")
        self.energy_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.energy_label)
        
        voice_layout.addWidget(self.status_frame)
        
        # Energy threshold control
        self.threshold_frame = QFrame()
        self.threshold_frame.setFrameShape(QFrame.StyledPanel)
        threshold_layout = QVBoxLayout(self.threshold_frame)
        
        self.threshold_label = QLabel("Energy Threshold: 0.0010")
        threshold_layout.addWidget(self.threshold_label)
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(1)
        self.threshold_slider.setMaximum(1000)
        self.threshold_slider.setValue(10)  # 0.0010 default
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        threshold_layout.addWidget(self.threshold_slider)
        
        voice_layout.addWidget(self.threshold_frame)
        
        # Transcript area
        self.voice_transcript_label = QLabel("Conversation Transcript:")
        self.voice_transcript = QTextEdit()
        self.voice_transcript.setReadOnly(True)
        voice_layout.addWidget(self.voice_transcript_label)
        voice_layout.addWidget(self.voice_transcript, 1)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Listening")
        self.start_button.clicked.connect(self.toggle_listening)
        self.start_button.setMinimumHeight(50)
        controls_layout.addWidget(self.start_button)
        
        self.test_speak_button = QPushButton("Test TTS")
        self.test_speak_button.clicked.connect(self.test_tts)
        controls_layout.addWidget(self.test_speak_button)
        
        self.clear_voice_button = QPushButton("Clear Transcript")
        self.clear_voice_button.clicked.connect(lambda: self.voice_transcript.clear())
        controls_layout.addWidget(self.clear_voice_button)
        
        voice_layout.addLayout(controls_layout)
    
    def setup_text_tab(self):
        """Set up the text chat tab UI"""
        text_layout = QVBoxLayout(self.text_tab)
        
        # Status display for text chat
        self.text_status_frame = QFrame()
        self.text_status_frame.setFrameShape(QFrame.StyledPanel)
        text_status_layout = QVBoxLayout(self.text_status_frame)
        
        self.text_status_label = QLabel("Status: Ready")
        self.text_status_label.setAlignment(Qt.AlignCenter)
        self.text_status_label.setFont(QFont("Arial", 12))
        text_status_layout.addWidget(self.text_status_label)
        
        text_layout.addWidget(self.text_status_frame)
        
        # Transcript area for text chat
        self.text_transcript_label = QLabel("Text Chat History:")
        self.text_transcript = QTextEdit()
        self.text_transcript.setReadOnly(True)
        text_layout.addWidget(self.text_transcript_label)
        text_layout.addWidget(self.text_transcript, 1)
        
        # Text input area
        input_frame = QFrame()
        input_frame.setFrameShape(QFrame.StyledPanel)
        input_layout = QVBoxLayout(input_frame)
        
        input_label = QLabel("Type your message:")
        input_layout.addWidget(input_label)
        
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Type here and press Enter...")
        self.text_input.returnPressed.connect(self.send_text_message)
        input_layout.addWidget(self.text_input)
        
        text_layout.addWidget(input_frame)
        
        # Control buttons for text chat
        text_controls_layout = QHBoxLayout()
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_text_message)
        self.send_button.setMinimumHeight(50)
        text_controls_layout.addWidget(self.send_button)
        
        self.clear_text_button = QPushButton("Clear History")
        self.clear_text_button.clicked.connect(lambda: self.text_transcript.clear())
        text_controls_layout.addWidget(self.clear_text_button)
        
        text_layout.addLayout(text_controls_layout)
    
    def setup_visual_tab(self):
        """Set up the visual speech tab UI"""
        visual_layout = QVBoxLayout(self.visual_tab)
        
        # Status display for visual chat
        self.visual_status_frame = QFrame()
        self.visual_status_frame.setFrameShape(QFrame.StyledPanel)
        visual_status_layout = QVBoxLayout(self.visual_status_frame)
        
        self.visual_status_label = QLabel("Status: Ready")
        self.visual_status_label.setAlignment(Qt.AlignCenter)
        self.visual_status_label.setFont(QFont("Arial", 12))
        visual_status_layout.addWidget(self.visual_status_label)
        
        # Energy level indicator for visual tab
        self.visual_energy_label = QLabel("Energy Level: 0.0000")
        self.visual_energy_label.setAlignment(Qt.AlignCenter)
        visual_status_layout.addWidget(self.visual_energy_label)
        
        visual_layout.addWidget(self.visual_status_frame)
        
        # Webcam display
        self.webcam_frame = QFrame()
        self.webcam_frame.setFrameShape(QFrame.StyledPanel)
        self.webcam_frame.setMinimumHeight(300)
        webcam_layout = QVBoxLayout(self.webcam_frame)
        
        self.webcam_label = QLabel("Webcam feed will appear here")
        self.webcam_label.setAlignment(Qt.AlignCenter)
        self.webcam_label.setStyleSheet("background-color: #e0e0e0;")
        self.webcam_label.setMinimumSize(640, 480)
        self.webcam_label.setMaximumSize(640, 480)
        webcam_layout.addWidget(self.webcam_label, 0, Qt.AlignCenter)
        
        visual_layout.addWidget(self.webcam_frame)
        
        # Transcript area for visual chat
        self.visual_transcript_label = QLabel("Visual Conversation Transcript:")
        self.visual_transcript = QTextEdit()
        self.visual_transcript.setReadOnly(True)
        visual_layout.addWidget(self.visual_transcript_label)
        visual_layout.addWidget(self.visual_transcript, 1)
        
        # Control buttons for visual chat
        visual_controls_layout = QHBoxLayout()
        
        self.webcam_toggle_button = QPushButton("Start Webcam")
        self.webcam_toggle_button.clicked.connect(self.toggle_webcam)
        self.webcam_toggle_button.setMinimumHeight(50)
        visual_controls_layout.addWidget(self.webcam_toggle_button)
        
        self.visual_listen_button = QPushButton("Start Listening")
        self.visual_listen_button.clicked.connect(self.toggle_visual_listening)
        self.visual_listen_button.setMinimumHeight(50)
        visual_controls_layout.addWidget(self.visual_listen_button)
        
        self.capture_button = QPushButton("Capture Frame")
        self.capture_button.clicked.connect(self.capture_frame)
        self.capture_button.setEnabled(False)
        visual_controls_layout.addWidget(self.capture_button)
        
        self.clear_visual_button = QPushButton("Clear Transcript")
        self.clear_visual_button.clicked.connect(lambda: self.visual_transcript.clear())
        visual_controls_layout.addWidget(self.clear_visual_button)
        
        visual_layout.addLayout(visual_controls_layout)
    
    def apply_styling(self):
        """Apply styling to the UI"""
        self.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 4px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QLabel {
                font-size: 14px;
            }
            QFrame {
                background-color: #f8f8f8;
                border-radius: 5px;
                padding: 10px;
            }
            QTextEdit {
                border: 1px solid #ccc;
                border-radius: 3px;
                font-size: 14px;
            }
            QLineEdit {
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 8px;
                font-size: 14px;
            }
            QTabWidget::pane {
                border: 1px solid #ccc;
                border-radius: 3px;
            }
            QTabBar::tab {
                background-color: #e1e1e1;
                border: 1px solid #ccc;
                border-bottom-color: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 12px;
                min-width: 100px;
            }
            QTabBar::tab:selected {
                background-color: #f8f8f8;
                border-bottom-color: #f8f8f8;
            }
            QTabBar::tab:!selected {
                margin-top: 2px;
            }
        """)
    
    def on_tab_changed(self, index):
        """Handle tab change events"""
        tab_names = ["voice", "text", "visual"]
        self.current_tab = tab_names[index]
        
        # Stop listening threads when switching tabs
        if self.listening:
            if self.current_tab != "voice":
                self.toggle_listening()
            elif self.current_tab != "visual":
                self.toggle_visual_listening()
    
    def toggle_listening(self):
        self.listening = not self.listening
        
        if self.listening:
            self.start_button.setText("Stop Listening")
            self.start_button.setStyleSheet("background-color: #d9534f;")
            self.status_label.setText("Status: Listening...")
            
            # Start microphone thread
            self.mic_thread = MicrophoneListener(self.energy_threshold)
            self.mic_thread.audio_captured.connect(self.process_audio)
            self.mic_thread.status_update.connect(self.update_status)
            self.mic_thread.energy_level.connect(self.update_energy_display)
            self.mic_thread.start()
            
            self.add_to_voice_transcript("Started listening")
        else:
            self.start_button.setText("Start Listening")
            self.start_button.setStyleSheet("background-color: #4CAF50;")
            self.status_label.setText("Status: Ready")
            
            # Stop microphone thread
            if self.mic_thread:
                self.mic_thread.stop()
                self.mic_thread.wait()
                self.mic_thread = None
            
            self.add_to_voice_transcript("Stopped listening")
    
    def toggle_visual_listening(self):
        """Toggle listening for visual speech tab"""
        self.listening = not self.listening
        
        if self.listening:
            # Check if webcam is active
            if not self.webcam_thread or not self.webcam_thread.isRunning():
                self.toggle_webcam()
            
            self.visual_listen_button.setText("Stop Listening")
            self.visual_listen_button.setStyleSheet("background-color: #d9534f;")
            self.visual_status_label.setText("Status: Listening...")
            
            # Start microphone thread
            self.mic_thread = MicrophoneListener(self.energy_threshold)
            self.mic_thread.audio_captured.connect(self.process_visual_audio)
            self.mic_thread.status_update.connect(self.update_visual_status)
            self.mic_thread.energy_level.connect(self.update_visual_energy_display)
            self.mic_thread.start()
            
            self.add_to_visual_transcript("Started listening")
        else:
            self.visual_listen_button.setText("Start Listening")
            self.visual_listen_button.setStyleSheet("background-color: #4CAF50;")
            self.visual_status_label.setText("Status: Ready")
            
            # Stop microphone thread
            if self.mic_thread:
                self.mic_thread.stop()
                self.mic_thread.wait()
                self.mic_thread = None
            
            self.add_to_visual_transcript("Stopped listening")
    
    def toggle_webcam(self):
        """Toggle webcam on/off"""
        if not self.webcam_thread or not self.webcam_thread.isRunning():
            # Start webcam
            self.webcam_toggle_button.setText("Stop Webcam")
            self.webcam_toggle_button.setStyleSheet("background-color: #d9534f;")
            self.capture_button.setEnabled(True)
            
            self.webcam_thread = WebcamCapture()
            self.webcam_thread.frame_ready.connect(self.update_webcam_display)
            self.webcam_thread.start()
            
            self.add_to_visual_transcript("Webcam started")
        else:
            # Stop webcam
            self.webcam_toggle_button.setText("Start Webcam")
            self.webcam_toggle_button.setStyleSheet("background-color: #4CAF50;")
            self.capture_button.setEnabled(False)
            
            if self.webcam_thread:
                self.webcam_thread.stop()
                self.webcam_thread.wait()
                self.webcam_thread = None
            
            self.webcam_label.setText("Webcam feed will appear here")
            self.webcam_label.setStyleSheet("background-color: #e0e0e0;")
            
            self.add_to_visual_transcript("Webcam stopped")
    
    def capture_frame(self):
        """Capture current webcam frame for visual speech"""
        if self.webcam_thread and self.webcam_thread.isRunning():
            # Create a unique filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(TEMP_DIR, f"captured_frame_{timestamp}.jpg")
            
            # Save the current frame
            if self.webcam_thread.save_frame(image_path):
                self.current_image_path = image_path
                self.add_to_visual_transcript(f"Frame captured: {image_path}")
                
                # Show a border around the webcam feed to indicate capture
                self.webcam_label.setStyleSheet("background-color: #e0e0e0; border: 3px solid #4CAF50;")
                
                # Reset border after a moment
                QTimer.singleShot(1000, lambda: self.webcam_label.setStyleSheet("background-color: #e0e0e0;"))
            else:
                self.add_to_visual_transcript("Failed to capture frame")
    
    def update_webcam_display(self, qt_image):
        """Update webcam display with new frame"""
        pixmap = QPixmap.fromImage(qt_image)
        self.webcam_label.setPixmap(pixmap.scaled(
            self.webcam_label.width(), 
            self.webcam_label.height(),
            Qt.KeepAspectRatio
        ))
    
    def process_audio(self, audio_data_base64, energy):
        # Start processing thread
        self.audio_thread = AudioProcessingThread(audio_data_base64, self.server_ip, self.server_port)
        self.audio_thread.finished_signal.connect(self.on_audio_processed)
        self.audio_thread.processing_signal.connect(self.set_processing_state)
        self.audio_thread.start()
        
        self.add_to_voice_transcript(f"Processing audio... (Energy: {energy:.6f})")
    
    def process_visual_audio(self, audio_data_base64, energy):
        """Process audio with visual context"""
        # Capture the current frame if we don't have a recent one
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            self.capture_frame()
        
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            self.add_to_visual_transcript("Error: No image available for visual speech")
            return
        
        # Start visual speech processing thread
        self.visual_thread = VisualSpeechProcessingThread(
            audio_data_base64, 
            self.current_image_path,
            self.server_ip,
            self.server_port
        )
        self.visual_thread.finished_signal.connect(self.on_visual_processed)
        self.visual_thread.processing_signal.connect(self.set_visual_processing_state)
        self.visual_thread.start()
        
        self.add_to_visual_transcript(f"Processing visual speech... (Energy: {energy:.6f})")
    
    def on_audio_processed(self, result):
        self.add_to_voice_transcript(result)
        if self.listening:
            self.status_label.setText("Status: Listening...")
        else:
            self.status_label.setText("Status: Ready")
    
    def on_visual_processed(self, result):
        self.add_to_visual_transcript(result)
        if self.listening:
            self.visual_status_label.setText("Status: Listening...")
        else:
            self.visual_status_label.setText("Status: Ready")
    
    def set_processing_state(self, is_processing):
        """Update the processing state in the microphone thread"""
        if self.mic_thread:
            self.mic_thread.set_processing(is_processing)
        
        if is_processing:
            self.status_label.setText("Status: Processing/Speaking...")
        elif self.listening:
            self.status_label.setText("Status: Listening...")
        else:
            self.status_label.setText("Status: Ready")
    
    def set_visual_processing_state(self, is_processing):
        """Update the processing state in the visual microphone thread"""
        if self.mic_thread:
            self.mic_thread.set_processing(is_processing)
        
        if is_processing:
            self.visual_status_label.setText("Status: Processing/Speaking...")
        elif self.listening:
            self.visual_status_label.setText("Status: Listening...")
        else:
            self.visual_status_label.setText("Status: Ready")
    
    def update_status(self, status):
        self.status_label.setText(f"Status: {status}")
    
    def update_visual_status(self, status):
        self.visual_status_label.setText(f"Status: {status}")
    
    def update_energy_display(self, energy):
        self.energy_label.setText(f"Energy Level: {energy:.6f}")
    
    def update_visual_energy_display(self, energy):
        self.visual_energy_label.setText(f"Energy Level: {energy:.6f}")
    
    def update_threshold(self):
        value = self.threshold_slider.value() / 10000.0  # Scale to small values
        self.energy_threshold = value
        self.threshold_label.setText(f"Energy Threshold: {value:.6f}")
        
        if self.mic_thread:
            self.mic_thread.set_energy_threshold(value)
    
    def add_to_voice_transcript(self, text):
        self.voice_transcript.append(f"[{time.strftime('%H:%M:%S')}] {text}")
    
    def add_to_text_transcript(self, text, is_user=False):
        prefix = "You" if is_user else "System"
        self.text_transcript.append(f"[{time.strftime('%H:%M:%S')}] {prefix}: {text}")
    
    def add_to_visual_transcript(self, text):
        self.visual_transcript.append(f"[{time.strftime('%H:%M:%S')}] {text}")
    
    def send_text_message(self):
        """Process text input and generate TTS response"""
        text = self.text_input.text().strip()
        if not text:
            return
        
        # Disable input while processing
        self.text_input.setEnabled(False)
        self.send_button.setEnabled(False)
        
        # Add user message to transcript
        self.add_to_text_transcript(text, is_user=True)
        
        # Clear input field
        self.text_input.clear()
        
        # Create and start the text processing thread
        self.text_thread = TextProcessingThread(text, self.server_ip, self.server_port)
        self.text_thread.started_signal.connect(self.on_text_processing_started)
        self.text_thread.finished_signal.connect(self.on_text_processing_finished)
        self.text_thread.error_signal.connect(self.on_text_processing_error)
        self.text_thread.start()
    
    def on_text_processing_started(self):
        self.text_status_label.setText("Status: Processing input...")
        self.add_to_text_transcript("Processing your message...", is_user=False)
    
    def on_text_processing_finished(self):
        self.text_status_label.setText("Status: Ready")
        self.add_to_text_transcript("Response completed", is_user=False)
        
        # Re-enable input
        self.text_input.setEnabled(True)
        self.send_button.setEnabled(True)
        self.text_input.setFocus()
    
    def on_text_processing_error(self, error_message):
        self.text_status_label.setText("Status: Error")
        self.add_to_text_transcript(f"Error: {error_message}", is_user=False)
        
        # Re-enable input
        self.text_input.setEnabled(True)
        self.send_button.setEnabled(True)
        self.text_input.setFocus()
    
    def test_tts(self):
        # Update status
        self.status_label.setText("Status: Testing TTS...")
        self.add_to_voice_transcript("Testing TTS...")
        
        # Set processing flag to avoid feedback if microphone is active
        if self.mic_thread:
            self.mic_thread.set_processing(True)
        
        # Run TTS in a separate thread
        def run_tts_and_reset():
            try:
                tts_stream("Hello, this is a test of the text to speech system.",
                          server_ip=self.server_ip,
                          server_port=self.server_port)
                # Add delay to ensure TTS is complete
                time.sleep(0.5)
            finally:
                # Reset processing flag
                if self.mic_thread and self.listening:
                    self.mic_thread.set_processing(False)
                    self.status_label.setText("Status: Listening...")
                else:
                    self.status_label.setText("Status: Ready")
        
        tts_thread = threading.Thread(target=run_tts_and_reset)
        tts_thread.daemon = True
        tts_thread.start()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Voice Chat Application')
    parser.add_argument('--ip', default='10.127.30.115', help='Server IP address')
    parser.add_argument('--port', default='5003', help='Server port')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Log startup information
    logger.info(f"Starting Voice Chat Application - Server: {args.ip}:{args.port}")
    
    # Start application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = EnhancedVoiceChatUI(args.ip, args.port)
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()