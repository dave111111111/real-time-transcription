import os
import signal
import numpy as np
import sounddevice as sd
import wave
from silero_vad import load_silero_vad, get_speech_timestamps
import torch
import time
from faster_whisper import WhisperModel
import threading
from pydub import AudioSegment

class SpeechRecognizer:
    def __init__(self, transcriber, sample_rate=16000, buffer_size=8000, silence_threshold=0.040):
        self.model = load_silero_vad()
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.silence_threshold = silence_threshold
        self.audio_buffer = []
        self.sentence_buffer = []
        self.recording_speech = False
        self.silence_start = None
        self.file_count = 1
        self.directory_name = "recordings"
        self.transcriber = transcriber

    def create_directories(self):
        """Creates 'recordings' and 'transcripts' directories if they don't exist."""
        directories = ['recordings', 'transcripts']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"✔ Directory '{directory}' created.")
            else:
                print(f"✔ Directory '{directory}' already exists.")

    def callback(self, indata, frames, time, status):
        """Callback function for the audio stream."""
        if status:
            print("⚠ Status:", status)
        self.audio_buffer.extend(indata[:, 0])

    def start_recording(self):
        self.create_directories()
        """Starts the audio stream and begins recording."""
        print("\n🎙 Recording... Speak in sentences.")
        with sd.InputStream(callback=self.callback, channels=1, samplerate=self.sample_rate):
            while True:
                if len(self.audio_buffer) >= self.buffer_size:
                    self.process_audio()

    def process_audio(self):
        """Processes the audio buffer to detect speech."""
        audio_data = torch.tensor(self.audio_buffer, dtype=torch.float32)
        speech_timestamps = get_speech_timestamps(audio_data, self.model, threshold=0.5)

        if speech_timestamps:
            print("\n🔊 Speech detected!")
            self.recording_speech = True
            self.silence_start = None
            int_audio = (np.array(self.audio_buffer) * 32767).astype(np.int16)
            self.sentence_buffer.extend(int_audio)
            self.audio_buffer.clear()
        else:
            if self.recording_speech:
                self.handle_silence()

    def handle_silence(self):
        """Handles silence detection and manages sentence completion."""
        if self.silence_start is None:
            self.silence_start = time.time()
        elif time.time() - self.silence_start > self.silence_threshold:
            if self.sentence_buffer:
                self.save_audio()
            self.reset_buffers()

    def save_audio(self):
        """Saves the recorded audio to a WAV file."""
        output_file = os.path.join(self.directory_name, f"output_sentence_audio_{self.file_count}.wav")
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(np.array(self.sentence_buffer, dtype=np.int16).tobytes())

        # Start transcription in a new thread
        transcription_thread = threading.Thread(target=self.transcriber.transcribe_audio_file, args=(output_file,))
        transcription_thread.start()

        self.file_count += 1

    def reset_buffers(self):
        """Resets the buffers for new recording."""
        self.sentence_buffer.clear()
        self.recording_speech = False
        self.silence_start = None


class Transcriber:
    def __init__(self, model_name="large-v3", sample_rate=16000, transcript_directory="./transcripts"):
        self.model = WhisperModel(model_name, device="cuda", compute_type="float16")
        self.sample_rate = sample_rate
        self.transcripts = []
        self.last_transcript = ""
        self.processing_buffer = []
        self.chunk_duration = 30
        self.transcript_file = self.generate_transcript_file(transcript_directory)
        self.current_transcription = ""  # New variable to store the real-time transcription

    def generate_transcript_file(self, transcript_directory):
        """Generate a new transcript file if one already exists."""
        os.makedirs(transcript_directory, exist_ok=True)
        base_filename = "transcript"
        extension = ".txt"
        file_count = 1

        # Check for existing transcript files and increment the count if necessary
        while os.path.exists(os.path.join(transcript_directory, f"{base_filename}_{file_count}{extension}")):
            file_count += 1

        # Return the unique filename
        new_file_path = os.path.join(transcript_directory, f"{base_filename}_{file_count}{extension}")
        print(f"📄 New transcript file created: {new_file_path}")
        return new_file_path

    def transcribe_audio_file(self, audio_file):
        print(f"🎧 Transcribing file: {audio_file}")
        audio_data, _ = self.load_audio(audio_file)
        self.process_audio_chunks(audio_data)

    def load_audio(self, audio_file):
        """Loads audio data from a WAV file."""
        with wave.open(audio_file, 'rb') as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            n_frames = wf.getnframes()
            audio_data = wf.readframes(n_frames)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
        return audio_array, frame_rate

    def process_audio_chunks(self, audio_array):
        chunk_size = self.sample_rate * self.chunk_duration
        num_chunks = int(np.ceil(len(audio_array) / chunk_size))

        for i in range(num_chunks):
            start_index = i * chunk_size
            chunk = audio_array[start_index:start_index + chunk_size]
            if len(chunk) > 0:
                self.transcribe_audio(chunk)

    def transcribe_audio(self, audio_chunk):
        segments, info = self.model.transcribe(audio_chunk, beam_size=10, task="transcribe")
        print("🌐 Detected language '%s' with probability %f" % (info.language, info.language_probability))

        transcript = ""
        for segment in segments:
            transcript += segment.text
            self.processing_buffer.append((segment.start, segment.end, segment.text))

        print(f"📝 Transcripts: \033[1m{transcript}\033[0m\n")
        print("-------------------------------------------------------------\n")

        self.current_transcription += transcript  # Accumulate the transcription over time

        self.update_transcript_file(transcript)

    def update_transcript_file(self, transcript):
        max_length = 80
        current_line = ""
        lines = []  # Initialize lines as an empty list to avoid UnboundLocalError

        try:
            with open(self.transcript_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    current_line = lines[-1].strip()
        except FileNotFoundError:
            pass

        current_line += transcript

        lines_to_write = []
        while len(current_line) > max_length:
            lines_to_write.append(current_line[:max_length])
            current_line = current_line[max_length:]

        with open(self.transcript_file, 'w') as f:
            if lines:
                f.writelines(lines[:-1])
            for line in lines_to_write:
                f.write(line + "\n")
            if current_line:
                f.write(current_line)

    def combine_and_cleanup_recordings(self, recordings_directory, output_file="combined_recording.wav"):
        audio_files = [f for f in os.listdir(recordings_directory) if f.endswith(('.wav', '.mp3'))]
        combined_audio = AudioSegment.empty()

        for file_name in audio_files:
            file_path = os.path.join(recordings_directory, file_name)
            audio_segment = AudioSegment.from_file(file_path)
            combined_audio += audio_segment

        combined_audio.export(output_file, format="wav")
        print(f"Combined audio saved as '{output_file}'")

        for file_name in audio_files:
            file_path = os.path.join(recordings_directory, file_name)
            os.remove(file_path)
            print(f"Deleted '{file_path}'")

        print("All recordings combined and original files deleted.")

def handle_exit(transcriber, recordings_directory="recordings"):
    transcriber.combine_and_cleanup_recordings(recordings_directory)


# Initialize transcriber and recognizer
transcriber = Transcriber()
recognizer = SpeechRecognizer(transcriber)

# Attach the cleanup function to SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, lambda *args: handle_exit(transcriber))

# Start recording
recognizer.start_recording()
