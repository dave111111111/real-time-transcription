# Real-Time Transcription 📜🎤

## Overview
This project is a real-time audio transcription system that captures and transcribes speech using **Silero VAD** and **Faster-Whisper**. The system detects voice activity, manages silence, and automatically saves audio segments along with their corresponding transcriptions.

## Features
- **Real-time Voice Activity Detection:** Utilizes Silero VAD to accurately detect when speech is present.
- **Automatic Transcription:** Leverages the Faster-Whisper model to transcribe audio segments efficiently.
- **File Management:** Automatically saves recorded audio files and appends transcriptions to a newly created transcript file for easy organization.
- **User-Friendly Interface:** Simple command-line interface to start recording and manage audio files.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/real-time-transcription.git
   cd real-time-transcription

2. Install requirements:
   ```bash
   pip install numpy sounddevice torch silero_vad faster_whisper transformers pydub
3. Installing ffmpeg

If you're on Windows, you can easily install FFmpeg using Chocolatey. Follow these steps:
1. Open a command prompt as Administrator.
2. Run the following command:
   ```bash
   choco install ffmpeg

### Usage
   ```bash
   python rec_speak_trans.py
```

### Recording
Once started, the system will:
- Continuously listen to audio input.
- Detect and record speech segments.
- Save each detected segment into the `recordings/` directory as a `.wav` file.

### Transcription
The system automatically:
- Starts a transcription thread for each detected segment.
- Saves transcript chunks to the `transcripts/` directory, generating a new file if one exists.

### Exit & Cleanup
Use `Ctrl+C` to exit. On exit, all individual audio files are combined into a single output file.

## Directory Structure
- `recordings/`: Stores individual audio segments in `.wav` format.
- `transcripts/`: Stores generated transcript files.

## Example Output
recordings/ ├── output_sentence_audio_1.wav └── output_sentence_audio_2.wav

transcripts/ └── transcript_1.txt

 ## How It Works

The real-time transcription system utilizes a combination of **voice activity detection** and **automatic transcription** to effectively capture and transcribe speech. Here’s a breakdown of the main components and their functionalities:

### 1. Speech Recognition Class

- **Initialization**: The `SpeechRecognizer` class is initialized with a transcriber instance, sample rate, buffer size, and silence threshold.
- **Audio Input**: The system captures audio input through the `sounddevice` library. It continuously listens for audio and stores it in a buffer.
- **Voice Activity Detection**: Using **Silero VAD**, the system detects whether speech is present in the captured audio. If speech is detected, it records the audio segment.
- **Managing Silence**: When silence is detected, the system saves the recorded audio to a `.wav` file and starts a transcription thread.
- **File Management**: The recorded audio segments are saved in the `recordings/` directory, and each segment is named incrementally.

### 2. Transcriber Class

- **Initialization**: The `Transcriber` class is initialized with a model name, sample rate, and directory for transcripts.
- **Transcribing Audio**: The transcriber takes each recorded audio file and processes it in chunks using the **Faster-Whisper** model. Each chunk is transcribed to text.
- **Saving Transcripts**: Transcriptions are saved in the `transcripts/` directory. If a transcript file already exists, a new file is generated with an incremented filename.
- **Audio Processing**: The audio is loaded from the saved `.wav` files, and the transcription is performed in manageable chunks for efficiency.

### 3. Combining and Cleaning Up

- **Exit Handler**: When the user exits the program (using Ctrl+C), the `handle_exit` function is triggered. This function combines all individual audio files into a single output file and deletes the original files.
- **Directory Structure**: The program creates necessary directories if they do not exist, ensuring organized storage of recordings and transcripts.

### Usage Flow

1. **Start Recording**: The user starts the program, which initializes the transcriber and speech recognizer.
2. **Capture Speech**: The system listens for audio, detects speech, and records it in real-time.
3. **Transcription Process**: As each audio segment is recorded, a transcription thread is initiated, and the transcribed text is saved to the appropriate file.
4. **Exit & Cleanup**: On exiting, the recorded audio segments are combined, and the original files are deleted to free up space.

This system efficiently combines real-time audio capture, intelligent speech detection, and automated transcription, making it a powerful tool for various applications such as note-taking, meeting transcription, and more.

## Credits
- **Silero VAD** by [snakers4/silero-vad](https://github.com/snakers4/silero-vad) - For efficient voice activity detection.
- **Faster-Whisper** by [SYSTRAN(https://github.com/SYSTRAN/faster-whisper) - Enables real-time transcription.
- **pydub** - For handling and processing audio files.

## License
This project is licensed under the Creative Commons NonCommercial (CC BY-NC)** License.. 
Feel free to modify any part of it according to your needs! Let me know if there's anything else I can assist you with.
If you want to contribute feel free.
