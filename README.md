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
