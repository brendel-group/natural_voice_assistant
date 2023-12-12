# Natural Voice Assistant 

This project is a demo for a full voice-to-voice assistant.It uses machine learning models for speech recognition, natural language processing, and speech synthesis.  The goal of this project is to build a fast and realistic voice assistant wich very low latency to ensure conversations that feel natural.  

## Installation
Before you start using the Assistant, ensure that you have Python 3.11 installed on your system. You can then install the necessary Python packages using pip and the requirements.txt file in the repository. To do so, execute the following command in your terminal:

bash
```bash
pip install -r requirements.txt
```
This will install all the dependencies needed for the project.

## Components
The voice Assistant is composed of three main components:

**Whisper Model for Automatic Speech Recognition (ASR):** This model is responsible for converting spoken words into text. We use the Whisper model for its efficiency and accuracy in transcribing human speech.

**Leo-Mistral Model for Large Language Model (LLM):** This is the core model that processes the transcribed text, understands the context, and generates appropriate responses.

**XTTS for Text-to-Speech (TTS):** The XTTS model is used for synthesizing human-like speech from the text responses generated by the LLM.

Voice Cloning in TTS
By default, the TTS system is using voice cloning. Users are required to provide a short audio sample (at least 3 seconds) of human voice, which the program uses for speech synthesis. This feature allows the assistant to speak in a voice that resembles the provided sample.

## Flexibility
The components used in the voice assistant are interchangeable. Theoretically, you can replace any of these models (ASR, LLM, or TTS) with other models of your choice, provided they offer similar functionalities and compatibility.

## Configuration
Before running the program, you need to provide a path to a local LLM and a voice sample for TTS. You can set these paths either directly in the code by changing the corresponding constant variables or by passing them as parameters when running the program.

## Running the Program
To start the Voice2Voice Assistant, use the following command in the console. You must at least provide the path to a local LLM and a voice example:

```bash
python voice2voice.py -m <path_to_llm> -w <whisper_model> -t <tts_model> -d <device> -v <path_to_voice_sample>
```
Parameters:<br>
-m, --llm: Path to the local Large Language Model.<br>
-w, --whisper: (Optional) Specify the Whisper model to use for ASR.<br>
-t, --tts: (Optional) Specify the TTS model.<br>
-d, --device: (Optional) Specify the device to use (e.g., 'cpu', 'gpu').<br>
-v, --voice: Path to the voice sample for voice cloning.<br>
