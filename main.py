import speech_recognition as sr        # Importing the speech recognition library
from llama_cpp import Llama           # Importing the Llama library for large language model processing
from TTS.api import TTS               # Importing the Text-to-Speech API
import whisper                         # Importing the Whisper library for speech recognition
import sounddevice as sd              # Importing the library for sound device manipulation
import time                            # Importing the time library for handling time-related tasks
import threading                       # Importing the threading library for concurrent execution
import queue                           # Importing the queue library for queue data structure
import pyaudio                         # Importing the PyAudio library for audio I/O
import argparse                        # Importing the argparse library for command-line argument parsing

# Constants for configuration
WHISPER_MODEL = "base"
LANGUAGE = "de"
LLM_PATH = "stablelm-zephyr-3b.Q4_K_M.gguf" 
SPEAKER_WAV = "voice.wav"
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
DEVICE = "cpu"
STOP_PHRASE = "Stoppe die Aufnahme"

class Voice2Voice():
    # Audio format settings for PyAudio
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 24000

    def __init__(self, whisper_model, llm, tts,voice_file,  wake_word=None):
        # Initialization of the Voice2Voice class
        self.recognizer = sr.Recognizer()  # Speech recognition instance
        self.recognizer.pause_threshold = 0.5  # Configuration for pause threshold
        self.whisper_model = whisper_model  # Whisper model for speech recognition
        self.llm = llm  # Large language model (LLM) instance
        self.tts = tts  # Text-to-Speech (TTS) instance
        self.tts_model = self.tts.synthesizer.tts_model  # TTS model
        # Getting conditioning latents for TTS
        self.gpt_cond_latent, self.speaker_embedding = self.tts_model.get_conditioning_latents(audio_path=[voice_file])
        self.audio = pyaudio.PyAudio()  # PyAudio instance for audio I/O
        self.audio_buffer = queue.Queue()  # Queue for audio buffer
        self.text_buffer = queue.Queue()  # Queue for text buffer
        self.wake_word = wake_word  # Optional wake word for activation

    def start_threads(self):
        # Start threads for text processing and audio playback
        text_thread = threading.Thread(target=self.tts_thread)
        text_thread.setDaemon(True)
        text_thread.start()

        audio_thread = threading.Thread(target=self.play_audio_thread)
        audio_thread.setDaemon(True)
        audio_thread.start()

    def transcribe_audio(self,whisper_model):
        # Transcribe audio using the Whisper model
        print("\nASR: Transcribing ...")
        stop = False
        start = time.time()
        transcription = whisper_model.transcribe("audio.wav", language=LANGUAGE, fp16=False)
        if len(transcription["segments"])>0 and transcription["segments"][0]['no_speech_prob'] < 0.2:
            text = transcription["text"]
            if STOP_PHRASE in text.lower():
                stop = True
        else:
            text = None
        print("ASR: <T> ", time.time()-start)
        print("ASR --> ", text)
        return text, stop

    def play_audio_thread(self):
        # Thread for playing audio
        stream = self.audio.open(format=self.FORMAT, channels=self.CHANNELS,
                            rate=self.RATE, output=True)
        while True:
            data = self.audio_buffer.get()
            stream.write(data)

    def tts_thread(self):
        # Thread for processing text-to-speech
        while True:
            text = self.text_buffer.get()
            if text is None:
                break
            print("\n TTS: synthesizing... ")
            start = time.time()
            stream_generator = self.tts_model.inference_stream(
                text,
                LANGUAGE,
                self.gpt_cond_latent,
                self.speaker_embedding)
            for i, chunk in enumerate(stream_generator):
                print("TTS: <T>", time.time()-start)
                print("TTS: --> Audio Chunk [",i,"] added to buffer")
                if i == 0:
                    print("\n\n\n###############")
                    print("  Total Latency for first response: ")
                    print("  [ ", time.time()-self.start_conversation," seconds ]")
                    print("###############\n\n\n")
                start = time.time()
                chunk = chunk.numpy().tobytes()
                self.audio_buffer.put(chunk)

    def prompt_llm(self,prompt):
        # Process text with the large language model
        print("\nLLM: inference ...")
        start = time.time()
        stream = self.llm(
            prompt,
            max_tokens=10000,  
            stream=True,
        )
        output_buffer = ""
        for output in stream:
            if "choices" in output and output["choices"]:
                text_output = output["choices"][0]["text"]
                output_buffer += text_output

                if any([t in output_buffer for t in [".","!","?"]]):
                    if len(output_buffer) < 3:
                        output_buffer = ""
                        continue
                    output_buffer = output_buffer.replace('assistant', '').strip() 
                    print("LLM: <T> ", time.time()-start)
                    print("LLM: -->  ", output_buffer)
                    start = time.time()
                    self.text_buffer.put(output_buffer)
                    output_buffer = ""

    def handle_voice_input(self):
        # Handle voice input from the user
        self.start_conversation = time.time()
        text, stop = self.transcribe_audio(self.whisper_model)
        if self.wake_word != None and self.wake_word.lower() not in text.lower():
            return False
        if text != None and not stop:
            self.prompt_llm(text)
        return stop

    def run(self):
        # Main run loop for the voice assistant
        self.start_threads()

        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.4)
            print("\n\nListening ...")
            while True:
                audio = self.recognizer.listen(source)
                with open("audio.wav", "wb") as f:
                    f.write(audio.get_wav_data())
                    stop = self.handle_voice_input()
                    if stop:
                        break

def main():
    # Main function to set up and run the voice assistant
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--llm", type=str, default=LLM_PATH)
    parser.add_argument("-w", "--whisper", type=str, default=WHISPER_MODEL)
    parser.add_argument("-t", "--tts", type=str, default=TTS_MODEL)
    parser.add_argument("-d", "--device", type=str, default=DEVICE)
    parser.add_argument("-v", "--voice", type=str, default=SPEAKER_WAV)
    args = parser.parse_args()

    # Initialize models
    whisper_model = whisper.load_model(args.whisper)
    llm = Llama(model_path=args.llm, n_gpu_layers=30)
    tts = TTS(args.tts, gpu=args.device=="cuda")
    model = Voice2Voice(whisper_model, llm, tts, args.voice)
    print("\n\n Using Cuda? ", args.device=="cuda")
    model.run()


if __name__ == "__main__":
    main()
