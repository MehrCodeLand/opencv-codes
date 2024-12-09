import speech_recognition as sr
import record_voice

record_voice.main
# Initialize the recognizer
recognizer = sr.Recognizer()

# Load the audio file
audio_file = "path_to_your_audio_file.wav"
with sr.AudioFile(audio_file) as source:
    audio_data = recognizer.record(source)

# Convert speech to text
try:
    text = recognizer.recognize_google(audio_data)
    print("Transcription: ", text)
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand the audio")
except sr.RequestError as e:
    print(f"Could not request results from Google Speech Recognition service; {e}")
