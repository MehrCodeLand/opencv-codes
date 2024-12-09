import speech_recognition as sr
import wave
import pyaudio
import time

# Initialize recognizer
recognizer = sr.Recognizer()

# Set the duration for recording
recording_duration = 15  # seconds

# Record audio from the microphone
def record_audio(duration):
    # Initialize the recognizer and microphone
    with sr.Microphone() as source:
        print("Recording for {} seconds...".format(duration))
        audio_data = recognizer.record(source, duration=duration)
        print("Recording complete.")
        return audio_data

# Save the recorded audio to a WAV file
def save_audio(audio_data, filename):
    with open(filename, "wb") as f:
        f.write(audio_data.get_wav_data())
    print("Audio saved to {}".format(filename))

# Main function
if __name__ == "__main__":
    audio = record_audio(recording_duration)
    save_audio(audio, "output.wav")
