import whisper
import speech_recognition as sr
import numpy as np 

model = whisper.load_model('base')

def main():
    """
    Real-time speech recognition function
    """
    r = sr.Recognizer()
    print('\033[32mSpeak...\033[0m')
    while True:
        with sr.Microphone(sample_rate=16000) as source:
            audio = r.listen(source).get_raw_data()
            buffer_audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768
            phrase = model.transcribe(buffer_audio, fp16=False)
            print(phrase['text'])


if __name__=='__main__':
    main()