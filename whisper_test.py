"""
A basic demonstration to show how OpenAI's Whisper can be used to 
accurately transcribe and perform ASR.
"""
import whisper

model = whisper.load_model("base")
result = model.transcribe("anjali_zoom_test.m4a")
print(result["text"])
