
from django.http import HttpResponse
from mangorest.mango import webapi
import whisper, hashlib, os, datetime, json, torch
from transformers import pipeline

def preprocess(text):
    """
    Remove timelines and return the result in this format:
    {SPEAKER}: {SENTENCES}
    """
    result = []
    lines = text.strip().split('\n')
    for line in lines:
        parts = line.split('|')
        speaker = parts[1].strip().split(':')[0]
        content = parts[1].strip().split(':')[1].strip()
        result.append(f"{speaker}: {content}")
    return '\n'.join(result)   

#-----------------------------models------------------------------------------------------------------------               
summarizer = pipeline("summarization", "knkarthick/MEETING-SUMMARY-BART-LARGE-XSUM-SAMSUM-DIALOGSUM-AMI", truncation=True)

#-----------------------------------------------------------------------------------------------------               

@webapi("/parrot/summarize_text/")
def summarizeText(request, **kwargs):
    post_data = request.POST.dict()
    transcription = post_data.get('transcription')
    
    input_cleanned_text = preprocess(transcription)
    print("Summarizing...")
    summary = summarizer(input_cleanned_text, min_length = 100,max_length=500)[0]['summary_text']

    print("\n\n" + summary)
    response = {'transcription': transcription, 'summary': summary}
    return HttpResponse(json.dumps(response), content_type='application/json')
