
from django.http import HttpResponse
from mangorest.mango import webapi
import whisper, hashlib, os, datetime, json, torch
from transformers import pipeline
import keybert
import math

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
summarizer = pipeline("summarization", "vmarklynn/bart-large-cnn-samsum-acsi-ami-v2", truncation=True)
kw_model = keybert.KeyBERT(model='all-mpnet-base-v2')
#-----------------------------------------------------------------------------------------------------               

@webapi("/parrot/summarize_text/")
def summarizeText(request, **kwargs):
    post_data = request.POST.dict()
    transcription = post_data.get('transcription')
    text = post_data.get('text')
    wordCount = post_data.get('wordCount')
    
    input_cleanned_text = preprocess(transcription)
    print("\n\n", input_cleanned_text, "\n\n")
    # print( "min: ", math.ceil(int(wordCount) * 0.1), "max: ", math.ceil(int(wordCount) * 0.25))
    print("\n\nSummarizing...")
    summary = summarizer(input_cleanned_text)[0]['summary_text']
    print("\n", summary, "\n")
    
    keywords = kw_model.extract_keywords(text, 
                                     keyphrase_ngram_range=(1, 1), 
                                     stop_words='english', 
                                     highlight=False,
                                     top_n=5)
    keywords_list_1= list(dict(keywords).keys())
    print(keywords_list_1)
    keywords = kw_model.extract_keywords(text, 
                                     keyphrase_ngram_range=(2, 2), 
                                     stop_words='english', 
                                     highlight=False,
                                     top_n=5)
    keywords_list_2= list(dict(keywords).keys())
    print(keywords_list_2)    
    keywords = kw_model.extract_keywords(text, 
                                     keyphrase_ngram_range=(3, 3), 
                                     stop_words='english', 
                                     highlight=False,
                                     top_n=5)    
    keywords_list_3 = list(dict(keywords).keys())
    print(keywords_list_3)
    
    response = {'transcription': transcription, 'summary': summary, 
                'keywords_list_1': keywords_list_1, 'keywords_list_2': keywords_list_2,
                'keywords_list_3': keywords_list_3,}
    return HttpResponse(json.dumps(response), content_type='application/json')

#-----------------------------------------------------------------------------------------------------               
@webapi("/parrot/summarize_summary/")
def summarizeSummary(request, **kwargs):
    post_data = request.POST.dict()
    summary_input = post_data.get('summary')
    wordCount = post_data.get('wordCount-summ')
    
    print( "min: ", math.ceil(int(wordCount) * 0.1), "max: ", math.ceil(int(wordCount) * 0.25))
    print("\n\nSummarizing again...")
    summary = summarizer(summary_input, min_length = math.ceil(int(wordCount) * 0.1), max_length = math.ceil(int(wordCount) * 0.25))[0]['summary_text']
    print("\n", summary, "\n")
    
    response = {'summary': summary}
    return HttpResponse(json.dumps(response), content_type='application/json')
