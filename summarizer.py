
from transformers import pipeline
import os, keybert

"""
Assumes input is as follows:
    text='''
    0:00:00 - 0:00:06 | SPEAKER_01: Yeah, we had a long 
    0:00:06 - 0:00:10 | SPEAKER_01: Morgan wants to make it hard.
    0:00:10 - 0:00:13 | None: The counter is not moving.
    0:00:13 - 0:00:16 | SPEAKER_01: It doesn't.
    0:00:16 - 0:00:18 | SPEAKER_00: I didn't even check yesterday.
    0:00:18 - 0:00:20 | SPEAKER_01: It didn't move
    0:00:20 - 0:00:22 | SPEAKER_01: I don't know if 
    0:00:22 - 0:00:24 | SPEAKER_01: Channel 3?
    '''

Remove timelines and return the result in this format:
{SPEAKER}: {SENTENCES}
"""
def cleanup(text):
    result = []
    lines = text.strip().split('\n')
    for line in lines:
        parts = line.split('|')
        speaker = parts[1].strip().split(':')[0]
        content = parts[1].strip().split(':')[1].strip()
        result.append(f"{speaker}: {content}")
    return '\n'.join(result)   

#-----------------------------models------------------------------------------------------------------               
summarizer = pipeline("summarization", "vmarklynn/bart-large-cnn-samsum-acsi-ami-v2", truncation=True)
kw_model = keybert.KeyBERT(model='all-mpnet-base-v2')
#-----------------------------------------------------------------------------------------------------               
def summarizeText(transcription, wordCount=1024):
    text = cleanup(transcription)
    summary = summarizer(text)[0]['summary_text']
    
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
    
    ret = { 'summary': summary, 
            'keywords_list_1': keywords_list_1, 
            'keywords_list_2': keywords_list_2,
            'keywords_list_3': keywords_list_3,}

    return ret
