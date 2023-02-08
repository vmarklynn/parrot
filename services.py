import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import whisper, pytube, hashlib, os, datetime
from pytube import YouTube
from nltk.tokenize import sent_tokenize
from  mangorest.mango import webapi
import pyannote
from pyannote.audio import Pipeline
from pyannote.core import Segment, Annotation, Timeline
import json
from django.http import HttpResponse

# Taken from https://github.com/yinruiqing/pyannote-whisper
class PyanWhisper:
    PUNC_SENT_END = ['.', '?', '!']
        
    def diarize_text(transcribe_res, diarization_result):
        timestamp_texts = PyanWhisper.get_text_with_timestamp(transcribe_res)
        spk_text = PyanWhisper.add_speaker_info_to_text(timestamp_texts, diarization_result)
        res_processed = PyanWhisper.merge_sentence(spk_text)
        return res_processed

    def get_text_with_timestamp(transcribe_res):
        timestamp_texts = []
        for item in transcribe_res['segments']:
            start = item['start']
            end = item['end']
            text = item['text']
            timestamp_texts.append((Segment(start, end), text))
        return timestamp_texts
    
    def add_speaker_info_to_text(timestamp_texts, ann):
        spk_text = []
        for seg, text in timestamp_texts:
            spk = ann.crop(seg).argmax()
            spk_text.append((seg, spk, text))
        return spk_text
    
    def merge_cache(text_cache):
        sentence = ''.join([item[-1] for item in text_cache])
        spk = text_cache[0][1]
        start = text_cache[0][0].start
        end = text_cache[-1][0].end
        return Segment(start, end), spk, sentence
    
    def merge_sentence(spk_text):
        merged_spk_text = []
        pre_spk = None
        text_cache = []
        for seg, spk, text in spk_text:
            if spk != pre_spk and pre_spk is not None and len(text_cache) > 0:
                merged_spk_text.append(PyanWhisper.merge_cache(text_cache))
                text_cache = [(seg, spk, text)]
                pre_spk = spk
            elif text[-1] in PyanWhisper.PUNC_SENT_END:
                text_cache.append((seg, spk, text))
                merged_spk_text.append(PyanWhisper.merge_cache(text_cache))
                text_cache = []
                pre_spk = spk
            else:
                text_cache.append((seg, spk, text))
                pre_spk = spk
        if len(text_cache) > 0:
            merged_spk_text.append(PyanWhisper.merge_cache(text_cache))
        return merged_spk_text

    def write_to_txt(spk_sent, file):
        with open(file, 'w') as fp:
            for seg, spk, sentence in spk_sent:
                line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sentence}\n'
                fp.write(line)

# our models                
model = whisper.load_model("base", device="cuda")
diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token="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")

bart = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

def transcribe_file(file ="/Users/snarayan/Desktop/data/audio/index.mp4", **kwargs):
    result = model.transcribe(file)
    return result

    

def splitIntoParas(tr, nLinesPerPara=4):
    n= nLinesPerPara
    l=tr.get('segments', [])
    ret = ""
    for i,j in enumerate(l[::n]):
        a, b = i*n, i*n + n
        o = "".join([j['text'] for j in l[a:b]])
        ret += o.strip() + "\n\n";
        #print(f'{a}-{b} {o} \n')
        
    return ret



test_url = "https://www.youtube.com/watch?v=DuSDVj9a4WM&list=PLEpvS3HCVQ5_ZlyF1_i-WSwBzLoDLxoc9"

#--------------------------------------------------------------------------------------------------------    
@webapi("/scribe/transcribe_youtube/")
def transcribe_youtube( url = test_url , force_download=False, force_transribe=False, **kwargs):    
    h = hashlib.md5(url.encode())
    file = "/tmp/" + str(h.hexdigest()) + ".mp4"
    
    if (force_download or not os.path.exists(file)):  
        file = YouTube(url).streams.filter(only_audio=True).first().download(filename=file)

    print( f"File: {file}")
    if (force_transribe or not os.path.exists(file +".txt")):  
        print( f"Calling transcription: {file}.txt")
        tr = model.transcribe(file)
        ret = splitIntoParas(tr)
        with open(file +".txt", "w") as f:
            f.write(ret)
        with open(file +".json", "w") as f:
            f.write(str(tr))
            
        transcription = ret
    else:
        with open(file +".txt", "r") as f:
            transcription = f.read()
        
    return transcription;


#--------------------------------------------------------------------------------------------------------    
@webapi("/parrot/transcribe_wavinput/")
def transcribe_wavinput(url, **kwargs):
    print("Hi: " + url)
    if url.method == 'POST':
        file = request.FILES['file']
        print("I'm in")
    # ret = "\n\n My Name is: " + n + "\n"
    # for g in kwargs:
    #     if (g =="request"):
    #         continue;
    #     ret += g + " " + kwargs.get(g) + "\n"
    
    # return ret
#--------------------------------------------------------------------------------------------------------    
@webapi("/parrot/uploadfile")
def uploadfile(request,  **kwargs):
    par = dict(request.GET)
    par.update(request.POST)

    DESTDIR ="/tmp/parrot/"
    print("uploadfile : ", DESTDIR, kwargs)
    
    if (not os.path.exists(DESTDIR)):
        os.makedirs(DESTDIR)
    
    
    ret = "Files:\n"
    for f in request.FILES.getlist('file'):
        content = f.read()
        filename = f"{DESTDIR}{str(f)}"
        print(f"++ Save file {filename} Content: {len(content)} :")
        with open(filename, "wb") as f:
            f.write(content)
        ret += filename + "\n"

    print(" Retuning ", ret )
    return ret

#--------------------------------------------------------------------------------------------------------    
@webapi("/parrot/processfile")
def processfile(request, force_transribe=False, **kwargs):
    print("processing file: ", kwargs)

    ret = uploadfile(request, **kwargs)
    f = ret.split('\n')[1]


    print( f"Calling transcription: {f}")
    result = model.transcribe(f)
    diarization = diarizer(f)
    final_result = PyanWhisper.diarize_text(result, diarization) 
    ret = ""
    # Write to a new file
    with open(f +".txt", "w") as new_f:
        for seg, spk, sent in final_result:
            line = f'{spk}:{sent}\n'
            new_f.write(line)                               
            ret += line
        transcription = ret
    
    input_ids = tokenizer.encode(transcription, return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        outputs = bart.generate(input_ids)
    # Generate Summary
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n\n" + summary)
    response = {'transcription': transcription, 'summary': summary}
    return HttpResponse(json.dumps(response), content_type='application/json')    
