#!/usr/bin/env python
#%%writefile "/tmp/trans.py"

import whisper, pytube, hashlib, os, datetime, sys, re, json, argparse
from pytube import YouTube
from nltk.tokenize import sent_tokenize
from  mangorest.mango import webapi
import colabexts
import colabexts.jcommon

model =None

def getmodel():
    global model;
    
    if model is None:
        model = whisper.load_model("base")
        
    return model

    
def transcribe_file(file ="/Users/snarayan/Desktop/data/audio/index.mp4", **kwargs):
    result = getmodel().transcribe(file)
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
        tr = getmodel().transcribe(file)
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
    if (not os.path.exists(DESTDIR)):
        os.makedirs(DESTDIR)
    
    ret = "File:\n"
    for f in request.FILES.getlist('file'):
        content = f.read()
        filename = f"{DESTDIR}{str(f)}"
        print(f"\nSaved file: {filename}")
        with open(filename, "wb") as f:
            f.write(content)
        ret += filename + "\n"

    print("Retuning ", ret)
    return ret

<<<<<<< HEAD
#-----------------------------------------------------------------------------------
def process(sysargs):
    print("Parsing and processing")
    
    if (sysargs.url.strip()):
        print( f"Transcribing {sysargs.url}")
        transcribe_youtube(sysargs.url.strip())
    
#-----------------------------------------------------------------------------------
sysargs=None
def addargs():
    global sysargs
    p = argparse.ArgumentParser(f"{os.path.basename(sys.argv[0])}:")
    p.add_argument('-u', '--url', type=str, default="", help="Youtube URL")
    try:
        sysargs, unknown=p.parse_known_args(sys.argv[1:])
    except argparse.ArgumentError as exc:
        print(exc.message )
        
    if (unknown):
        print("Unknown options: ", unknown)
        #p.print_help()
    return sysargs    
#-----------------------------------------------------------------------------------
if __name__ == '__main__':
    if (not colabexts.jcommon.inJupyter()):
        t1 = datetime.datetime.now()
        sysargs = addargs()
        ret = process(sysargs)
        t2 = datetime.datetime.now()
        print(f"#All Done in {str(t2-t1)} ***")
    else:
        pass
