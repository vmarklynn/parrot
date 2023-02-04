
import whisper, pytube, hashlib, os, datetime
from pytube import YouTube
from nltk.tokenize import sent_tokenize
from  mangorest.mango import webapi

model = whisper.load_model("base")


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
