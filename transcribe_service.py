import whisper, hashlib, os, datetime, json, torch, pyannote, logging, sys
from mangorest.mango import webapi
from pyannote.audio import Pipeline
from pyannote.core import Segment, Annotation, Timeline
from transformers import pipeline

import parrot.summarizer as summarizer

logging.basicConfig( level=logging.INFO,
    format='%(levelname)s:%(name)s %(asctime)s %(filename)s:%(lineno)s:%(funcName)s: %(message)s',
    #handlers=[ logging.FileHandler("/tmp/stream.log"), logging.StreamHandler()],
    handlers=[ logging.StreamHandler()],
)
logger = logging.getLogger( "app.audio" )

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
                
#-----------------------------models-----------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
transcriber = whisper.load_model("base.en", device=device)
diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                use_auth_token="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")
#---------------------------------------------------------------------------------------------------- 
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
#--------------------------------------------------------------------------------------------------------    
def _transcribe_process(file):
    result = transcriber.transcribe(file)
    diarization = diarizer(file)
    final_result = PyanWhisper.diarize_text(result, diarization)
    
    # Write final result to a new file
    ret = ""
    with open(file+".txt", "w") as new_f:
        for seg, spk, sent in final_result:
            start = str(datetime.timedelta(seconds=int(seg.start)))
            end = str(datetime.timedelta(seconds=int(seg.end)))
            line = f'{start} - {end} | {spk}:{sent}\n'
            new_f.write(line)                               
            ret += line
        transcription = ret
    
    response = { 'file_url': file, 
                 'transcription': transcription,  
                 'text': result["text"]
                }

    return response
#--------------------------------------------------------------------------------------------------------    
@webapi("/parrot/processfile")
def processfile(request, **kwargs):
    files = uploadfile(request, **kwargs).split("\n")
    if ( len(files) <= 0 ):   return "WARNING: No files given!"
    file = files[1]

    if os.path.exists(file+".json"):
        return open(file+".json", "r").read()

    if os.path.exists(file+".processing"): return f"WARNING: {file}.being processed!"
    open(file+".processing", "w").write("STARTED")

    ret = _transcribe_process(file)
    srt = summarizer.summarizeText( ret['transcription'])

    with open (file +".json", "w") as f:
        f.write(json.dumps(ret))

    ret.update(srt)
    return ret
