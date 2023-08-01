# A Framework for Abstractive Summarization of Conversational Meetings
#### Project Repository for Research Paper

Vincent Marklynn<sup>1</sup>, Anjali Sebastian<sup>1</sup>, Yong Long Tan<sup>1</sup>, Wan D. Bae<sup>1</sup>,
Shayma Alkobaisi<sup>2</sup>, and Sada Narayanappa<sup>3</sup> <br>
<sup>1</sup>Dept. of Computer Science, Seattle University, {vmarklynn, asebastian, ytan, baew}@seattleu.edu <br>
<sup>2</sup>College of Information Technology, United Arab Emirates University, shayma.alkobaisi@uaeu.ac.ae <br>
<sup>3</sup>Lockheed Martin Spaces, Inc, sada.narayanappa@lmco.com

### Research Paper Abstract
---
We aim to develop a system framework and conduct analysis of speech to text specifically in conversations with more than two speakers in a meeting environment. Unlike extractive summarization, abstractive summarization creates summaries by synthesizing new words and sentences that maintain the original meaning of the source. This presents new challenges researchers and developers face when developing language processing models for text generation. With recent advances in automatic speech recognition and natural language processing models such as OpenAI’s Whisper and Meta’s BART, we simplify the process of speech recognition and abstractive summarization of long meetings. Our proposed framework consists of three phases; speech to text conversion and text summarization work as a pipeline and the models are integrated to a web user interface. We demonstrate the development of the proposed pipeline and the applications of the trained models. We also show both quantitative and qualitative analysis on the model performance comparing to the BART base model. Our model with summarizing long meeting dialogues improved summarization by 139.6% over the base model in the ROUGE-LSUM metric. Many companies and organizations can benefit from our solution in various applications. Besides granting accessibility accommodations to hard of- hearing people, our framework provides accurate and insightful analysis to industry and academia.

### User Interface 
Here's a basic example of how to navigate through the UI to obtain a transcript, summary, and keywords:

#### Upload a recording file (.wav)
Click on the `Choose File/Browse` button to select your recording file, then click the `Upload File` button.

#### Wait for processing
The processing time depends on the length of your recording. With a single GPU (RTX 2060 with Max-Q), it can take approximately 3 minutes to process an 18-minute audio file.

#### Edit the generated transcript (Optional)
A media player is provided so you can listen to the audio while following the transcript. The transcription box below allows you to edit the transcribed content if necessary. For instance, you might want to correct a word, number, or sentence. There is also an option for you to change the speaker tags identified by our system, for example, from 'SPEAKER_00' to a specific name, and 'SPEAKER_01' to another name, and so forth. Note that any changes made will only be saved in the storage after clicking the `Change` button. 

#### Get a summary and keywords
In the top-left corner of the current page, click on the blue `Summary` link to navigate to the second page. Here, you will find the `Get my summary` button – click on it. The processing time for generating a summary and keywords depends on the length of your recording.

### Navigating through the Repository
1. data: contains all the full transcripts generated from the audio files of AMI and ICSI corpus using both whisper and pyannote.
2. formatted_data_v3: contains the data after preprocessing. We are trying to make appropriate input to the BART model. The formatted data takes the data and converts it to the speaker: Dialogue format that the BART model can be trained on.
3. docs - Documentation of the framework
- parrot_presentation.pptx presentation on the project. 
- Youtube link to presentation https://www.youtube.com/watch?v=QHgWTdenAyM
- Youtube Link to demo https://www.youtube.com/watch?v=3fHrQ8I0j4c
- Final Report - the final submitted report for the project
  
4. reference_txt:  Human-made summaries for reference purposes taken from  Yale-LILY/QMSum: Dataset for NAACL 2021 paper: "QMSum: A New Benchmark for Query-based Multi-domain Meeting Summarization" (github.com)  
5. templates: all Django template files for the user interface
6. mini_samples: small audio samples to test the Django UI
7. notebooks: all code files are here. Detailed info on each code file is as follows
- audio_to_text.ipynb - Code for making the pipeline for transcribing AMI and ICSI meetings into text. input = audio meeting files output = the text files found in the data folder.
- parrot_train.ipynb - Script for training our base model and fine-tuning BART 

**The summarization model BART<sub>ASC</sub> is up on hugging face**  Link: https://huggingface.co/vmarklynn/bart-large-cnn-samsum-acsi-ami-v2
Other experimental models trained are also available here. 

- preprocess.ipynb - This is the code for converting the text data to the formatted data i.e. data that can be used to train the BART Model. The formatted data will be in speaker: Dialogue format.
- summarization.ipynb - is our summarization service that generates summary and keywords on the django interface.
- transcribe.ipynb -  is our service that generates the transcription service on the UI. This has the code for generating a transcript on the first page of the UI. For combining the outputs of whisper and pyannote we used code segments from - https://github.com/yinruiqing/pyannote-whisper

8. Environment Files: environment.yml (conda environment exported from AWS)
9. **Project Setup:** project_setup.md - how to set up our project and the required environment


### 1.Research Objectives
- Analyze and summarize audio data in conversations with more than two speakers in a meeting environment.
- Speaker Diarization - Identifying how many speakers are there in the conversation and when each speaker spoke.  
- Summarization – there are two approaches - Extractive vs. Abstractive. We use the abstractive method as there are not many people doing this. 

### 2. Related Work
BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension 
BART is pre-trained by corrupting the input text using some noise schemes and learning a model to reconstruct the original input. It reads the corrupted input in both directions, left to right or vice versa. The bidirectional encoder produces a set of hidden states that capture the meaning and context of our input text. Then, this collection of hidden states will get pushed to the autoregressive decoder. The decoder is another component that generates the output text one element at a time, where each element is conditioned on the previously generated elements.

### 3. Tools Used
- Whisper -  https://github.com/openai/whisper  
- Pyannote - Partitioning speakers (diarization) - Amazon Transcribe 
- Summarization Algorithm – BART 
- NLTK - https://www.nltk.org/
- Django Framework  

### 4. Datasets Used
- AMI Meeting Corpus - https://groups.inf.ed.ac.uk/ami/corpus/ 
- ICSI Meeting Corpus - https://groups.inf.ed.ac.uk/ami/icsi/ 

BART hasn’t been trained on any long conversation dataset that includes multiple speakers. We plan to feed some of the AMI and ICIS corpus to BART and utilize another portion of the data to validate the outcome. 

### 5. Metrics for assessing performance
- Our main evaluation metric will be ROUGE (Recall-Oriented Understudy for Gisting Evaluations). 
- It compares automatically produced summaries against reference summaries that are human-produced. 
- For example, if a machine summary produced “the cat was found under the bed,” it would be compared to “the cat was under the bed.” 
- ROUGE measures the following: **Recall:** How much of the reference summary is recovered from the reference? **Precision:** How much of the system summary was relevant? 
- ROUGE needs reference summaries. For reference summaries, we are using previous human summaries created for the data set.
