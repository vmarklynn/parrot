# PARROT SUMMARIZATION FOR MEETINGS
#### Project for Data Science Capstone @ Seattle University
#### Team Members: Vincent Marklynn, Long Yong Tan, Anjali Sebastian
#### Advisor: Dr. Wan Bae , Sponsor: Dr. Sada Narayanappa

### Information on all Modules
1. data: contains all the full transcripts generated from the audio files of AMI and ICSI corpus using both whisper and pyannote.
2. formatted_data_v3: contains the data after preprocess. We are trying to make appropriate for input to the BART model. The formatted data takes the data and converts it to speaker : Dialogue format that the BART model can be trained on.
3. docs - Important documents on the project
- parrot_presentation.pptx presentation on the project. 
- Youtube link to presentation https://www.youtube.com/watch?v=QHgWTdenAyM
- Youtube Link to demo https://www.youtube.com/watch?v=3fHrQ8I0j4c

- Final Report - the final submitted report for the project


4. reference_txt:  Human made summaries for reference purposes taken from  Yale-LILY/QMSum: Dataset for NAACL 2021 paper: "QMSum: A New Benchmark for Query-based Multi-domain Meeting Summarization" (github.com)  
5. templates: all Django template files
6. mini_samples: small audio samples to test the Django UI
7. notebooks: all code files are here. Detailed info on each code file
- audio_to_text.ipynb - Code for making the pipeline for transcribing AMI and ICSI meetings into text. input = audio meeting files output = the text files found in data folder.
- parrot_train.ipynb - Script for training our base model and fine tuning BART 
Our model is up on hugging face  https://huggingface.co/vmarklynn/bart-large-cnn-samsum-acsi-ami-v2
Other models we trained are also available here. 
- preprocess.ipynb - This is the code for convereting the text data to the formatted data i.e. data that can be used to train the BART Model. The formatted data will be in speaker : Dialogue format.
- summarization.ipynb - is our summarization service that generates summary and keywords on the django interface.
- transcribe.ipynb -  is our service that generates the transciption service on the UI . This has the code for generating transcript in the first page of the UI. For combining the ouputs of whisper and pyannote we used code segments from - https://github.com/yinruiqing/pyannote-whisper 
8. Environment Files: environment.yml (conda environment exported from AWS)
9. **Project Setup:** project_setup.md - how to setup our project and required environment


### 1.Project Objectives
- Analyze and summarize audio data in conversations with more than two speakers in a meeting environment.
- Speaker Diarization - Identifying how many speakers are there in teh converstion and when each speaker spoke.  
- Summarization – there are two approaches - Extractive vs. Abstractive. We use the abstractive method as there are not many people doing this. 

### 2. Related Work
BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension 
BART is pretrained by corrupting the input text using some noise schemes and learning a model to reconstruct the original input. It reads the corrupted input in both directions, left to right or vice versa. The bidirectional encoder produces a set of hidden states that capture the meaning and context of our input text. Then, this collection of hidden states will get pushed to the autoregressive decoder. The decoder is another component that generates the output text one element at a time, where each element is conditioned on the previously generated elements.

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
- It compares automatically produced summaries against reference summaries which are human produced. 
- For example, if a machine summary produced “the cat was found under the bed,” it would be compared against “the cat was under the bed.” 
- ROUGE measures the following: **Recall:** how much of the reference summary is recovered from the reference? **Precision:** how much of the system summary was relevant? 
- ROUGE needs reference summarries. For reference summary we are using previous human summaries created for the data set.
