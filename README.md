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
---

Feel free to email us (asebastian@seattleu.edu, ytan@seattleu.edu) if you want to see a demonstration of the framework.
Here's a basic example of how to navigate through the UI to obtain a transcript, summary, and keywords:

#### 1. Upload a recording file (.wav)
Click on the `Choose File` button to select your recording file, then click the `Upload File` button.

#### 2. Wait for processing
The processing time depends on the length of your recording. With a single GPU (RTX 2060 with Max-Q), it can take approximately 3 minutes to process an 18-minute audio file.

#### 3. Edit the generated transcript (Optional)
A media player is provided so you can listen to the audio while following the transcript. The transcription box below allows you to edit the transcribed content if necessary. For instance, you might want to correct a word, number, or sentence. There is also an option for you to change the speaker tags identified by our system, for example, from 'SPEAKER_00' to a specific name, and 'SPEAKER_01' to another name, and so forth. Note that any changes made will only be saved in the storage after clicking the `Change` button. 

#### 4. Get a summary and keywords
In the top-left corner of the current page, click on the blue `Summary` link to navigate to the second page. Here, you will find the `Get my summary` button – click on it. The processing time for generating a summary and keywords depends on the length of your recording.

### Navigating through the Repository
---

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
10.  __init__.py - a file that imports modules from transcribe_service.py and summarizer_service.py if they exist in this main project directory else it notifies that the service files are missing.
11. services.py - a Python file autogenerated by the Jupyter notebook located at `/notebooks/whisper_trans.ipynb`
12. summarizer_service.py - another Python file autogenerated by the Jupyter notebook located at `/notebooks/summarizer.ipynb`
13. transcribe_service.py - another Python file autogenerated by the Jupyter notebook located at `/notebooks/transcribe.ipynb`
14. urls.py - a file that is part of the standard structure of a Django application. It is used to define URL routing for the Django web application.

### Summary of Research Work
---

##### 1.Research Objectives
- Analyze and summarize audio data in conversations with more than two speakers in a meeting environment.
- Speaker Diarization - Identifying how many speakers are there in the conversation and when each speaker spoke.  
- Summarization – there are two approaches - Extractive vs. Abstractive. We use the abstractive method since it is more challenging and produces more human-friendly summaries.
- We develop a system framework consisting of three phases: 1. Speech-to-text conversion 2. Text summarization 3. The user interface for the model usage  

##### 2. Related Work
- Bidirectional Auto-Regressively Transformer (BART) (Lewis et al., 2020): Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
- QMSum (Zhong et al., 2021) is a novel benchmark designed for evaluating the performance of summarization models in the context of multidomain meetings.
- Whisper (Radford et al., 2022) is a system designed for automatic speech recognition, developed by OpenAI. https://github.com/openai/whisper
- Pyannote (Bredin et al., 2020) is a Python library that provides tools for various tasks in multimedia processing.
- Our summarizing spoken dialogues is closely related to a method of summarization for spontaneous speech (Furui et al., 2004). Their approach diverges from ours in that it is fundamentally
extractive while we aim at abstractive summarization, generating new words and sentences to maintain the essence of the source material.

##### 3. Tools Used
- Python 3.8
- Jupyter Notebook, VSCode, Datalore
- FFMEG
- Whisper Open AI
- Pyannote
- PyTorch
- CUDA
- Hugging Face Transformers
- AWS
- BART
- Github
- Django
- HTML, CSS, Javascript

##### 4. Datasets used
- AMI Corpus (Commission, accessed on April 13, 2023; Kraaij et al., 2005) - https://groups.inf.ed.ac.uk/ami/corpus/ 
- ICSI Corpus (International Computer Science Institute Speech Group, accessed on April 13, 2023; Janin et al., 2003) ICSI Meeting Corpus - https://groups.inf.ed.ac.uk/ami/icsi/ 
- reference summaries obtained from the paper (Zhong et al., 2021)

##### 5. Metrics for assessing performance
Compare automatically produced summaries against reference summaries that are human-produced. For example, if a machine summary produced “the cat was found under the bed,” it would be compared to “the cat was under the bed.”. In our analysis, we used three ROUGE methods:
1. Rouge-N: This measures the overlap between the generated text and the reference text in terms of n-grams.
2. Rouge-L: This measures the overlap between the generated text and the reference text in terms of the longest common subsequences.
3. Rouge-Lsum: It computes the Rouge-L score for multiple sentences, and then averages the scores.


