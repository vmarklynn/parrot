Instructions for Mac and Linux
### 1. Setup Anaconda
### 2. Clone/ Download the following 2 repos so they come under 1 directory
- Parrot fromhttps://github.com/vmarklynn/parrot
```git clone https://github.com/vmarklynn/parrot.git``` <br>

- Learn Django from https://github.com/sada-narayanappa/LearnDjango
```git clone https://github.com/sada-narayanappa/LearnDjango.git``` 

### 3. Setup environment for whisper, pyannote, and django
a. Run the following command from (base) of anaconda.
```cd parrot``` then
```conda env create -f environment.yml```
```source activate whisper_django```

b. Install ffmpeg for whisper
##### On Mac
```brew install ffmpeg```
##### On Linux:
on Ubuntu or Debian
```sudo apt update && sudo apt install ffmpeg```
 
on Arch Linux
```sudo pacman -S ffmpeg```
 
using yum
```sudo yum install ffmpeg```

Linux ffmpeg issues
If you get error "Requires the ffmpeg CLI and `ffmpeg-python` package to be installed." after installing then try
find your libopenh under the environment
```ls ~/anaconda3/envs/whisper_django/lib/libopenh264*``` <br>
 
```cp /home/st/asebastian/anaconda3/envs/whisper_django/lib/libopenh264.so /home/st/asebastian/anaconda3/envs/whisper_django/lib/libopenh264.so.5```


### 4. Link Django to Parrot
```cd LearnDjango```
<br>```ln -s ../parrot .```
 
Link the static part. find ....  
### 5. Running the app
```cd LearnDjango```
<br> ``` sh ./run.sh```


 

   