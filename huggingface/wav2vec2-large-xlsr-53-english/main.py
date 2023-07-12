from gravityai import gravityai as grav
from huggingsound import SpeechRecognitionModel
import os


# @misc{grosman2021xlsr53-large-english,
#   title={Fine-tuned {XLSR}-53 large model for speech recognition in {E}nglish},
#   author={Grosman, Jonatas},
#   howpublished={\url{https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english}},
#   year={2021}
# }

# Make sure that caches are downloaded to /tmp , which is a writable directory
# These are environment keys we need to setup, so that when we fetch models, they'll be stored under /tmp
hugging_hub_env_key = "HUGGINGFACE_HUB_CACHE"
hugging_hub_env_value = "/tmp/huggingface/hub/.cache"

hugging_trans_env_key = "TRANSFORMERS_CACHE"
hugging_trans_env_value = "/tmp/huggingface/transformers/.cache"

# Only lasts during the duration of the python process; this will not set the environment variables of 
# the container
os.environ[hugging_hub_env_key] = hugging_hub_env_value
os.environ[hugging_trans_env_key] = hugging_trans_env_value

model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")

"""
This is an arbitrary function that you specify when you make your
grav.wait_for_requests(doIt) call.  

Important Notes: 
- In this case, we are calling the function "doIt", but it can be any python user function name.
  e.g. you can call it "pumperSnickle" if you want; don't care. Just as long as it takes two arguments,
  an input file path and an output file path (see next note below)
- This arbitrary function takes two arguments: an input file path, and output file path.
  - You can name the variables that you use to store these paths whatever you want!  In this case,
    the input file path is stored in the variable "dataPath" and the output file path is stored in
    the variable "outPath"

- The variables of dataPath (used to store your input file path) and outPath (used to store your output file path) 
  can be named whatever you want!
- Both dataPath (or whatever you call the variable storing your input file path)
  and outPath must be used while in this function!
- When the inputFile is read into our system, we don't exactly know the form it will take.
  e.g. a zip file can either be called "my_cool_archive.zip" or "my_cool_archive" (without the .zip)
  They are both zip files, at the end of the day, so rather than guess what the form is, we marshal
  the file into a <some_guid>.dat while we are in the container.
  - e.g. your input file path might be something like /some/path/f603a318-e518-479a-9fa1-7b305b6563b6.dat
  - *sigh* this is just a long-winded way of saying: "don't base any logic off of your input file suffix", 
    because it won't work while in the container!  Also, make sure that any function you feed your input file
    can handle a .dat file.
"""
async def doIt(dataPath, outPath):
    audio_paths = [dataPath]
    response = transcribe(audio_paths)
    with open(outPath, 'w') as outFile:
         outFile.write(response[0]["transcription"])

def transcribe(audio_paths):
    transcriptions = model.transcribe(audio_paths)
    return transcriptions

grav.wait_for_requests(doIt)