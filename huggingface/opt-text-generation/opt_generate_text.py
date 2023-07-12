from gravityai import gravityai as grav
from transformers import AutoTokenizer, OPTForCausalLM
import torch
import json
import os

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

# Sets environment variables in the container itself
# might or might not work :-)
#os.system("export {0}={1}".format(hugging_hub_env_key, hugging_hub_env_value))
#os.system("export {0}={1}".format(hugging_trans_env_key, hugging_trans_env_value))


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
def doIt(dataPath, outPath):

   dataFile = json.load(open(dataPath))[0]
   # these are assumed to be required fields; notice that we don't check for None
   inputText = dataFile['text']
   max_length = dataFile['max_length']

   # Optional fields are passed on as 'None', so be sure to handle accordingly
   if 'max_new_tokens' in dataFile and dataFile['max_new_tokens']:
         max_new_tokens = dataFile['max_new_tokens']
   else:
      max_new_tokens = None
   
   if 'min_length' in dataFile and dataFile['min_length']:
      min_length = dataFile['min_length']
   else:
      min_length = 10
   
   if 'do_sample' in dataFile and dataFile['do_sample']:
      do_sample = dataFile['do_sample']
   else:
      do_sample = False
   
   if 'early_stopping' in dataFile and dataFile['early_stopping']:
      early_stopping = dataFile['early_stopping']
   else:
      early_stopping = False
   
   if 'num_beams' in dataFile and dataFile['num_beams']:
      num_beams = dataFile['num_beams']
   else:
      num_beams = 1
   
   if 'temperature' in dataFile and dataFile['temperature']:
      temperature = dataFile['temperature']
   else:
      temperature = 1
   
   if 'top_k' in dataFile and dataFile['top_k']:
      top_k = dataFile['top_k']
   else:
      top_k = 50
  
   if 'top_p' in dataFile and dataFile['top_p']:
      top_p = dataFile['top_p']
   else:
      top_p = 1
   
   if 'repetition_penalty' in dataFile and dataFile['repetition_penalty']:
      repetition_penalty = dataFile['repetition_penalty']
   else:
      repetition_penalty = 1
   
   if 'no_repeat_ngram_size' in dataFile and dataFile['no_repeat_ngram_size']:
      no_repeat_ngram_size = dataFile['no_repeat_ngram_size']
   else:
      no_repeat_ngram_size = 0
   
   if 'encoder_no_repeat_ngram_size' in dataFile and dataFile['encoder_no_repeat_ngram_size']:
      encoder_no_repeat_ngram_size = dataFile['encoder_no_repeat_ngram_size']
   else:
      encoder_no_repeat_ngram_size = 0
   
   if 'num_return_sequences' in dataFile and dataFile['num_return_sequences']:
      num_return_sequences = dataFile['num_return_sequences']
   else:
      num_return_sequences = 1
   
   input_ids = tokenizer(inputText, return_tensors="pt").input_ids
   
   output = model.generate(input_ids,
      max_length = max_length,
      max_new_tokens = max_new_tokens,
      min_length = min_length,
      do_sample = do_sample,
      early_stopping = early_stopping,
      num_beams = num_beams,
      temperature = temperature,
      top_k = top_k,
      top_p = top_p,
      repetition_penalty = repetition_penalty,
      no_repeat_ngram_size = no_repeat_ngram_size,
      encoder_no_repeat_ngram_size = encoder_no_repeat_ngram_size,
      num_return_sequences = num_return_sequences)
   
   with open(outPath, 'w') as f:
      text_generated = []
      for i in range(len(output)):
         text_generated.append(tokenizer.decode(output[i], skip_special_tokens=True))
      results = {"generated_text":text_generated}
      json.dump(results, f)

# Actually fetches the model from huggingface
# Make sure that cache_dir specifies a directory under /tmp
#
# For huggingface transformer models, I highly recommend you use the path specified under 
# hugging_trans_env_value; no guarantees that it will work otherwise
model = OPTForCausalLM.from_pretrained("facebook/opt-350m", cache_dir=hugging_trans_env_value)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m", cache_dir=hugging_trans_env_value)

grav.wait_for_requests(doIt)
