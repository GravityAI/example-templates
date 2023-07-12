# Overview

This model utilizes facebook/wav2vec2-large-xlsr-53 using training data associated with Common Voice 6.1.

See: https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english

for more details

# Project contents
```
wav2vec2-large-xlsr-53-english/
|-> .vscode/
|    |-> launch.json     // just contains a basic debug setup for VSCode
|-> 4507-16021-0012.wav  // an example input file.  This input file will be run as a test during the build; see gravityai-build.json
|-> out.dat              // this is the output after feeding in 4507-16021-0012.wav. Not needed during the build
|-> gravityai-build.json // this is gravityai's build file.
|                        //  
|                        // - If you are using gravityAI's pypi library (highly recommended), 
|                        //   you will need to specify UseGaiLib: true
|                        // - Tests[] will contain build time tests to be run. In this case, we are specifying a test
|                        //   which will consume some input file, and we don't care about the output.  
|                        //   This is essentially a hack to avoid "container diagnostics" which will throw a bunch of random
|                        //   data at your model during the build.  If your model isn't ready to be thoroughly vetted, this is a hack 
|                        //   to avoid and build your model.  If you are confident that your model will handle a variety of data
|                        //   input, then you can remove Tests[]
|-> main.py              // entrypoint containing code that hooks into gravityAI's library
|-> README.md // That's me. Hi!
|-> requirements.txt // Your little slice of python dependencies for your project.  Lucky you!
```