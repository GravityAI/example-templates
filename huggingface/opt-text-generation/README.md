# Overview

This is a basic model that utilizes the OPT model as proposed in Open Pre-trained Transformer Language Models by Meta AI.
For more information see: 
https://huggingface.co/docs/transformers/model_doc/opt
https://arxiv.org/pdf/2205.01068.pdf

# Project contents
```
huggingface_opt-text-generation/
|-> .vscode/
|    |-> launch.json //just contains a basic debug setup for VSCode
|-> dataInput.json //an example input file.  This input file will be run as a test during the build; see gravityai-build.json
|-> dataInput2.json //another example of an input file; used for debugging purposes only
|-> dataOutput.json //this is the output after feeding in dataInput.json.  Not needed during the build
|-> dataOutput2.json //this is the output after feeding in dataInput2.json. Not needed during the build
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
|-> README.md // That's me. Hi!
|-> requirements.txt // Your little slice of python dependencies for your project.  Lucky you!
```