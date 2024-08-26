# UnderstandQuery Plugin

UnderstandQuery is a Plugin for Understand which allows you to ask and LLM questions about your code from directly within Understand. There are two versions of the code, with and without context. The version with context uses a RAG setup to embed the codebase and use it as context when sending a query to the LLM. The version without context only uses a user-highlighted snippet of code in the query to the LLM. 

## Installation

UnderstandQuery requires the following packages, which can be installed using the package manager [pip](https://pip.pypa.io/en/stable/).

```bash
pip install langchain
pip install langchain-community
pip install langchain-core
pip install chromadb
pip install sentence-transformers
pip install tree-sitter==0.21.3 tree-sitter-languages
pip install langchain_huggingface
```

This program can be run with either llama.cpp or llamafile. 

### Running with llama.cpp
An LLM in a GUFF format must be downloaded locally, and the model_path variable must be updated with the path to this model. The model we have used is `DeepSeek-Coder-V2-Lite-instruct-Q4_K_M.gguf`, which can be dowloaded from [Hugging Face](https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF). Update the `model_path` to point to the local LLM GUFF file.

### Running with llamafile
If using llamafile, you need to download a Llamafile model (will have .llamafile as an extention). Before running the program, you need to start up llamafile by running that file as an executable. (For example, on linux  in the terminal: $./mistral-7b-instruct-v0.2.Q4_K_M.llamafile)

## Setting up Understand

If you want to be able to use both the "with context" and "without context" versions of UnderstandQuery, you will need to add two seperate User Tools in Understand.

In Understand, go to Tools > User Tools > Configure

Select "New"

Add the following information:
* **Menu Text:** UnderstandQuery
* **Command:** /path/to/your/python
* **Parameters:**
    * **(no context):** /path/to/understandLLM_noContext.py $PromptForSelect"Select an option=Generate a summary;Ask a question"  $PromptForText"Enter your question"
    * **(with context):** /path/to/understandLLM_withContext.py $CurProjectDir $PromptForSelect"Select an option=Generate a summary;Ask a question"  $PromptForText"Enter your question"
* **Input:** Selected Text
* **Output:** Capture
* **Add to:** Context Menu

## Usage

Within Understand, highlight a snippet of code you'd like to use with the LLM. Right click, and select User Tools > Your Tool Name.
There will be a pop-up box that asks you to select either "Generate a summary" or "Ask a question". Note that the text box "Enter your question" will be present for both options, but the question that you type is only used for "Ask a question" and any text entered there is ignored if you select "Generate a summary". 
