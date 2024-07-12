# Understand LLM Plugins

These are two Understand Plugins for using an LLM to interact with your code in Understand.

## Installation

You may need to install some or all the the follwoing pip packages:

```bash
pip install langchain
pip install langchain-community
pip install langchain-core
pip install chromadb
pip install sentence-transformers
pip install tree-sitter==0.21.3 tree-sitter-languages

```
(There may be some missing packages in this list, troubleshoot based on errors)

To use HuggingFace, create a Hugging Face Read-only API token, and set HUGGINGFACEHUB_API_TOKEN as an enviroment variable. 

## Setting up Understand

If you want to be able to use both the "with context" and "without context" tools, you will need to add two seperate User Tools in Understand.

In Understand, go to Tools > User Tools > Configure

Select "New"

Add the following information:
* **Menu Text:** Your name for the tool
* **Command:** /path/to/your/python
* **Parameters:**
    * **(no context):** /path/to/understandLLM_noContext.py $PromptForSelect"Select an option=Generate a summary;Ask a question"  $PromptForText"Enter your question"
    * **(with context):** /path/to/understandLLM_withContext.py $CurProjectDir $PromptForSelect"Select an option=Generate a summary;Ask a question"  $PromptForText"Enter your question"
* **Input:** Selected Text
* **Output:** Capture
* **Add to:** Context Menu

## Usage

To use these programs, you'll need to either use the HuggingFace API (this means that you're not running locally), or set up Llamafile or Llamacpp. There is code for all three options, you just need to toggle comments to switch between which LLM option you want to use. 

Within Understand, highlight a snippet of code you'd like to use with the LLM. Right click, and select User Tools > Your Tool Name.
There will be a pop-up box that asks you to select either "Generate a summary" or "Ask a question". Note that the text box "Enter your question" will be present for both options, but the question that you type is only used for "Ask a question" and any text entered there is ignored if you select "Generate a summary". 