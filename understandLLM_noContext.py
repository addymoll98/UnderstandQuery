# You can run this program with HuggingFace (not local), Llamafile, or Llamacpp (both local)
# Toggle between the below comments to switch between models

### FOR RUNNING WITH HUGGINGFACE ###

from langchain_huggingface import HuggingFaceEndpoint

from langchain_huggingface import HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    temperature= 0.1,
    top_k= 30,
    max_new_tokens= 512,
    repetition_penalty= 1.03
)

##### FOR RUNNING WITH LLAMAFILE #####

# from langchain_community.llms.llamafile import Llamafile

# llm = Llamafile()

#### FOR RUNNING WITH LLAMACPP #####

# from langchain_community.llms import LlamaCpp

# zephr_path = "/Users/adelinemoll/Documents/LLM/zephyr-7b-beta.Q2_K.gguf" # Mac Path, REPLACE WITH YOUR PATH TO YOUR LOCAL LLM MODEL
# # zephr_path = "/home/adelinemoll/Public/LLM/zephyr-7b-beta.Q2_K.gguf" # Linux Path
# llm = LlamaCpp(model_path=zephr_path, verbose=False, n_ctx=4096)

#####################################

# Input the highlighted code snippet from Understand into Contents
contents = []
while True:
    try:
        line = input()
    except EOFError:
        break
    contents.append(line)

contents= '\n'.join(contents)

# Set up the prompt templates for summarizing the code or for asking a question about the code.
SUMMARIZE_CODE_PROMPT_TEMPLATE = """
Generate a concise summary of the provided C++ code. Keep the summary to 5 sentences or less. 

Instructions: 
- Provide one paragraph.
- Use 5 sentences or less.
- Describe dependencies, important functions and classes, and relevant information from comments.

Restrictions:
- Do not engage in any conversation.
- Only describe the provided C++ code.

Code:
{code}

Summary:
"""

QUESTION_PROMPT_TEMPLATE = """
You are an expert C++ developer. Answer the following question based on the given code. Provide a detailed and specific answer.

Code:
{code}

Question:
{question}

Answer:
"""

# Get the User Option and Question from Understand
import sys
userOption = sys.argv[1] # In Understand, the user is prompted to select "Summarize Code" or "Ask a question"
if userOption == "Ask a question":
    userQuestion = sys.argv[2] # This is the users question from Understand
    prompt = QUESTION_PROMPT_TEMPLATE.format(code=contents, question=userQuestion)
    print("Question: " + userQuestion)
else:
    prompt = SUMMARIZE_CODE_PROMPT_TEMPLATE.format(code=contents)
    print("Here is a summary of this code: ")

# Invoke the LLM and print the response
result = llm.invoke(prompt)
print("LLM Response: " + result)
