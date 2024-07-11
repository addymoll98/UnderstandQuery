import sys

### FOR RUNNING WITH HUGGINGFACE ###
import os;
from langchain_huggingface import HuggingFaceEndpoint

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
llmHUGGINGFACE = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    temperature= 0.1,
    top_k= 30,
    max_new_tokens= 512,
    repetition_penalty= 1.03
)

### FOR RUNNING WITH OPENAI ###

# import os
# os.environ["OPENAI_API_KEY"] = ""
# from langchain_openai import OpenAI
# llmOPENAI = OpenAI(openai_api_key="OPENAP_API_KEY", openai_organization="proj_RRvxgAHHyE2h3ZOVM07xZGIZ")

##### FOR RUNNING WITH LLAMAFILE #####

# from langchain_community.llms.llamafile import Llamafile
# llmLLAMAFILE = Llamafile()

#### FOR RUNNING WITH LLAMACPP #####

from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate    
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

zephr_path = "/Users/adelinemoll/Documents/LLM/zephyr-7b-beta.Q2_K.gguf" # Mac Path
# zephr_path = "/home/adelinemoll/Public/LLM/zephyr-7b-beta.Q2_K.gguf" # Linux Path
llmLLAMACPP = LlamaCpp(model_path=zephr_path, verbose=False, n_ctx=4096)

#####################################



contents = []
while True:
    try:
        line = input()
    except EOFError:
        break
    contents.append(line)

contents= '\n'.join(contents)

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

userOption = sys.argv[1]
if userOption == "Ask a question":
    userQuestion = sys.argv[2]
    prompt = QUESTION_PROMPT_TEMPLATE.format(code=contents, question=userQuestion)
    print("Question: " + userQuestion)
else:
    prompt = SUMMARIZE_CODE_PROMPT_TEMPLATE.format(code=contents)
    print("Here is a summary of this code: ")

# result = llmLLAMACPP.invoke(contents)
result = llmHUGGINGFACE.invoke(prompt)
print("LLM Response: " + result)
