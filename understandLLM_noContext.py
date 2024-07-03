### FOR RUNNING WITH HUGGINGFACE ###

import os;
from langchain_huggingface import HuggingFaceEndpoint

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    temperature= 0.1,
    top_k= 30,
    max_new_tokens= 512,
    repetition_penalty= 1.03
)

######################################

##### FOR RUNNING WITH LLAMAFILE #####

# llm = Llamafile()

#####################################

#### FOR RUNNING WITH LLAMACPP #####

# from langchain_community.llms import LlamaCpp
# from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
# from langchain_core.prompts import PromptTemplate    
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# zephr_path = "/Users/adelinemoll/Documents/LLM/LangChain/zephyr-7b-beta.Q2_K.gguf"
# llm = LlamaCpp(model_path=zephr_path, verbose=True, n_ctx=4096, callback_manager=callback_manager)

#####################################

contents = []
while True:
    try:
        line = input()
    except EOFError:
        break
    contents.append(line)

contents= '\n'.join(contents)

SUMMARIZE_CODE_PROMPT = """
    Generate a concise summary of the provided C++ code. Keep the summary to 5 sentences or less. 

    Instructions: 
    - Provide one paragraph.
    - Use 5 sentences or less.
    - Describe dependencies, important functions and classes, and relevant information from comments.

    Restrictions:
    - Do not say "Summary" or "Output".
    - Do not engage in any conversation.
    - Only describe the provided C++ code.
    - Start directly with the summary, with no precursors.
    """

contents = contents + "\n" + SUMMARIZE_CODE_PROMPT
result = llm.invoke(contents)
print(result)