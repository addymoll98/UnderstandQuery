import Function_Analyzer_Helper

def describe_functions(sorted):
    import os

    ### FOR RUNNING WITH HUGGINGFACE ###

    # from langchain_huggingface import HuggingFaceEndpoint
    # llm = HuggingFaceEndpoint(
    #     repo_id="HuggingFaceH4/zephyr-7b-beta",
    #     task="text-generation",
    #     max_new_tokens=512,
    #     do_sample=False,
    #     repetition_penalty=1.03,
    # )
        
    #####################################

    ##### FOR RUNNING WITH LLAMAFILE #####

    # from langchain_community.llms.llamafile import Llamafile
    # llm = Llamafile()

    #####################################

    ##### FOR RUNNING WITH LLAMACPP #####

    from langchain_community.llms import LlamaCpp
    from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
    from langchain_core.prompts import PromptTemplate    
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    zephr_path = "/home/adelinemoll/Public/LLM/zephyr-7b-beta.Q2_K.gguf"
    llm = LlamaCpp(model_path=zephr_path, verbose=True, n_ctx=4096, callback_manager=callback_manager)

    #####################################


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

    print(len(sorted))
    descriptions_file = open("descriptions.txt", "a")
    for i in range(50,53):
    # for file in sorted:
        for function in sorted[i].functions:
            if function.fullname.startswith("scpp"):
                function_text = function.content
                function_text = function_text + "Generate a description of the code above. Use 5 sentances or less."
                print(f"Generating summary for {function.fullname}")
                result = llm.invoke(f"{function_text} \n {SUMMARIZE_CODE_PROMPT}")
                descriptions_file.write(f"{function.fullname}\n")
                descriptions_file.write(f"{result}\n\n")
    descriptions_file.close()





