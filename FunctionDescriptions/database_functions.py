import os
import understand
import Function_Analyzer_Helper

# UND_PATH = "/Users/adelinemoll/Documents/LLM/LangChain/SCpp/SCpp.und" # Mac Path
UND_PATH = "/home/adelinemoll/Public/LLM/LangChain/SCpp/SCpp.und" # Linux Path
REPO_PATH = "/home/adelinemoll/Public/LLM/LangChain/SCpp" # Linux Path

def capture_files(db):
    file_list = []
    # Capture files
    captured_files = db.ents("file ~unnamed")
    for file in sorted(captured_files,key= lambda ent: ent.longname()):
        file_list.append(Function_Analyzer_Helper.Understand_File(
            file.relname(), 
            file.depends(), 
            file.dependsby(), 
            file.contents(),
            functions=[]
        ))
    print(f"Number of files: {len(file_list)}")
    return file_list

def capture_functions(db):
    function_list = []
    # Capture functions
    captured_functions = db.ents("function ~unknown ~unresolved")
    for function in sorted(captured_functions,key= lambda ent: ent.longname()):
        function_list.append(Function_Analyzer_Helper.Understand_Function(
            function.longname(), 
            function.contents(),
            function.refs()
        ))
    print(f"Number of functions: {len(function_list)}")
    return function_list

def sort_functions(understand_files, understand_functions):
    print("in sort_functions")
    for function in understand_functions:
        for file in understand_files:
            if file.relname == str(function.ref_list[0].file().relname()):
                file.append_func(function)
                break 
    return understand_files