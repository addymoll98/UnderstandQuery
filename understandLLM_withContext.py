# Get the Repo Path from Understand
import sys
repo_path = sys.argv[1]
print(f"Repo Path: {repo_path}")

# Load documents
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
loader = GenericLoader.from_filesystem(
    repo_path,
    glob="**/*",
    suffixes=[".cpp", ".hpp"],  # Update to match C++ file extensions
    exclude=[],
    parser=LanguageParser(language=Language.CPP, parser_threshold=500),  # Update to use CPP language parser
)
documents = loader.load()

# Split documents
cpp_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.CPP, chunk_size=1000, chunk_overlap=1  # Update to use CPP language
)
texts = cpp_splitter.split_documents(documents)

# Embed documents
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
all_embeddings = embeddings.embed_documents([text.page_content for text in texts])

# Ensure the list of embeddings matches the text chunks
if len(all_embeddings) != len(texts):
    raise ValueError("Mismatch between the number of embeddings and text chunks.")

# Create Chroma DB
from langchain_community.vectorstores import Chroma
db = Chroma.from_documents(texts, embeddings)
retriever = db.as_retriever(
    search_type="mmr",  # Also test "similarity"
    search_kwargs={"k": 8},
)

### FOR RUNNING WITH HUGGINGFACE ###

import os
print(os.getenv('HUGGINGFACEHUB_API_TOKEN'))

from langchain_huggingface import HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    temperature= 0.1,
    top_k= 30,
    max_new_tokens= 512,
    repetition_penalty= 1.03
)

print(os.getenv('HUGGINGFACEHUB_API_TOKEN'))

##### FOR RUNNING WITH LLAMAFILE #####

# from langchain_community.llms.llamafile import Llamafile
# llm = Llamafile()

#### FOR RUNNING WITH LLAMACPP #####

# from langchain_community.llms import LlamaCpp
# zephr_path = "/Users/adelinemoll/Documents/LLM/zephyr-7b-beta.Q2_K.gguf" # Mac Path, REPLACE WITH YOUR PATH TO YOUR LOCAL LLM MODEL
# # zephr_path = "/home/adelinemoll/Public/LLM/zephyr-7b-beta.Q2_K.gguf" # Linux Path
# llm = LlamaCpp(model_path=zephr_path, verbose=False, n_ctx=4096)

#####################################

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Create the chain for retreiving context from the documents
prompt = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
        (
            "user",
            "Given the provided user question, which includes C++ code, generate a search query to look up portions of the codebase that are related to this question.",
        ),
    ]
)

retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

# Create the chain for calling the LLM
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
A user is asking a question about a snippet of C++ code within their codebase.
Your job is to anwer the users question as simply and concisely as possible.
Use the provided context to help you answer the quesion.
If you cannot answer, do not make up an answer, instead, respond with 
"I do not have enough context to answer this question". \n\n{context}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
    ]
)
document_chain = create_stuff_documents_chain(llm, prompt)

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
Generate a concise summary of the provided C++ function. 
Use the provided context from other parts of the codebase if it helps summarize the function. 
Keep the summary to 5 sentences or less. 

Instructions: 
- Provide one paragraph.
- Use 5 sentences or less.
- Describe dependencies, important functions and classes, and relevant information from comments.

Restrictions:
- Do not engage in any conversation.
- Only describe the provided C++ function.

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
userOption = sys.argv[2]
if userOption == "Ask a question":
    userQuestion = sys.argv[3]
    prompt = QUESTION_PROMPT_TEMPLATE.format(code=contents, question=userQuestion)
    print("Question: " + userQuestion)
else:
    prompt = SUMMARIZE_CODE_PROMPT_TEMPLATE.format(code=contents)
    print("Here is a summary of this code: ")

# Create a chain that combines the retreiver_chain and document_chain
qa = create_retrieval_chain(retriever_chain, document_chain)

# Invoke the LLM and print the response
result = qa.invoke({"input": prompt})
print(result['answer'])

