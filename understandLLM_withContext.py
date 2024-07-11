import sys
import os
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.llms.llamafile import Llamafile

# Construct the full path to your repository
repo_path = sys.argv[1]
print(f"Repo Path: {repo_path}")

# print(f"Repository path: {repo_path}")

# Debug: Check if repo_path exists
if not os.path.exists(repo_path):
    print(f"Repo path does not exist: {repo_path}")
else:
    print(f"Repo path exists: {repo_path}")

# Load documents
loader = GenericLoader.from_filesystem(
    repo_path,
    glob="**/*",
    suffixes=[".cpp", ".hpp"],  # Update to match C++ file extensions
    exclude=[],
    parser=LanguageParser(language=Language.CPP, parser_threshold=500),  # Update to use CPP language parser
)
documents = loader.load()


# Debug: Print the number of loaded documents
print(f"Loaded {len(documents)} documents.")

# Check if documents are loaded and continue
if not documents:
    print("No documents were loaded. Please check the repository path and file patterns.")
else:
    print(f"Loaded {len(documents)} documents.")
    # Split documents
    cpp_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.CPP, chunk_size=1000, chunk_overlap=1  # Update to use CPP language
    )
    texts = cpp_splitter.split_documents(documents)

    # Debug: Print the number of text chunks and first few text chunks
    print(f"Split into {len(texts)} chunks.")
    # print(texts[:5])  # Print first 5 text chunks

    # Embed documents
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Embed all documents
    all_embeddings = embeddings.embed_documents([text.page_content for text in texts])

    # Debug: Verify generated embeddings
    if not all_embeddings:
        print("No embeddings were generated. Check the input documents and the embedding model.")
    else:
        print(f"Generated {len(all_embeddings)} embeddings.")

    # Ensure the list of embeddings matches the text chunks
    if len(all_embeddings) != len(texts):
        raise ValueError("Mismatch between the number of embeddings and text chunks.")

    # Create Chroma DB
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever(
        search_type="mmr",  # Also test "similarity"
        search_kwargs={"k": 8},
    )

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


### FOR RUNNING WITH OPENAI ###

# import os
# os.environ["OPENAI_API_KEY"] = ""
# from langchain_core.prompts import PromptTemplate
# from langchain_openai import OpenAI
# llm = OpenAI(openai_api_key="OPENAP_API_KEY")

### FOR RUNNING WITH HUGGINGFACE ###

from langchain_huggingface import HuggingFaceEndpoint
llmHUGGINGFACE = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

#####################################

##### FOR RUNNING WITH LLAMAFILE #####

# llm = Llamafile()

#####################################

##### FOR RUNNING WITH LLAMACPP #####

# from langchain_community.llms import LlamaCpp
# from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
# from langchain_core.prompts import PromptTemplate    
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# zephr_path = "/Users/adelinemoll/Documents/LLM/zephyr-7b-beta.Q2_K.gguf"
# llmLLAMACPP = LlamaCpp(model_path=zephr_path, verbose=True, n_ctx=4096, callback_manager=callback_manager)

####################################

llm = llmHUGGINGFACE
# llm = llmLLAMACPP

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
            # "Summarize the function provided by the user based on this context from the entire codebase:\n\n{context}",
        ),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
    ]
)
document_chain = create_stuff_documents_chain(llm, prompt)


contents = []
while True:
    try:
        line = input()
    except EOFError:
        break
    contents.append(line)

contents= '\n'.join(contents)

# contents = """
# Model::input_vector_t interpolatedInput(const Model::input_vector_v_t &U, double t, double total_time, bool first_order_hold) { const size_t K = U.size(); const double time_step = total_time / (K - 1); const size_t i = std::min(size_t(t / time_step), K - 2); const Model::input_vector_t u0 = U.at(i); const Model::input_vector_t u1 = first_order_hold ? U.at(i + 1) : u0; const double t_intermediate = std::fmod(t, time_step) / time_step; const Model::input_vector_t u = u0 + (u1 - u0) * t_intermediate; return u; } 
# """


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

userOption = sys.argv[2]
print(userOption)
if userOption == "Ask a question":
    userQuestion = sys.argv[3]
    prompt = QUESTION_PROMPT_TEMPLATE.format(code=contents, question=userQuestion)
    print("Question: " + userQuestion)
else:
    prompt = SUMMARIZE_CODE_PROMPT_TEMPLATE.format(code=contents)
    print("Here is a summary of this code: ")

qa = create_retrieval_chain(retriever_chain, document_chain)
# request = contents + "\n" + SUMMARIZE_CODE_PROMPT
result = qa.invoke({"input": prompt})
print(result['answer'])

