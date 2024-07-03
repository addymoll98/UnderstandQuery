import re
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader  # Import PDF loader and directory loader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_text_by_hierarchical_paragraphs(text):
    # Regular expression to match hierarchical paragraphs up to three levels like 1.1, 1.1.1, etc.
    paragraph_pattern = re.compile(r'^\d+\.\d+\.\d+\. |\d+\.\d+\. ')
    lines = text.split("\n")
    chunks = []
    current_chunk = ""

    for line in lines:
        if paragraph_pattern.match(line.strip()):
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
        current_chunk += line + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# def split_text_by_rules(text):
#     # Regular expression to match the rule identifiers
#     rule_pattern = re.compile(r'\d+\.\d+ [A-Z]{3}\d{2}-[A-Z]+')
#     lines = text.split("\n")
#     chunks = []
#     current_chunk = ""

#     for line in lines:
#         if rule_pattern.match(line.strip()):
#             if current_chunk:
#                 chunks.append(current_chunk.strip())
#                 current_chunk = ""
#         current_chunk += line + "\n"
#     if current_chunk:
#         chunks.append(current_chunk.strip())
#     return chunks

# Paths to your PDFs
pdf_path_1 = "/Users/adelinemoll/Documents/LLM/UnderstandAIAPI/DocumentAnalysis/dafman91-119.pdf"
# pdf_path_2 = "/Users/adelinemoll/Documents/LLM/UnderstandAIAPI/DocumentAnalysis/cert-cpp/cert-cpp.pdf"

# Extract text
dafman_text = extract_text_from_pdf(pdf_path_1)
# certCpp_text = extract_text_from_pdf(pdf_path_2)

# Split text
dafman_chunks = split_text_by_hierarchical_paragraphs(dafman_text)
# certCpp_chunks = split_text_by_rules(certCpp_text)

# Get the current working directory
current_directory = os.getcwd()

# Construct the full path to your repository
repo_path = os.path.join(current_directory, "cert-cpp")  # Update to your PDF folder path

# Debug: Check if repo_path exists
if not os.path.exists(repo_path):
    print(f"Repo path does not exist: {repo_path}")
else:
    print(f"Repo path exists: {repo_path}")

# Load documents
loader = DirectoryLoader(repo_path, loader_cls=PyMuPDFLoader)  # Load PDFs from the directory
documents = loader.load()

print(f"Loaded {len(documents)} documents.")
# Split documents
pdf_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1)  # General text splitter
texts = pdf_splitter.split_documents(documents)

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


# Convert chunks to Document objects
# from langchain.docstore.document import Document
# documents = [Document(page_content=chunk) for chunk in certCpp_chunks]

# # Embed CERT C++ Documents
# from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# # all_embeddings = embeddings.embed_documents(certCpp_chunks)

# # Create Chroma database from documents
# from langchain_community.vectorstores import Chroma
# db = Chroma.from_documents(documents, embeddings)

retriever = db.as_retriever(
    search_type="mmr",  # Also test "similarity"
    search_kwargs={"k": 8},
)

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

### FOR RUNNING WITH HUGGINGFACE ###
import os
# from langchain_community.llms import HuggingFaceHub
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
# llm = HuggingFaceHub(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
#     model_kwargs={
#         "max_new_tokens": 512,
#         "top_k": 10,
#         "temperature": 0.1,
#         "repetition_penalty": 1.03,
#     },
#     verbose = False
# )

from langchain_huggingface import HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

# from transformers import TextGenerationPipeline, LlamaTokenizer
# # Initialize the pipeline with your model parameters
# llm = TextGenerationPipeline(
#     model="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
#     tokenizer=LlamaTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta"),
#     device=0,  # specify GPU device if available, e.g., 0 for GPU 0
#     max_new_tokens=512,
#     top_k=10,
#     temperature=0.1,
#     repetition_penalty=1.03
# )

#####################################

##### FOR RUNNING WITH LLAMAFILE #####

# llm = Llamafile()

#####################################

##### FOR RUNNING WITH LLAMACPP #####

# from langchain_community.llms import LlamaCpp
# from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
# from langchain_core.prompts import PromptTemplate    
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# zephr_path = "/Users/adelinemoll/Documents/LLM/LangChain/zephyr-7b-beta.Q2_K.gguf"
# llm = LlamaCpp(model_path=zephr_path, verbose=True, n_ctx=4096, callback_manager=callback_manager)

####################################

prompt = ChatPromptTemplate.from_messages(
    [
        ("user", "{input}"),
        (
            "user",
            "Generate a search query based on the users input.",
        ),
    ]
)

retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant helping a user sort through document requirements. The user has a list of general coding requirements they must follow for a critical software development project. The user's goal is to determine if any of these DAFMAN requirements can be checked, at least in part, using the CERT C++ standards available in static analysis tools. Given the context provided, inform the user if any CERT C++ standards help confirm that the provided DAFMAN requirement has been met in the code. \n\n{context}. ",
            # "Respond to the user question only with one word, \"Yes\" or \"No\"",
        ),
        ("user", "{input}"),
    ]
)
document_chain = create_stuff_documents_chain(llm, prompt)

qa = create_retrieval_chain(retriever_chain, document_chain)

response_file = open("responses.txt", "a")
for i in range(100, 200):
    print("Generating a response")
    requirement = dafman_chunks[i]
    question = f"{dafman_chunks[i]}\n\nQuestion:\nCan any CERT C++ standards help confirm that this requirement has been met?"
    result = qa.invoke({"input": question})
    response_file.write(f"DAFMAN Requirement: \n{dafman_chunks[i]}\n\n")
    # response_file.write(f"INPUT: {request_text}\n")
    # response_file.write(f"RESPONSE: {result}\n\n")
    # response_file.write(f"RESPONSE: {result['answer']}\n\n")
    response_file.write(f"RESPONSE: {result['answer']}\n\n")

response_file.close()