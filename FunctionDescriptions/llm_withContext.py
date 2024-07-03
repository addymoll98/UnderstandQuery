import Function_Analyzer_Helper

def describe_functions(sorted):
    import os
    from langchain_community.document_loaders.generic import GenericLoader
    from langchain_community.document_loaders.parsers import LanguageParser
    from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.llms import HuggingFaceHub
    from langchain_community.llms.llamafile import Llamafile

    # Construct the full path to your repository
    repo_path = "/Users/adelinemoll/Documents/LLM/LangChain/SCpp"

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


    ### FOR RUNNING WITH HUGGINGFACE ###

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 512,
            "top_k": 30,
            "temperature": 0.1,
            "repetition_penalty": 1.03,
        },
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

    # zephr_path = "/Users/adelinemoll/Documents/LLM/LangChain/zephyr-7b-beta.Q2_K.gguf"
    # llm = LlamaCpp(model_path=zephr_path, verbose=True, n_ctx=4096, callback_manager=callback_manager)

    ####################################

    prompt = ChatPromptTemplate.from_messages(
        [
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
            (
                "user",
                "Given the provided C++ function, generate a search query to look up portions of the codebase that are related to this function.",
            ),
        ]
    )

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarize the function provided by the user based on this context from the entire codebase:\n\n{context}",
            ),
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm, prompt)

    SUMMARIZE_CODE_PROMPT = """
    Generate a concise summary of the provided C++ function. 
    Use the provided context from other parts of the codebase if it helps summarize the function. 
    Keep the summary to 5 sentences or less. 

    Instructions: 
    - Provide one paragraph.
    - Use 5 sentences or less.
    - Describe dependencies, important functions and classes, and relevant information from comments.

    Restrictions:
    - Do not say "Summary" or "Output".
    - Do not engage in any conversation.
    - Only describe the provided C++ function.
    - Start directly with the summary, with no precursors.
    """

    qa = create_retrieval_chain(retriever_chain, document_chain)

    descriptions_file = open("descriptions.txt", "a")
    for file in sorted:
        for function in file.functions:
            if function.fullname.startswith("scpp"):
                function_text = function.content
                request = function_text + "\n" + SUMMARIZE_CODE_PROMPT
                print(f"Generating summary for {function.fullname}")
                result = qa.invoke({"input": request})
                descriptions_file.write(f"{function.fullname}\n\n")
                descriptions_file.write(f"{result}\n\n")
    descriptions_file.close()






