from langchain_community.llms import LlamaCpp

model_path = "/Users/adelinemoll/Documents/LLM/LangChain/zephyr-7b-beta.Q2_K.gguf"
model = LlamaCpp(model_path=model_path)

model.invoke("What is the capitol of AZ?")