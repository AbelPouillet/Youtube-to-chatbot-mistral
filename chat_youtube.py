from langchain.document_loaders import YoutubeLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
import os

# Set your HuggingFace API token
os.environ["HUGGINGFACE_API_TOKEN"] = "your-huggingface-token-here"

# Load YouTube video
loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=O5xeyoRL95U&ab_channel=LexFridman", add_video_info=False)
documents = loader.load()

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()

# Set up Mistral model and QA chain
llm = HuggingFaceHub(
	repo_id="mistralai/Mistral-7B-Instruct-v0.1",
	model_kwargs={"temperature": 0.0, "max_length": 512}
)
chain = load_qa_chain(llm, chain_type="stuff")

# Query the model
query = "What did the president say about Ketanji Brown Jackson"
docs = retriever.get_relevant_documents(query)
output = chain.run(input_documents=docs, question=query)
print(output)