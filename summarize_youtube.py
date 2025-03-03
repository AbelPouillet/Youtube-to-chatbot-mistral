from langchain.document_loaders import YoutubeLoader
from langchain.llms import HuggingFaceHub
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
import os

# Set your HuggingFace API token
os.environ["HUGGINGFACE_API_TOKEN"] = "your-huggingface-token-here"

# Initialize Mistral model
llm = HuggingFaceHub(
	repo_id="mistralai/Mistral-7B-Instruct-v0.1",
	model_kwargs={"temperature": 0.0, "max_length": 512}
)

# Load and process YouTube content
loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=O5xeyoRL95U&ab_channel=LexFridman", add_video_info=False)
docs = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)
print(split_docs)

# Create and run summarization chain
chain = load_summarize_chain(llm, chain_type="map_reduce")
chain.run(split_docs)