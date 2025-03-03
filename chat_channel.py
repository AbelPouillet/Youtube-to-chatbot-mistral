from langchain.document_loaders import YoutubeLoader
import scrapetube
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
import os

# Set your HuggingFace API token
os.environ["HUGGINGFACE_API_TOKEN"] = "your-huggingface-token-here"

channel_id = "UC03sxjXYe4mSLqr5etxOXGA"
videos = scrapetube.get_channel(channel_id)
pages = []
for v in videos:
    videoId = v['videoId']
    url = "https://www.youtube.com/watch?v="+videoId
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
    pages += loader.load_and_split()    

# Use HuggingFace embeddings (all-MiniLM-L6-v2 is a good default choice)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
docsearch = Chroma.from_documents(pages, embeddings).as_retriever()    

query = "What is Vadoo ?"
docs = docsearch.get_relevant_documents(query)

# Use Mistral model through HuggingFace
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.0, "max_length": 512}
)
chain = load_qa_chain(llm, chain_type="stuff")
output = chain.run(input_documents=docs, question=query)