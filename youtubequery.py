import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import YoutubeLoader
from langchain.llms import HuggingFaceHub
from langchain.docstore.document import Document

class YoutubeQuery:
    def __init__(self, huggingface_api_key = None) -> None:
        os.environ["HUGGINGFACE_API_TOKEN"] = huggingface_api_key
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.1",
            model_kwargs={"temperature": 0.0, "max_length": 512}
        )
        self.chain = None
        self.db = None

    def ask(self, question: str) -> str:
        if self.chain is None:
            response = "Please, add a video."
        else:
            docs = self.db.get_relevant_documents(question)
            response = self.chain.run(input_documents=docs, question=question)
        return response

    def ingest(self, url: str) -> str:
        documents = YoutubeLoader.from_youtube_url(url, add_video_info=False).load()
        splitted_documents = self.text_splitter.split_documents(documents)
        self.db = Chroma.from_documents(splitted_documents, self.embeddings).as_retriever()
        self.chain = load_qa_chain(self.llm, chain_type="stuff")
        return "Success"

    def forget(self) -> None:
        self.db = None
        self.chain = None