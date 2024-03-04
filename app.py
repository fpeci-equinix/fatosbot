#import required libraries
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import ConfluenceLoader, Docx2txtLoader, TextLoader, PyPDFLoader, ConfluenceLoader, UnstructuredPowerPointLoader, UnstructuredURLLoader, SeleniumURLLoader, WebBaseLoader, UnstructuredExcelLoader,JSONLoader, OutlookMessageLoader, GitLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA

from langchain.prompts.chat import (ChatPromptTemplate,
 SystemMessagePromptTemplate,
 HumanMessagePromptTemplate)
from langchain.llms import LlamaCpp
from langchain.retrievers import BM25Retriever,EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from cohere import Client
#
import chainlit as cl

from getpass import getpass
#
import os
from configparser import ConfigParser
env_config = ConfigParser()

# Retrieve the cohere api key from the environmental variables
def read_config(parser: ConfigParser, location: str) -> None:
 assert parser.read(location), f"Could not read config {location}"
#
CONFIG_FILE = os.path.join(".", ".env")
read_config(env_config, CONFIG_FILE)
api_key = env_config.get("cohere", "api_key").strip()
os.environ["COHERE_API_KEY"] = api_key

#text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)

system_template = """Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

Begin!
 - - - - - - - - 
{summaries}"""

messages = [SystemMessagePromptTemplate.from_template(system_template),HumanMessagePromptTemplate.from_template("{question}"),]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

#Decorator to react to the user websocket connection event.
@cl.on_chat_start
async def init():
#  files = None
#  #Wait for the user to upload a PDF file
#  while files is None:
#     files = await cl.AskFileMessage(
#     content="Please upload a PDF file to begin!",
#     accept=["application/pdf"],
#     max_size_mb=20,
#     timeout=180,
#     ).send()
#     file = files[0]
 msg = cl.Message(content=f"Processing …")
 await msg.send()
 
 # Read the PDF file
#  pdf_stream = BytesIO(file.content)
#  pdf = PyPDF2.PdfReader(pdf_stream)
#  pdf_text = ""
#  for page in pdf.pages:
#     pdf_text += page.extract_text()
 # Split the text into chunks
 #texts = text_splitter.split_text(pdf_text)
 
 documents = []
 for file in os.listdir("docs"):
    if file.endswith(".pdf"):
        pdf_path = "./docs/" + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif file.endswith('.docx') or file.endswith('.doc'):
        doc_path = "./docs/" + file
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())        
    elif file.endswith('.pptx'):
        text_path = "./docs/" + file
        loader = UnstructuredPowerPointLoader(text_path)
        documents.extend(loader.load())
    elif file.endswith('.msg'):
        text_path = "./docs/" + file
        import extract_msg
        os.system('python -m extract_msg --out ./docs' + text_path)        
    elif file.endswith('.json'):
        text_path = "./docs/" + file
        loader = JSONLoader(file_path=text_path, jq_schema='.data',text_content=False)
        documents.extend(loader.load())
    elif file.endswith('.xlsx') or file.endswith('.xls'):
        text_path = "./docs/" + file
        loader = UnstructuredExcelLoader(text_path)
        documents.extend(loader.load())
    elif file.endswith('.mp4'):
        video_path = "./docs/" + file
        model = whisper.load_model("base")
        my_clip = mp.VideoFileClip(video_path)
        audio_path="./"+ video_path.strip("mp4")+"mp3"
        my_clip.audio.write_audiofile(audio_path)
        result = model.transcribe(audio_path)
        text_path = audio_path.strip("mp3")+"txt"
        with open(text_path, "w", encoding="utf-8") as txt:
         txt.write(result["text"])
        loader = TextLoader(text_path)
        documents.extend(loader.load())
 configi = {"confluence_url":"https://equinixjira.atlassian.net/wiki",
          "username":'fatos.peci@eu.equinix.com',
          "api_key":'xxxxxx',
          "space_key":"GSRE"
          } 
 
 confluence_url = configi.get("confluence_url",None)
 username = configi.get("username",None)
 api_key = configi.get("api_key",None)
 space_key = configi.get("space_key",None)
 loader = ConfluenceLoader(
    url=confluence_url,
    username = username,
    api_key= api_key
 )

 documents.extend(loader.load(
    space_key=space_key,
    limit=100
    ))
#  documents.extend(loader.load(
#     space_key="GLNA",
#     #include_attachments=True,
#     limit=100
#     ))
#  documents.extend(loader.load(
#     space_key="IC",
#     #include_attachments=True,
#     limit=100
#     ))
 
 loader = GitLoader(
    repo_path="./docs/Network2Code",
    file_filter=lambda file_path: file_path.endswith(".yml"),
)
documents.extend(loader.load())
 loader = DirectoryLoader('./docs/logs', glob="**/*.txt", use_multithreading=True, loader_cls=TextLoader)
 documents.extend(loader.load())

 urls = [
    "https://nmsreports.corp.equinix.com/data/device_inventory.json",
    "https://nmsreports.corp.equinix.com/data/cx_latency_daily.json",
 ]

 headers = {
    "Authorization": "Basic ZnBlY2k6MXFheXhzdzIkRQ==",
    "Content-Type": "application/json"
 }
 for url in urls:
  response = requests.get(url, headers=headers, verify=False)
  data = response.json()
  with open('./docs/'+url.split("/")[-1], 'w') as f:
    json.dump(data, f)
  loader = JSONLoader(file_path='./docs/'+url.split("/")[-1], jq_schema='.data',text_content=False)
  documents.extend(loader.load())
 #text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=10)
 text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
 #text_splitter = TokenTextSplitter(chunk_size=50, chunk_overlap=0)
 documents = text_splitter.split_documents(documents)
 

 # Create metadata for each chunk
 metadatas = [{"source": f"{i}-pl"} for i in range(len(documents))]
 # Create a Chroma vector store
 model_id = "embed/bge-large-en-v1.5"
 embeddings = HuggingFaceBgeEmbeddings(model_name= model_id,
 model_kwargs = {"device":"cpu"})

#  os.environ["OPENAI_API_TYPE"] = "azure"
#  os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
#  os.environ["OPENAI_API_BASE"] = "https://eqix-az-hack-ncentral.openai.azure.com/"
#  os.environ["OPENAI_API_KEY"] = "xxxxxxxxxxx"
#  embeddings = AzureOpenAIEmbeddings(
#     azure_deployment="northcentral_embeddings",
#     openai_api_version="2023-05-15-preview",
# )
 #
 bm25_retriever = BM25Retriever.from_documents(documents)
 bm25_retriever.k=5
 # Store the embeddings in the user session
 cl.user_session.set("embeddings", embeddings)
 url = "http://localhost:6334/"
 docsearch = await cl.make_async(Qdrant.from_documents)(
 documents, embeddings,url=url,
    prefer_grpc=True,collection_name="my_documents",force_recreate=True,
 )
#  record_manager = SQLRecordManager(
#         f"qdrant/my_documents", db_url="sqlite:///record_manager_cache.sql"
#     )
#  record_manager.create_schema()
#  llm = AzureOpenAI(engine="eqix_secure_gpt_35T_16k",
#                    model="gpt-3.5-turbo", 
#                    temperature=0.7,
#                    stream=True,
# )
#  index(
#         documents,
#         record_manager,
#         docsearch,
#         cleanup="full",
#         source_id_key="source",        
#     )

#  llm = AzureChatOpenAI(
#     deployment_name="eqix_secure_gpt_35T_16k",
#     openai_api_version="2023-03-15-preview",
#     temperature=0.7,
# )

 llm = LlamaCpp(streaming=True,
 model_path="model/zephyr-m.gguf",
 max_tokens = 3900,
 temperature=0,
 top_p=1,
 stream=True,
 n_gpu_layers=0,
 verbose=True,
 n_ctx=8192)
 #Hybrid Search
 qdrant_retriever = docsearch.as_retriever(search_kwargs={"k":5})
 ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever,qdrant_retriever],
 weights=[0.5,0.5])
 #Cohere Reranker
 #
 #compressor = CohereRerank(client=Client(api_key=os.getenv("COHERE_API_KEY")),user_agent='langchain')
 #
 compressor = LLMChainExtractor.from_llm(llm)
 compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
 
 
 chain = RetrievalQA.from_chain_type(
 llm = llm,
 chain_type="stuff",
 retriever=compression_retriever,
 return_source_documents=True,
 )
 # Save the metadata and texts in the user session
 cl.user_session.set("metadatas", metadatas)
 cl.user_session.set("documents", documents)
 # Let the user know that the system is ready
 msg.content = f"Everything processed. You can now ask questions!"
 await msg.update()
 #store the chain as long as the user session is active
 cl.user_session.set("chain", chain)

 @cl.on_message
 async def process_response(res:cl.Message):
    # retrieve the retrieval chain initialized for the current session
    chain = cl.user_session.get("chain") 
    # Chinlit callback handler
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])
    cb.answer_reached = True
    print("in retrieval QA")
    #res.content to extract the content from chainlit.message.Message
    print(f"res : {res.content}")
    response = await chain.acall(res.content, callbacks=[cb])
    print(f"response: {response}")
    answer = response["result"]
    sources = response["source_documents"]
    source_elements = []

    # # Get the metadata and texts from the user session
    # metadatas = cl.user_session.get("metadatas")
    # all_sources = [m["source"] for m in metadatas]
    # documents = cl.user_session.get("documents")

    if sources:
        found_sources = []

        # Add the sources to the message
        for source in sources:
            print(source.metadata)
            try :
                source_name = source.metadata["source"]
            except :
                source_name = ""
            # Get the index of the source
            text = source.page_content
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer, elements=source_elements).send()
