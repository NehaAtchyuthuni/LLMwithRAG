import os
import pandas as pd
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

openai.api_key = os.environ['OPEN_AI_KEY']
embeddings_model = "text-embedding-3-large"
path = "C:/Users/yugan/Documents/Science Fair/Sampledocs/"
data_path = "C:/Users/yugan/Documents/Science Fair/parsed_pdf_docs_with_embeddings.csv"
content = []
chunks = []

def get_embeddings(text):
    embeddings = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        encoding_format="float"
    )
    return embeddings.data[0].embedding

for filename in os.listdir(path):
    filepath = os.path.join(path, filename)
    if os.path.isfile(filepath):
        # Process the file here
        loader = PyPDFLoader(filepath)
        doc = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
           chunk_size=1500,
           chunk_overlap=200
        )
        chunks.extend(text_splitter.split_documents(doc))

# Create an array to store the page content
for i, row in enumerate(chunks):
  content.append(row.page_content)

# Create a data frame to store the content and the corresponding embeddings
df = pd.DataFrame(content, columns=['content'])
df['embeddings'] = df['content'].apply(lambda x: get_embeddings(x))
# We'll save to a csv file here
df.to_csv(data_path, index=False)



