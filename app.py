# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, render_template, request
import openai
from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
from pdfminer.high_level import extract_text
import os
import concurrent.futures
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from rich import print
from ast import literal_eval

openai.api_key = os.environ['OPEN_AI_KEY']

current_dir = os.getcwd()

data_path = "parsed_pdf_docs_with_embeddings -backup-withoutCorrectivedoc.csv"
df = pd.read_csv(os.path.join(current_dir,'parsed_pdf_docs_with_embeddings -backup-withoutCorrectivedoc.csv'))
df["embeddings"] = df.embeddings.apply(literal_eval).apply(np.array)
embeddings_model = "text-embedding-3-large"


def get_embeddings(text):
    embeddings = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        encoding_format="float"
    )
    return embeddings.data[0].embedding


system_prompt = '''
    You will be provided with an input prompt and content as context that can be used to reply to the prompt.

    You will do the following things:
    1. First, you will internally assess whether the content provided is relevant to reply to the input prompt. 
    2. If the content is relevant, use elements found in the content to craft a reply to the input prompt.
    3. If the content is not relevant, do not use your own knowledge base, and say that you don't know the answer. 
    4. Answer or reply should not exceed 50 words approximately and should be in bullet format. 
    5. Please list the source links at the end
'''
model = "gpt-4o-mini"


def search_content(df, input_text, top_k):
    embedded_value = get_embeddings(input_text)
    df["similarity"] = df.embeddings.apply(
        lambda x: cosine_similarity(np.array(x).reshape(1, -1), np.array(embedded_value).reshape(1, -1)))
    res = df.sort_values('similarity', ascending=False).head(top_k)
    return res


def get_similarity(row):
    similarity_score = row['similarity']
    if isinstance(similarity_score, np.ndarray):
        similarity_score = similarity_score[0][0]
    return similarity_score


def generate_output(input_prompt, similar_content, threshold=0.5):
    content = similar_content.iloc[0]['content']

    # Adding more matching content if the similarity is above threshold
    if len(similar_content) > 1:
        for i, row in similar_content.iterrows():
            similarity_score = get_similarity(row)
            if similarity_score > threshold:
                content += row['content']

    print('########## content start ############')
    print(content)
    print('########## content end ############')
    prompt = "INPUT PROMPT:\n" + input_prompt + "\n-------\nCONTENT:\n" + content
    completion = openai.chat.completions.create(
        model=model,
        temperature=0.5,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return completion.choices[0].message.content

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    question = None
    response = None
    combined = None
    if request.method == 'POST':
        question = request.form['question']
        combined = request.form['text'] + '\nQuestion: '+ question
        matching_content = search_content(df, question, 1)

        for i, match in matching_content.iterrows():
            print("Similarity:", get_similarity(match))
            # print(match)
        response = generate_output(question, matching_content)
        combined = combined + '\nResponse: '+ response
    return render_template('index.html', response=combined)

if __name__ == '__main__':
    app.run()
