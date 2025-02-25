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
import tkinter as tk


# Set your OpenAI API key
openai.api_key = "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"

data_path = "C:/Users/yugan/Documents/Science Fair/parsed_pdf_docs_with_embeddings.csv"
df = pd.read_csv(data_path)
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
    3. If the content is not relevant, use your knowledge base to answer. 
    4. Answer or reply should not exceed 50 words approximately and should be in bullet format. 
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
    #print(content)
    print('generate output')
    # Adding more matching content if the similarity is above threshold
    if len(similar_content) > 1:
        for i, row in similar_content.iterrows():
            similarity_score = get_similarity(row)
            if similarity_score > threshold:
                content += row['content']

    prompt = "INPUT PROMPT:\n"+ input_prompt +"\n-------\nCONTENT:\n"+content
    print(prompt)
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

root = tk.Tk()
root.title("Neha's AI Counselor with Enhanced RAG ")

# Create the chatbot's text area
text_area = tk.Text(root, bg="light green", width=100, height=30)
text_area.tag_configure("bold", font=("Arial", 10, "bold"))
text_area.pack()

# Create the user's input field
input_field = tk.Entry(root, width=50)
input_field.pack()

background_text = "How may I help you ?"
input_field.insert(0, background_text)
input_field.config(fg='gray')

def on_entry_click(event):
    if input_field.get() == background_text:
        input_field.delete(0, tk.END)
        input_field.config(fg='black')

def on_entry_focusout(event):
    if not input_field.get():
        input_field.insert(0, background_text)
        input_field.config(fg='gray')

input_field.bind("<FocusIn>", on_entry_click)
input_field.bind("<FocusOut>", on_entry_focusout)

# Create the send button
send_button = tk.Button(root, text="Submit", command=lambda: send_message())
send_button.pack()

response= ""
def send_message():
  # Get the user's input
  user_input = input_field.get()

  # Clear the input field
  input_field.delete(0, tk.END)

  # Generate a response from the chatbot
  if user_input.lower() == "quit" or user_input.lower() == "exit" or user_input.lower() == "end":
      exit()

  matching_content = search_content(df, user_input, 1)

  for i, match in matching_content.iterrows():
      print("Similarity:")
      print(get_similarity(match))
  response = generate_output(user_input, matching_content)

  # Display the response in the chatbot's text area
  text_area.insert(tk.END, user_input+ "\n", "bold")
  text_area.insert(tk.END, response+ "\n\n")

root.mainloop()