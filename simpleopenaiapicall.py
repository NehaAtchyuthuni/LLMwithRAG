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
openai.api_key = "mykey"

# Open ai model
model = "gpt-4o-mini"

# Use your own knowledge to reply
system_prompt = '''
    You will be provided with an input prompt.
    You will do the following things:
    Answer or response should not exceed 100 words. 
    Please list the source links at the end
'''

def generate_output(input_prompt):
    prompt = "INPUT PROMPT:\n"+ input_prompt
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
root.title("Neha's AI Counselor using GPT 4.0 mini")

# Create the chatbot text area
text_area = tk.Text(root, bg="beige", width=100, height=30)
text_area.tag_configure("bold", font=("Arial", 10, "bold"))
text_area.pack()

# Create the user input field
input_field = tk.Entry(root, width=50)
input_field.pack()

# Watermark text for user input field
background_text = "Enter your question here"
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

# Create the Submit button
send_button = tk.Button(root, text="Submit", command=lambda: send_message())
send_button.pack()
response= ""
def send_message():

# Get the user input
  user_input = input_field.get()

# Clear the input field
  input_field.delete(0, tk.END)

# Generate a response from the chatbot
  if user_input.lower() == "quit" or user_input.lower() == "exit" or user_input.lower() == "end"  :
      exit()
  response = generate_output(user_input)

# Display the response in the chatbot text area
  text_area.insert(tk.END, user_input+ "\n", "bold")
  text_area.insert(tk.END, response+ "\n\n")
root.mainloop()
