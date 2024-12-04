import os
import openai
import pandas as pd
import re
import nltk
from docx import Document
from pdfminer.high_level import extract_text
from langchain_text_splitters import RecursiveCharacterTextSplitter

nltk.download('punkt')
nltk.download('punkt_tab')
API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
openai.api_key = API_KEY
model_name = "text-embedding-ada-002"
input_folder = "files"

data = []

# Tokenization function using NLTK
tokenizer = nltk.tokenize.word_tokenize


# Function to clean the text
def clean_text(text):
    # Remove extra line breaks and spaces
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = text.strip()  # Remove leading/trailing spaces
    return text


def extract_docx(file_path):
    doc = Document(file_path)
    content = "\n".join([para.text for para in doc.paragraphs])
    return content


def extract_pdf(file_path):
    try:
        content = extract_text(file_path)
        if content.strip():  # If there's actual text
            return content
        else:
            print(f"Warning: No text found in {file_path}")
            return ""
    except Exception as e:
        print(f"Error extracting PDF {file_path}: {e}")
        return ""


def divide_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=True,
    )
    texts = text_splitter.split_text(text)
    return texts


# Generate embeddings for each text file

# for i, file_name in enumerate(files):
count = 0

for file_name in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file_name)

    if file_name.endswith(".docx"):
        print(f"Scraping DOCX file: {file_name}")
        text = extract_docx(file_path)

    elif file_name.endswith(".pdf"):
        print(f"Scraping PDF file: {file_name}")
        text = extract_pdf(file_path)

    # Clean the text
    text = clean_text(text)

    # Split the text into chunks based on token limit
    text_chunks = divide_chunks(text)

    # Process each chunk separately
    for chunk_num, chunk in enumerate(text_chunks):
        # Generate the embeddings for the chunk
        response = openai.Embedding.create(input=chunk, model=model_name)
        embeddings = response["data"][0]["embedding"]

        # Tokenize the chunk and get the number of tokens
        tokens = tokenizer(chunk)
        num_tokens = len(tokens)

        # Append the data to the list
        data.append((f"{count}_{chunk_num}", chunk, num_tokens, embeddings))

        # Print progress
        print(f"Processed file {count + 1}/{len(input_folder)} - chunk {chunk_num + 1}/{len(text_chunks)}")

    # Convert the data to a DataFrame
    df = pd.DataFrame(data, columns=["index", "text", "n_tokens", "embeddings"])

    # Save the cleaned and embedded data to CSV
    df.to_csv(f"processed/embedded_files.csv", index=False)

    count = count + 1
