import os
import openai
import pandas as pd
import nltk
import tiktoken
nltk.download('punkt')

# STORE THE API KEY IN AN ENVIRONMENT VARIABLE
API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
openai.api_key = API_KEY
model_name = "text-embedding-ada-002"
input_folder = "files"
output_file = "processed/embedded_files.csv"

encoding = tiktoken.encoding_for_model(model_name)
df = pd.read_csv("chunks/output.csv")


def num_tokens_from_string(string: str) -> int:
    return len(encoding.encode(string))


def generate_embeddings(df, output_file):
    """
    Generate embeddings for text chunks in a DataFrame and save the results to a CSV file
    """
    data = []
    model_name = "text-embedding-ada-002"
    max_tokens = 4096

    for index, row in df.iterrows():
        text = row["text"]
        num_tokens = num_tokens_from_string(text)

        if num_tokens > max_tokens:
            print(f"Warning: Skipping chunk at index {index} because it exceeds max token limit.")
            continue

        try:
            response = openai.Embedding.create(input=text, model=model_name)
            embeddings = response["data"][0]["embedding"]

            data.append({
                "index": index,
                "directory": row["directory"],
                "filename": row["filename"],
                "filetype": row["filetype"],
                "text": text,
                "n_tokens": num_tokens,
                "embeddings": embeddings
            })

            print(f"Processed chunk at index {index}.")
        except Exception as e:
            print(f"Error processing chunk at index {index}: {e}")

    result_df = pd.DataFrame(data)
    result_df.to_csv(output_file, index=False)
    print(f"Embeddings saved to {output_file}.")


generate_embeddings(df, output_file="processed/embedded_files.csv")
