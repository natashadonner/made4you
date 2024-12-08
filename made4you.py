import nltk

nltk.download('punkt')
import pandas as pd
import openai
import numpy as np
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
import gradio
import tiktoken
import os

# STORE THE API KEY IN AN ENVIRONMENT VARIABLE
openai.api_key = os.getenv("OPENAI_API_KEY", "").strip()
# ADD YOUR NAME HERE
name_of_user = "Natasha's"
MAX_TOKENS = 4096
TOKEN_MARGIN = 500
count = 0
chat_history = []


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


tokenizer = nltk.tokenize.word_tokenize


def count_tokens(messages, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    total_tokens = 0
    for message in messages:
        total_tokens += len(encoding.encode(message['content']))
    return total_tokens


def combine_question(df, model="gpt-4", q="What is a spotlist?", chat_history="", max_len=100, debug=False, max_tokens=1000, stop_sequence=None):
    try:
        prompt = f"""Create a SINGLE standalone question. The question should be based on the New question plus the Chat history. 
        If the New question can stand on its own you should return the New question. 
        New question: "{q}", Chat history: "{chat_history}"."""

        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "Combine the new question with chat history to form a standalone question."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            stop=stop_sequence
        )

        return response["choices"][0]["message"]["content"].strip()

    except Exception as e:
        if debug:
            print(f"Error encountered: {e}")
        return ""


def create_context(question, df, max_len=1500, size="ada"):
    combined_question = combine_question(
        df=df,
        model="gpt-4",
        q=question,
        chat_history=chat_history
    )

    print(f"Combined Question: {combined_question}\n")

    q_embeddings = openai.Embedding.create(input=combined_question, engine='text-embedding-ada-002')['data'][0]['embedding']
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    df = df.sort_values('distances', ascending=True)

    returns = []
    cur_len = 0

    for i, row in df.iterrows():
        text = row['text']
        text_len = num_tokens_from_string(text, "p50k_base")

        if cur_len + text_len > max_len:
            break
        returns.append(text)
        cur_len += text_len

    combined_context = "\n\n".join(returns)
    print(f"Total tokens used: {cur_len}")
    return combined_context


def chatbot(question, clear_memory):

    global chat_history
    global count

    df = pd.read_csv('processed/embedded_files.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

    if clear_memory or count == 0:
        count = 0
        chat_history = [
            {"role": "system", "content": """You are an assistant whose role is to provide accurate and helpful answers to any questions asked.
            Answer the question below based on the provided context.
            ### Instructions:
            1. Thoroughly analyze the context and extract only the relevant information needed to answer the question. If any part of the context contains unnecessary information, exclude it from your answer. Ensure your analysis is both comprehensive and focused on relevance.
            2. Include all applicable and critical details from the context in your response, but do not add unrelated or redundant information.
            3. After addressing the question, you may add any additional insights or useful details from the context if they directly enhance the response. This additional content should remain concise and relevant.
            4. If the question cannot be answered based on the context provided, respond with: "This question cannot be answered based on the context provided."
            5. Answer in the language the question is asked in.
            Base your response solely on the given context. Do not introduce external information."""}
        ]

    chat_history.append({"role": "user", "content": question})
    context = create_context(question, df)
    print("Context: " + context)

    total_tokens = count_tokens(chat_history)
    if total_tokens > (MAX_TOKENS - TOKEN_MARGIN):
        while total_tokens > (MAX_TOKENS - TOKEN_MARGIN) and len(chat_history) > 1:
            chat_history.pop(1)  # Remove the oldest message (except the initial system message)
            total_tokens = count_tokens(chat_history)

    if context:
        chat_history.append({"role": "system", "content": f"Context:\n{context}"})

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=chat_history,
        max_tokens=1000,
        temperature=0.7
    )

    assistant_response = response["choices"][0]["message"]["content"].strip()
    chat_history.append({"role": "assistant", "content": assistant_response})
    print(chat_history)

    return assistant_response


iface = gradio.Interface(
    fn=chatbot,
    inputs=[
        gradio.Textbox(lines=5, label="Your Question"),
        gradio.Checkbox(label="Clear Memory"),
    ],
    outputs="text",
    title=f"{name_of_user} Chatbot",
    description="Ask questions and get accurate answers.",
    theme='bethecloud/storj_theme'
)
iface.launch(share=True)
