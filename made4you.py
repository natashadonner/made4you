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


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


tokenizer = nltk.tokenize.word_tokenize


def create_context(question, df, max_len=1500, size="ada"):
    """
    Skapa en kontext genom att välja de mest relevanta text-chunks baserat på embeddings.
    """

    # Create embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    # Sort chuns by distance to question
    df = df.sort_values('distances', ascending=True)

    returns = []
    cur_len = 0

    # Iterate over the chunks and add them to the context until we reach the max length
    for i, row in df.iterrows():
        text = row['text']
        text_len = num_tokens_from_string(text, "p50k_base")  # Count the number of tokens in the text

        if cur_len + text_len > max_len:
            break
        returns.append(text)
        cur_len += text_len

    combined_context = "\n\n".join(returns)

    print(f"Total tokens used: {cur_len}")
    return combined_context


def answer_question(df, model="gpt-3.5-turbo-instruct", question="What is a spotlist?", context="", size="ada", debug=False,
                    max_tokens=1500, stop_sequence=None):
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    prompt = f"""Answer the question below based on the provided context.
    ### Question:
    {question}
    ### Context:
    {context}
    ### Instructions:
    1. Thoroughly analyze the context and extract only the relevant information needed to answer the question. If any part of the context contains unnecessary information, exclude it from your answer. Ensure your analysis is both comprehensive and focused on relevance.
    2. Include all applicable and critical details from the context in your response, but do not add unrelated or redundant information.
    3. After addressing the question, you may add any additional insights or useful details from the context if they directly enhance the response. This additional content should remain concise and relevant.
    4. If the question cannot be answered based on the context provided, respond with: "This question cannot be answered based on the context provided."
    Base your response solely on the given context. Do not introduce external information.
    """

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=prompt,
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""


context_with_memory = ""
global chat_history
chat_history = ""
memory = ""
global count
count = 0


def combine_question(df, model="gpt-3.5-turbo", q="What is a spotlist?", chat_history="", max_len=100, debug=False, max_tokens=1000, stop_sequence=None):
    """
    Combine a new question with chat history to create a standalone question.

    Parameters:
        - df: Placeholder for unused parameter (remove if unnecessary).
        - model: The OpenAI model to use (default is GPT-3.5-turbo).
        - q: The new question.
        - chat_history: Previous chat history.
        - max_len: Maximum length for the output question.
        - debug: If True, prints debugging information.
        - max_tokens: Maximum tokens for the model to return.
        - stop_sequence: Token(s) where the model will stop generating.

    Returns:
        - A standalone, combined question.
    """
    try:
        # Construct the system prompt
        prompt = f"""Create a SINGLE standalone question. The question should be based on the New question plus the Chat history. 
        If the New question can stand on its own you should return the New question. 
        New question: "{q}", Chat history: "{chat_history}"."""

        # Debug print for prompt
        if debug:
            print("Prompt:", prompt)

        # Call OpenAI's Completion API
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "Combine the new question with chat history to form a standalone question."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            stop=stop_sequence
        )

        # Extract and return the response
        return response["choices"][0]["message"]["content"].strip()

    except Exception as e:
        if debug:
            print(f"Error encountered: {e}")
        return ""


def combine_question1(df, model="gpt-3.5-turbo-instruct", q="What is a spotlist?", chat_history="", max_len=100, size="ada",
                     debug=False, max_tokens=1000, stop_sequence=None):
    prompt = f"""Create a SINGLE standalone question. The question should be based on the New question plus the Chat history. 
    If the New question can stand on its own you should return the New question. New question: \"{q}\", Chat history: \"{chat_history}\".""",

    """
    #messages = [{"role": "system", "content": "Combine the following question q with the information from memory to create a new specific question.Follow these rules: Analyze q and memory, considering that memory contains the previous question. Determine if q is a follow-up question to the memory. For example: Memory: What is a Campaign? Question: Can I add a tag to it? System: Can I add a tag to a Campaign? If it is a follow up question, replace anaphoric words in q to specifically target the subject in memory. If it's not a follow up question, simply respond with only q(Don't change q at all) and don't combine q and memory.q: \"{q}\", Memory: \"{memory}\"."}]
    messages = [{"role": "system", "content": "Create a SINGLE standalone question. The question should be based on the New question plus the Chat history. New question: \"{q}\", Chat history: \"{chat_history}\"."}]

    
    messages.append({"role": "user", "content": q})
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply
    """
    try:
        # Create a completions using the question and context

        response = openai.Completion.create(
            prompt=prompt,
            temperature=0.7,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""


def chatbot(question, clearMemory):

    df = pd.read_csv('processed/embedded_files.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

    global chat_history
    global count

    if clearMemory:
        count = 0
        chat_history = ""

    if count > 0:
        print("Question: " + question + "chat_history: " + chat_history + "\n")
        question = combine_question(df, q=question, chat_history=chat_history)
        print("Combined questions: " + question + "\n")
        chat_history = question
    else:
        print("Question: " + question + "\n")

    count = count + 1
    context = create_context(question, df)
    responses = answer_question(df, question=question, context=context)
    print("Context created: " + context + "\n")

    if count == 1:
        chat_history = chat_history + (str(question))  # Add current question to the deque

    chat_history = "User: " + chat_history + "System: " + (str(responses))
    return responses


iface = gradio.Interface(
    fn=chatbot,
    inputs=[
        gradio.Textbox(lines=5),
        gradio.Checkbox(label="clearMemory")],
    outputs="text",
    title=f"{name_of_user} Chatbot",
    description="Select your user type and ask a question",
    flagging_options=["Correct", "Wrong"],
    # theme='freddyaboulton/dracula_revamped'
    theme='bethecloud/storj_theme'

)
iface.launch(share=True)
