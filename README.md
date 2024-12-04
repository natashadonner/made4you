## Local Chatbot "Made4You" with OpenAI API

### Purpose

This project enables you to create a personalized chatbot running locally on your computer with your own files.
By managing your own instance, you can ensure greater control over your data and use your own OpenAI API key for security. 
Learn more about data security with OpenAI [here](https://www.openai.com/security).

### Getting Started with Your Local Chatbot

#### Step 1: Clone the Repository

Begin by cloning the repository to your local machine using SSH or HTTPS. Choose a suitable directory on your computer where you would like to set up the project. Here's how you can clone using SSH:

```bash 
git clone git@github.com:yourusername/yourprojectname.git
```

#### Step 2: Set Up Your OpenAI API Key
- If you already have an OpenAI API key, you may skip this step. If not, follow these instructions to obtain one:
- Visit OpenAI's website and sign up for an account if you do not already have one.
- Once logged in, navigate to the API section and generate a new API key.
- To ensure the security of your key, add it to your environment variables on your computer rather than directly embedding it in your code. This prevents accidental exposure of your key, especially if you plan to push your code to a public repository.

#### Step 3: Set Up Environment Variable
For detailed instructions on how to set an environment variable for your OpenAI API key, refer to this comprehensive guide
[here](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety).


#### Step 4: Open project
Open your project in your preferred IDE or in the terminal. 

#### Step 5: Bash
Use bash to install python on your computer if you haven't already.
For IOS users, you can use homebrew to install python:

```bash
brew install python
```

#### Step 6: Create virtual environment:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Great job! Now you are ready to run your chatbot locally.

#### Step 7: Run the Chatbot
To run the chatbot, first insert files in the files directory. For know it only works with pdf and docs. 
Then execute the following command in your terminal:

```bash
python made4you.py
```

#### Step 8: You can now open the chatbot in your browser by visiting the URL that is displayed in the terminal.

#### Step 9: Interact with your chatbot by asking questions or providing prompts. The chatbot will respond with information from the files you uploaded.

## Troubleshooting

If you encounter any issues, ensure the following:

- Your Python version is up-to-date.
- All dependencies are installed (pip install -r requirements.txt).
- Your OpenAI API key is correctly set in your environment variables.

For additional support, feel free to raise an issue in the repository.

