import os
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load Ollama LLM
def load_llm():
    return OllamaLLM(model="mistral")  # or llama3

# Step 2: Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
Answer the following question using only the information in the context.
If the answer cannot be found in the context, say "I don't know."


Chat History:
{chat_history}

Context:
{context}

Current Question:
{question}

Answer:
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["chat_history", "context", "question"]
    )

# Step 3: Load FAISS vectorstore
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 4: Chat history
chat_history = []

# Step 5: Build QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 6: Interactive Loop
while True:
    user_query = input("You: ")
    if user_query.lower() in ["exit", "quit"]:
        print("Exiting chatbot...")
        break

    history_str = ""
    for msg in chat_history:
        history_str += f"{msg['role'].capitalize()}: {msg['content']}\n"

    response = qa_chain.invoke({
        'question': user_query,
        'chat_history': history_str.strip()
    })

    bot_response = response["result"]
    print("Bot:", bot_response)

    chat_history.append({"role": "user", "content": user_query})
    chat_history.append({"role": "assistant", "content": bot_response})
