import os
import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    """
    Load or initialize the FAISS vectorstore from local path.
    """
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db


def set_custom_prompt(custom_prompt_template):
    """
    Return a PromptTemplate accepting chat_history, context, and question.
    """
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["chat_history", "context", "question"]
    )


def load_llm():
    """
    Initialize and return the Ollama LLM.
    """
    return OllamaLLM(model="mistral")  # fallback: llama3


def main():
    st.title("MediBot - Medical Book Chatbot")

    # Initialize chat history in session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'history_pairs' not in st.session_state:
        st.session_state.history_pairs = []  # list of (user, assistant)

    # Display existing messages
    for msg in st.session_state.messages:
        st.chat_message(msg['role']).markdown(msg['content'])

    # Prompt user for input
    user_input = st.chat_input("Ask your medical question…")
    if not user_input:
        return

    # Append and show user message
    st.chat_message('user').markdown(user_input)
    st.session_state.messages.append({'role': 'user', 'content': user_input})

    # Build a single string of the entire chat history for prompting
    history_str = ""
    for user, bot in st.session_state.history_pairs:
        history_str += f"User: {user}\nAssistant: {bot}\n"

    # Define custom prompt including chat history
    CUSTOM_PROMPT_TEMPLATE = '''
Chat History:
{chat_history}

Context:
{context}

Current Question:
{question}

Answer using only the above context. If you don’t know, say "I don't know."
'''

    try:
        # Load vectorstore
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error("Failed to load the vector store.")
            return

        # Build conversational retrieval chain
        conv_chain = ConversationalRetrievalChain.from_llm(
            llm=load_llm(),
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            combine_docs_chain_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
        )

        # Invoke chain with proper inputs
        result = conv_chain({
            'question': user_input,
            'chat_history': st.session_state.history_pairs
        })
        answer = result.get("answer", "")

        # Display and store assistant reply
        st.chat_message('assistant').markdown(answer)
        st.session_state.messages.append({'role': 'assistant', 'content': answer})
        st.session_state.history_pairs.append((user_input, answer))

    except Exception as e:
        st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
