# -HealthBot-Conversational-AI-for-Medical-Query-Assistance
HealthBot: Conversational AI for Medical Query Assistance Developed a medical chatbot using LangChain and Ollama-hosted LLMs. Supports multi-turn conversations with persistent memory and custom prompt templates. Delivers accurate, context-aware responses for health-related queries in real time.

**🔧 Tech Stack:**

- **LLM Backend:** Ollama with custom fine-tuned models  
- **Frameworks:** LangChain, Streamlit  
- **Features:** Persistent chat memory, custom prompt templates, contextual understanding
  
**💡 Use Case:**  
Ideal for building AI assistants that simulate interactions with virtual health consultants or for exploring LLM applications in the healthcare domain (educational/demonstrative use).

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Hirmi25/healthbot.git
   cd healthbot

2. Create and activate a virtual environment:
   ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate

4. Install dependencies
   ```bash
  pip install -r requirements.txt

5. Set up your .env file with:
   ```bash
  OLLAMA_API_KEY=your_key

6. Create Memory for LLM
   ```bash
  python create_memory_for_llm.py

7. Run the app:
   ```bash
  streamlit run app.py
