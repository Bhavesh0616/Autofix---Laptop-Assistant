# Autofix---Laptop-Assistant

# 🤖 AutoFix AI – Your Laptop-Fixing Sidekick

AutoFix AI is an offline, autonomous, open-source AI assistant that diagnoses and fixes laptop issues using conversational AI, a FAISS-based knowledge base, and Robotic Process Automation (RPA). It's a proof-of-concept step toward building a self-evolving intelligence—RAVEN.

---

## 🚀 Vision

Current chatbots often rely on paid APIs, always-on internet, and lack true autonomy. AutoFix breaks those boundaries:

- 🧠 Operates offline using open-source models like **Mistral** or **Qwen**
- 🛠️ Diagnoses laptop problems with **NLP**, **entity recognition**, and **web scraping**
- ⚙️ Applies fixes with **Python-based RPA**
- 🧩 Uses **FAISS vector database** for local knowledge retrieval
- 🔐 Sandboxed and access-controlled for **security and trust**
- 🔄 Learns from past interactions using feedback loops and memory storage

---

## 📂 Project Structure

| File | Description |
|------|-------------|
| `test.py` | Core chatbot code with RAG, FAISS, RPA, security, and feedback modules |
| `rpa_actions.py` | Executes system-level tasks like turning off Wi-Fi or opening apps |
| `fixes_history.json` | Stores all past queries and applied solutions for learning |
| `feedback.json` | Logs user feedback and ratings to improve future interactions |
| `web_cache.json` | Caches scraped search results to improve offline resilience |
| `autofix.log` | Chat log with timestamped user queries and responses |
| `laptop_knowledge_index.faiss` | FAISS vector index of embedded laptop repair knowledge |
| `🚀 Problem Statement.txt` | Full project vision and roadmap |
| `text_gen.ipynb` | Jupyter notebook for testing local model generation |

---

## 🛠️ Features

### 🔍 Intelligent Troubleshooting
- Entity extraction using **spaCy**
- Web scraping with **BeautifulSoup**
- Pattern analysis on logs (stubbed for extension)
- Lightweight **ranking algorithm** for solution selection

### 🧠 Retrieval-Augmented Generation (RAG)
- Uses **sentence-transformers** to embed knowledge
- Queries **FAISS** vector index for relevant fixes
- Combines retrieved info with **Hugging Face local LLMs**

### 🤖 Autonomous Repair Actions
- Executes RPA tasks like:
  - `turn_off_wifi()`
  - `open_file_manager()`
  - `clear_temp_files()`
- Decision engine determines auto vs. user-confirmed actions

### 🔐 Security Features
- **User authentication**
- **Sandboxing** to isolate commands
- Access-controlled design, extendable to Docker/VM containers

### 📈 Feedback & Learning
- All fixes and feedback stored in JSON files
- Can be used to fine-tune responses or retrain models

---

## ⚙️ How to Run

> 💡 Requires Python 3.10+, transformers, faiss-cpu, sentence-transformers, spaCy, and more.

1. Clone the repo:
```bash
   git clone https://github.com/your-username/autofix-ai.git
   cd autofix-ai
```

2. Install Dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Run the chatbot:
```bash
python test.py
```

🧪 Example Interactions

```bash

AutoFix Chatbot: Hey there! I’m your laptop-fixing sidekick. What’s up?

You: my laptop is slow  
→ Try closing background apps or increasing virtual memory.

You: turn off wifi  
→ Wi-Fi turned off—disconnected from the network!

You: how to open file explorer  
→ Press Win + E to open File Explorer.
```

🔮 Future Plans
AutoFix is just the beginning. The final goal is RAVEN—a self-evolving general AI that can learn, adapt, and scale to solve humanity’s grandest challenges like:

Consciousness transfer 🧠

Interstellar survival 🚀

Fully autonomous self-healing systems 💡

🧠 Credits
This project was created by Bhavesh Walankar as a futuristic AI experiment to explore the boundaries of offline intelligence, automation, and human-machine synergy.

📄 License
This project is open-source under the MIT License.



