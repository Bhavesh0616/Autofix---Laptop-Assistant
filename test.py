import os
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import spacy  # New import
import requests
from bs4 import BeautifulSoup
import subprocess
import getpass
import logging
import time
import threading

# Memory functions
def save_fix(query, response):
    history_file = "F:\\Project-R\\Autofix\\fixes_history.json"
    try:
        with open(history_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {"fixes": []}
    data["fixes"].append({"query": query, "response": response})
    with open(history_file, "w") as f:
        json.dump(data, f, indent=4)

def retrieve_fix(query):
    history_file = "F:\\Project-R\\Autofix\\fixes_history.json"
    try:
        with open(history_file, "r") as f:
            data = json.load(f)
        for fix in data["fixes"]:
            if query.lower() in fix["query"].lower():
                return fix["response"]
    except FileNotFoundError:
        return None
    return None

# Entity extraction
nlp = spacy.load("en_core_web_sm")

def extract_entities(query):
    doc = nlp(query)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    return entities, nouns

# Web scraping function
def scrape_solutions(query):
    cache_file = "F:\\Project-R\\Autofix\\web_cache.json"
    try:
        with open(cache_file, "r") as f:
            cache = json.load(f)
    except FileNotFoundError:
        cache = {}
    if query in cache:
        return cache[query]
    try:
        url = f"https://www.google.com/search?q={query}+laptop+fix"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        results = [result.text for result in soup.find_all("div", class_="BNeawe s3v9rd AP7Wnd")[:3]]
        cache[query] = results
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=4)
        return results
    except Exception as e:
        return [f"Error scraping: {e}"]
    
def scrape_solutions(query):
    cache_file = "F:\\Project-R\\Autofix\\web_cache.json"
    try:
        with open(cache_file, "r") as f:
            cache = json.load(f)
            print(f"Cache loaded: {cache}")
    except FileNotFoundError:
        cache = {}
        print("Cache file not found, creating new cache")
    if query in cache:
        print(f"Cache hit for {query}")
        return cache[query]
    try:
        url = f"https://www.google.com/search?q={query}+laptop+fix"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        results = [result.text for result in soup.find_all("div", class_="BNeawe s3v9rd AP7Wnd")[:3]]
        cache[query] = results
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=4)
        print(f"Cache updated with {query}: {results}")
        return results
    except Exception as e:
        cache[query] = [f"Error scraping: {e}"]
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=4)
        print(f"Cache updated with error for {query}: {e}")
        return [f"Error scraping: {e}"]

# Ranking algorithm function
def rank_solutions(query, retrieved_infos):
    query_words = set(query.lower().split())
    ranked = []
    for info in retrieved_infos:
        info_words = set(info.lower().split())
        score = len(query_words.intersection(info_words))
        ranked.append((info, score))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[0][0] if ranked else retrieved_infos[0]

# Pattern analysis function
def analyze_logs():
    log_file = "C:\\Windows\\System32\\winevt\\Logs\\System.evtx"  # Example log file
    if not os.path.exists(log_file):
        return "Log file not found."
    # Dummy analysis (real log parsing needs more setup)
    return "Found pattern: High CPU usage detected—close background apps."

# New RPA actions
def clear_temp_files():
    try:
        os.system("del /q /s /f %temp%\\*")
        return "Temp files cleared!"
    except Exception as e:
        return f"Oops, couldn’t clear temp files. Error: {e}"

def check_updates():
    try:
        os.system("wuauclt /detectnow")  # Windows update check
        return "Checking for updates—check Settings for progress!"
    except Exception as e:
        return f"Oops, couldn’t check updates. Error: {e}"
    
# Sandbox function
def run_in_sandbox(command):
    try:
        temp_dir = "F:\\Project-R\\Autofix\\temp_sandbox"
        os.makedirs(temp_dir, exist_ok=True)
        subprocess.run(command, shell=True, cwd=temp_dir, capture_output=True, text=True)
        return "Operation completed in sandbox."
    except Exception as e:
        return f"Sandbox error: {e}"
    
# Access control function
def authenticate_user():
    valid_users = {"admin": "password123", "user1": "pass456"}  # Hardcoded for demo
    attempts = 3
    while attempts > 0:
        username = input("Enter username: ")
        password = getpass.getpass("Enter password: ")
        if valid_users.get(username) == password:
            return True
        print(f"Invalid credentials. {attempts - 1} attempts left.")
        attempts -= 1
    return False
    
# Feedback function
def save_feedback(query, response, rating):
    feedback_file = "F:\\Project-R\\Autofix\\feedback.json"
    try:
        with open(feedback_file, "r") as f:
            feedback_data = json.load(f)
    except FileNotFoundError:
        feedback_data = {"feedback": []}
    feedback_data["feedback"].append({"query": query, "response": response, "rating": rating})
    with open(feedback_file, "w") as f:
        json.dump(feedback_data, f, indent=4)

# Model aur FAISS setup

# model_id = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128, temperature=0.7)
llm = HuggingFacePipeline(pipeline=pipe)

system_prompt = "I am AutoFix, a laptop repair chatbot. I answer questions concisely, focusing on laptop issues. If the user asks for an action, I execute it and confirm. I keep responses short and conversational."

# FAISS Database Setup
laptop_knowledge = [
    "To open LinkedIn on your laptop, type 'linkedin.com' in your browser or click the LinkedIn app icon if installed.",
    "To open File Explorer, press Win + E.",
    "If your laptop is slow, try closing background apps or increasing virtual memory.",
    "To turn off Wi-Fi, go to network settings or use the command 'netsh wlan disconnect'.",
    "A laptop battery typically lasts 3-5 years depending on usage."
]

fine_tune_data = [
    {"query": "my laptop is slow", "response": "Try closing background apps or increasing virtual memory."},
    {"query": "how to open file explorer", "response": "Press Win + E to open File Explorer."},
    {"query": "how to open linkedin", "response": "Type 'linkedin.com' in your browser or click the app icon."}
]

fine_tune_texts = [f"Question: {item['query']} Answer: {item['response']}" for item in fine_tune_data]
laptop_knowledge.extend(fine_tune_texts)

# embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(laptop_knowledge)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index_file = "F:\\Project-R\\Autofix\\laptop_knowledge_index.faiss"
if os.path.exists(index_file):
    index = faiss.read_index(index_file)
else:
    index.add(np.array(embeddings, dtype=np.float32))
    faiss.write_index(index, index_file)

from rpa_actions import turn_off_wifi, open_file_manager, open_linkedin

def chatbot():
    if not authenticate_user():
        print("AutoFix Chatbot: Access denied. Exiting.")
        exit()
    print("AutoFix Chatbot: Hey there! I’m your laptop-fixing sidekick. What’s up? (Type 'exit' to quit)")
    while True:
        query = input("You: ")
        entities, nouns = extract_entities(query)
        print(f"Entities: {entities}, Nouns: {nouns}")
        web_results = scrape_solutions(query) if any(keyword in query.lower() for keyword in ["how to", "my laptop", "fix", "clear", "check"]) else []
        print(f"Web results for {query}: {web_results}")
        log_analysis = analyze_logs() if "my laptop" in query.lower() and "slow" in query.lower() else ""
        if query.lower() == "exit":
            print("AutoFix Chatbot: Catch you later, tech warrior!")
            break
        try:
            with subprocess.Popen(["python", "-c", ""], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as sandbox:
                if "turn off wifi" in query.lower():
                    response = turn_off_wifi()
                elif "open" in query.lower() and "file manager" in query.lower():
                    response = open_file_manager()
                elif "open" in query.lower() and "linkedin" in query.lower():
                    response = open_linkedin()
                elif "clear temp files" in query.lower():
                    response = clear_temp_files()
                elif "check updates" in query.lower():
                    response = check_updates()
                else:
                    past_response = retrieve_fix(query)
                    if past_response:
                        response = past_response
                    else:
                        query_embedding = embedder.encode([query])
                        distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k=3)
                        retrieved_infos = [laptop_knowledge[i] for i in indices[0]]
                        if web_results:
                            retrieved_infos.extend(web_results)
                        if log_analysis:
                            retrieved_infos = [log_analysis] if not retrieved_infos else retrieved_infos + [log_analysis]
                        retrieved_info = rank_solutions(query, retrieved_infos)
                        if not retrieved_info or "Error" in retrieved_info:
                            full_prompt = f"{system_prompt}\n\nQuestion: {query}\nSuggest a creative fix for this laptop issue:\nAnswer:"
                            response = llm.invoke(full_prompt)
                            if "Answer:" in response:
                                response = response.split("Answer:")[1].strip()
                            response = response.split("\n")[0].strip()
                        else:
                            if query.lower().startswith("how to"):
                                full_prompt = f"{system_prompt}\n\nRetrieved Info: {retrieved_info}\nWeb Results: {web_results}\nLog Analysis: {log_analysis}\nQuestion: {query}\nAnswer:"
                            else:
                                full_prompt = f"{system_prompt}\n\nRetrieved Info: {retrieved_info}\nWeb Results: {web_results}\nLog Analysis: {log_analysis}\nQuestion: {query}\nAnswer:"
                            response = llm.invoke(full_prompt)
                            if "Answer:" in response:
                                response = response.split("Answer:")[1].strip()
                            response = response.split("\n")[0].strip()
                    save_fix(query, response)
                    logging.info(f"Query: {query}, Response: {response}")
                    print(f"AutoFix Chatbot: {response}")
                    rating = input("Rate this response (1-5): ")
                    while not (rating.isdigit() and 1 <= int(rating) <= 5):
                        rating = input("Invalid rating. Please enter a number between 1-5: ")
                    save_feedback(query, response, int(rating))
        except Exception as e:
            print(f"AutoFix Chatbot: Oops, glitch in the matrix! Error: {e}")

# Logging setup
logging.basicConfig(filename="F:\\Project-R\\Autofix\\autofix.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


# Integration in main (if __name__ mein call karo)
if __name__ == "__main__":
    print("Device set to use cpu")
    chatbot()