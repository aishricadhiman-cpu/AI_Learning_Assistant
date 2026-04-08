#   AI Learning Assistant  
### A RAG-based Machine Learning & Deep Learning Knowledge System

---

##  Objectives of the Project

The primary objective of this project is to build an **intelligent AI-powered learning assistant** that can:

- Help students understand **Machine Learning (ML)** and **Deep Learning (DL)** concepts
- Provide **context-aware answers** from trusted academic sources
- Reduce dependency on constant faculty availability
- Deliver **accurate, source-backed explanations** instead of generic responses

This system leverages **Retrieval-Augmented Generation (RAG)** to combine:
- Knowledge retrieval from books 
- Reasoning ability of Large Language Models   

---

##  Overview of the Model

This project is a **Retrieval-Augmented Generation (RAG) system**.

###  What is RAG?

RAG is a hybrid approach where:
1. Relevant information is **retrieved from a knowledge base**
2. An LLM generates answers using that retrieved context

This ensures:
- Higher accuracy 
- Reduced hallucination   
- Source-backed responses  

---

##  Knowledge Base (Books Used)

The system is built using content from:

### 1. *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*  
 Author: **Aurélien Géron**

- Covers practical ML & DL concepts
- Industry-relevant techniques
- Strong foundation for beginners and practitioners

---

### 2. *Deep Learning with Python*  
 Author: **François Chollet** (Creator of Keras)

- Focuses on deep learning fundamentals
- Practical implementation using Keras
- Conceptual clarity with real-world examples

---

## ️ Core Technologies Used

- **LLM:** Groq (`llama-3.3-70b-versatile`)
- **Embeddings Model:** Hugging Face (`BAAI/bge-base-en-v1.5`)
- **Vector Database:** ChromaDB
- **Framework:** LangChain
- **Frontend:** Streamlit

---

##  Features

-  Answers based on **trusted ML/DL books**
-  Semantic search using **vector embeddings**
-  Context-aware responses using LLM
-  **Source citations with page numbers**
- ️ Guardrails to filter irrelevant queries
-  Fast retrieval with ChromaDB
-  Conversation memory using **WindowBufferMemory**
-  Advanced retrieval using:
  - **MMR (Max Marginal Relevance)** → diversity
  - **Embedding filter** → relevance
-  Returns **top 2–3 most relevant chunks**

---

##  How It Works (Pipeline Explanation)

### Step-by-step flow:

1. **User Query Input**
   - User asks a question in natural language

2. **Query → Vector Conversion**
   - Query is converted into embedding using:
     ```
     BAAI/bge-base-en-v1.5
     ```

3. **Similarity Search**
   - The query vector is compared with stored document vectors
   - Similarity scores are computed

4. **Retriever (Advanced Strategy)**
   - Uses:
     - MMR → ensures diverse results
     - Embedding filtering → ensures relevance
   - Selects **top 2–3 most relevant chunks**

5. **Context Passing to LLM**
   - Retrieved chunks are passed to:
     ```
     Groq LLM (llama-3.3-70b-versatile)
     ```

6. **Answer Generation**
   - LLM generates a contextual answer using retrieved content

7. **Source Attribution**
   - Displays:
     - Book name   
     - Page number  

---

##  Project Structure

```
ml_dl_tutor/
│
├── data/ # PDF books
├── chroma_db/ # Vector database
├── src/ # Core modules
│ ├── loader.py
│ ├── cleaner.py
│ ├── splitter.py
│ ├── embeddings.py
│ ├── vectorstore.py
│ ├── retriever.py
│ ├── chain.py
│ ├── guardrail.py
│
├── app.py # Streamlit app
├── ingest.py # Data ingestion
├── config.py
├── requirements.txt
└── README.md

```

---

##  Real-World Impact & Applications

This type of system has strong real-world relevance:

###  Educational Institutions
- Institutes can build **custom RAG assistants** using their syllabus and books
- Students can:
  - Resolve doubts anytime 
  - Learn at their own pace  
  - Access **trusted academic content instantly**

###  Coaching Centers & EdTech Platforms
- Provide **24/7 doubt-solving assistants**
- Reduce faculty workload
- Scale learning support for thousands of students

###  Self-learners
- Acts as a **personal AI tutor**
- Helps in understanding complex ML/DL concepts with references

---

##  Business Opportunities

RAG-based systems like this can be used to:

- Build **AI tutors for universities**
- Develop **subscription-based learning assistants**
- Create **domain-specific knowledge systems** (law, medicine, finance)
- Integrate into **EdTech platforms (like Byju’s, Unacademy)**

 This project demonstrates how AI can transform **knowledge accessibility and education delivery**

---

##  Future Improvements

-  Add support for **mathematical problem-solving and numerical queries**
-  Improve handling of **analytical and logic-based questions**
-  Enhance **memory capacity** for longer and more context-aware conversations
-  Expand knowledge base to cover broader AI domains:
  - Natural Language Processing (NLP)
  - Computer Vision
  - Reinforcement Learning
-  Integrate **cloud-based vector databases** (e.g., Pinecone, Weaviate) for scalability
-  Optimize retrieval speed and response latency
-  Develop a **chat-based conversational interface** with history tracking
-  Improve retrieval using hybrid search (keyword + semantic)

---

## 🧾 Conclusion

This project demonstrates the practical implementation of a **Retrieval-Augmented Generation (RAG)** system for building an intelligent and reliable learning assistant.

By combining:
- High-quality academic resources  
- Efficient vector-based retrieval 
- Powerful Large Language Models  

…the system provides **accurate, context-aware, and source-backed responses**, significantly improving the learning experience.

Such systems have the potential to:
- Transform traditional education  
- Enable personalized learning  
- Provide scalable academic support  

 Overall, this project showcases how AI can be effectively leveraged to bridge the gap between **information access and conceptual understanding**.

---

##  Author

**Aishrica**  
MSc Data Science @ IIIT Lucknow

