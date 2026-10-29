## SHRIHARIHARAN S

AI/ML Engineer | Generative AI |Machine Learning | Data Science

Kanchipuram, Tamil Nadu, India | +91 9585298060 | shrihariharan1999@gmail.com

[www.linkedin.com/in/shrihariharan-shankar-654b1a160/ | github.com/Shrihariharan1999](https://www.linkedin.com/in/shrihariharan-shankar-654b1a160/)

## PROFESSIONAL SUMMARY

AI/ML and Generative AI Engineer with 4+ years of professional IT experience, including enterprise payment implementation and test engineering at Wipro, an M. Tech in Computer Science (AI &amp; Data Science) from VIT Chennai (CGPA 9.28/10), and hands-on experience building AI applications. I work primarily with Python and have built machine learning and deep learning solutions, QLoRA fine-tuned LLMs, hybrid RAG systems using FAISS and BM25 with reranking, and multi-agent applications using LangChain, LangGraph, MCP, and FastAPI. My recent work includes VoyageAI, a live multi-agent travel planner deployed on Google Cloud, an autonomous data analyst for CSV/Excel data, an Enterprise RAG platform, and an AI automation testing agent. At Wipro, I owned 30+ Mastercard payment-modernization projects across requirements, implementation, testing, and production rollout. During my Intel internship, I worked on Python-based automation for post-silicon validation, cache coherency, seamless runtime patching of processor microcode components such as uCode/pCode, structured log analysis, and failure investigation, while also building RAG-based and agentic AI applications. I enjoy turning AI concepts into practical systems with APIs, evaluation, guardrails, testing, and deployment.

## TECHNICAL SKILLS

- Generative AI &amp; Agentic Systems: LLMs, RAG, Hybrid Retrieval (FAISS + BM25), Cross-Encoder Reranking, LangChain, LangGraph, deepagents, MCP, Tool Calling, Multi-Agent Systems, Human-in-the-Loop, Guardrails, Self-Healing, LLM-as-aJudge, LLM Evaluation, Prompt Engineering, LoRA, QLoRA, PEFT, LangSmith
- Machine Learning &amp; Data Science: Regression, Classification, Decision Trees, Random Forest, SVM, KNN, XGBoost, LightGBM, CatBoost, AdaBoost, Gradient Boosting, K-Means, DBSCAN, PCA, Feature Engineering, Hyperparameter Tuning, Cross-Validation, SHAP, SMOTE, Data Analysis &amp; Visualization, ML Pipelines
- Deep Learning &amp; NLP: ANN, CNN, RNN, LSTM, GRU, Transfer Learning, Computer Vision, Time-Series Forecasting, GANs, Transformers, Self/Multi-Head Attention, Tokenization, Embeddings, TF-IDF, Word2Vec, NER, Sentiment Analysis
- Languages &amp; Frameworks: Python, SQL, C, PyTorch, TensorFlow, Keras, Scikit-learn, Pandas, NumPy, OpenCV, NLTK, Hugging Face Transformers, Sentence-Transformers, Pydantic, Streamlit
- Deployment, Cloud &amp; APIs: FastAPI, REST APIs, Uvicorn, AsyncIO, Docker, Google Cloud Run, Firebase (Auth, Firestore, Hosting), Apache Kafka, MySQL, SQLite, Git, Postman, Pytest, Model Deployment · OpenAI, Google Gemini, NVIDIA AI APIs, Groq, Ollama

## PROFESSIONAL EXPERIENCE

## Test Engineer, Wipro Limited

Chennai, Tamil Nadu | Sep 27, 2021 - Aug 10, 2026 | Sabbatical Jun 2024 - Aug 2026 for full-time M. Tech; relieved after graduation

- Delivered 30+ Mastercard payment modernization projects across North America, APAC, Latin America, and EMEA, covering implementation, testing, and production rollout.
- Executed BIN 8 transition testing by validating customer-provided retain and return ranges and performing mainframe SQL checks against approved migration requirements.
- Validated 180-day transaction activity, including SMS and DMS transactions, and assessed mainframe dependencies to determine whether customer approval or further investigation was required.
- Supported Token Replenishment by validating ICA and card details, reviewing eligible Token Pool ranges, and verifying range assignments against applicable business rules before MDES System Manager configuration.
- Explored AI-assisted testing to suggest additional test scenarios, compare candidate ranges against approved requirements, and explain validation exceptions, with final verification performed using SQL results and established business rules.

## Graduate Technical Intern (Internship), Intel Corporation

Bangalore, Karnataka | Jul 2, 2025 - May 27, 2026

- Built RAG-based and Agentic AI applications using Lang Chain, multi-agent workflows, and retrieval-augmented generation during Intel's internal AI learning programs and Hackathon.
- Performed post-silicon validation of Intel Xeon (Forest Series) server processors, focusing on cache coherency and system-level validation.
- Validated live-patching (Seamless Patching) mechanisms for Rapid Series processors across multiple runtime scenarios while maintaining platform stability.
- Developed Python automation scripts to accelerate OOBMSM validation, reducing manual testing effort through automated workflows and for Validation.
- Conducted structured log analysis, debugging, and failure investigation to identify and isolate hardware and software validation issues. Worked with enterprise validation processes, improving reliability and efficiency of large-scale server testing.

## AI/ML, GENERATIVE AI &amp; DATA SCIENCE PROJECTS

## VoyageAI: AI Tour Planner, Live Full-Stack Multi-Agent Travel Assistant

FastAPI, LangChain, LangGraph, deepagents, Gemini, React, Firebase Auth, Firestore, Docker, Google Cloud Run, LangSmith

[Live: voyageai-702fe.web.app | Code: github.com/Shrihariharan1999/Agentic\_AI/tree/main/ai-tour-planner](https://github.com/Shrihariharan1999/Agentic_AI/tree/main/ai-tour-planner)

- Built a multi-agent travel planner that turns natural-language requests into date-aware, day-by-day itineraries; a planner agent orchestrates a weather sub-agent and 8 live-data tools (Google Places/Routes, flights, hotels, train availability, weather, web search).
- Grounded and controlled the agents: prices, URLs, and schedules come only from tool results and the agent stops to ask the user instead of guessing when constraints cannot be met.
- Secured the FastAPI backend with Firebase ID-token verification and per-user ownership checks; chats persist in Firestore.
- Deployed React on Firebase Hosting (Google Sign-In) and a Dockerized FastAPI backend on Cloud Run, traced with LangSmith.

Agentic AI Platform: Autonomous Data Analyst, Web Testing &amp; MCP Tool-Using Agents

LangGraph-style agents, FastAPI, Streamlit, SQLite, FastMCP, Playwright, Gemini, LangSmith · github.com/Shrihariharan1999/Agentic\_AI

- Autonomous AI Analyst: Built a multi-agent application with Coordinator, Analyst, and Validator agents that converts naturallanguage questions on CSV/Excel datasets into statistical analysis, visualizations, and web-researched answers, with SQLite for session and dataset-version tracking. Developed an LLM-as-a-Judge validation workflow that reviews generated reports, identifies issues, and automatically retries the draft before producing the final Markdown/HTML report, with file access sandboxed through a custom MCP server.
- AI Automation Testing: Built a Playwright-based web-testing agent supporting human-in-the-loop approval, self-healing retries, and LangSmith tracing for browser-based test execution.
- AI Code Interpreter: Developed a separate Python-based code-interpreter application within the same repository for executing AIgenerated analytical/code workflows in response to user requests.

## Enterprise RAG System

Python, LangChain, FAISS, BM25, Sentence-Transformers, Gemini · github.com/Shrihariharan1999/Enterprise\_RAG

- Built a modular RAG pipeline (ingestion, chunking, vector store, retrieval) supporting 8 document formats, combining FAISS semantic search with BM25 keyword search and cross-encoder reranking.
- Added spell-correction(optional), duplicate-file detection, page-level citations, and persistent chat history for traceable answers.

## LLM Fine-Tuning with QLoRA

[Python, PyTorch, Hugging Face Transformers, PEFT, QLoRA · github.com/Shrihariharan1999/LLM\_FineTuning\_QLORA\_2026](https://github.com/Shrihariharan1999/LLM_FineTuning_QLORA_2026)

- Fine-tuned open-source LLMs with QLoRA (4-bit quantization plus LoRA adapters), training only a small slice of parameters so the job ran on a single consumer GPU instead of a cluster.
- Benchmarked fine-tuned outputs against the base model to quantify the actual gain from adaptation.

## Machine Learning, Deep Learning &amp; NLP Portfolio

Python, Scikit-learn, TensorFlow, Keras, PyTorch, NLTK, LangChain

GitHub: Machine\_Learning\_MTECH | Deep\_Learning\_2026 | Machine\_Learning\_2026

- 87-notebook, 11-module portfolio spanning supervised/unsupervised ML, ensembles, and CNNs/LSTMs/GANs/Transformers implemented from scratch (attention and positional encoding by hand).
- Includes a generative-AI module (prompt engineering, LoRA, RAG, LLM evaluation, GenAI ethics) and applied case studies: Titanic survival, cirrhosis-outcome prediction, and English-to-French translation.

## MindMate: AI Mental Health Support Platform

Next.js 14, FastAPI, Gemini 2.5 Flash, Llama 3.3 (Groq), DeepFace/MTCNN · github.com/Shrihariharan1999/MindMate

- The frontend runs on Next.js with a FastAPI backend, JWT auth, and real-time WebSocket chat. Llama 3.3, accessed through Groq, acts as an LLM judge that scores each conversation for safety, honesty, and empathy, with a rule-based fallback if the model call fails. Gemini chat, DeepFace/MTCNN emotion detection, VADER sentiment, and Edge-TTS voice feed a Recharts mood dashboard.

## HACKATHON

DATASET'24 Hackathon: Telecom Energy Optimization (Sponsored by Prodapt and partnered with Nokia), VIT Chennai Python, Apache Kafka, LLMs, React | Certificate: Drive Link

- Built a real-time Kafka pipeline and LLM-driven model to predict network load and auto-switch cell towers to power-saving mode.
- Delivered in a 4-person team over a 36-hour hackathon and presented to industry judges.

## EDUCATION &amp; ACHIEVEMENTS

## M. Tech, Computer Science &amp; Engineering (AI &amp; Data Science) | 2024 - 2026

Vellore Institute of Technology, Chennai · CGPA: 9.28/10 · Meritorious Award for academic performance, Semester 1 and Semester 2

B.E., Electronics and Instrumentation Engineering | 2017 - 2021

Panimalar Engineering College · CGPA: 8.66/10