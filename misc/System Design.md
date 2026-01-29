Task: Build a chatbot that will do the following:
1. Generate summary of uploaded PDF, and,
2. RAG over the same PDF.

Tech Stack:
1. Use OpenAI embedding and language models, API Key for which will be provided by me via a .env file in the project root,
2. Use LangChain (langchain==1.2.7), LangGraph (langgraph==1.0.7) and LangSmith (langsmith==0.6.6) wherever required,
2. Use Python (3.11.2), ChromaDB and Flask API (Flask==3.1.2, flask-cors==6.0.2) for development and deployment,
3. Use Streamlit (streamlit==1.53.1) for the frontend interface.

Instructions:
1. Memory management:
	a. Memory will be preserved ONLY for the active chat session,
	b. No requirement for long-term/persistent memory,
	c. No cross-session persistence of any kind, starting a new chat means a clean slate.
2. Data Management:
	a. User uploaded documents (.pdf files) will be preserved ONLY for the active chat session,
	b. Embeddings generated for the RAG pipeline, shall be stored into the vector database ONLY for the active chat session,
	c. No cross-session persistence of any kind, starting a new chat means a clean slate.

UI/UX (from an user standpoint):
1. Upon visiting the website/webpage, the user is given an upload button to upload a PDF file, where, without uploading a PDF no other interaction is possible,
2. Right after upload, a popup that blocks all interaction until the following is completed:
	a. PDF is parsed,
	b. A concise summary of the entire parsed PDF is generated,
    c. The vector database is populated with embeddings from the parsed PDF.
3. Post upload and processing, a summary is displayed to the user on a clean UI without the upload button or the popup anymore, ONLY a chat area (where the summary is shown), text input box and a new chat button is available now.
4. User uses the text input box to type their queries into regarding the PDF, receiving answers grounded strictly in the report content,
6. The new chat option clears all memory, uploaded documents and embeddings to start fresh,
7. Closing the browser tab or window also clears all memory, uploaded documents and embeddings to start fresh on next visit.