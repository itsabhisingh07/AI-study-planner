#  AI Study Planner Agent

An interactive, multi-agent AI application that researches technical topics and generates comprehensive, highly detailed study guides in Markdown format. 

##  Features
* **Multi-Agent Architecture:** Utilizes a graph-based workflow (Orchestrator -> Worker -> Reducer) to plan, research, and compile documents.
* **Interactive UI:** A clean Streamlit frontend for users to easily input topics and instantly download or read their generated guides.
* **File Management:** Built-in sidebar to track, view, and delete past research sessions.

## 🛠️Tech Stack
* **Framework:** LangGraph & LangChain
* **LLM:** Llama-3.3-70b (via Groq)
* **Frontend:** Streamlit
* **Language:** Python

##  How to Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/itsabhisingh07/ai-study-planner.git](https://github.com/itsabhisingh07/ai-study-planner.git)

   Install dependencies:

```bash
    pip install -r requirements.txt

    Add your API Key: Create a .env file in the root directory and add GROQ_API_KEY=your_key_here.
    Run the app:
    ```

```bash
python -m streamlit run app.py
```