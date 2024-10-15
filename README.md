# Project Overview: Medical AI Chatbot

This is a medical AI chatbot that can help you with your medical queries. It is built using Python. The chatbot uses a deep learning model to predict the disease based 
on the symptoms provided by the user. It also provides information about the disease and possible treatments. The chatbot is designed to be user-friendly and easy to use. 
The chatbot is still in development and new features are being added regularly. If you have any feedback or suggestions, please feel free to contact me. We hope you find 
the chatbot helpful and informative. Thank you for using our medical AI chatbot!

![Screenshot 2024-10-15 205759](https://github.com/user-attachments/assets/5fa33f81-307d-4f38-ad7b-d635cee85430)


## Project Explanation

### Retrieval-Augmented Generation (RAG) Model

The chatbot leverages a **RAG model** to deliver precise medical information:
- **Retrieval:** It fetches relevant information from a vector database built from the dataset "Current Medical Diagnosis and Treatment 2023 BY MAXINE A. PAPADAKIS, STEPHEN J. MCHPHEE, MICHAEL W. RABOW."
- **Generation:** A **Large Language Model (LLM)** generates responses based on the retrieved information.

### Large Language Model (LLM)

The **LLM** is trained on medical data, enabling it to understand user queries and produce accurate, human-like responses.

### Vector Database

A **Vector Database** stores numerical representations (embeddings) of the medical text. It performs similarity searches to retrieve the most relevant information for the user’s query.

### Workflow

1. **User Query:** The chatbot converts the user’s query into a vector.
2. **Information Retrieval:** The vector database retrieves the most similar documents.
3. **Response Generation:** The LLM generates a detailed response using the retrieved documents.
4. **User Interaction:** The chatbot provides the response to the user.

This architecture ensures that the chatbot delivers accurate, relevant, and context-aware medical information.

# Technologies Used: 
  - LLM,
  - NLP
  - RAG Pipeline,
  - Hugging Face,
  - Streamlit

# Installation
To get started with Legal_AI_Chatbot, follow these steps:
 1. Clone the repository:
    ```bash
    https://github.com/vidyanandk/MediChat
    ```
 2. Navigate to the project directory:

    ```bash
    pip install -r requirements.txt
    ```

     Then,
    ```bash
    python -m streamlit run model.py
    ```
    Or
    ```bash
    streamlit run model.py
    ```

## Contibuting
Contributions are always welcome! Just raise an issue, we will discuss it.

## Contact
For any questions or inquiries, please contact us.
