# Natural-Language-Query-Agent_Guide

## User Guide for PDF Chat Assistant

This user guide provides an overview and instructions for using the PDF Chat Assistant, a Streamlit application that allows users to upload PDF files and ask questions about their content. The app processes the PDFs, creates an index for quick retrieval, and provides detailed answers based on the uploaded documents.

### Key Components

1. **Libraries and API Tokens**:
    - **Libraries**: The application uses various libraries such as `os`, `streamlit`, `langchain`, `PyPDF2`, and `FAISS` for PDF processing, text splitting, embedding generation, and similarity searches.
    - **API Token**: The `REPLICATE_API_TOKEN` environment variable must be set to authenticate and use the Replicate API.

2. **Functions**:
    - **get_pdf_text(pdf_docs)**: Extracts text from a list of PDF documents.
    - **get_text_chunks(text)**: Splits the extracted text into manageable chunks for processing.
    - **get_vector_store(text_chunks)**: Creates a FAISS vector store from the text chunks for efficient similarity search.
    - **get_conversational_chain()**: Sets up a conversational chain using a language model for answering questions.
    - **user_input(user_question)**: Handles user input, performs a similarity search, and returns an answer to the user's question.

3. **Main Application**:
    - **Streamlit Interface**: The application uses Streamlit to create a web interface where users can upload PDFs, ask questions, and receive answers.
    - **Menu and Sidebar**: The sidebar allows users to upload PDF files and trigger processing.

### How to Use

1. **Setup**:
    - Ensure you have the required libraries installed (`streamlit`, `PyPDF2`, `langchain`, `FAISS`).
    - Set the `REPLICATE_API_TOKEN` environment variable with a valid token.

2. **Running the Application**:
    - Run the script using Streamlit: `streamlit run your_script_name.py`.
    - This will open a web interface in your default web browser.

3. **Uploading PDFs**:
    - Use the file uploader in the sidebar to upload one or more PDF files.
    - Click the "Submit & Process" button to start processing the PDFs. This will extract text from the PDFs, split the text into chunks, and create a FAISS vector store.

4. **Asking Questions**:
    - Enter your question in the text input field on the main page.
    - The app will search the processed text for relevant information and provide a detailed answer.

### Detailed Function Descriptions

- **get_pdf_text(pdf_docs)**:
    - Loops through each PDF file and extracts text from all pages.
    - Returns a single string containing the combined text of all PDFs.

- **get_text_chunks(text)**:
    - Uses `RecursiveCharacterTextSplitter` to divide the text into chunks of 100,000 characters with an overlap of 1,000 characters.
    - Returns a list of text chunks.

- **get_vector_store(text_chunks)**:
    - Generates embeddings for the text chunks using a HuggingFace model.
    - Creates a FAISS vector store from the embeddings and saves it locally.

- **get_conversational_chain()**:
    - Sets up a language model with a specified prompt template for answering questions.
    - Returns a chain configured for question-answering.

- **user_input(user_question)**:
    - Loads the FAISS vector store and performs a similarity search for the user question.
    - Uses the conversational chain to generate a detailed answer based on the search results.
    - Displays the answer on the Streamlit interface.

- **main()**:
    - Configures the Streamlit page and layout.
    - Handles user interactions such as file uploads and question submissions.
    - Calls appropriate functions to process PDFs and handle user queries.

### Error Handling

- If the FAISS index file is not found, an error message is displayed, prompting the user to ensure the index file is correctly generated and saved.

This guide should help you understand and use the PDF Chat Assistant effectively. If you encounter any issues or have questions, refer to the comments in the code for additional insights.
