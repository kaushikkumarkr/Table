# Import necessary libraries
import streamlit as st
from transformers import AutoModelForTableQuestionAnswering, AutoTokenizer, pipeline
import pandas as pd
import torch

# Load model & tokenizer
model_name = 'google/tapas-large-finetuned-wtq'
tapas_model = AutoModelForTableQuestionAnswering.from_pretrained(model_name)
tapas_tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initializing pipeline
nlp = pipeline('table-question-answering', model=tapas_model, tokenizer=tapas_tokenizer)

# Define function for question answering
def qa(query, data):
    result = nlp({'table': data, 'query': query})
    answer = result['cells']
    return answer

# Main function for Streamlit app
def main():
    st.title("Table Question Answering Chatbot")
    
    # File upload
    st.write("Please upload your CSV file:")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read uploaded data
        data = pd.read_csv(uploaded_file)
        data = data.astype(str)
        
        # Display first 5 rows of data
        st.write("First 5 rows of the uploaded data:")
        st.write(data.head())
        
        # Question-Answering session
        question = st.text_input("Ask your question:")
        
        if st.button("Get Answer"):
            if question:
                answer = qa(question, data)
                st.write("Answer:", answer)
            else:
                st.warning("Please enter a question.")
    
# Run the app
if __name__ == "__main__":
    main()
