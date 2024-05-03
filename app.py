# Import necessary libraries
import streamlit as st
import pandas as pd

# Main function for Streamlit app
def main():
    st.title("Table Question Answering Chatbot")
    
    # Lazy loading of Tapas model & tokenizer
    @st.cache(allow_output_mutation=True)
    def load_tapas_model():
        from transformers import AutoModelForTableQuestionAnswering, AutoTokenizer
        model_name = 'google/tapas-large-finetuned-wtq'
        tapas_model = AutoModelForTableQuestionAnswering.from_pretrained(model_name)
        tapas_tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tapas_model, tapas_tokenizer

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
                tapas_model, tapas_tokenizer = load_tapas_model()
                from transformers import pipeline
                nlp = pipeline('table-question-answering', model=tapas_model, tokenizer=tapas_tokenizer)
                result = nlp({'table': data, 'query': question})
                answer = result['cells']
                st.write("Answer:", answer)
            else:
                st.warning("Please enter a question.")
    
# Run the app
if __name__ == "__main__":
    main()
