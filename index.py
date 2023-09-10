import os, streamlit as st
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, ServiceContext
from langchain.llms.openai import OpenAI
import time

# Uncomment to specify your OpenAI API key here (local testing only, not in production!), or add corresponding environment variable (recommended)

st.title("💬 Ask llama 🦙")


api_key = st.text_input("🔑 Your API Key", "")
if os.environ['OPENAI_API_KEY'] == None: 
    os.environ['OPENAI_API_KEY'] = api_key 

    

temperature_option = st.slider("🌡️ Temperature", min_value=0, max_value=10, value=5, step=1)/10


# Define a simple Streamlit app


option = st.selectbox(
    'Choose a model :',
    ('🧠 GPT 4', '🚀 GPT 3.5 Turbo'), label_visibility="hidden")

if option == '🧠 GPT 4':
    model_name_option = "gpt-4"
elif option == '🚀 GPT 3.5 Turbo':
    model_name_option = "gpt-3.5-turbo"

query = st.text_input("What would you like to ask? (source: DATA)", "")


# If the 'Submit' button is clicked
if st.button("Submit"):
    with st.spinner('Wait for it...'):
        if not query.strip():
            st.error(f"Please provide the search query.")
        else:
            try:
            # This example uses text-davinci-003 by default; feel free to change if desired
                llm_predictor = LLMPredictor(llm=OpenAI(temperature=temperature_option, model_name=model_name_option))

            # Configure prompt parameters and initialise helper
                max_input_size = 4096
                num_output = 256
                max_chunk_overlap = 20

                prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

            # Load documents from the 'data' directory
                documents = SimpleDirectoryReader('data').load_data()
                service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
                index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
            
                response = index.query(query)
                st.success(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
