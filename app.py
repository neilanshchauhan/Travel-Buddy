import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Load the GROQ
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Travel Buddy")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-it")

# Create a prompt template for the travel assistant
prompt_template = ChatPromptTemplate.from_template(
    """
    You are a helpful and knowledgeable travel assistant. Your task is to help users plan their trips by providing personalized travel itineraries and suggesting local attractions based on their preferences. 
    Always consider the user's input on location type and budget when making recommendations.
    First tell the recommended places according to the user's input, and then explain why these places are suitable.
    Location: {location}
    Type: {loc_type}
    Budget: {budget}

    Suggestions:
    """
)

# Get user input for the travel assistant
location = st.text_input("Enter a place you wish to visit")

loc_type = st.selectbox(
    "Select the type of place you prefer:",
    ("Peaceful", "Food Hub", "Nature", "Historical", "Spiritual", "Beautiful", "Family")
)

budget = st.selectbox(
    "Select your budget range:",
    ("High", "Moderate", "Low")
)

# Create an output parser
output_parser = StrOutputParser()

# Chain the prompt template, language model, and output parser
chain = prompt_template | llm | output_parser

# Generate and display the response based on user input
if st.button("Submit"):
    if location:
        response = chain.invoke({
            'location': location,
            'loc_type': loc_type,
            'budget': budget
        })
        st.write(response)
    else:
        st.write("Please enter a location to get recommendations.")
