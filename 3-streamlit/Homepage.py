import streamlit as st

def app():
    st.title("Welcome to the Predictive Maintenance App!")
    st.write(
        """
        **Project Summary**: 
        This application demonstrates a predictive maintenance workflow 
        and a RAG-based chatbot that answers maintenance-related questions
        using domain documents.
        
        - Page 1: Introduction (You are here!)
        - Page 2: Predictive Maintenance
        - Page 3: RAG Chatbot
        """
    )