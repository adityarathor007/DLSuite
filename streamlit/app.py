import streamlit as st

# Create tabs
tab1, tab2, tab3 = st.tabs(["Home", "Analytics", "Settings"])

# Add content to each tab
with tab1:
    st.header("Home")
    st.write("Welcome to the home page!")

with tab2:
    st.header("Analytics")
    st.line_chart([1, 2, 3, 4])

with tab3:
    st.header("Settings")
    st.text_input("Enter your name:")
