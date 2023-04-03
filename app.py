import streamlit as st
import numpy as np
import pandas as pd

# Setting the page title
st.set_page_config(page_title="Agent-Item Preferences", page_icon=":clipboard:")

# Setting the sidebar
st.sidebar.title("Note")
st.sidebar.write("This tool is only for demonstration purpose")

# Setting the main page
st.title("Agent-Item Preferences")

# Getting user inputs for number of agents and items
col1, col2 = st.columns(2)
with col1:
    n_agents = st.number_input("Enter the number of agents (2-10)", min_value=2, max_value=10, value=2)
with col2:
    n_items = st.number_input("Enter the number of items (2-20)", min_value=2, max_value=20, value=2)

# Button to upload preferences from CSV file
uploaded_file = st.file_uploader("Upload a CSV file of preferences (optional)", type="csv")
if uploaded_file is not None:
    # Reading the CSV file into a pandas DataFrame
    preferences_df = pd.read_csv(uploaded_file, header=None)
    # Checking if the DataFrame conforms to the expected size
    if preferences_df.shape != (n_agents, n_items):
        st.warning(f"The uploaded file does not have the expected size ({n_agents} rows x {n_items} columns).")

# Button to generate random preferences
if st.button("Generate random preferences"):
    # Generating a table of random preferences
    preferences = np.random.uniform(-10, 10, size=(n_agents, n_items))
    # Displaying the preferences table
    st.write("Randomly generated preferences:")
    st.write(preferences)

# Button to get WEF1+PO Allocation
if st.button("Get WEF1+PO Allocation"):
    # Showing a progress bar
    progress_bar = st.progress(0)
    for i in range(10000):
        progress_bar.progress((i + 1) / 10000)
    # Displaying the allocation
    st.write("WEF1+PO Allocation:")
    st.write("To be implemented...")
