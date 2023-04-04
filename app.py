import math

import streamlit as st
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils import *

# Setting the page title
st.set_page_config(page_title="Agent-Item Preferences", page_icon=":clipboard:")

# Setting the sidebar
st.sidebar.title("Note")
st.sidebar.write("This tool is only for demonstration purpose")

# Setting the main page
st.title("Agent-Item Preferences")

# Getting user inputs for number of agents and items
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        n_agents = st.number_input("Enter the number of agents (2-20)", min_value=2, max_value=20, value=2, step=1)

    with col2:
        n_items = st.number_input("Enter the number of items (4-40, multiples of 2)", min_value=4, max_value=40, value=4, step=2)

    submit_button = st.form_submit_button("Submit")

# Generate or upload preferences based on user input
if submit_button:
    with st.form("preference_form"):
        col1, col2 = st.columns(2)

        with col1:
            random_seed = st.number_input("Enter the random seed (optional)", min_value=0)

        with col2:
            uploaded_file = st.file_uploader("Upload preference matrix (CSV file)", type="csv")

        generate_button = st.form_submit_button("Generate preferences")

    if generate_button:
        with st.spinner("Generating preferences..."):
            if n_items < n_agents:
                st.warning("Number of items must be greater than or equal to the number of agents.")
                n_items = n_agents * 2

            if n_items % 2 != 0:
                st.warning("Number of items must be multiples of 2.")
                n_items = n_items + 2 - n_items % 2

            n_agents = int(n_agents)
            n_items = int(n_items)

            if random_seed is not None:
                np.random.seed(random_seed)

            if uploaded_file is not None:
                with st.spinner("Uploading preferences..."):
                    # Reading the CSV file into a pandas DataFrame
                    preferences = pd.read_csv(uploaded_file, header=None).to_numpy()
                    # Checking if the DataFrame conforms to the expected size
                    if preferences.shape != (n_agents, n_items):
                        st.warning(f"The uploaded file does not have the expected size ({n_agents} rows x {n_items} columns).")
                    # Checking if the DataFrame has positive or negative utilities for all agents
                    elif not np.all(np.logical_or(preferences >= 0, preferences <= 0)):
                        st.warning("The uploaded file must have either positive or negative utilities for all agents.")
                    else:
                        st.write("Uploaded preferences:")
                        st.write(preferences)
                    
                    train(n_items, n_agents, preferences)
            else:
                preferences = np.concatenate((np.random.uniform(0,10,(n_agents, math.floor(n_items / 2))), np.random.uniform(-10,0,(n_agents, math.ceil(n_items / 2)))), axis=1)
                st.write("Randomly generated preferences:")
                st.write(preferences)
                train(n_items, n_agents, preferences)

            # Button to regenerate random preferences
            if st.button("Regenerate random preferences"):
                preferences = np.concatenate((np.random.uniform(0,10,(n_agents, math.floor(n_items / 2))), np.random.uniform(-10,0,(n_agents, math.ceil(n_items / 2)))), axis=1)
                st.write("Randomly generated preferences:")
                st.write(preferences)
                train(n_items, n_agents, preferences)
