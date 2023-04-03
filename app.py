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
col1, col2 = st.columns(2)

with col1:
    n_agents = st.number_input("Enter the number of agents (2-20)", min_value=2, max_value=20, value=2, step=1)
    
with col2:
    n_items = st.number_input("Enter the number of items (4-40, multiples of 2)", min_value=4, max_value=40, value=4, step=2)

if n_items < n_agents:
    st.warning("Number of items must be greater than or equal to the number of agents.")
    n_items = n_agents * 2
    
if n_items % 2 != 0:
    st.warning("Number of items must be multiples of 2.")
    n_items = n_items + 2 - n_items % 2
    
n_agents = int(n_agents)
n_items = int(n_items)

preferences = np.concatenate((np.random.uniform(0,10,(n_agents, math.floor(n_items / 2))), np.random.uniform(-10,0,(n_agents, math.ceil(n_items / 2)))), axis=1)

# Button to upload preferences from CSV file
uploaded_file = st.file_uploader("Upload a CSV file of preferences (optional)", type="csv")
if uploaded_file is not None:
    # Reading the CSV file into a pandas DataFrame
    preferences = pd.read_csv(uploaded_file, header=None).to_numpy()
    # Checking if the DataFrame conforms to the expected size
    if preferences.shape != (n_agents, n_items):
        st.warning(f"The uploaded file does not have the expected size ({n_agents} rows x {n_items} columns).")
    # Checking if the DataFrame has positive or negative utilities for all agents
    elif not np.all(np.logical_or(preferences >= 0, preferences <= 0)):
        st.warning("The uploaded file must have either positive or negative utilities for all agents.")

# Button to generate random preferences
if st.button("Regenerate random preferences"):
    # Generating a table of random preferences
    preferences = np.concatenate((np.random.uniform(0,10,(n_agents, math.floor(n_items / 2))), np.random.uniform(-10,0,(n_agents, math.ceil(n_items / 2)))), axis=1)
    # Displaying the preferences table
    st.write("Randomly generated preferences:")

st.write(preferences)

# Get heuristics
heurs = np.zeros((n_items,n_agents))
for i in range(n_items):
  A, M = get_cost_matrix(i, preferences, n_agents)
  heurs[i] = find_barycenter(A, M)

# Display completion message
st.write("Stage 2 completed: heuristics found!")

# Set up optimization
ps = nn.Parameter(torch.from_numpy(heurs[:,:-1]))
aten = torch.from_numpy(preferences).requires_grad_(False)
nsteps = st.slider("Select number of optimization steps", 2000, 50000, 10000)
alpha = st.slider('Choose an alpha value', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
st.write('Selected alpha:', alpha)

# Button to get WEF1+PO Allocation
if st.button("Get WEF1+PO Allocation"):
    optimizer = Adam([ps])
    all_max_prox = torch.inf
    saved_args = None
    
    # Display progress bar
    with st.spinner("Running optimization..."):
        progress_bar = st.progress(0)
        
        stop_button = st.button("Stop Optimization")

        for i in tqdm(range(nsteps)):
            loss = compute_loss(ps, aten, n_agents, alpha)
            loss.backward()
            optimizer.step()
            
            if stop_button:
                # if the stop button has been clicked, break out of the loop
                st.warning("Optimization stopped by user")
                break
                    
            # Compute allocation and max approx
            if i % 1000 == 0:
                print("loss",loss)
                prs = 1 - ps.sum(axis=1)
                all_ps = torch.cat([ps, prs.unsqueeze(-1)], axis=-1)
                intargs = torch.argmax(all_ps,axis=1)
                intps = torch.zeros(all_ps.shape)

                for i in range(n_items):
                    intps[i][intargs[i]] = 1

                intE = torch.zeros((n_agents,n_agents))

                for j in range(n_agents):
                    for k in range(n_agents):
                        intE[j][k] = torch.max(torch.tensor([0.0]), sum(aten[j] * intps[:, k]) / (k + 1) - sum(aten[j] * intps[:, j]) / (j + 1))

                max_approx = -torch.inf
                for i in range(n_agents):
                    for j in range(n_agents):
                        if intE[i][j] > 0:
                            if max(aten[i] * intps[:, j]) / (j + 1) < intE[i][j]:
                                approx = intE[i][j] / (max(aten[i] * intps[:, j]) / (j + 1))
                                if approx != torch.inf and max_approx < approx:
                                    max_approx = approx
                print(f"                 Approx = {max_approx}")

                if all_max_prox > max_approx:
                    all_max_prox = max_approx
                    saved_args = intargs
        
            progress_bar.progress((i + 1) / nsteps)

        progress_bar.empty()
    
    st.write("Stage 3 completed!")
    # Displaying the allocation
    st.write("WEF1+PO Allocation:")
    st.write(saved_args.detach().numpy().tolist())
    st.write(f"Estimated epsilon-WEF1: {all_max_prox}")
    # st.write(f"Total utilitarian welfare: {}")
