import streamlit as st
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import pandas as pd
import ot
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
    n_agents = st.number_input("Enter the number of agents (2-10)", min_value=2, max_value=10, value=2)
with col2:
    n_items = st.number_input("Enter the number of items (2-20)", min_value=2, max_value=20, value=2)

# Button to upload preferences from CSV file
uploaded_file = st.file_uploader("Upload a CSV file of preferences (optional)", type="csv")
if uploaded_file is not None:
    # Reading the CSV file into a pandas DataFrame
    preferences = pd.read_csv(uploaded_file, header=None).to_numpy()
    # Checking if the DataFrame conforms to the expected size
    if preferences.shape != (n_agents, n_items):
        st.warning(f"The uploaded file does not have the expected size ({n_agents} rows x {n_items} columns).")

# Button to generate random preferences
if st.button("Generate random preferences"):
    # Generating a table of random preferences
    preferences = np.random.uniform(-10, 10, size=(n_agents, n_items))
    # Displaying the preferences table
    st.write("Randomly generated preferences:")
    st.write(preferences)

# Get heuristics
heurs = np.zeros((20,10))
for i in range(20):
  A, M = get_cost_matrix(i, preferences)
  heurs[i] = find_barycenter(A, M)

# Display completion message
st.write("Stage 2 completed: heuristics found!")

# Set up optimization
ps = nn.Parameter(torch.from_numpy(heurs[:,:-1]))
aten = torch.from_numpy(preferences).requires_grad_(False)
nsteps = st.slider("Select number of optimization steps", 1000, 50000, 10000)

# Button to get WEF1+PO Allocation
if st.button("Get WEF1+PO Allocation"):
    optimizer = Adam([ps])
    all_max_prox = torch.inf
    saved_args = None
    
    # Display progress bar
    with st.spinner("Running optimization..."):
        for i in tqdm(range(nsteps)):
            loss = compute_loss(ps)
            loss.backward()
            optimizer.step()
            
            # Compute allocation and max approx
            if i % 1000 == 0:
                prs = 1 - ps.sum(axis=1)
                all_ps = torch.cat([ps, prs.unsqueeze(-1)], axis=-1)
                intargs = torch.argmax(all_ps,axis=1)
                intps = torch.zeros(all_ps.shape)

                for i in range(20):
                    intps[i][intargs[i]] = 1

                intE = torch.zeros((10,10))

                for j in range(10):
                    for k in range(10):
                        intE[j][k] = torch.max(torch.tensor([0.0]), sum(aten[j] * intps[:, k]) / (k + 1) - sum(aten[j] * intps[:, j]) / (j + 1))

            max_approx = -torch.inf
            for i in range(10):
                for j in range(10):
                    if intE[i][j] > 0:
                        if max(aten[i] * intps[:, j]) / (j + 1) < intE[i][j]:
                            approx = intE[i][j] / (max(aten[i] * intps[:, j]) / (j + 1))
                            if approx != torch.inf and max_approx < approx:
                                max_approx = approx
                    print(f"                 Approx = {max_approx}")

            if all_max_prox > max_approx:
                all_max_prox = max_approx
                saved_args = intargs
        
        # progress_bar.progress((i + 1) / 10000)
    
    st.write("Stage 3 completed: heuristics found!")
    # Displaying the allocation
    st.write("WEF1+PO Allocation:")
    st.write(saved_args.detach().numpy().item())
    st.write(f"Estimated epsilon-WEF1: {all_max_prox}")
    # st.write(f"Total utilitarian welfare: {}")
