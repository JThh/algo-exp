from torch.optim import Adam
from tqdm import tqdm

import numpy as np
import ot
import torch
import torch.nn as nn
import streamlit as st

# Stage 2: Find heuristics via Wasserstein barycenters
def get_cost_matrix(item, prefs, nagents):
  # Construct cost matrix M
  a = np.zeros((nagents,nagents))
  # Populate individual allocation measures
  for i in range(nagents):
    a[i][i] = 1

  A = a.T

  envy = np.zeros((nagents,nagents,nagents))
  M = np.zeros((nagents,nagents,))

  # Get envy
  for i in range(nagents):
    p = A[i]
    for j in range(nagents):
      for k in range(nagents):
        envy[i][j][k] = prefs[j][item] * p[k] / (k + 1) - prefs[j][item] * p[j] / (j + 1)

  print("Average envy for n extreme allocations", np.mean(np.sum(envy**2, axis=(1, 2))))

  # Get cost
  for i in range(nagents):
    for j in range(i, nagents):
      M[j][i] = M[i][j] = np.sum((envy[i] - envy[j])**2, axis=(0, 1))

  M /= M.max()
  return A, M

def find_barycenter(A, M, reg=1e-2, numItermax=100000):
  weights = np.array([1 / A.shape[1]] * A.shape[1])
  print(A.shape[1], weights)
  bary_wass = ot.bregman.barycenter(A, M, reg, weights=weights, numItermax=numItermax)
  return bary_wass

def compute_loss(ps, aten, nagents, alpha=0.01):
  prs = 1 - ps.sum(axis=1)
  all_ps = torch.concat([ps, prs.unsqueeze(-1)], axis=-1)

  E = torch.zeros((nagents,nagents))

  for j in range(nagents):
    for k in range(nagents):
      E[j][k] = torch.max(torch.tensor([0.0]), sum(aten[j] * all_ps[:, k]) / (k + 1) - sum(aten[j] * all_ps[:, j]) / (j + 1))

  V = torch.zeros(nagents)

  for i in range(nagents):
    V[i] = sum(aten[i] * all_ps[:, i]) / (i + 1)

  reg = sum([torch.pow(_p, 2) if _p < 0.5 else torch.pow(1 - _p, 2) for _ps in all_ps for _p in _ps])

  J = torch.sum(E**2,axis=[0,1]) + alpha * sum(V)**2 + reg  # Only for PO

  return J

def get_WEF1(intps, n_agents, aten):
    intE = torch.zeros((n_agents,n_agents), requires_grad=False)

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
    # print(f"                 Approx = {max_approx}")
    return max_approx

def train(n_items, n_agents, preferences):
    heurs = np.zeros((n_items,n_agents))
    for i in range(n_items):
        A, M = get_cost_matrix(i, preferences, n_agents)
        heurs[i] = find_barycenter(A, M)

    # Display completion message
    st.write("Stage 2 completed: heuristics found!")
    aten = torch.from_numpy(preferences).requires_grad_(False)

    if st.button("Get Heuristic Allocation"):
        heur_intargs = torch.argmax(torch.from_numpy(heurs),axis=1)
        heur_intps = torch.zeros(heurs.shape, requires_grad=False)
        for i in range(n_items):
            heur_intps[i][heur_intargs[i]] = 1
        st.write(f"Heuristic allocation: {get_WEF1(heur_intps, n_agents, aten)}")

    # Set up optimization
    ps = nn.Parameter(torch.from_numpy(heurs[:,:-1]))
    nsteps = st.slider("Select number of optimization steps", 2000, 50000, 10000)
    alpha = st.slider('Choose an alpha value', min_value=0.0, max_value=1.0, value=0.01, step=0.01)
    st.write('Selected alpha:', alpha)

    # Button to get WEF1+PO Allocation
    if st.button("Get WEF1+PO Allocation"):
        optimizer = Adam([ps])
        all_max_prox = torch.inf
        saved_args = None
        saved_PO = False
        
        # Display progress bar
        with st.spinner("Running optimization..."):
            progress_bar = st.progress(0)
            stop_button = st.button("Stop Optimization")

            for step in tqdm(range(nsteps)):
                loss = compute_loss(ps, aten, n_agents, alpha)
                loss.backward()
                optimizer.step()
                
                if stop_button:
                    # if the stop button has been clicked, break out of the loop
                    st.warning("Optimization stopped by user")
                    break
                        
                # Compute allocation and max approx
                if step % 1000 == 0:
                    print("loss",loss)
                    prs = 1 - ps.sum(axis=1)
                    all_ps = torch.cat([ps, prs.unsqueeze(-1)], axis=-1)
                    intargs = torch.argmax(all_ps, axis=1)
                    intps = torch.zeros(all_ps.shape, requires_grad=False)
                    for i in range(n_items):
                        intps[i][intargs[i]] = 1
                    max_approx = get_WEF1(intps, n_agents, aten)
                    print(f"                 Approx = {max_approx}")

                    # if max_approx == -torch.inf:
                    #     saved_args = intargs
                    #     saved_PO = True
                    #     for j in range(10):
                    #         if torch.any(aten[j] * intps[:, j] < 0, 0):
                    #             saved_PO = False
                                
                    #     st.write("WEF1 found! Break out of the loop...")
                    #     break

                    if max_approx != -torch.inf and all_max_prox > max_approx:
                        all_max_prox = max_approx
                        saved_args = intargs
                        saved_PO = True
                        for j in range(n_agents):
                            for i in range(n_items):
                                if aten[j, i] * intps[i, j] < 0:
                                    if torch.any(aten[:,i] >= 0):
                                        saved_PO = False

                progress_bar.progress((step + 1) / nsteps)

            progress_bar.empty()
        
        st.write("Stage 3 completed!")
        # Displaying the allocation
        st.write("WEF1+PO Allocation:")
        if saved_args is None:
            st.write(intargs.detach().numpy().tolist())
            st.write(f"Estimated epsilon-WEF1: {max_approx}")
            # st.write(f"PO: {saved_PO}")
        else:
            st.write(saved_args.detach().numpy().tolist())
            st.write(f"Estimated epsilon-WEF1: {all_max_prox}")
            st.write(f"PO: {saved_PO}")
        # st.write(f"Total utilitarian welfare: {}")