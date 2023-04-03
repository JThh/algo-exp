import numpy as np
import ot
import torch
from torch.optim import Adam
from tqdm import tqdm

# Stage 2: Find heuristics via Wasserstein barycenters
def get_cost_matrix(item, prefs):
  # Construct cost matrix M
  a = np.zeros((10,10))
  # Populate individual allocation measures
  for i in range(10):
    a[i][i] = 1

  A = a.T

  envy = np.zeros((10,10,10))
  M = np.zeros((10,10))

  # Get envy
  for i in range(10):
    p = A[i]
    for j in range(10):
      for k in range(10):
        envy[i][j][k] = prefs[j][item] * p[k] / (k + 1) - prefs[j][item] * p[j] / (j + 1)

  print("Average envy for n extreme allocations", np.mean(np.sum(envy**2, axis=(1, 2))))

  # Get cost
  for i in range(10):
    for j in range(i, 10):
      M[j][i] = M[i][j] = np.sum((envy[i] - envy[j])**2, axis=(0, 1))

  M /= M.max()
  return A, M

def find_barycenter(A, M, reg=1e-2, weights=np.array([0.1] * 10), numItermax=100000):
  bary_wass = ot.bregman.barycenter(A, M, reg, weights=weights, numItermax=numItermax)
  return bary_wass

def compute_loss(ps, aten):
  prs = 1 - ps.sum(axis=1)
  all_ps = torch.concat([ps, prs.unsqueeze(-1)], axis=-1)

  E = torch.zeros((10,10))

  for j in range(10):
    for k in range(10):
      E[j][k] = torch.max(torch.tensor([0.0]), sum(aten[j] * all_ps[:, k]) / (k + 1) - sum(aten[j] * all_ps[:, j]) / (j + 1))

  V = torch.zeros(10)

  for i in range(10):
    V[i] = sum(aten[i] * all_ps[:, i]) / (i + 1)

  reg = sum([torch.pow(_p, 2) if _p < 0.5 else torch.pow(1 - _p, 2) for _ps in all_ps for _p in _ps])

  J = - sum(V)**2 + reg  # Only for PO

  return J

def train(params, aten, nsteps=10000):
  optimizer = Adam([params])
  all_max_prox = torch.inf
  saved_args = None
  for i, _ in enumerate(tqdm(list(range(nsteps)))):
    loss = compute_loss(params)
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
      print(f"#==================# Steps = {i} #==================#")
      prs = 1 - params.sum(axis=1)
      all_ps = torch.concat([params, prs.unsqueeze(-1)], axis=-1)
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

  return saved_args