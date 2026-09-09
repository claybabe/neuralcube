# 2026 - copyright - all rights reserved - clayton thomas baber

import os
import glob
import numpy as np
from time import time
from tqdm import tqdm
from cube import Cube
from model import RubikDistancePredictor, RubikEnsemble
from torch import tensor, argsort, float32
from tkinter import Tk, filedialog
from collections import defaultdict

if __name__ == "__main__":
  root = Tk()
  root.withdraw()
  
  model_paths = []
  all_endpoints = []

  num_models = int(input("number of models? "))
  for _ in range(num_models):
    run_dir = filedialog.askdirectory(initialdir="checkpoints", title="Select Run Directory")
    
    if not run_dir:
      continue

    # 1. Find Checkpoint File (.ckpt)
    ckpts = glob.glob(os.path.join(run_dir, "*.ckpt"))
    if not ckpts:
      raise FileNotFoundError(f"No .ckpt file found in {run_dir}")
    
    # Filter out 'last.ckpt' if specific accuracy checkpoints exist
    best_ckpts = [c for c in ckpts if not c.endswith("last.ckpt")]
    chosen_ckpt = sorted(best_ckpts)[-1] if best_ckpts else ckpts[0]
    model_paths.append(chosen_ckpt)

    # 2. Load Endpoints (.npy)
    endpoints_path = os.path.join(run_dir, "endpoints.npy")
    if os.path.exists(endpoints_path):
      run_endpoints = np.load(endpoints_path)
      all_endpoints.append(run_endpoints)
    else:
      print(f"Warning: 'endpoints.npy' not found in {run_dir}")

  if not all_endpoints:
    raise FileNotFoundError("No endpoints found across selected run directories.")

  # Concatenate arrays if multiple runs selected, deduplicate, and convert to list
  concatenated_endpoints = np.vstack(all_endpoints)
  cases = set(tuple(ep) for ep in concatenated_endpoints)
  
  model = RubikEnsemble(model_paths)

  cube = Cube()
  total = len(cases)
  spacer = len(str(total))
  
  outcomes = {"success":0, "dead":0, "timeout":0, "saved":0}
  memo = {}

  start_time = time()
  with tqdm(total=total, desc="Evaluating") as pbar:
    for checked, case in enumerate(cases, start=1):
      cube.setState(case)
      history = defaultdict(int)
      
      outcome = None
      for i in range(42):
        state = cube.getState()
        probe =  tensor(cube.getProbe(), dtype=float32)

        predictions = model(probe).squeeze()
        
        choices = argsort(predictions)

        choice = history[state]

        if choice < 18:
          action = choices[choice]
          history[state] += 1
        else:
          outcome = "dead"
          break

        cube.act(action)
        state = cube.getState()
        if state in memo:
          outcomes["saved"] += 1
          outcome = memo[state]
          break

        if cube.isSolved():
          outcome = "success"
          break
      
      if outcome is None:
        outcome = "timeout"

      outcomes[outcome] += 1
      for visited in history:
        memo[visited] = outcome

      pbar.set_postfix(
        accu = f"{outcomes['success']/checked:6.4f}",
        dead = f"{outcomes['dead']:0{spacer}d}",
        save = f"{outcomes['saved']:0{spacer}d}",
        solv = f"{outcomes['success']:0{spacer}d}",
        tout = f"{outcomes['timeout']:0{spacer}d}"
      )
      pbar.update(1)
    

  end_time = time()
  print(f"done evaluating: {outcomes['success']/total} accuracy. finished in {end_time - start_time}\n", "\n".join(model_paths))
