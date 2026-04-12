# 2025 - copyright - all rights reserved - clayton thomas baber

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
  for _ in range(int(input("number of models? "))):
    model_path = filedialog.askopenfilename(initialdir="lightning_logs")
    model_paths.append(model_path)
  
  model = RubikEnsemble(model_paths)

  cube = Cube()
  cases = set()
  for orbit in Cube.orbits:
    cube.reset()
    cube.algo(orbit[:20])
    cases.add(cube.getState())
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
  print(f"done evaluating: {outcomes['success']/total} accuracy. {end_time - start_time}\n", "\n".join(model_paths))
