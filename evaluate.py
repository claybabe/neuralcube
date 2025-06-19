# 2025 - copyright - all rights reserved - clayton thomas baber

from tqdm import tqdm
from cube import Cube
from model import RubikDistancePredictor
from torch import tensor, argmin, float32
from tkinter import Tk, filedialog

def sprout(cube):
    sprouts = []
    for i in range(18):
        seedling = Cube(cube)
        seedling.act(i)
        sprouts.append(seedling.toOneHot())
    return sprouts

if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    
    model_path = filedialog.askopenfilename()
    model = RubikDistancePredictor.load_from_checkpoint(model_path, map_location='cpu')

    cube = Cube()
    cases = set()
    for orbit in Cube.orbits:
        cube.reset()
        cube.algo(orbit[:6])
        cases.add(cube.getState())
    total = len(cases)
    spacer = len(str(total))
    success = 0
    cycle = 0
    timeout = 0
    
    with tqdm(total=total, desc="Evaluating") as pbar:
        for case in cases:
            cube.setState(case)
            history = set()
            history.add(cube.getState())
            
            for i in range(21):
                probe =  tensor(sprout(cube), dtype=float32)
                predictions = model(probe)
                choice = argmin(predictions)
                #print(predictions)
                cube.act(choice)
                if cube.getState() in history:
                    cycle += 1
                    cube.reset()
                    break
                if cube.isSolved():
                    success += 1
                    break
                history.add(cube.getState())
            
            if not cube.isSolved():
                timeout += 1

            pbar.set_postfix(cycl = f"{cycle:0{spacer}d}", timo = f"{timeout:0{spacer}d}", accu=f"{success/total:6.4f}")
            pbar.update(1)
    print(f"done evaluating {model_path}")
