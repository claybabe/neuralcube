# 2025 - copyright - all rights reserved - clayton thomas baber

from tqdm import tqdm
from cube import Cube
from model import RubikDistancePredictor
from torch import tensor, argsort, float32
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
    
    models = []
    model_paths = []
    for _ in range(int(input("number of models? "))):
        model_path = filedialog.askopenfilename()
        model_paths.append(model_path)
        models.append(RubikDistancePredictor.load_from_checkpoint(model_path, map_location='cpu'))

    cube = Cube()
    cases = set()
    for orbit in Cube.orbits:
        cube.reset()
        cube.algo(orbit[:6])
        cases.add(cube.getState())
    total = len(cases)
    spacer = len(str(total))
    success = 0
    dead = 0
    timeout = 0
    
    with tqdm(total=total, desc="Evaluating") as pbar:
        for case in cases:
            cube.setState(case)
            history = set()
            history.add(cube.getState())
            
            for i in range(21):
                probe =  tensor(sprout(cube), dtype=float32)

                predictions = tensor([0]*18, dtype=float32)
                for model in models:
                    predictions += model(probe).squeeze()
                
                chosen = None
                choices = argsort(predictions)

                for choice in choices:
                    test = Cube(cube)
                    test.act(choice)
                    if test.getState() not in history:
                        chosen = choice
                        break
                
                if chosen is None:
                    dead += 1
                    cube.reset()
                    break

                cube.act(choice)
                
                if cube.isSolved():
                    success += 1
                    break

                history.add(cube.getState())
            
            if not cube.isSolved():
                timeout += 1

            pbar.set_postfix(accu=f"{success/total:6.4f}", dead=f"{dead:0{spacer}d}", solv = f"{success:0{spacer}d}", tout = f"{timeout:0{spacer}d}")
            pbar.update(1)
    print(f"done evaluating: {success/total} accuracy.\n", "\n".join(model_paths))
