# 2025 - copyright - all rights reserved - clayton thomas baber

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
    success = 0
    for orbit in Cube.orbits:
        cube.reset()    
        cube.algo(orbit[:6])
        history = set()
        history.add(cube.getState())
        for i in range(21):
            probe =  tensor(sprout(cube), dtype=float32)
            predictions = model(probe)
            choice = argmin(predictions)
            #print(predictions)
            cube.act(choice)
            if cube.getState() in history:
                print("unfortunately there was a CYCLE !!!!!")
                cube.reset()
                break
            if cube.isSolved():
                print(f"SOLVED in {i+1}")
                success += 1
                break
            history.add(cube.getState())
            #print(i)
        if not cube.isSolved():
            print("COULDNT SOLVE IN 21")
    print(f"successfull {success/len(Cube.orbits)} times")
