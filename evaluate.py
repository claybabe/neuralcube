# 2023 - copyright - all rights reserved - clayton thomas baber

from cube import Cube
from model import BrownianAntipodalNavigator
from torch import Generator, randint, tensor, argmax
from torch.nn.functional import one_hot
from tkinter import Tk, filedialog

if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    model_path = filedialog.askopenfilename()

    navigator = BrownianAntipodalNavigator()
    navigator.load_from_checkpoint(model_path)

    cube = Cube()
    goal = cube.toColorHot()

    generator = Generator()
    generator.manual_seed(12345678)

    results = {"solved": 0, "cyclefail": 0, "timeout": 0, "longest": 0}
    for i in range(100):
        cube.act(int(randint(0, 17, (1,), generator=generator)))
        pig = Cube()
        pig.setState(cube.getState())
        start = pig.toColorHot()
        history = set()
        history.add(pig.getState())
        broke = False
        length = 0
        for j in range(20):
            length += 1
            x = tensor(start + cube.toColorHot() + goal).float()
            x = int(argmax(navigator(x)))
            pig.act(x)
            if pig.getState() in history:
                results["cyclefail"] += 1
                broke = True
                break
            if pig.isSolved():
                broke = True
                break
            history.add(pig.getState())
        if pig.isSolved():
            results["solved"] += 1
        else:
            if not broke:
                results["timeout"] += 1
        results["longest"] = max(results["longest"], length)
    print(results)
