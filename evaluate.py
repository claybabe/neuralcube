# 2023 - copyright - all rights reserved - clayton thomas baber

from cube import Cube
from model import BrownianAntipodalNavigator
from torch import Generator, randint, tensor, argmax, rand, ones
from torch.nn.functional import one_hot
from tkinter import Tk, filedialog
import torch

def getChoice(pig, pigstory, sample):
    turtle = Cube()
    checks = 0
    for choice in getChoices(sample):
        turtle.setState(pig)
        turtle.act(choice)
        if turtle.getState() in pigstory:
            checks += 1
        else:
            return choice, checks
    return False, checks

def getChoices(sample):
    n = tensor(18)
    mask = ones(n)
    guesses = []
    for _ in range(int(n)):
        guess = argmax(sample * mask)
        guesses.append(int(guess))
        mask += one_hot(guess, n) * -1
    return guesses

if __name__ == "__main__":
    torch.manual_seed(0)
    root = Tk()
    root.withdraw()
    models = []
    for _ in range(int(input("number of models? "))):
        model_path = filedialog.askopenfilename()
        navigator = BrownianAntipodalNavigator()
        navigator.load_from_checkpoint(model_path)
        models.append(navigator)

    cube = Cube()
    goal = cube.toColorHot()

    generator = Generator()
    generator.manual_seed(12345678)

    results = {"solved": 0, "cyclefail": 0, "timeout": 0, "longest": 0, "checks": 0}
    choices = {}
    for i in range(42):
        cube.act(int(randint(0, 17, (1,), generator=generator)))
        pig = Cube()
        pig.setState(cube.getState())
        start = pig.toColorHot()
        history = set()
        broke = False
        length = 0
        actions = []
        for j in range(42):
            length += 1
            history.add(pig.getState())

            x = tensor(start + pig.toColorHot() + goal).float()
            a = tensor([0]*18).float()
            for model in models:
                a += model(x)

            x, checks = getChoice(pig.getState(), history, a)
            results["checks"] += checks
            if x is False:
                results["cyclefail"] += 1
                print("cyclefail")
                broke = True
                break
            if x not in choices:
                choices[x] = 0
            choices[x] += 1
            pig.act(x)
            actions.append(x)
            if pig.isSolved():
                broke = True
                break
        if pig.isSolved():
            results["solved"] += 1
        else:
            if not broke:
                results["timeout"] += 1
                print("timeout")
        results["longest"] = max(results["longest"], length)
        print(actions, "\n", pig)
    print(results, choices)
