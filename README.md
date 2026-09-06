# neuralcube

neural simulations of rubik's cube

## Debian 13
install prereqs `sudo apt install git python3-venv python3-tk`

clone repo `git clone https://github.com/claybabe/neuralcube`

`cd neuralcube`

create environment `python3 -m venv env`

activate environment `source env/bin/activate`

install requirements `pip install -r requirements.txt`

test the cube `python cube.py`

build the data `python dataset.py`

train the model `python model.py`

evaluate ensembles `python evaluate.py`

interactive visual simulation `python simulate.py`


qaz wsx edc rfv tgb yhn - perform single action

uj ik ol                - rotate the entire cube

Home                    - stop solve and reset

End                     - stop solve

Enter                   - start solve attempt

. (period)              - single step of solve

1                       - single step to largest probed distance

2                       - go to endpoint of path

3                       - increment to next endpoint



# Acknowledgments & Data Credits

This project utilizes dataset assets generated during the God's Number is 20 computational proof:

    Dataset: htm4.zip (20-move Half-Turn Metric positions)

    Credits: Tomas Rokicki, Herbert Kociemba, Morley Davidson, and John Dethridge

    Source: Cube20.org

We are deeply grateful to Tomas Rokicki and the Cube20 team for making these fundamental combinatorial assets available to the Rubik's Cube research community.