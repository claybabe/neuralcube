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


# Tests

## cube.py
    --- Testing Notations ---
    Notation L1 | Cycle Length: 4 | PASSED
    Notation L2 | Cycle Length: 2 | PASSED
    Notation L3 | Cycle Length: 4 | PASSED
    Notation U1 | Cycle Length: 4 | PASSED
    Notation U2 | Cycle Length: 2 | PASSED
    Notation U3 | Cycle Length: 4 | PASSED
    Notation F1 | Cycle Length: 4 | PASSED
    Notation F2 | Cycle Length: 2 | PASSED
    Notation F3 | Cycle Length: 4 | PASSED
    Notation R3 | Cycle Length: 4 | PASSED
    Notation R2 | Cycle Length: 2 | PASSED
    Notation R1 | Cycle Length: 4 | PASSED
    Notation D3 | Cycle Length: 4 | PASSED
    Notation D2 | Cycle Length: 2 | PASSED
    Notation D1 | Cycle Length: 4 | PASSED
    Notation B3 | Cycle Length: 4 | PASSED
    Notation B2 | Cycle Length: 2 | PASSED
    Notation B1 | Cycle Length: 4 | PASSED
    All notation tests passed. Unique states across notation cycles: 19

    --- Testing Rotations ---
    Rotation  0 | Cycle Length: 1 | PASSED
    Rotation  1 | Cycle Length: 4 | PASSED
    Rotation  2 | Cycle Length: 2 | PASSED
    Rotation  3 | Cycle Length: 4 | PASSED
    Rotation  4 | Cycle Length: 4 | PASSED
    Rotation  5 | Cycle Length: 2 | PASSED
    Rotation  6 | Cycle Length: 4 | PASSED
    Rotation  7 | Cycle Length: 4 | PASSED
    Rotation  8 | Cycle Length: 2 | PASSED
    Rotation  9 | Cycle Length: 4 | PASSED
    Rotation 10 | Cycle Length: 3 | PASSED
    Rotation 11 | Cycle Length: 3 | PASSED
    Rotation 12 | Cycle Length: 2 | PASSED
    Rotation 13 | Cycle Length: 2 | PASSED
    Rotation 14 | Cycle Length: 2 | PASSED
    Rotation 15 | Cycle Length: 2 | PASSED
    Rotation 16 | Cycle Length: 2 | PASSED
    Rotation 17 | Cycle Length: 3 | PASSED
    Rotation 18 | Cycle Length: 3 | PASSED
    Rotation 19 | Cycle Length: 3 | PASSED
    Rotation 20 | Cycle Length: 3 | PASSED
    Rotation 21 | Cycle Length: 3 | PASSED
    Rotation 22 | Cycle Length: 3 | PASSED
    Rotation 23 | Cycle Length: 2 | PASSED
    All rotation tests passed. Unique state representations across non-identity rotations: 24

    --- Testing Paths in htm4.zip ---
    Testing paths in 'htm4.txt': 1130279lines [35:31, 530.16lines/s]  
    Path Cycle Length Distribution:
      Length   2: 10842 paths
      Length   3: 61 paths
      Length   4: 3817 paths
      Length   6: 16787 paths
      Length   8: 17604 paths
      Length   9: 572 paths
      Length  10: 1281 paths
      Length  12: 92423 paths
      Length  14: 1112 paths
      Length  16: 2254 paths
      Length  18: 51017 paths
      Length  20: 4698 paths
      Length  21: 3 paths
      Length  22: 32 paths
      Length  24: 100779 paths
      Length  28: 2492 paths
      Length  30: 61921 paths
      Length  33: 45 paths
      Length  35: 354 paths
      Length  36: 82504 paths
      Length  40: 23062 paths
      Length  42: 34442 paths
      Length  44: 1557 paths
      Length  45: 1174 paths
      Length  48: 14436 paths
      Length  56: 15734 paths
      Length  60: 89342 paths
      Length  63: 1667 paths
      Length  66: 8321 paths
      Length  70: 2954 paths
      Length  72: 56198 paths
      Length  77: 34 paths
      Length  80: 104 paths
      Length  84: 43920 paths
      Length  90: 62623 paths
      Length  99: 198 paths
      Length 105: 2639 paths
      Length 110: 86 paths
      Length 112: 581 paths
      Length 120: 49663 paths
      Length 126: 31927 paths
      Length 132: 11023 paths
      Length 140: 1724 paths
      Length 144: 10029 paths
      Length 154: 2569 paths
      Length 165: 367 paths
      Length 168: 26970 paths
      Length 180: 39176 paths
      Length 198: 14069 paths
      Length 210: 33583 paths
      Length 231: 804 paths
      Length 240: 5723 paths
      Length 252: 12763 paths
      Length 280: 606 paths
      Length 315: 1211 paths
      Length 330: 6288 paths
      Length 336: 4458 paths
      Length 360: 14123 paths
      Length 420: 13840 paths
      Length 462: 15166 paths
      Length 495: 232 paths
      Length 504: 3568 paths
      Length 630: 7748 paths
      Length 720: 1655 paths
      Length 840: 4899 paths
      Length 990: 5795 paths
      Length 1260: 630 paths

    Action Distribution:
      0: 1177848
      1: 1427698
      2: 1162614
      3: 1104772
      4: 1556747
      5: 1078287
      6: 1139187
      7: 1545529
      8: 1145420
      9: 1202154
    10: 1464655
    11: 1201424
    12: 1063000
    13: 1524894
    14: 1085884
    15: 1098650
    16: 1508922
    17: 1117895

    All tests complete successfully.

## dataset.py

    --- Initializing PathDatasetProcessor (htm4.zip) ---
    [1/7] Unzipping & parsing 'htm4.txt': 1130279lines [00:03, 307659.59lines/s]
    [2/7] Trie prefix filtering: 100%|███████████████████████████████████████████████████████████████| 1130279/1130279 [00:02<00:00, 425929.61path/s]
    [3/7] Farthest Point Sampling: 100%|███████████████████████████████████████████████████████████████████████████| 16/16 [00:00<00:00, 66.01seed/s]
    [4/7] Generating cyclic shifts: 100%|████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 41120.63shift/s]
    [5/7] Applying anti-action inversions: 100%|████████████████████████████████████████████████████| 1/1 [00:00<00:00, 6668.21it/s, total_paths=160]
    [6/7] Expanding 24-orientation rotations: 100%|█████████████████████████████████████████████████████████████| 24/24 [00:00<00:00, 124583.29rot/s]
    [7/7] Deduplicating paths: 100%|█████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 155.39it/s, input=3840, unique=3840]
    Dataset pipeline completed. Final unique paths: 3840

    Processing Paths: 100%|█████████████████████████████████████████████████████████████████████████████████████| 3840/3840 [00:12<00:00, 301.64it/s]
    Done! Distribution: {0: 1, 1: 18, 2: 243, 3: 3240, 4: 28080, 5: 372891, 6: 54576, 7: 743982, 8: 79410, 9: 745848, 10: 79128, 11: 726948, 12: 103464, 13: 724932, 14: 99756, 15: 703764, 16: 126108, 17: 701748, 18: 122436, 19: 680940, 20: 146916}
    Train batch inputs shape: torch.Size([6, 324])
    Train batch targets shape: torch.Size([6])
    (tensor([0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
            0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
            0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
            0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
            0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
            0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.,
            0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
            0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
            0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
            0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
            0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.,
            0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
            0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.,
            1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.,
            1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.,
            1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]), tensor(19.))


    (tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.,
            0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
            0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.,
            0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.,
            0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
            1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
            0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
            0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
            0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
            0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
            1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
            0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
            0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.,
            0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.,
            1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
            0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
            0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.,
            0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.]), tensor(10.))


    (tensor([0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.,
            0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
            0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.,
            0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.,
            0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
            0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0.,
            1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
            0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.,
            1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
            0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.,
            0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
            0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.,
            0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
            0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
            0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.,
            1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.]), tensor(19.))


    (tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.,
            0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
            0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.,
            0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
            0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
            0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
            0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.,
            0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
            0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
            0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.,
            1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.,
            0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
            0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.,
            0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.]), tensor(13.))


    (tensor([0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
            0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
            0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.,
            1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
            1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
            0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
            0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
            0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
            1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
            0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.,
            0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.,
            0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.,
            0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
            0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.]), tensor(5.))


    (tensor([0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
            1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
            0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
            0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
            0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
            0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
            0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0.,
            0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
            1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.,
            0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
            0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.,
            0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.,
            0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
            0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
            0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.,
            0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.]), tensor(18.))


    done with test
