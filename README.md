# FPLBot
Automatically do your FPL team using an MLP to predict expected points for each player over a window of gameweeks and finding the optimal transfer plan to maximise the accumulated xP of the best starting XI using tree traversal algorithms.

## Installation


## Usage
To be able to access your current squad (including current selling price, which is not available on the public API) and (optionally) submit the transfers and final XI / Captain automatically, you must set the following environment variables in a .env file in the root directory. You may find your FPL_ID by logging in to the FPL website and clicking on 'my team', your ID will be in the URL.

```
.env
EMAIL=<your fpl email>
PASSWORD=<your fpl password>
FPL_ID=<your fpl ID>
```

If you do not want to login to FPL, you can instead set controller to None in run.py, but you must then call FPLBot.build_current_squad(squad_dict) with a dictionary like that in test_squad.json before calling FPLBot.do_team().

Otherwise, after setting the environment variables, run as follows.
```
python run.py
```
By default, the transfers and squad plan will not be committed to FPL. If you do want to commit the changes, set bot.do_team(commit=True) in run.py 

## Output
This example was immediately before GW10 of the 25/26 season, using the squad in test_squad.json. The team had 2 free transfers.
```
Featurising players for GW10: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 746/746 [00:39<00:00, 19.11it/s]
Featurising players for GW11: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 746/746 [00:03<00:00, 230.11it/s]
Featurising players for GW12: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 746/746 [00:03<00:00, 234.79it/s]
Featurising players for GW13: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 746/746 [00:03<00:00, 226.97it/s]
Featurising players for GW14: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 746/746 [00:03<00:00, 233.25it/s]
Traversing Possible Transfers...
Traversal: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [03:45<00:00, 45.10s/it, expanded=27710, kept=15]

TRANSFERS
================
OUT: Cole Palmer | IN: Bukayo Saka
OUT: Eliezer Mayenda Dossou | IN: Jean-Philippe Mateta

Starting XI
================
GK: Emiliano Martínez Romero | xP = 1.98 
DEF: Daniel Muñoz Mejía | xP = 3.69 
DEF: Marcos Senesi Barón | xP = 3.28 
DEF: Gabriel Gudmundsson | xP = 3.11 
MID: Bukayo Saka | xP = 6.36 (VC)
MID: Elliot Anderson | xP = 3.02 
MID: Amad Diallo | xP = 2.8 
MID: Morgan Rogers | xP = 2.56 
FWD: Jean-Philippe Mateta | xP = 7.44 (C)
FWD: Danny Welbeck | xP = 5.36 
FWD: João Pedro Junqueira de Jesus | xP = 3.15 

Bench
================
GK: Alisson Becker | xP = 0.0 
DEF: Ezri Konsa Ngoyo | xP = 1.57 
DEF: Jeremie Frimpong | xP = 0.0 
MID: Dan Ndoye | xP = 0.55 
```


## XP Prediction 
Stuff about the XP prediction models and customization

## Graph traversal 
Stuff about graph traversal and customisation. Show figures of graph traversal comparisons