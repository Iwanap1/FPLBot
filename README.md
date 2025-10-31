# FPLBot
Automatically do your FPL team using an MLP to predict expected points for each player over a window of gameweeks and finding the optimal transfer plan to maximise the accumulated xP of the best starting XI using tree traversal algorithms


# Usage
```
python run.py
```

## Output
```
Logging in to FPL...
Featurising players for GW10: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 746/746 [00:38<00:00, 19.61it/s]
Featurising players for GW11: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 746/746 [00:03<00:00, 235.07it/s]
Featurising players for GW12: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 746/746 [00:03<00:00, 236.39it/s]
Featurising players for GW13: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 746/746 [00:03<00:00, 235.50it/s]
Featurising players for GW14: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 746/746 [00:03<00:00, 232.29it/s]
Traversing Possible Transfers...
Traversal: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [03:12<00:00, 38.41s/it, expanded=2e+4, kept=15]

TRANSFERS
================
OUT: João Pedro Junqueira de Jesus | IN: Igor Thiago Nascimento Rodrigues
OUT: Dan Ndoye | IN: Iliman Ndiaye

Starting XI
================
GK: Emiliano Martínez Romero | xP = 1.9838573932647705 
DEF: Daniel Muñoz Mejía | xP = 3.6903984546661377 
DEF: Marcos Senesi Barón | xP = 3.2834737300872803 
DEF: Gabriel Gudmundsson | xP = 3.1139111518859863 
MID: Bukayo Saka | xP = 6.358638286590576 
MID: Iliman Ndiaye | xP = 6.004890441894531 
MID: Elliot Anderson | xP = 3.0208497047424316 
MID: Amad Diallo | xP = 2.8033227920532227 
FWD: Jean-Philippe Mateta | xP = 7.44330358505249 (C)
FWD: Igor Thiago Nascimento Rodrigues | xP = 6.549025058746338 (VC)
FWD: Danny Welbeck | xP = 5.355681419372559 

Bench
================
GK: Alisson Becker | xP = 0.0 
DEF: Ezri Konsa Ngoyo | xP = 1.5690187215805054 
DEF: Jeremie Frimpong | xP = 0.0 
MID: Morgan Rogers | xP = 2.560279130935669 
```