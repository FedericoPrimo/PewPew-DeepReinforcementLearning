auto selector cuda or mips se disponibile invece di config

# DQN
Ad ora la nostra pipeline in DQN è:

Frame grezzo Atari (210×160 RGB)
    ↓
[AtariWrapper + FrameStack - preprocessing classico] (fa un preprocess all'immagine tipo maxandskipframe e altra roba, guarda doc per maggiori info)
    ↓
Osservazione (4, 84, 84)
    ↓
[DQNAgent - CNN interna]
    ↓
Q-values per ogni azione


## possibili cambiamneti e info sistema

optimizer ad ora è adam
cambiare sampling azioni in caso di esplorazione policy al posto di sample nell'actionspace ma con un sampling limitato da qualcosa
loss usa huber al posto di MSE
qnetwork ha copia con hard update e replay buffer

# PPO

Ad ora DQN è sequenziale mentre PPO è parallelizzato e delegato a sb3, uniformare la cosa.
Inoltre, PPO potrebbe non avere lo stesso preprocessing di DQN
Inoltre, PPO non usa la cnn di matti ma ne usa una default di sb3

# Generali

Runna questo comando per metriche interattive

tensorboard --logdir logs/ --port 6006

per vederlo vai su localhost:6006 nel browser

ancora  ho solo greyscale su 4 frame in dqn, tanto il masking è da fare successivamente, ad ora uso quello della libreria, valutare se modificare quello di matti per replicare il suo funzionamento