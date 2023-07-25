# ReinforcementLearning-OpenAI-AntV2
Benchmark di algoritmi di RL per addestramento di AntV2

Progetto valido per il corso di Fondamenti di Intelligenza Artificiale (A.A. 2021/2022) presso l'Università degli Studi di Verona

Obiettivo del progetto è stato quello di approfondire la tematica del reinforcement learning andando ad utilizzare differenti algoritmi, confrontandone le prestazioni, per addestrare AntV2 del phisics engine MuJoCo, tramite la libreria OpenAI.

In particolare sono stato approfonditi i seguenti algoritmi: PPO, TD3 e SAC:

- Proximal Policy Optimization (PPO): Si basa sull’idea di effettuare il miglior aggiornamento dei parametri della policy possibile in modo che la 
nuova policy non differisca troppo dalla policy precedente

- Twin Delayed DDPG (TD3): TD3 è il successore di DDPG, va a risolvere il problema di DDPG (DDPG, come riportato in OpenAI Spinning Guide, presenta un errore, secondo cui la funzione Q appresa 
inizia a sovrastimare notevolmente i valori Q, il che porta ad errori nelle policy stimate, in quanto vengono 
usati gli errori nella funzione Q), andando ad imparare due Q-Function, ed utilizzando la più piccola per aggiornare il target, a loro volta i sample delle Q-Function vengono aggiornate in base al target.

- Soft Actor Critic (SAC): L’idea dietro SAC è che non cerca solo di massimizzare i reward, ma anche l’entropia della policy.
