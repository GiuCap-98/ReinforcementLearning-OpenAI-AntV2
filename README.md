# ReinforcementLearning-OpenAI-AntV2
Benchmark di algoritmi di RL per addestramento di AntV2

Progetto valido per il corso di Fondamenti di Intelligenza Artificiale (A.A. 2021/2022) presso l'Università degli Studi di Verona

Obiettivo del progetto è stato quello di approfondire la tematica del reinforcement learning andando ad utilizzare differenti algoritmi, confrontandone le prestazioni, per addestrare __AntV2__ del phisics engine __MuJoCo__, tramite la libreria __OpenAI Gym__.

In particolare sono stato approfonditi i seguenti algoritmi: PPO, TD3 e SAC:

- __Proximal Policy Optimization (PPO)__: Si basa sull’idea di effettuare il miglior aggiornamento dei parametri della policy possibile in modo che la 
nuova policy non differisca troppo dalla policy precedente

- __Twin Delayed DDPG (TD3)__: TD3 è il successore di DDPG, va a risolvere il problema di DDPG (DDPG, come riportato in OpenAI Spinning Guide, presenta un errore, secondo cui la funzione Q appresa 
inizia a sovrastimare notevolmente i valori Q, il che porta ad errori nelle policy stimate, in quanto vengono 
usati gli errori nella funzione Q), andando ad imparare due Q-Function, ed utilizzando la più piccola per aggiornare il target, a loro volta i sample delle Q-Function vengono aggiornate in base al target.

- __Soft Actor Critic (SAC)__: L’idea dietro SAC è che non cerca solo di massimizzare i reward, ma anche l’entropia della policy.
