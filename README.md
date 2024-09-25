# Tarea 2 - Reinforcement Learning PUC 2024-2
### Javier Campos & Pedro Palma

Repositorio con el código de la T2 del curso Aprendizaje Reforzado 2024-2.

A continuación, instrucciones para replicar los experimentos.

## MDP
Los algoritmos principales se encuentran en `iterative_policy_evaluation.py` y en `value_iteration.py`. En `MDPs/experiment.py`, se encuentra definida la clase `Experiment`, que se encarga de inicializar ambientes y algoritmos para correr los experimentos de la tarea.
### Warning: La tarea "Evaluate greedy policy para GamblersProblem con p=0.55 se demora al rededor de 5min en correr
1. Correr `MDPs/Main.py`.
2. Seleccionar el problema (Grid/Cookie/Gamblers). 
3. Dar un valor para `size`/`p`.
4. Seleccionar el experimento a replicar.
En caso de ser un gráfico deberá ingresar luego el nombre para guardarlo

## Monte Carlo
En `MonteCarlo/experiment.py` creamos la clase `Experiment`, que se encarga de inicializar ambientes y algoritmos para correr los experimentos de la tarea. El algoritmo MonteCarlo principal está en el archivo `monte_carlo.py`, en la clase `MonteCarlo`.

1. Correr `MonteCarlo/Main.py`.
2. Seleccionar el problema (Blackjack/Cliff). En el caso de cliff, deberá ingresar un entero positivo como el ancho del cliff.

3. Seleccionar el experimento a replicar.
En caso de ser un gráfico deberá ingresar luego el nombre para guardarlo

