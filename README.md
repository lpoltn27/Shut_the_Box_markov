# Shut the Box – Markov Chains and Optimal Strategies

This repository contains the computational part of my bachelor's thesis on the dice game **Shut the Box**.

The aim of the project is to analyze the game mathematically using **finite Markov chains** and to determine optimal playing strategies using **dynamic programming and the Bellman principle**.

## About the Project

Shut the Box is a dice game in which numbered tiles are closed depending on the result of a dice roll.

In this project, the game is modeled as a finite Markov chain. The possible configurations of open and closed tiles represent the states of the Markov chain.

The project includes:

- mathematical modeling of Shut the Box
- representation of game states
- calculation of legal moves
- construction of transition matrices
- analysis of absorbing states
- calculation of probabilities and expected values
- comparison of different playing strategies
- determination of an optimal strategy using the Bellman principle

## Jupyter Notebooks

The repository contains two Jupyter notebooks.

### Markov Chain Model

The first notebook implements the Markov chain model of Shut the Box.

It contains the state space, transition probabilities and transition matrix used to analyze the stochastic behavior of the game.

### Optimal Strategy

The second notebook focuses on finding an optimal playing strategy.

Dynamic programming and the **Bellman principle** are used to determine decisions that minimize the expected remaining score.

## Bachelor's Thesis

The corresponding bachelor's thesis provides the mathematical background and a detailed explanation of the models and algorithms used in the notebooks.

## Technologies

The computational analysis was implemented using:

- Python
- Jupyter Notebook
- NumPy

## Author

Lukas Poltnig

Bachelor's thesis project, 2026.
