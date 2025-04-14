# DFSM-for-MILPs
This repository contains code files and the instructions on running them for the Decision-Focused Surrogate Modeling for Mixed-Integer
Linear Optimization paper.

# Data generation and surrogate model training

For the hybrid vehicle control case study, the data file generates and solves 700 MILP instances for a horizon length of T=30, the cost vector and the function used to generate the problem specific parameters can be modified as needed. 

For the production scheduling case study, the data file generates and solves 700 MILP instances for a horizon length of H=40 and I=13 batches, M=4 machines and T=8 time slots, the production running costs and the functions used to generate processing times, residence times and due times can be modified as needed. 

For training the surrogate models, adjust the number of training data points (K) and the number of added cuts (V) as needed. 


# Citation

@article{ \
dixit2025decisionfocused, \
title={Decision-Focused Surrogate Modeling for Mixed-Integer Linear Optimization}, \
author={Shivi Dixit and Rishabh Gupta and Qi Zhang}, \
journal={Transactions on Machine Learning Research}, \
issn={2835-8856}, \
year={2025}, \
url={https://openreview.net/forum?id=A6tOXkkE4Z}, \
note={} \
}
