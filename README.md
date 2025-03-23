# Evolutionary Algorithms Project

## About
This repository contains our work on the Evolutionary Algorithms Practical Assignment. We implemented two optimization methods:
- A **Genetic Algorithm (GA)** for solving the Ising Ring problem (F19) and the Low Autocorrelational Binary Sequences (LABS) problem.
- An **Evolution Strategy (ES)** for the same problems, using real-valued encoding with appropriate mutation and recombination operators.

The detailed methodology, experiments, and results are described in the report.

## Repository Structure
- **GA (2).py:**  
  Contains the implementation of a Genetic Algorithm using various genetic operators (uniform crossover, n-point crossover, inversion mutation, tournament selection) to solve the optimization problems.
  
- **ES (2).py:**  
  Contains the implementation of an Evolution Strategy with real-valued individuals, including mutation, recombination, and tournament selection operators.
  
- **EA_Practical_Assignment.pdf:**  
  The full report detailing our methodology, experimental setup, results, and discussion of our evolutionary algorithm approaches.

## How to Run
1. Ensure you have Python installed along with the required packages (refer to any documentation within the code for dependencies).
2. Run the GA implementation:
   ```bash
   python GA\ \(2\).py
