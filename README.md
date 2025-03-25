# Find Me If You Can! - Reinforcement Learning Assignment

## Overview

This repository contains the code for the "Find me if you can!" assignment for the ECE 595 course on Reinforcement Learning. The assignment involves implementing a bandit algorithm with an application in Positioning, Navigation, and Timing (PNT) systems. The task focuses on simulating how a target can select three anchor nodes simultaneously to improve its position estimation accuracy using a bandit algorithm. The accuracy of the position estimation will be determined using the trilateration technique. Trilateration is a method used to determine the position of a target based on the time differences of arrival (TDOA) of signals from multiple reference points, known as anchor nodes. By measuring the pseudoranges, of the target from each selected anchor node, the target’s position can be estimated by intersecting spheres centered at the anchor nodes.

## Instructions

Follow the steps below to complete the assignment:

### Step 0: Initialize the Problem Parameters
- The problem considers five anchor nodes for the target to select from.
- The total number of time-steps or rounds for the simulation is provided in the starter code.
- Anchor node positions and an exemplary target position are given for reproducible results.
- An initial estimate of the target’s position is provided to predict the final estimated position.
- Two ε values are considered: ε = 0.01 and ε = 0.3.
### Step 1: Implement the Bandit Algorithm
1. Calculate the real ranges between the target and each anchor node, incorporating some noise for pseudoranges.
2. Choose three anchor nodes at each time step based on the simple bandit algorithm.
3. Use selected anchor nodes, their positions, target’s estimated position, and measured pseudoranges to determine the Jacobian matrix.
### Step 2: Calculate GDOP and Update Estimates
1. Calculate the GDOP value GDOP(A) for choosing the action A.
2. Measure the reward R(A) from the selected anchor nodes.
3. Update N(A) and Q(A) based on the reward.
4. Update the position estimate of the target using the residual matrix.
### Step 3: Plot and Analyze Results
1. Run the algorithm for the specified number of rounds with ε values 0.01 and 0.3.
2. Record and display:
3. GDOP versus steps of the simple bandit algorithm
4. Reward versus steps of the simple bandit algorithm
5. Euclidean distance between the real and estimated position of the target versus steps

## How to Run
1. Clone the repository to your local machine.
2. Navigate to the repository directory.
3. Execute the Python script for the assignment.

## Files Included
- bandit_algorithm.py: Contains the implementation of the bandit algorithm.
- utils.py: Utility functions for calculating pseudoranges and Jacobian matrix.
- plot_results.py: Script to generate plots for analysis.

## Results
The results of the simulation will be stored in the results directory.

# Did You Find Me?!!!

Congratulations on completing the assignment! If you have any questions or need further assistance, feel free to reach out.

Author: Zachary Montoya
Professor: Erini Eleni Tsiropoulou, PhD
Date: September 5, 2023

