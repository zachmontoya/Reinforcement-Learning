# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import logging
from itertools import combinations


### Step 1: Initialize the problem parameters.
num_anchor_nodes = ["A", "B", "C", "D", "E"]
number_of_nodes_selected_per_action = 3
total_steps = 100000
# total_steps = 10000

## Set the logging level
logging.basicConfig(filename='run.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

## Initialize anchor node positions and target position
anchor_positions = np.array([[11, 30, 10], [5, 40, -20], [15, 40, 30], [5, 35, 20], [15, 35, -10]], dtype=float)
anchor_positions_dictionary = {"A": anchor_positions[0],"B": anchor_positions[1],"C": anchor_positions[2],"D": anchor_positions[3],"E": anchor_positions[4]}

## Intializing the unique combinations of anchor nodes
combinations_of_three = np.array(list(combinations(num_anchor_nodes, 3)))

## Shuffle the combinations of anchor nodes - this is incredibly important as the greedy action is selected via a argsort and the Q-values are initialized to 0.
np.random.shuffle(combinations_of_three)
logging.info(f'Unique Combinations of Anchor Nodes:, {combinations_of_three}')
target_position = [10, 35, 0.1]

## Defining the Master Matrix for the Q Values and Action Counts
## Create a 10x3 matrix with the last column as nested arrays of strings
master_matrix  = np.zeros((10, 3), dtype=object) 
for i in range(len(master_matrix)):
    master_matrix[i,2] = combinations_of_three[i]
    # Column 1 is for the Q Values
    # Column 2 is for the Action Counts
    # Column 3 is for the Actions AKA unique combinations of 3 elements

## Define two epsilon values
epsilons = [0.01, 0.3]

## Initialize arrays to store values
gdop_values = []
reward_values = []
distance_errors = []
selected_positions = np.zeros((3, 3)).astype(float)

## Initialize lists for epsilon-specific values
gdop_epsilon = [[] for x in range(len(epsilons))]
reward_epsilon = [[] for x in range(len(epsilons))]
distance_errors_epsilon = [[] for x in range(len(epsilons))]

## Calculate the centroid of anchor node positions
centroid = np.mean(anchor_positions, axis=0)

## Set the initial position estimate as the centroid
position_initial_estimate = centroid

## Function to calculte the Jacobian matrix
def calculate_jacobian(selected_positions, position_estimate, pseudoranges):
    J = np.zeros((3, 3)).astype(float)
    # #  J = (position_estimate - selected_positions) / (pseudoranges)
    for i in range(3): # 3 rows, f, g, h representing the 3 selected anchor nodes
        for j in range(3): # 3 columns, x, y, z representing the 3 coordinates of the target
            J[i,j] = (position_estimate[j] - selected_positions[i][j])/(pseudoranges[i]) #-(xa-xt)/(A) S.T. A = SQRT((xa-xt)^2+(ya-yt)^2+(za-zt)^2)
    return J

## Function to calculate Euclidean distance
def euclidean_distance(a, b):
    return np.linalg.norm(a - b) # Root Sum Squared of X, Y, Z

## Function to calculate GDOP (Geometric Dilution of Precision)
def calculate_gdop(jacobian):
    is_linearly_dependent = np.linalg.matrix_rank(jacobian) < min(jacobian.shape)
    if is_linearly_dependent: # If the Jacobian is linearly dependent, use the pseudo-inverse --------------------------------------------------
        print("\nJacobian is linearly dependent")
    else:
        G = np.linalg.inv(np.dot(jacobian.T, jacobian))
    gdop = np.sqrt(np.trace(G))
    return gdop

## Function to calculate reward based on GDOP
def calculate_reward(gdop):
    return np.sqrt(10/3) / gdop if gdop > 0 else 0

## Function to calculate factorial
def factorial(n):
    ## single line to find factorial
    return 1 if (n==1 or n==0) else n * factorial(n - 1)

## Determining the number of unique combinations of anchor nodes
number_of_combinations = int((factorial(len(num_anchor_nodes))) / (factorial(number_of_nodes_selected_per_action) * factorial(len(num_anchor_nodes) - 3)))

### Step 2: Implement the Bandit Algorithm.

## Loop through the epsilon values
for e in range (len(epsilons)):
    print("\n\nEspsilon: ", epsilons[e])

    ## Initializing the 'position_estimate' to 'position_initial_estimate'
    position_estimate = position_initial_estimate.copy() # Center of the anchor nodes

    ## Initialize action counts for each epsilon
    for i in range(len(master_matrix)):
        master_matrix[i,1] = float(0)
    print("Action Counts Reset to: ", master_matrix[:,1])

    ## Initialize Q-values for each epsilon
    for i in range(len(master_matrix)):
        master_matrix[i,0] = float(0)
    print("Q-Values Reset to: ", master_matrix[:,0])
    
    ## Main loop for the epsilon-greedy bandit algorithm
    for step in range(total_steps):
        
        logging.info(f'Step: {step}')
        
        # print every 10% of the total steps
        if step % (total_steps/10) == 0:
            print("Step: ", step)

        ## Exploration: Choose random actions
        if np.random.rand() <= epsilons[e]: # np.random.rand() returns a random number between 0 and 1, less than epsilon means exploration
            Action = np.random.choice(number_of_combinations, 1, replace=False)
            for i in range(len(selected_positions)):
                selected_positions[i] = anchor_positions_dictionary[master_matrix[Action,2][0][i]]
            logging.info(f'Explore best action: {Action}')
            logging.info(f'EXPLORE selected positions: {selected_positions}')
        
        ## Exploitation: Choose actions with highest Q-values
        else: # this is complimentary (1-epsilon), thererfore exploitation
            ## best_action = np.argmax(Q_values) # choose the action with the highest Q-value
            Action = np.argsort(master_matrix[:,0])[-1:]
            logging.info(f'Action: {Action}')
            for i in range(len(selected_positions)):
                selected_positions[i] = anchor_positions_dictionary[master_matrix[Action,2][0][i]]
            logging.info(f'Exploit best actions {Action}')
            logging.info(f'Exploit selected positions {selected_positions}')

        ## Code for determining pseudoranges
        pseudoranges = [euclidean_distance(selected_positions[i], position_estimate) + np.random.uniform(-0.0001, 0.0001, 1)[0] for i in range(3)]
        logging.info(f'Pseudoranges: {pseudoranges}')

        ## Determine the 'jacobian' matrix J(A) from the selected anchor nodes
        jacobian = calculate_jacobian(selected_positions,position_estimate,pseudoranges) 
        logging.info(f'Jacobian: {jacobian}')

        ## Determine the 'gdop' value GDOP(A) from the calculated 'jacobian'
        gdop = calculate_gdop(jacobian)
        logging.info(f'GDOP:  {gdop}')

        ## Determine the 'reward' R(A) using the 'gdop' value
        reward = calculate_reward(gdop)
        logging.info(f'Reward: {reward}')
        
        ## Update action counts N(A)
        master_matrix[Action,1] += 1
        logging.info(f'Action Counts: {master_matrix[:,1]}')

        ## Update Q-values Q(A)
        master_matrix[:,0][Action] = master_matrix[:,0][Action] + (1/master_matrix[:,1][Action]) * (reward - master_matrix[:,0][Action])
        # Q(A) = Q(A) + (1/N(A)) * (R(A) - Q(A)) S.T. A = Action
        logging.info(f'Q-Values: {master_matrix[:,0]}')

        ## Update position estimate
        # RES = np.array([pseudoranges]) - np.array([euclidean_distance(selected_positions[i],target_position) + np.random.uniform(-0.0001, 0.0001, 1)[0] for i in range(3)]) 
        RES = np.array([pseudoranges]) - np.array([euclidean_distance(selected_positions[i],position_estimate) for i in range(3)])
        RES = np.vstack(RES[0])
        is_linearly_dependent = np.linalg.matrix_rank(jacobian) < min(jacobian.shape)
        if is_linearly_dependent: # If the Jacobian is linearly dependent, use the pseudo-inverse -------------------------------------------------------------------------------------------------
            exit("\nError: Jacobian is Linearly Depedent")
        else:
            DelXYZ = np.dot(np.dot(np.linalg.inv(np.dot(jacobian.T,jacobian)),jacobian.T), RES) # (J^T * J)^-1 * J^T * ResidualMatrix
        logging.info(f'Current Position Estimate: {position_estimate}')
        position_estimate = position_estimate - DelXYZ.T[0]
        logging.info(f'DelXYZ: {DelXYZ.T[0]}')
        logging.info(f'New Position Estimate: {position_estimate}')

        ## Store GDOP(A), R(A), Euclidean distance error for each step of 'total_steps'
        gdop_values.append(gdop)
        reward_values.append(reward)
        distance_errors.append(np.linalg.norm(target_position - position_estimate))

        ## Store GDOP values, rewards, Euclidean distance errors for each epsilon
        gdop_epsilon[e].append(gdop)
        reward_epsilon[e].append(reward)
        distance_errors_epsilon[e].append(np.linalg.norm(target_position - position_estimate))

        ##logging.info the results at the end of the epsilon loop
        if step == total_steps - 1:
            print("\n\nFinal Position Estimate: ", position_estimate)
            print("Target Position: ", target_position)
            print("Q-Values: ", master_matrix[:,0])
            print("Action Counts: ", master_matrix[:,1])
            logging.info(f'Final Position Estimate: {position_estimate}')
            logging.info(f'Target Position: {target_position}')
            logging.info(f'Q-Values: {master_matrix[:,0]}')
            logging.info(f'Action Counts: {master_matrix[:,1]}')
            

### Step 3: Plot and analyze the results.

## Plot GDOP vs. Steps for each epsilon
plt.figure(figsize=(10, 5))
for e in range(len(epsilons)):
    plt.plot(gdop_epsilon[e], label=f'Epsilon = {epsilons[e]}')
plt.xlabel('Steps')
plt.ylabel('GDOP')
plt.legend()
plt.title('GDOP vs. Steps for each epsilon')
plt.show()

## Plot Reward vs. Steps for each epsilon
plt.figure(figsize=(10, 5))
for e in range(len(epsilons)):
    plt.plot(reward_epsilon[e], label=f'Epsilon = {epsilons[e]}')
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.legend()
plt.title('Reward vs. Steps for each epsilon')
plt.show()

## Plot Distance Error vs. Steps
plt.figure(figsize=(10, 5))
for e in range(len(epsilons)):
    plt.plot(distance_errors_epsilon[e], label=f'Epsilon = {epsilons[e]}')
plt.xlabel('Steps')
plt.ylabel('Distance Error')
plt.title('Euclidean Distance Error vs. Steps')
plt.show()
