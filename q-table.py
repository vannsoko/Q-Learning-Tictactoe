from tictactoe_environment import TicTacToeENV
import numpy as np
import random
from tqdm import tqdm
import pickle
import threading
import sys
import array
# import mathplotlib.pyplot as plt
# import time


# lock = True if sys.maxsize < 2**63-1 else False

env = TicTacToeENV("console")
testenv = TicTacToeENV("console")

q_table = np.zeros(177148, dtype=np.float64)
# q_table = array.array('d', 177148 * [0.0]) # Becomes a RawArray for 64 bit architechtures

# 2. Hyperparamètres de l'apprentissage
alpha = 0.5  # Taux d'apprentissage (vitesse d'assimilation des nouvelles informations)
gamma = 0.99  # Facteur d'escompte (importance accordée aux récompenses futures)
epsilon = 1.0  # Taux d'exploration initial (100% au hasard au début)
epsilon_min = 0.05  # Taux d'exploration minimal (pour toujours garder un peu de hasard)
epsilon_decay = 0.9996  # Vitesse de réduction de l'exploration
episodes = 15000  # Nombre de parties jouées pour s'entraîner

test_episodes = 100


# PLT

# plt.ion()


# 2. Fonction pour jouer 100 parties sans exploration (epsilon = 0)
def evaluate_agent(eval_env, q_table, nb_episodes=100, epsilon=0):
    results = {1: 0, 0: 0, -1: 0, -100: 0}  # Winning, draw, lossing, illegal move
    for _ in tqdm(
        range(1, nb_episodes + 1), desc=f"Evaluation Agent Q-Table epsilon: {epsilon}"
    ):
        state, _ = eval_env.reset()
        done = False
        while not done:
            state_index = state_to_index(state)
            valid_actions = [int(i) for i in state if i==0]

            if random.uniform(0, 1) < epsilon or state_index == 0:
                # action = eval_env.action_space.sample()
                action = random.choice(valid_actions)
            else:
                # action = np.argmax(q_table[state_index: state_index+9])
                max = float("-inf")
                action = valid_actions[0]
                for i in valid_actions:
                    if (val := q_table[state_index+i])>max:
                        max = val
                        action = i
            state, reward, finished, truncated, info = eval_env.step(action)
            done = finished or truncated

            if done:
                results[reward] += 1

    return results


def state_to_index(state, action=0) -> int:
    state_value = 0

    # Same idea as a bitmask but with 3 possibilites
    for i in range(9):
        state_value += int(state[i]) * (3**i)


    # Multiply times 9 to create space for the 9 possible (valid or not) actions per state 
    state_value *= 9

    state_value += action
    
    return state_value


def rotate_state(state: np.array, action: int) -> (np.array, int):
    # rotation 90 degree à droite
    n_state = np.zeros(9, dtype=np.int8)
    
    for i in range(3):
        n_state[i] = state[i+6]
        n_state[i+1] = state[i+3]
        n_state[i+2] = state[i]

    rot_action_dic = {0:6, 1:3, 2:0, 3:7, 4:4, 5:1, 6:8, 7:5, 8:2}
    n_action = rot_action_dic.get(action)

    return n_state, n_action


#   q_table[state_tuple][action] = nouvelle_valeur
def update_q_table(state: np.array, action: int, value: int):
    global q_table
    
    state_index = state_to_index(state, action)
    q_table[state_index] = value


    # rotation
    rot_state, rot_action = state, action
    for i in range(3):
        rot_state, rot_action = rotate_state(rot_state, rot_action)
        rot_state_index = state_to_index(rot_state, rot_action)
        q_table[rot_state_index] = value



print("début de l'entraînement")

def agent_action():
    ...


def func(worker_id, episodes=episodes, alpha=alpha, epsilon=epsilon, gamma=gamma, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay):
    for episode in tqdm(range(1, episodes + 1), desc=f"Entraînement Q-Table {worker_id=}"):
        
        alpha *= 0.9995
        state, _ = env.reset()
    
        termine = False
        if episode > 2000:
            epsilon = epsilon * epsilon_decay if epsilon > epsilon_min else epsilon_min
    
        while not termine:
            state_index = state_to_index(state)

            valid_actions = [int(i) for i in state if i==0]

            if episode < 2000 or random.uniform(0, 1) < epsilon:
                action = random.choice(valid_actions)
            else:
                max = float("-inf")
                action = valid_actions[0]
                for i in valid_actions:
                    if  (val := q_table[state_index+valid_actions[i]]) > max:
                        max = val
                        action = valid_actions[i]
                # action = np.argmax(q_table[state_index:state_index+9])
    
            next_state, reward, finished, truncated, info = env.step(action)
            next_state_index = state_to_index(next_state)
            termine = finished or truncated
    
    
            # Calcul de la valeur future maximbale
            # Si la partie est finie, il n'y a plus de futur, donc la valeur est 0
            if termine:
                max_future_q = 0
            else:
                next_valid_actions = [int(i) for i in next_state if i==0]
                max_future_q = np.max([q_table[next_state_index+val] for val in next_valid_actions])
#                max_future_q = np.max(q_table[next_state_index:next_state_index+9])
    
            #  Q(s, a) <- Q(s, a) + alpha [ R + gamma * max_{a'} Q(s', a') - Q(s, a) ]
    
            # MISE À JOUR DE LA Q-TABLE (Équation de Bellman)
            ancienne_valeur = q_table[state_index + action]
            nouvelle_valeur = ancienne_valeur + alpha * (
                reward + gamma * max_future_q - ancienne_valeur
            )
        
            # q_table[state_index + action] = nouvelle_valeur
            update_q_table(state, action, nouvelle_valeur)
    
            # L'agent passe à l'état suivant
            state = next_state
    
            # Affichage de la progression
            if (episode + 1) % 5000 == 0:
                print(
                    f"Épisode {episode + 1}/{episodes} terminé. Epsilon actuel: {epsilon:.3f}"
                )
    
        #  print(
        #      f"\nEntraînement terminé ! L'agent a exploré {len(q_table)} états différents du plateau."
        #  )
    
        if (episode) % 500 == 0 or episode == 1:
            results = evaluate_agent(testenv, q_table, 100, 0)
            epsilon_results = evaluate_agent(testenv, q_table, 100, 0.3)
            print("-" * 20)
            print(f"epsilon: {epsilon}")
            print(
                f"results:\nwins: {results.get(1)}%\ndraws: {results.get(0)}%\n losses: {results.get(-1)}%\n illegal: {results.get(-100)}%"
            )
            print("-" * 10)
            print(
                f"epsilon_results:\nwins: {epsilon_results.get(1)}%\ndraws: {epsilon_results.get(0)}%\n losses: {epsilon_results.get(-1)}%\n illegal: {epsilon_results.get(-100)}%"
            )
            print("-" * 20)
            with open(f"tictactoe_q_table_{episode}.pkl", "wb") as file:
                pickle.dump(q_table, file)


func(1)
print("Training complete! Saving the agent...")

# Open a file in 'wb' (write binary) mode
with open("tictactoe_q_table_final.pkl", "wb") as file:
    pickle.dump(q_table, file)

print("Agent successfully saved to 'tictactoe_q_table.pkl'!")
