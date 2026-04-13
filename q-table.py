from tictactoe_environment import TicTacToeENV
from collections import defaultdict
import numpy as np
import random
from tqdm import tqdm
import pickle
# import mathplotlib.pyplot as plt
# import time

env = TicTacToeENV("console")
testenv = TicTacToeENV("console")

q_table = defaultdict(lambda: np.zeros(9))

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
            s_tuple = tuple(state)

            # Exploitation pure : on prend la meilleure action connue
            if random.uniform(0, 1) < epsilon or s_tuple == np.zeros(9):
                action = eval_env.action_space.sample()
            else:
                action = np.argmax(q_table[s_tuple])

            state, reward, finished, truncated, info = eval_env.step(action)
            done = finished or truncated

            if done:
                results[reward] += 1

    return results


print("début de l'entraînement")

for episode in tqdm(range(1, episodes + 1), desc="Entraînement Q-Table"):
    alpha *= 0.9995
    state, _ = env.reset()
    state_tuple = tuple(state)

    termine = False
    if episode > 2000:
        epsilon = epsilon * epsilon_decay if epsilon > epsilon_min else epsilon_min

    while not termine:
        if episode < 2000 or random.uniform(0, 1) < epsilon:
            # valid_actions = [i for i, v in enumerate(state) if v == 0]
            # action = random.choice(valid_actions)
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state_tuple])

        next_state, reward, finished, truncated, info = env.step(action)
        next_state_tuple = tuple(next_state)
        termine = finished or truncated

        if next_state_tuple not in q_table:
            q_table[next_state_tuple] = np.zeros(9)

        # Calcul de la valeur future maximbale
        # Si la partie est finie, il n'y a plus de futur, donc la valeur est 0
        if termine:
            max_future_q = 0
        else:
            max_future_q = np.max(q_table[next_state_tuple])

        #  Q(s, a) <- Q(s, a) + alpha [ R + gamma * max_{a'} Q(s', a') - Q(s, a) ]

        # MISE À JOUR DE LA Q-TABLE (Équation de Bellman)
        ancienne_valeur = q_table[state_tuple][action]
        nouvelle_valeur = ancienne_valeur + alpha * (
            reward + gamma * max_future_q - ancienne_valeur
        )
        q_table[state_tuple][action] = nouvelle_valeur

        # L'agent passe à l'état suivant
        state_tuple = next_state_tuple

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

print("Training complete! Saving the agent...")

# Open a file in 'wb' (write binary) mode
with open("tictactoe_q_table.pkl", "wb") as file:
    pickle.dump(q_table, file)

print("Agent successfully saved to 'tictactoe_q_table.pkl'!")
