# 🎮 Q-Learning Tic-Tac-Toe

A reinforcement learning agent trained to play Tic-Tac-Toe using **tabular Q-Learning**, built from scratch with a custom [Gymnasium](https://gymnasium.farama.org/)(OpenAI-Gym) environment. Trained against the **Minimax** algorithm (perfect player) with some random moves, so the agent can learn to win.
The opponent is a perfect **Minimax** player — making this a non-trivial benchmark for RL convergence.

---

## 📌 Overview

| Property | Value |
|---|---|
| Algorithm | Q-Learning (tabular) |
| Environment | Custom `gymnasium.Env` |
| Opponent | Minimax AI with variable stochasticity ($5\%$ to $100\%$ random move probability) |
| State space | Discrete board configurations (up to 5 478 reachable states) |
| Action space | `Discrete(9)` — one cell per board position |
| Reward shaping | `+1` win · `0` draw · `-1` loss · `-100` illegal move |

The agent learns entirely through trial and error, starting with **pure random exploration** and progressively shifting toward **exploitation** of learned Q-values via an ε-greedy policy with decay.

---

## 🧠 Algorithm: Tabular Q-Learning

The agent maintains a Q-table mapping `(state, action)` pairs to expected returns. After each step, the table is updated using the **Bellman equation**:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a) \right]$$

| Symbol | Meaning | Value |
|--------|---------|-------|
| `α` (alpha) | Learning rate | `0.5` → decays via `× 0.9995` per episode |
| `γ` (gamma) | Discount factor | `0.99` |
| `ε` (epsilon) | Exploration rate | `1.0` → decays to `0.05` after episode 2 000 |
| Episodes | Training games | `15 000` |

**Key design choices:**
- The first 2 000 episodes use pure random exploration to seed the Q-table with diverse states before exploitation begins.
- Learning rate `α` decays alongside `ε` to stabilize convergence in later training.
- Illegal moves terminate the episode immediately with a heavy penalty (`-100`), teaching the agent board validity implicitly.

---

## 🌍 Custom Environment

The environment (`TicTacToeENV`) is a `gymnasium.Env` subclass.

```
Board encoding:  1 = agent  ·  -1 = opponent  ·  0 = empty
Observation:     np.array of shape (9,), dtype=int8, values in {-1, 0, 1}
Action:          int in [0, 8] → cell index (row-major order)
```

```
0 | 1 | 2
--+---+--
3 | 4 | 5
--+---+--
6 | 7 | 8
```

### Opponent Strategy

The environment's `step()` method triggers an **immediate Minimax counter-move** after every valid agent action — the agent is always playing against an optimal opponent. On `reset()`, the opponent also plays first with probability 0.5, with an 80 % chance of playing optimally (20 % random), to diversify starting states.

### Reward Structure

| Outcome | Reward |
|---------|--------|
| Agent wins | `+1` |
| Draw | `0` |
| Agent loses | `-1` |
| Illegal move | `-100` + episode terminates |

---

## 🗂️ Project Structure

```
.
├── tictactoe_environment.py   # Custom Gymnasium environment + Minimax opponent
├── q-table.py                 # Q-Learning training loop + evaluation
├── tictactoe_q_table.pkl      # Saved Q-table after training (generated)
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy gymnasium tqdm
```

### Training

```bash
python q-table.py
```

The script will:
1. Train for 15 000 episodes against the Minimax opponent.
2. Evaluate the agent every 500 episodes (win/draw/loss/illegal rates over 100 test games).
3. Save intermediate Q-table snapshots (`tictactoe_q_table_<episode>.pkl`).
4. Save the final Q-table to `tictactoe_q_table.pkl`.

### Training Output (example)

```
--------------------
epsilon: 0.312
results:
wins:    42%
draws:   55%
losses:   3%
illegal:  0%
--------------------
```

> Against a perfect Minimax player, **wins are impossible from all positions** — the theoretical optimum is ~100 % draws with 0 % losses. Any win indicates the opponent started from a disadvantaged random position.

---

## 📊 Evaluation

The `evaluate_agent()` function runs `n` episodes with a fixed ε (default `0` for pure exploitation) and reports outcome distributions:

```python
results = evaluate_agent(env, q_table, nb_episodes=100, epsilon=0)
# → {1: wins, 0: draws, -1: losses, -100: illegal_moves}
```

A second evaluation with `epsilon=0.3` is run in parallel to assess robustness under partial exploration.

---

## 📄 License

MIT
