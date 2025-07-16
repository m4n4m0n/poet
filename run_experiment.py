import os
import numpy as np
import pandas as pd
from evotorch import Problem
from evotorch.algorithms import Cosyne
from config import (
    __N,
    __Na,
    __Ne,
    __Nre,
    __T,
    __S,
    __nruns,
    pbounds,
    CACHE_DIR,
)
from ufuncs import fitness

def create_initial_environments(__Nre):
    """
    Erstellt initiale Zufalls-Umgebungen als DataFrame.
    """
    envs = pd.DataFrame({
        'initial_speed': np.random.randint(pbounds['initial_speed'][0],
                                           pbounds['initial_speed'][1] + 1,
                                           size=__Nre),
        'speed_up': np.random.uniform(pbounds['speed_up'][0],
                                      pbounds['speed_up'][1],
                                      size=__Nre),
        'perc_paddle': np.random.uniform(pbounds['perc_paddle'][0],
                                         pbounds['perc_paddle'][1],
                                         size=__Nre),
        'shrink': np.random.uniform(pbounds['shrink'][0],
                                    pbounds['shrink'][1],
                                    size=__Nre),
        'max_speed_paddle': np.random.randint(pbounds['max_speed_paddle'][0],
                                              pbounds['max_speed_paddle'][1] + 1,
                                              size=__Nre),
        'steps_per_frame': np.random.randint(pbounds['steps_per_frame'][0],
                                             pbounds['steps_per_frame'][1] + 1,
                                             size=__Nre),
        'sight': np.random.uniform(pbounds['sight'][0],
                                   pbounds['sight'][1],
                                   size=__Nre),
    })
    return envs

def main():
    print("Starte POET-Experiment...")

    # Environments vorbereiten
    env_df = create_initial_environments(__Nre)

    # Problem dimension
    len_x = int(__N * (__N - 1) + __N * 9 + 1)

    problem = Problem(
        "min",
        lambda x: fitness(x, env_row=env_df.iloc[0], __nruns=__nruns),
        solution_length=len_x,
        initial_bounds=(-1.0, 1.0),
        num_actors=__Na
    )

    searcher = Cosyne(
        problem,
        popsize=__Na,
        tournament_size=2,
        mutation_stdev=0.3,
    )

    for step in range(__S):
        searcher.step()
        best_eval = searcher.status['pop_best'].evals.item()
        print(f"Step {step+1}/{__S}: Best = {best_eval}")

    print("Experiment abgeschlossen!")

if __name__ == "__main__":
    main()