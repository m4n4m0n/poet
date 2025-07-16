import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from evotorch import Problem, SolutionBatch
from evotorch.algorithms import Cosyne
from evotorch.logging import PandasLogger
from smart_pong.pong import pong
from smart_pong.evotorch_ODIN import set_params, evaluate
from sklearn.neighbors import KNeighborsRegressor
from threadpoolctl import threadpool_limits
from ufuncs import mtrx_to_stake_diagr, mutated_env

#Pong Parameters
__nruns = 5     # Number of Runs in fitness function

#Network Parameters
__N = 256       # Number of Neurons

#Environment Parameters
__Ne = 8        # Number of environments
__Nre = __Ne*4  # Number of random initialized environments 
__Na = 60       # Number of Agents

#POET Parameters
__T = 30        # Number of Poet Steps
__S = 200        # Number of Steps for Evolution Strategy
 

def fitness(x: torch.Tensor) -> torch.Tensor:
    """ determine fitness of genome """
    net = set_params(x)

    score = []
    with threadpool_limits(limits=int(1), user_api='blas'):
        for i in range(__nruns):
            net.reset_state()
            score.append(pong(net=net, player= net, dur=1000,
                              init_speed= int(env.iloc[__idx, 0]), 
                              speed_up= env.iloc[__idx, 1],
                              perc_paddle= env.iloc[__idx, 2], 
                              shrink= env.iloc[__idx, 3], 
                              max_speed_paddle= int(env.iloc[__idx, 4]), 
                              steps_per_frame=int(env.iloc[__idx, 5]),
                              ball_noise=0.0,
                              avoid_wall=False,
                              ) 
                         )
    score.sort()
    return np.mean(score)


def mutate(env):
    env = mutated_env(env, int(__Nre/__Ne))
    scores = np.zeros(__Nre)
    for i in range(__Nre): 
        __idx = i 
        ges_score = 0
        for j in range(__Ne):
            key = f'searcher{j}'
            networks = problems[key].population.values
            for k in range(len(agents)):
                ges_score += fitness(networks[k])
        scores[__idx] = ges_score    
    abs_diff = np.abs(scores - np.mean(scores))
    closest_indices = np.argsort(abs_diff)[:__Ne]
    env = env.loc[closest_indices]
    env = env.reset_index(drop=True)
    return env

            
#%% Create Environments

"""pong(dur=850, init_speed=6, speed_up=1.01, speed_up_interval=20, 
        ball_noise=0.0, perc_paddle=0.2, shrink=0.1, max_speed_paddle=8,
        control_speed = True, direction=1, inp_scale=1, steps_per_frame=20,
        inp_pos_ball_rel=True, inp_pos_ball_differential_dist=False,
        inp_pos_paddle=True, inp_speed_paddle=False, inp_speed_ball=False,
        sight=0.66, delay=0, timed=False, easy_reset=1,
        player=None, net=None, training=False,
        plot=False, plot_guide=True, FPS=25, debug=False,
        avoid_wall=False, _return='obj1', T=0, **kwags):"""


env = pd.DataFrame({'initial_speed': np.random.randint(1, 20, size=__Nre), 
                   'speed_up': np.random.uniform(low=1.0, high=1.01, size=__Nre),
                   'perc_paddle': np.random.uniform(low=0.1, high=0.2, size=__Nre), 
                   'shrink': np.random.uniform(low=0.0, high=0.2, size=__Nre), 
                   'max_speed_paddle': np.random.randint(2, 10, size=__Nre),
                   'steps_per_frame': np.random.randint(5, 30, size=__Nre)
                   })
scores = np.zeros(__Nre)

len_x = int(__N * (__N-1) + __N * 9 + 1)
problem = Problem("min", fitness, 
                  solution_length=len_x,
                  initial_bounds=(-1.0, 1.0),
                  num_actors= 60
                 )

networks = problem.generate_batch(popsize=__Na)
for i in range(__Nre): 
    __idx = i 
    problem.evaluate(networks)
    score = 0
    x = networks._evdata
    for j in range(__Na): 
        score += x[j]
    scores[i] = score / __Na

abs_diff = np.abs(scores - np.mean(scores))
closest_indices = np.argsort(abs_diff)[:__Ne]

for i in closest_indices:
    if scores[i] > 20:
        print("NOO")

fix_test_env = env.loc[np.argsort(np.abs(scores - np.quantile(scores, 0.75)))[0]]

env = env.loc[closest_indices]
env = env.reset_index(drop=True)


#%% Create Agents and initialize ES
problems = {}
for i in range(__Ne):
    __idx = i 
    key = f'key{i}'  
    problems[key] = __idx
    key = f'problem{i}'
    problem = Problem("min", fitness, solution_length=len_x,
                      initial_bounds=(-1.0, 1.0),num_actors= 60
                     )
    problems[key] = problem
    key = f'searcher{i}'
    searcher = Cosyne(problem, popsize = __Na, tournament_size= 2, 
                      mutation_stdev=0.3, )
    problems[key] = searcher 
    key = f'logger{i}'
    logger = PandasLogger(searcher)
    problems[key] = logger
    
#%% new environments created by mutation


file_ss = open("searcher_states.txt","w")
file_scores = open("searcher_scores.txt","w")

for t in range(__T):
    print("Poet " + str (t+1) + ". Epoche")
    
    if t > 0:
        env = mutate(env)                   ## new environments created by mutation
    
    file_ss.write("Poet " + str (t+1) + ". Epoche\t")
    for i in range(__Ne): 
        file_scores.write(f"{i}. env: " + str(env.iloc[i]))
    
    for m in range(__Ne):                   ## each agent independently optimized
        print("ES " + str (m+1) + ". Environment")  
        file_ss.write("Poet " + str (t+1) + ". Epoche; ES Environment:" + str(m))
        for n in range(__S):
            problems[f'searcher{m}'].step()
            file_ss.write(problems[f'searcher{m}'].status._to_string())
    
    
    transfered = np.zeros((__Ne,__Ne))                                           # Matrix um Werte aus Environments zu zählen
    new_agents = {}
    if len(env) > 1:                        
        for m in range(__Ne):               ## create transfer agents 
            __idx = problems[f'key{m}']
            for n in range(__Ne):
                if n == 0: 
                    agents = problems[f'searcher{n}'].population.clone()
                    if problems[f'key{n}'] != m:                                 # Falls nicht ursprüngliches Environment, 
                        problem.evaluate(agents)                                 # Agenten neue Fitnesswerte zuweisen
                else: 
                    new_batch = problems[f'searcher{n}'].population.clone()
                    if problems[f'key{n}'] != m:                                 # Falls nicht ursprüngliches Environment,
                        problem.evaluate(new_batch)                              # Agenten neue Fitnesswerte zuweisen
                    agents = SolutionBatch.cat([agents, new_batch])                     
            best = agents.argsort()[:__Na]        
            new_agents[f'key{m}'] = __idx
            new_agents[f'agents{m}'] = agents[best]
            bins = [x for x in range(0, __Na*__Ne+1, __Na)]
            transfered[m] = np.bincount(np.searchsorted(bins,best, 
                                                        side= 'right'))[1:]      # 1. Wert steht für die Anzahl der Agenten aus dem 1. Environment (index < __Na),
                                                                                 # der 2. aus dem 2. Environment (__Na < index < __Na * 3) usw.
       
        
        for m in range(__Ne):               ## transfer the agents
            __idx = problems[f'key{m}']
            overwrite = problems[f'searcher{m}'].population.access_values()
            overwrite_evals = problems[f'searcher{m}'].population.access_evals()
            for i in range(__Na):
                overwrite[i] = new_agents[f'agents{m}'].values[i]
                overwrite_evals[i] = new_agents[f'agents{m}'].evals[i] 
   
        plot = mtrx_to_stake_diagr(transfered, __Ne)                             # visualize ratio of transfered agents
        plot.savefig(f'transfer_{t+1}epoche.jpg')
    
    file_scores.write(f"Poet {t}. Epoche: \t")
    for m in range(__Ne): 
        score = fitness(problems[f'searcher{m}'].status['pop_best'].values)
        file_scores.write(f'env {m} best: ' + str(score) + '\t')
    file_scores.write('\n\n')  
    
    
file_ss.close()
file_scores.close()

for i in range(__Ne):
    pds = problems[f'logger{i}'].to_dataframe()
    pds.to_pickle("logger" + str(i) + ".pkl")