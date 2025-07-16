import torch
import time
import matplotlib.pyplot as plt
import numpy as np

from evotorch import Problem
from evotorch.algorithms import Cosyne
from smart_pong.evotorch_ODIN import set_params
from smart_pong.pong import pong
from threadpoolctl import threadpool_limits
from ufuncs import mtrx_to_stake_diagr, Environment, mutate_envs, write_params, progress, transfer, progress_test, ev_step


debug =False

#Pong Parameters
__nruns = 5     # Number of Runs in fitness function
__N = 256       # Number of Neurons

#Environment Parameters
__Ne = 5        # Number of environments
__faktor = 3
max_admitted = 3
capacity = 8
 
#Network Parameters
__Na = 5 if debug else 60      # Number of Agents (must be an even number for transfer)
  
#POET Parameters
__T = 5 if debug else 30       # Number of Poet Steps
__S = 20 if debug else 200      # Number of Steps for Evolution Strategy
       
array =  True #if not debug else False
num_actors = __Na       # Parallelisierung
 

def fitness(x: torch.Tensor) -> torch.Tensor:
    """ determine fitness of genome """
    net = set_params(x)
    with open("current_parameters.txt", "r") as text:
        par = text.read()
    par  = par.split(" ")[:-1]
    
    score = []
    with threadpool_limits(limits=int(1), user_api='blas'):
        for i in range(__nruns):
            net.reset_state()
            score.append(pong(net=net, dur=1000,
                              init_speed= int(par[0]), 
                              speed_up= float(par[1]),
                              perc_paddle= float(par[2]), 
                              shrink= float(par[3]), 
                              max_speed_paddle= int(par[4]), 
                              steps_per_frame= int(par[5]),
                              ball_noise=0.0,
                              avoid_wall=False,
                              ) 
                         )
       
    return np.mean(score)

            
#%% Create Environments

env = []
for i in range(__Ne*__faktor):
    env.append(Environment())
    
scores = np.zeros(__Ne*__faktor)

len_x = int(__N * (__N-1) + __N * 9 + 1)
problem = Problem("min", fitness, 
                  solution_length=len_x,
                  initial_bounds=(-1.0, 1.0),
                  num_actors= num_actors
                 )

networks = problem.generate_batch(popsize=__Na)
for i in range(__Ne*__faktor):
    write_params(env[i])
    problem.evaluate(networks)
    score = 0
    evdata = networks._evdata
    for j in range(__Na): 
        score += evdata[j]
    scores[i] = score / __Na

abs_diff = np.abs(scores - np.mean(scores))
closest_indices = np.argsort(abs_diff)[:__Ne]

test_env = env[np.argsort(np.abs(scores - np.quantile(scores, 0.75)))[0]]       # Festes Testproblem Erstellen (in Liste wegen Funktionsnutzung)

env_old = env
env = []
for i in closest_indices:
    env.append(env_old[i])


#%% Create Agents and initialize ES

batches = []
for m in range(__Ne):
    write_params(env[m])
    batch = problem.generate_batch(__Na)
    batches.append(batch)

searcher = Cosyne(problem, popsize = __Na, tournament_size= 2, 
                  mutation_stdev=0.3, )

#logger = PandasLogger(searcher)

    
#%% new environments created by mutation

plot_scores = plt.figure(1)
plt.title("Scores of Environments")
plot_scores_n = plt.figure(2)
plt.title("Scores normalized of Environments")
plot_test_scores = plt.figure(3)
plt.title("Test Scores")
plot_test_scores_n = plt.figure(4)
plt.title("Test Scores normalized")
blue = 0.0
all_test_scores = np.zeros((0,2))
all_scores = np.zeros((0,2))

for t in range(__T):
    ev_best = []
    ev_worst = []
    start_t = time.time()
    old_test_scores = progress_test(t, test_env, batches, problem, array= array)
    old_scores = progress(t, env, batches, problem, array = array)
    print("Poet " + str (t+1) + ". Epoche") 
    
    batches, ev_best, ev_worst = ev_step(env, batches, problem, searcher, t, __S, ev_best, ev_worst) # evolutionary step
    
    new_test_scores = progress_test(t, test_env, batches, problem, array = array, compare = old_test_scores)

    new_scores = progress(t, env, batches, problem, array = array, compare = old_scores)

    
    transfered, batches, all_agents = transfer(env, batches, problem) 
    plot = mtrx_to_stake_diagr(transfered, __Ne)                                # visualize ratio of transfered agents
    
    plt.figure(1)
    plt.plot(np.vstack((old_scores, new_scores)), color =  [0.3, blue, 0.6, 0.4])
    plt.figure(2)
    plt.plot(np.vstack((old_scores/old_scores, new_scores/old_scores)), color =  [0.3, blue, 0.6, 0.4])
    plt.figure(3)
    plt.plot(np.vstack((old_test_scores, new_test_scores)), color =  [0.3, blue, 0.6, 0.4])
    plt.figure(4)
    plt.plot(np.vstack((old_test_scores/old_test_scores, new_test_scores/old_test_scores)), color =  [0.3, blue, 0.6, 0.4])
    
    blue = blue + (1.0 /__T)
    
    
    if not debug:                                                               # Save figures and Arrays 
        plot.savefig(f'transfer_{t+1}epoche.jpg')
        plot_scores.savefig('scores.jpg')
        plot_scores_n.savefig('scores_normalised.jpg')
        plot_test_scores.savefig('test_scores.jpg')
        plot_test_scores_n.savefig('test_scores_normalised.jpg')
        
        all_test_scores = np.vstack((all_test_scores, 
                                      np.array(np.vstack((old_test_scores.T,
                                                          new_test_scores.T))).T))
        np.save("test_scores_array.npy", all_test_scores)
        
        all_scores = np.vstack((all_scores, 
                                np.array(np.vstack((old_scores.T,
                                                    new_scores.T))).T))
        np.save("scores_array.npy", all_scores)
        
        
        
        
    
    env = mutate_envs(env, __faktor, max_admitted, capacity, problem, all_agents, debug)
    for i in range(len(env)- __Ne):
        write_params(env[__Ne+i])
        batch = problem.generate_batch(__Na)
        batches.append(batch)
    __Ne = len(env)

    end_t = time.time()
    print((end_t - start_t)/60)
    

      
