import os
import torch
import time

import matplotlib.pyplot as plt
import numpy as np

from evotorch import Problem
from evotorch.algorithms import Cosyne, MAPElites
from smart_pong.evotorch_ODIN import set_params
from smart_pong.little_helpers import bin_avg
from smart_pong.mapping import elite_map
from smart_pong.pong import pong
from threadpoolctl import threadpool_limits
from ufuncs import Environment, mutate_trans, write_params, transfer, ev_step, progress, plot_san


debug = False
run_id = 1 if debug else 127
__N = 256       # Number of Neurons
# create folder for results if not existing
if not os.path.exists(f"./Cache/{run_id}/"):
    os.makedirs(f"./Cache/{run_id}/")

#Environment Parameters
__Ne = 3 if debug else 8            # Number of environments at the beginning
__faktor = 3 if debug else 1        #factor for mutated ones per environment for competition 
max_admitted = 2 if debug else 8    #max env that are added to optimization 
capacity = 1 if debug else 8        #total capacity of environments
__M = 2 if debug else 4             #Mutation of environments after x poet steps             

#POET Parameters
__T = 15 if debug else 60           # Number of Poet Steps
__S = 15 if debug else 400           # Number of Steps for Evolution Strategy

#MAP Parameters 
__map_depth = 1
num_actors = 15 if debug else 60    # Parallelisierung
__Na = int(num_actors*2/3)          # Number of Agents (must be an even number for transfer)
  


def fitness(x: torch.Tensor) -> torch.Tensor:
    """ determine fitness of genome """
    net = set_params(x)
    with open("current_parameters.txt", "r") as text:
        par = text.read()
    par  = par.split(" ")[:-1]
    with open("nruns.txt", "r") as text:
        __nruns = text.read()
    __nruns = int(__nruns)
    
    score = []         
    with threadpool_limits(limits=int(1), user_api='blas'):
        for i in range(__nruns):
            net.reset_state()
            score.append(pong(net=net, dur=1000,
                              init_speed= float(par[0]), 
                              speed_up= float(par[1]),
                              perc_paddle= float(par[2]), 
                              shrink= float(par[3]), 
                              max_speed_paddle= float(par[4]), 
                              steps_per_frame= int(par[5]),
                              sight = float(par[6]),
                              ball_noise=0.0,
                              avoid_wall=False,
                              _return = 'wob') 
                         )
   
    means = np.mean(score, axis = 0 )
    return means


#%% Create Environments

env = []
# for i in range(__Ne*__faktor):                                                #for random environments
#     env.append(Environment())

len_x = int(__N * (__N-1) + __N * 9 + 1)
problem = Problem("min", fitness, 
                  solution_length=len_x,
                  initial_bounds=(-np.pi, np.pi),
                  eval_data_length=3,
                  num_actors= num_actors
                 )

env.append(Environment(initial_speed= 4, speed_up = 1.015, perc_paddle= 0.1, 
                	 shrink= 0.2, max_speed_paddle= 6,steps_per_frame= 13, sight= 0.67))
env.append(Environment(initial_speed= 5, speed_up = 1.01, perc_paddle= 0.15, 
                	 shrink= 0.15, max_speed_paddle= 4,steps_per_frame= 7, sight = 0.8))
env.append(Environment(initial_speed= 6, speed_up = 1.0, perc_paddle= 0.12, 
                	 shrink= 0.18, max_speed_paddle= 5,steps_per_frame= 10, sight = 0.6))
env.append(Environment(initial_speed= 4, speed_up = 1.007, perc_paddle= 0.2, 
                	 shrink= 0.2, max_speed_paddle= 5,steps_per_frame= 5, sight = 0.6))
env.append(Environment(initial_speed= 7, speed_up = 1.002, perc_paddle= 0.18, 
                	 shrink= 0, max_speed_paddle= 8,steps_per_frame= 15, sight = 1))
env.append(Environment(initial_speed= 5, speed_up = 1.005, perc_paddle= 0.18, 
                	 shrink= 0.1, max_speed_paddle= 6,steps_per_frame= 20, sight = 0.8))
env.append(Environment(initial_speed= 4, speed_up = 1.006, perc_paddle= 0.2, 
                	 shrink= 0.08, max_speed_paddle= 6,steps_per_frame= 9, sight = 0.75))
env.append(Environment(initial_speed= 6, speed_up = 1.002, perc_paddle= 0.12, 
                	 shrink= 0.02, max_speed_paddle= 7,steps_per_frame= 12, sight = 0.78))

env = [env[i] for i in range(__Ne)]

test_env = []
test_env.append(Environment(initial_speed=6, speed_up=1.01, perc_paddle=0.2, 
                            shrink=0.1, max_speed_paddle=8, steps_per_frame = 20, sight = 0.5))



#%% Create Agents and initialize ES

batches = []

for m in range(__Ne):
    write_params(env[m])
    batch = problem.generate_batch(__Na)
    batches.append(batch)

searcher = Cosyne(problem, popsize = __Na, tournament_size= 2, 
                  mutation_stdev=0.2, mutation_probability= 0.3)

#logger = PandasLogger(searcher)

    
#%% 

plot_scores = plt.figure(1)
plt.title("Scores of Environments")
plot_scores_n = plt.figure(2)
plt.title("Scores normalized of Environments")
plot_test_scores = plt.figure(3)
plt.title("Test Scores")
plot_test_scores_n = plt.figure(4)
plt.title("Test Scores normalized")
all_test_scores = np.zeros((0,2))
all_scores = np.zeros((0,2))
feature_grid = MAPElites.make_feature_grid(
    lower_bounds=[0.2, 0.2],         
    upper_bounds=[3, 5],
    num_bins=17,
    dtype="float32",
)
elite = elite_map(problem, searcher, __map_depth, feature_grid, 5, 'challenge')
save_env = {}

#%%
for t in range(__T):
    start_t = time.time()
    print("Poet " + str (t+1) + ". Epoche") 
    
    old_test_scores = progress(t, test_env, batches, problem, run_id, MAP = True)
    batches, status = ev_step(env, batches, problem, searcher, 
                                              t, __S, run_id, True, True) 
    
    elite.fill(status['best_solution'])
    
    new_test_scores = progress(t, test_env, batches, problem, run_id,         
                                    compare = old_test_scores, MAP = True)

    if __Ne > 1:
        transfered, batches, bin_cnt = transfer(env, batches, problem) 
        plt.figure()
        plt.hist(bin_cnt)
        plt.savefig(f"./Cache/{run_id}/bincount{t}.jpg")
        for i in range(__Ne):
            if not env[i].color():
                env[i].generate_random_color()
        san = plot_san(env, transfered)
        san.savefig(f"./Cache/{run_id}/fluct{t}.png", bbox_inches="tight", dpi=150)
        np.save(f'./Cache/{run_id}/transfered{t}.npy', transfered)
        
    for i in range(__Ne):
        if debug: 
            plt.figure(1)
            plt.plot(np.array((t, t+0.9)), np.array((status['values'][i][2][0], status['values'][i][2][-1])), color = env[i].color())
            plt.figure(2)
            plt.plot(np.array((t, t+0.9)), np.array((1, status['values'][i][2][-1]/status['values'][i][2][0])), color = env[i].color())
        else:
            plt.figure(1)
            plt.plot(np.array((t, t+0.9)), np.array((status['values'][i][2][0:10].mean(), status['values'][i][2][-11:-1].mean())), color = env[i].color())
            plt.figure(2)
            plt.plot(np.array((t, t+0.9)), np.array((1, status['values'][i][2][-11:-1].mean()/status['values'][i][2][0:10].mean())), color = env[i].color())
        plt.figure(3)
        plt.plot(np.array((t, t+0.9)), np.vstack((old_test_scores, new_test_scores))[:,i], color = env[i].color())
        plt.figure(4)
        plt.plot(np.array((t, t+0.9)), np.vstack((old_test_scores/old_test_scores, 
                            new_test_scores/old_test_scores))[:,i], color = env[i].color())
    
    if not debug:                                                           # Save figures and Arrays 
        plot_scores_n.savefig(f'./Cache/{run_id}/scores_normalised{t}.jpg')
        plot_test_scores_n.savefig(f'./Cache/{run_id}/test_scores_normalised{t}.jpg')
        plot_scores.savefig(f'./Cache/{run_id}/scores{t}.jpg')
        plot_test_scores.savefig(f'./Cache/{run_id}/test_scores{t}.jpg')
        
        all_test_scores = np.vstack((all_test_scores, 
                                      np.array(np.vstack((old_test_scores.T,
                                                          new_test_scores.T))).T))
        np.save(f"./Cache/{run_id}/test_scores_array.npy", all_test_scores)
        
        all_scores = np.vstack((all_scores, 
                                np.array(np.vstack((status['values'][i][2][0],
                                                    status['values'][i][2][-1])).T)))
        np.save(f"./Cache/{run_id}/scores_array.npy", all_scores)
        
        np.save(f"./Cache/{run_id}/elite_evals.npy", np.array([elite._map[k].evals for k in range(__map_depth)]))
 
        np.save(f"./Cache/{run_id}/elite_values.npy", [elite._map[k].values for k in range(__map_depth)])
        
        np.save(f"./Cache/{run_id}/batches.npy", batches)
        
        status.to_pickle(f"./Cache/{run_id}/status_ep{t}.pkl")
        
        save_env.update({f'epoche {t}': env.copy()})
        np.savez(f"./Cache/{run_id}/environments.npz", save_env)
    
        #visuelle Ausgabe des Status
        labels = ['best', 'worst', 'mean']
        if not os.path.exists(f"./Cache/{run_id}/poet{t}/"):
            os.makedirs(f"./Cache/{run_id}/poet{t}/")
        for i in range(3):
            plt.figure()
            for j in range(__Ne):
                plt.plot(bin_avg(status['values'][j][i], __S if debug else int(__S/10)), label=labels[i]+f" Env {env[j]._idx}", color = env[j].color())
            plt.legend(loc = 'lower left')
            plt.savefig(f"./Cache/{run_id}/poet{t}/" + labels[i] + ".jpg")
    
    if t < (__T-1):  
        env, batches = mutate_trans(env, problem, batches, transfered, MAP = True)
        __Ne = len(env)
    # if (t+1) % __M == 0: 
    #     print("Mutation")
    #     env, batches = mutate_one(env, problem, batches, MAP = True)

    end_t = time.time()
    print((end_t - start_t)/60)
    

      
