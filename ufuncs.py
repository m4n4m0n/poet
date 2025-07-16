import random
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evotorch import SolutionBatch
from typing import Optional
from parameters import pbounds
from pySankey.sankey import sankey
from sklearn.neighbors import NearestNeighbors


def to_hex(color):
    color = tuple([int(x*255) for x in color])
    return '#%02x%02x%02x' % color


def mtrx_to_stake_diagr(matrix, number_environments): 
    rand_color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_environments)]
     
    fig = plt.figure()
    for m in range(number_environments):
        if m == 0:
            percentages= [i  for i in matrix[:, m]]
            plt.bar(range(number_environments), percentages, color = rand_color[m], width=0.85, label=f'{m+1}. environment')
            
        else: 
            new_percentages = [i for i in matrix[:, m]]
            plt.bar(range(number_environments), new_percentages,color = rand_color[m], edgecolor='white',
                    bottom = percentages, width=0.85, label=f'{m+1}. environment')
            for i in range(number_environments): 
                percentages[i] = percentages[i] + new_percentages[i]
    plt.xticks(range(number_environments), range(1, number_environments+1))
    plt.xlabel("environments")
    plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    fig.show()
    
    return fig

def plot_san(env, transfered):
    col_dict = {f"Env {env[i]._idx}" :  to_hex(env[i].color()) for i in range(len(env))}
    sankey(left=[ "Env " + str(env[i]._idx) for i in np.nonzero(transfered)[1]], 
            right=[ "Env " + str(env[i]._idx) for i in np.nonzero(transfered)[0]], 
            leftWeight= transfered.ravel()[transfered.ravel() != 0], 
            rightWeight=transfered.ravel()[transfered.ravel() != 0], 
            colorDict= col_dict,
            aspect=20, fontsize=20 
            )
    
    san = plt.gcf()
    san.set_size_inches(6, 6)
    san.set_facecolor("w")
    return san


def write_params(env):
    with open("current_parameters.txt", "w") as text:
        for i in range(len(env.as_list())):
            text.write(str(env.as_list()[i]) + " ")


def mutate_envs(envs, faktor, max_admitted, capacity, problem, batches, debug, mc_fakt, MAP = False):
    if (faktor > 0 ) and (max_admitted > 0):
        envs_new = []
        for m in range(len(envs)):
            for i in range(faktor):
                envs_new.append(envs[m].mutate_child())                      
        scores_env = np.zeros(len(envs_new))
        agents = batches[0].clone()
        for m in range(len(batches)-1):
            agents = agents.concat(batches[m].clone())
        
        for i in range(len(envs_new)):
            write_params(envs_new[i])
            problem.evaluate(agents)
            if MAP:
                scores_env[i] = agents.evals[:,1].mean().item()
            else:
                scores_env[i] = agents.evals.mean().item()
                
        if not debug: 
            envs_new = [i for i, j in zip(envs_new, scores_env) if j > 5 and j < (12 - (8*mc_fakt))]        # Minimal criterion, do not use for debug - list will be empty
       
        if len(envs_new):
            envs_new = sort_by_novelty(envs_new, envs)
            
            if len(envs_new) > max_admitted:
                for i in range(max_admitted): 
                    envs.append(envs_new[i])
                    write_params(envs_new[i])
                    problem.evaluate(agents)
                    batches.append(agents.take_best(len(batches[0])))
            else:
                for i in range(len(envs_new)): 
                    envs.append(envs_new[i])
                    write_params(envs_new[i])
                    problem.evaluate(agents)
                    batches.append(agents.take_best(len(batches[0])))
            
            if len(envs)>capacity:
                ages = []
                for i in range(len(envs)):
                    ages.append((envs[i].age,envs[i]._idx))
                ages = sorted(ages, key = lambda x: x[1])
                for i in range(len(envs)- capacity):
                    for j, o in enumerate(envs):
                        if o._idx == ages[i][1]:
                            del envs[j]
                            del batches[j]
    return envs, batches

#Mutation of Environments, if environments don't improve at least 2% they're deleted 
def mutate_comp(envs, faktor, capacity, problem, batches, debug, status, MAP = False):
    envs_new = envs
    envs = []
    for m in range(len(envs_new)):
        for i in range(faktor):
            envs.append(envs_new[m].mutate_child())                      
    scores_env = np.zeros(len(envs))
    agents = batches[0].clone()
    for m in range(len(batches)-1):
        agents = agents.concat(batches[m])
   
    delete = []
    for m in range(len(envs_new)):
        if not debug: 
            if status['values'][m][2][-11:-1].mean() > (0.98 * status['values'][m][2][0:10].mean()):
                delete.append(m)
        else:
            if status['values'][m][2][-1] > (0.98 * status['values'][m][2][0]):
                delete.append(m)
                
    for i in range(len(delete)): 
        del envs_new[delete[-(i+1)]]
        del batches[delete[-(i+1)]]

    if len(envs_new) < capacity: 
            
        for i in range(len(envs)):
            write_params(envs[i])
            problem.evaluate(agents)
            if MAP:
                scores_env[i] = agents.evals[:,1].mean().item()
            else:
                scores_env[i] = agents.evals.mean().item()
                
        if not debug: 
            envs = [i for i, j in zip(envs, scores_env) if j > 5 and j < 15]        # Minimal criterion, do not use for debug - list will be empty

        if len(envs):
            novel = sort_by_novelty(envs, envs_new)
            
            i = 0
            while len(envs_new) < capacity and i < len(novel): 
                envs_new.append(novel[i])
                write_params(novel[i])
                problem.evaluate(agents)
                batches.append(agents[:(len(batches[0]))])
                i += 1
    return envs_new, batches


def mutate_one(envs, problem, batches, MAP = False):
    envs_new = []
    agents = batches[0].clone()
    for m in range(len(batches)-1):
        agents = agents.concat(batches[m].clone())
    __Na = len(batches[0])
    batches = []
    print(agents.evals)
    for m in range(len(envs)):
        sc_check = False
        children = []
        sc_children = []
        count = 0 
        while sc_check == False and count <= 10 : 
            child = envs[m].mutate_child()
            write_params(child)
            problem.evaluate(agents)
            if MAP: 
                score = agents.evals[:,1].mean().item()
                if score > 5 and score < 12: 
                    sc_check = True
                else: 
                    children.append(child)
                    sc_children.append(score)
            else: 
                score = agents.evals.mean().item()
                if score > 5 and score < 12: 
                    sc_check = True
                else: 
                    children.append(child)
                    sc_children.append(score)
            print(score)
            count = count +1
        if sc_check: 
            envs_new.append(envs[m].mutate_child())
        else:
            idx = sc_children.index(min(sc_children))
            envs_new.append(children[idx])
        batches.append(agents.take_best(__Na))
        print(batches[m].evals)
        print(agents.evals)
    return envs_new, batches


def mutate_trans(envs, problem, batches, transfered, MAP = False):
    __Ne = len(envs)
    __Na = len(batches[0])
    agents = batches[0].clone()
    for m in range(len(batches)-1):
        agents = agents.concat(batches[m].clone())
    __Na = len(batches[0])
    
    delete = []
    for m in range(__Ne):
        if transfered[:,m].sum() < __Na:
            delete.append(m)
                
    for i in range(len(delete)): 
        del envs[delete[-(i+1)]]
        del batches[delete[-(i+1)]]
    print(__Ne)
    print(len(envs))
    for m in range((__Ne - len(envs))):
        sc_check = False
        while sc_check == False: 
            new = Environment()
            write_params(new)
            problem.evaluate(agents)
            if MAP: 
                score = agents.evals[:,1].mean().item()
                if score > 5 and score < 12: 
                    sc_check = True
            else: 
                score = agents.evals.mean().item()
                if score > 5 and score < 12: 
                    sc_check = True
            print(score)
        envs.append(new)
        batches.append(agents.take_best(__Na))
    return envs, batches


def sort_by_novelty(env, env_old):
    if len(env) > 1:
        env_m = np.zeros((len(env), 6))
        for i in range(len(env)):
            env_m[i, 0] = env[i].initial_speed 
            env_m[i, 1] = env[i].speed_up
            env_m[i, 2] = env[i].perc_paddle
            env_m[i, 3] = env[i].shrink
            env_m[i, 4] = env[i].max_speed_paddle
            env_m[i, 5] = env[i].steps_per_frame
        
        env_old_m = np.zeros((len(env_old), 6))
        for i in range(len(env_old)):
            env_old_m[i, 0] = env_old[i].initial_speed 
            env_old_m[i, 1] = env_old[i].speed_up
            env_old_m[i, 2] = env_old[i].perc_paddle
            env_old_m[i, 3] = env_old[i].shrink
            env_old_m[i, 4] = env_old[i].max_speed_paddle
            env_old_m[i, 5] = env_old[i].steps_per_frame
            
    
        neigh = len(env_old)

        kneight = NearestNeighbors(n_neighbors= neigh).fit(env_old_m)
        distances, indices = kneight.kneighbors(env_m)
        sum_distances = sum(distances.T)
        indices = np.argsort(sum_distances)
        new_env = []
        for i in range(len(indices)): 
            idx = indices[-(i+1)]
            new_env.append(env[idx])
        return new_env


def ev_step(env, batches, problem, searcher, t, __S, run_id, 
            save_best_values = False, MAP = False):
    __Ne = len(env)
    __Na = len(batches[0])
    
    if save_best_values: 
        _status = pd.DataFrame(columns=['env_idx', 'best', 'worst', 'values', 
                                        'best_solution'])
    else: 
        _status = pd.DataFrame(columns=['env_idx', 'best', 'worst', 'values'])
    
    for m in range(__Ne):                   
        start_t = time.time()
        write_params(env[m])
        problem.evaluate(batches[m])
        overwrite = searcher.population.access_values()                         #! Muss als Variable gespeichert 
        overwrite_evals = searcher.population.access_evals()
        for i in range(__Na):
            overwrite[i] = batches[m].values[i]
            overwrite_evals[i] = batches[m].evals[i] 
        print("ES " + str(m+1) + ". Environment") 
        
        step_data = np.zeros((3, __S))
        
        if save_best_values:
            step_best_values = []
        
        for i in range(__S):
            searcher.step()
            if MAP: 
                step_data[0, i] = searcher.population.evals[:,0].min().item()
                step_data[1, i] = searcher.population.evals[:,0].max().item()
                step_data[2, i] = searcher.population.evals[:,0].mean().item()
            else: 
                step_data[0, i] = searcher.population.evals.min().item()
                step_data[1, i] = searcher.population.evals.max().item()
                step_data[2, i] = searcher.population.evals.mean().item()
            
            
            if save_best_values: 
                if MAP:
                    best_idx = searcher.population.evals[:,0].min(dim=0)[1].item() 
                else:
                    best_idx = searcher.population.evals.min(dim=0)[1].item()
                step_best_values.append(searcher.population[best_idx])         
        env[m].mature()                                                          # Genutzes Environment 'altert'
            
        best = min(step_data[0])
        worst = max(step_data[1])
        
        if save_best_values:
            best = step_best_values[np.where(step_data[0] == best)[0][0]]
            
        with open(f"./Cache/{run_id}/searcherstates.txt", "a") as text:
            text.write("\n Poet " + str (t+1) + ". Epoche; ES Environment:" + str(m)+ "\n")
            text.write("Best: " + str(best) + "\t\t Worst: " + str(worst))
            text.write("\n\n")

        if save_best_values & MAP:
            _status.loc[f'{len(_status)}'] = [env[m].idx, 
                                                  best.evals[0].item(),
                                                  worst, step_data, best]
        elif MAP:
                _status.loc[f'{len(_status)}'] = [env[m].idx, best.evals.item(),
                                                  worst, step_data, best]
        else:
            _status.loc[f'{len(_status)}'] = [env[m].idx, best, worst, step_data]
        
        overwrite = batches[m].access_values()
        overwrite_evals = batches[m].access_evals()
        for i in range(__Na):
            overwrite[i] = searcher.population.values[i]
            overwrite_evals[i] = searcher.population.evals[i]
        end_t = time.time()
        print((end_t - start_t)/60)
        
    return batches, _status


def progress(t, env, batches, problem, run_id, compare = None, MAP = False):
    scores = np.zeros(len(batches))
    with open("nruns.txt", "w") as text:                                        # damit hier über 10 pong spiele gemittelt wird -> weniger sprünge 
        text.write(str(10))
    
    write_params(env[0])    
    for m in range(len(batches)):
        if m > 0 & len(env) > 1:                                                # Falls Test-Environment nicht params überschreiben, da gleichbleiben
            write_params(env[m])
        networks = batches[m].clone()
        assert len(networks) > 0
        problem.evaluate(networks)
        if MAP:
            scores[m] = networks.evals[:,1].mean().item()
        else:
            scores[m] = networks.evals.mean().item()
        
        
    if compare is not None:
        with open(f"./Cache/{run_id}/progress.txt", "a") as text:               # Überwachung der neuen Environments
            text.write ("Poet " + str (t+1) + ". Epoche\n")
            if len(env)>1: 
                text.write('Environments:\n\n')
            else:
                text.write('Batches on Test Env: \n')
            for n in range(len(batches)):
                if len(env) > 1:
                    text.write(f"env {n} \n" + str(env[m].string) + "\n\n")
                text.write(f' {n} old score: ' + str(compare[n]) + 
                           '  new score: ' + str(scores[n]) + '\n\n')
            text.write('\n\n\n')  
    
    with open("nruns.txt", "w") as text:                                        # zurück auf 5 Spiele stellen für Optimierung 
        text.write(str(5))
    return scores
   
        
def transfer(env, batches, problem):
    __Ne = len(env)
    __Na = len(batches[0])                                                      # Matrix um Werte aus Environments zu zählen   
    new_idx =[]
    transfered = np.zeros((__Ne, __Ne))
    bin_cnt  = np.zeros((1)) 
    agents = batches[0].clone()
    for m in range(__Ne-1):
        agents = SolutionBatch.cat([agents, batches[m+1].clone()])
    batches = []
    for m in range(__Ne):               ## create transfer agents 
        write_params(env[m])
        problem.evaluate(agents)   
        best = agents.evals[:,1].argsort()[:__Na]
        new_idx.append(best)
        bins = [x for x in range(0, __Na*len(env)+1, __Na)]
        transfered[m] = np.bincount(np.searchsorted(bins,best,side= 'right'),
                                    minlength = len(env)+1)[1:]                 # 1. Wert steht für die Anzahl der Agenten aus dem 1. Environment (index < __Na),
                                                                                # der 2. aus dem 2. Environment (__Na < index < __Na * 3) usw.
        bin_cnt = np.hstack((bin_cnt, (best%10)))
        batches.append(agents.take_best(__Na))
    return transfered, batches, bin_cnt


env_count = []

class Environment:
    """
    Envrionment is an class that holds parameters for creating pong games. 
    They are able to generate Children, that have similar but not same parameters. 
    
    """
    def __init__(
            self,
            initial_speed: Optional[float] = None, 
            speed_up: Optional[float] = None,
            perc_paddle: Optional[float] = None, 
            shrink: Optional[float] = None, 
            max_speed_paddle: Optional[float] = None,
            steps_per_frame: Optional[int] = None,
            sight: Optional[float] = None
            ):
        
            if initial_speed is None:
                initial_speed = np.random.randint(pbounds['initial_speed'][0],
                                                  pbounds['initial_speed'][1])
            self._initial_speed = self._check_bounds(initial_speed, 'initial_speed')
        
            if speed_up is None: 
                speed_up = np.random.uniform(pbounds['speed_up'][0],
                                             pbounds['speed_up'][1])
            self._speed_up = self._check_bounds(speed_up, 'speed_up')
            
                
            if perc_paddle is None: 
                perc_paddle = np.random.uniform(pbounds['perc_paddle'][0],
                                                pbounds['perc_paddle'][1])
            self._perc_paddle = self._check_bounds(perc_paddle, 'perc_paddle')
            
            
            if shrink is None: 
                shrink = np.random.uniform(pbounds['shrink'][0],
                                           pbounds['shrink'][1])
                
            self._shrink = self._check_bounds(shrink, 'shrink')
            
                
            if max_speed_paddle is None:
                max_speed_paddle = np.random.randint(pbounds['max_speed_paddle'][0],
                                                     pbounds['max_speed_paddle'][1])
            self._max_speed_paddle = self._check_bounds(max_speed_paddle, 'max_speed_paddle')
            
                
            if steps_per_frame is None: 
                steps_per_frame = np.random.randint(pbounds['steps_per_frame'][0],
                                                    pbounds['steps_per_frame'][1])
            self._steps_per_frame = self._check_bounds(steps_per_frame, 'steps_per_frame')
            
            if sight is None: 
                sight = np.random.uniform(pbounds['sight'][0],
                                          pbounds['sight'][1])
            self._sight = self._check_bounds(sight, 'sight')
            
            self._idx = len(env_count)
            env_count.append(self._idx)
            self._ancestors = []
            self.age = 0
            self._color = None
            self.string = f"""Environment {self._idx}: 
                \t initial_speed: {self._initial_speed}, 
                \t speed up: {self._speed_up}, 
                \t perc_paddle: {self._perc_paddle} 
                \t shrink: {self._shrink}, 
                \t max_speed_paddle: {self._max_speed_paddle}, 
                \t steps_per_frame: {self._steps_per_frame}
                \t sight: {self._sight}"""
            
    def __repr__(self):
        return f"Environment {self._idx}"

    def __str__(self):
        return self.to_string()
    
    def to_string(self):
        self.string = f"""Environment {self._idx}: 
            \t initial_speed: {self._initial_speed}, 
            \t speed up: {self._speed_up}, 
            \t perc_paddle: {self._perc_paddle} 
            \t shrink: {self._shrink}, 
            \t max_speed_paddle: {self._max_speed_paddle}, 
            \t steps_per_frame: {self._steps_per_frame}
            \t sight: {self._sight}"""
        return self.string
    
    @property
    def idx(self):
        return self._idx
        
    @property
    def initial_speed(self):
        return self._initial_speed
	
    @initial_speed.setter
    def initial_speed(self, new_value):
        self._initial_speed = self._check_bounds(new_value, 'initial_speed')
        
    @property
    def speed_up(self):
        return self._speed_up
	
    @speed_up.setter
    def speed_up(self, new_value):
        self._speed_up = self._check_bounds(new_value, 'speed_up')

    @property
    def perc_paddle(self):
        return self._perc_paddle
	
    @perc_paddle.setter
    def perc_paddle(self, new_value):
        self._perc_paddle = self._check_bounds(new_value, 'perc_paddle')
    
    @property
    def shrink(self):
        return self._shrink
	
    @shrink.setter
    def shrink(self, new_value):
        self._shrink = self._check_bounds(new_value, 'shrink')
    
    @property
    def max_speed_paddle(self):
        return self._max_speed_paddle
	
    @max_speed_paddle.setter
    def max_speed_paddle(self, new_value):
        self._max_speed_paddle = self._check_bounds(new_value, 'max_speed_paddle') 
       
    @property
    def steps_per_frame(self):
        return self._steps_per_frame
	
    @steps_per_frame.setter
    def steps_per_frame(self, new_value):
        self._steps_per_frame = self._check_bounds(new_value, 'steps_per_frame')
        
    @property
    def sight(self):
        return self._sight
    
    @sight.setter
    def sight(self, new_value):
        self._sight = self._check_bounds(new_value, 'sight')
        
    def color(self):
        return self._color
    
    def set_color(self, new_color):
        self._color = new_color
        
    def generate_random_color(self):
        r = random.random()
        g = random.random()
        b = random.random()
        self._color = tuple((r, g, b))
    
    def _check_bounds(self, var, var_name):
        if var < pbounds[var_name][0]:
            #print (f'Der Wert für {var_name} ist zu klein. Er wird auf {pbounds[var_name][0]} gesetzt ')
            var = pbounds[var_name][0]
        elif var > pbounds[var_name][1]:
            #print (f'Der Wert für {var_name} ist zu groß. Er wird auf {pbounds[var_name][1]} gesetzt ')
            var = pbounds[var_name][1]  
        return var 
            
    def mature(self):
        self.age = self.age + 1
    
    def mutate_child(self):
        initial_speed = self._initial_speed + np.random.normal(0, 0.5)
        speed_up = self._speed_up  + np.random.normal(0, 0.001)
        perc_paddle = self._perc_paddle + np.random.normal(0,0.03)
        shrink = self._shrink + np.random.normal(0, 0.01)
        max_speed_paddle = self._max_speed_paddle + np.random.normal(0, 0.3)
        steps_per_frame = self._steps_per_frame + np.random.randint(-2, 2)
        sight = self._sight + np.random.normal(0, 0.02)
        
        child = Environment(initial_speed, speed_up, perc_paddle, shrink, 
                            max_speed_paddle, steps_per_frame, sight)
        
        for i in self._ancestors: child._ancestors.append(i)
        child._ancestors.append(self._idx)
        
        return child
    
    def info(self):
        return print(f"""Index: {self._idx}
        Ancestors: {self._ancestors}
        Age: {self.age}""")
        
    def as_list(self):
        return [self._initial_speed, self._speed_up, self._perc_paddle, 
                self._shrink, self._max_speed_paddle, self._steps_per_frame, 
                self._sight]
        
        
