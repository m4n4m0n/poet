# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Martina Kraußer
# Licensed under the MIT License.
# See LICENSE file or https://opensource.org/licenses/MIT

# Network Parameters
__N = 256             # Number of neurons in the network
__Na = 60             # Number of agents

# Environment Parameters
__Ne = 8              # Number of environments
__Nre = __Ne * 4      # Number of random initialized environments

# POET Parameters
__T = 30              # Number of POET steps
__S = 200             # Number of ES steps

# Fitness Function
__nruns = 5           # Number of runs in fitness function

# Bounds for environment parameters
pbounds = {
    'initial_speed': (4, 15),
    'speed_up': (1, 1.01),
    'perc_paddle': (0.1, 0.2),
    'shrink': (0, 0.2),
    'max_speed_paddle': (2, 10),
    'steps_per_frame': (5, 20),
    'sight': (0.25, 1.0)
}

# Cache / Paths
CACHE_DIR = "./Cache/"