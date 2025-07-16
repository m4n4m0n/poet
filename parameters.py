#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 13:20:40 2023

@author: martkr
"""


pbounds = {'initial_speed': (4, 15), 
          'speed_up': (1, 1.01),
          'perc_paddle': (0.1, 0.2), 
          'shrink': (0, 0.2), 
          'max_speed_paddle': (2, 10),
          'steps_per_frame': (5, 20),
          'sight': (0.25, 1.0)
          }