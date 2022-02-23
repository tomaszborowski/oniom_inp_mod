#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script parsing Gaussian log file for combined 3 calculations:
    1. MM calculations for the real system
    2. QM calculations for the model system
    3. MM calculations for the model system
(This order must be preserved !!!)

and calculating ONIOM extrapolated energy:
    E_ONIOM = E(1) + E(2) - E(3)

@author: borowski
last update: 23.02.2022
"""
import sys, re

sys_argv_len = len(sys.argv)
if sys_argv_len > 1:
    oniom_log = sys.argv[1]
else:
    oniom_log = None


oniom_log_f = open(oniom_log, 'r')

E_1 = None
E_2 = None
E_3 = None
    
while True:
    line = oniom_log_f.readline()
    if not line:
        break
    match_flag_E = re.search("Energy=", line)
    match_flag_SCF = re.search("SCF Done:", line)
    if match_flag_E:
        if not E_1:
            E_1 = eval(line.split()[1])
        else:
            E_3 = eval(line.split()[1])
    if match_flag_SCF:
        E_2 = eval(line.split()[4])

oniom_log_f.close()

E_ONIOM = E_1 + E_2 - E_3

print("E_ONIOM [a.u.] = ", E_ONIOM)
print("\n Components: \n")
print("E_1 = ", E_1)
print("E_2 = ", E_2)
print("E_3 = ", E_3)
