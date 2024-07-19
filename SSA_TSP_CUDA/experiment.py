import subprocess
import os, sys

# files = ['fnl4461.tsp','pla7397.tsp','ida8197.tsp','ja9847.tsp',\
#         'rl11849.tsp','usa13509.tsp','brd14051.tsp','d18512.tsp',\
#         'ido21215.tsp','sw24978.tsp','irx28268.tsp']

# files = [   'fnl4461.tsp','pla7397.tsp','ida8197.tsp','ja9847.tsp', \
#             'rl11849.tsp','usa13509.tsp','brd14051.tsp', 'd18512.tsp', \
#             'ido21215.tsp','sw24978.tsp','irx28268.tsp' \
    
#             'burma14.tsp', 'ulysses16.tsp', 'gr17.tsp', 'ulysses22.tsp', 'fri26.tsp', 'bays29.tsp',     \
#             'eil51.tsp', 'berlin52.tsp', 'kroA100.tsp', 'kroB100.tsp', 'kroC100.tsp', 'kroD100.tsp',    \
#             'kroE100.tsp', 'pr76.tsp', 'rat99.tsp', 'eil76.tsp', 'rd100.tsp', 'kroA150.tsp',            \
#             'kroA200.tsp', 'pr107.tsp', 'pr124.tsp', 'pr136.tsp', 'pr144.tsp', 'pr152.tsp', 'rat195.tsp',            \
#             'lin105.tsp', 'lin318.tsp', 'ts225.tsp', 'tsp225.tsp', 'pr226.tsp', 'pr264.tsp',            \
#             'pr299.tsp', 'pr439.tsp', 'pr1002.tsp', 'rd400.tsp', 'linhp318.tsp','rat575.tsp', 'rat783.tsp'
#         ]

files = ['gr9882.tsp','fi10639.tsp','xia16928.tsp','vm22775.tsp','lsb22777.tsp','xrh24104.tsp','bbz25234.tsp']

# files = [fn.lower() for fn in files]

print(files)

iterations = 100
local_search = 1
ssa_num = 80
pchange = 0.05
N = 1

for t in range(N):
    for fn in files:
        vtkpath = fn 
        cppath = vtkpath + 'out.txt'
        cfgpath = 'config_new.cfg'

        cmd = './gpussa --test tsp/' + fn + \
            ' --outdir results'  +         \
            ' --alg ssa_gpu'  +            \
            ' --iter ' + str(iterations) +  \
            ' --ls ' + str(local_search) +  \
            ' --ssa ' + str(ssa_num) +      \
            ' --pc ' + str(pchange)
        
        print(cmd)
        # output = os.system(cmd)
        ex = subprocess.Popen(cmd, shell=True)
        ex.wait()
        ex.kill()

# ./gpussa --test tsp/xia16928.tsp --outdir results --alg ssa_gpu --iter 100 --ls 1 --ssa 80 --pc 0.91