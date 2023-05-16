import json, sys, yaml
import numpy as np
import subprocess

program='ds_t' # 'ds_t' 'ds_guenter_t' 'ds_curv_t'
for m in [1,10,50]: # [1,10,50] [1,10,50] [1,10,100,1000]
    file_name = 'out_'+program+str(m)+'.json'
    f = open(file_name, "w")
    print(file_name)
    y = []
    for i in [0,1,2,3,4,5]:
        if program=='ds_curv_t' :
            NR = 2
            NZ = 30*np.round( 2**(2*i/3))
            NP = 5*2**i
            mR = 1
            mZ = m
        else :
            NP = 5*2**i
            NR = 6*np.round( 2**(2*i/3))
            NZ=NR
            mR=mZ=m
        s = "%i %i %i %i %i %i "% (3,NR,NZ,NP,mR,mZ)
        print( i, NR, NZ, NP)
        p1 = subprocess.Popen(["echo",s+'./'+program], stdout = subprocess.PIPE)
        p2 = subprocess.Popen(['./'+program], stdin=p1.stdout, stdout = subprocess.PIPE)
        p1.stdout.close()
        y.append( yaml.safe_load( p2.stdout.read()))

    json.dump( y, f, sort_keys=True, indent=2)
    f.close()
