#!/bin/bash
# submit with qsub.py 
        
export NAME=feltorsim
export DATA_DIR=data/$NAME
export DESCR=fnltr5blobti1
export INPUT=inputcomp.txt
export GEOMINPUT=geometry_params.txt
#$ -N feltorsim9
#$ -o feltorsim9.out
#$ -P fermi
#$ -q kepler
#$ -j yes
#$ -cwd
            
#$ -pe smp 8 
            
#make feltor_hpc
export OMP_NUM_THREADS=$NSLOTS
export LD_LIBRARY_PATH=/opt/intel/lib/intel64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/sw/netcdf/4.3/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/sw/hdf5/1.8.10-p1/lib:$LD_LIBRARY_PATH
./feltor_hpc $INPUT $GEOMINPUT $DATA_DIR/$DESCR.nc  &> $DATA_DIR/$DESCR.info

