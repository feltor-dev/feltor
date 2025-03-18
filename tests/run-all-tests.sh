#!/bin/bash

for dir in ../inc/dg/backend ../inc/dg/topology ../inc/dg ../inc/file
do
    current=$(pwd)
    cd $dir
    echo "##############################################################"
    echo "GO TO DIRECTORY" $dir
    for dev in cpu gpu omp
    do
        echo "#########################"
        echo "DIRECTORY" $dir "TEST DEVICE" $dev
        make tests -j 4 device=$dev
        ./tests
        make mpi-tests -j 4 device=$dev
        for num in 1 2 3 4 6 8
        do
            if [[ "$dev" == "omp" ]];
            then
                export OMP_NUM_THREADS=1
            fi
            mpirun -n $num --oversubscribe ./mpi-tests
        done
        make clean
        echo "#########################"
    done
    cd $current
    echo "##############################################################"
    echo
done
# Clean this directory
rm *.o
