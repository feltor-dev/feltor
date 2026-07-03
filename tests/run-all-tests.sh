#!/bin/bash
# This script runs all tests on a local machine. It should be executed from
# time to time because the automated tests on github do not run e.g. for
# multi-GPU systems

for dir in ../inc/dg/backend ../inc/dg/topology ../inc/dg ../inc/file ../inc/geometries ../inc/matrix
do
    current=$(pwd)
    cd $dir
    echo "##############################################################"
    echo "GO TO DIRECTORY" $dir
    for dev in cpu gpu omp
    do
        echo "#########################"
        echo "DIRECTORY" $dir "TEST DEVICE" $dev
        make clean # Delete leftovers from previous tests
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
