/*
The MIT License (MIT)

Copyright (c) 2018 Siegfried Höfinger, Matthias Wiesenberger

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <sched.h>

#ifndef __CUDACC__
int sched_getcpu(void); //forward declare glibc function to avoid compiler warnings
#endif

//**************************************************************************************//
//Determine on which CPU a thread in an MPI, MPI+OpenMP, or MPI+CUDA program executes.  //
//    pure MPI:                                                                         //
//    mpicc thread_affinity.c -o thread_affinity                                        //
//    hybrid MPI+CPU                                                                    //
//    mpicc thread_affinity.c -o thread_affinity -fopenmp                               //
//    hybrid MPI+GPU                                                                    //
//    nvcc --compiler-bindir mpiicc thread_affinity.c -o thread_affinity                //
//**************************************************************************************//
int main(int argc, char **argv)
{
    int i,l, task, ntasks, cpuid;
    char node_name[MPI_MAX_PROCESSOR_NAME];

    // determine MPI task and total number of MPI tasks
#ifdef _OPENMP
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &task);
    if(provided < MPI_THREAD_FUNNELED)
        if(task==0)printf("Warning: mpi implementation does not support threads!\n");
#else
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &task);
#endif
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    // determine total number of OpenMP threads, thread number,
    // associated core and hostname and print in correct order
#ifdef _OPENMP
    int j;
    int nthreads = omp_get_max_threads();
#elif defined  __CUDACC__
    int num_devices=0;
    cudaGetDeviceCount(&num_devices);
    if(num_devices == 0)
    {
        printf("No CUDA capable devices found\n");
        return -1;
    }
    int device = task % num_devices; //assume # of gpus/node is fixed
    cudaSetDevice( device);
#endif//__CUDACC__
    for( i=0; i<ntasks; i++)
    {
        if(task==i)
        {
            MPI_Get_processor_name(&node_name[0], &l);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static,1) ordered private(cpuid)
            for ( j=0; j<nthreads; j++) {
                int thread = omp_get_thread_num();
                cpuid = sched_getcpu();
                #pragma omp ordered
                printf("node %s, task %3d of %3d tasks, thread %3d of %3d threads, on cpu %3d\n",
                        &node_name[0], task, ntasks, thread, nthreads, cpuid);
            }
#elif defined __CUDACC__
            cpuid = sched_getcpu();
            printf("node %s, task %3d of %3d tasks, GPU %3d out of %3d, on cpu %3d\n",
                    &node_name[0], task, ntasks, device, num_devices, cpuid);
#else //pure MPI
            cpuid = sched_getcpu();
            printf("node %s, task %3d of %3d tasks, on cpu %3d\n",
                    &node_name[0], task, ntasks, cpuid);
#endif
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}
