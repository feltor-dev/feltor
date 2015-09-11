#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <string>
#include <utmpx.h>

int main( int argc, char**argv)
{ 
    MPI_Init( &argc, &argv);
    int nprocs, pID;
    MPI_Comm_rank( MPI_COMM_WORLD, &pID);
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs);
    int namelen;
    std::string name('x', 128);

    MPI_Get_processor_name( &name[0], &namelen);
    name.resize( namelen);
    for( unsigned i=0; i<nprocs; i++)
    {
        
        if(pID==i)
        {
#pragma omp parallel
{
    int nthreads = omp_get_num_threads();
    int tID = omp_get_thread_num(); 
#pragma omp critical
    std::cout << "Hello from thread "<<tID<<" out of "<<nthreads<<" from process "<<pID<<" out of "<<nprocs<<" on node "<<name<<" and id "<<sched_getcpu()<<std::endl;
}
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;

}
