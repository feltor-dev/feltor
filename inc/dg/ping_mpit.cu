#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP
#include <iostream>
#include <string>
#include <utmpx.h> //for sched_getcpu function

int main( int argc, char**argv)
{ 
    MPI_Init( &argc, &argv);
    int nprocs, pID;
    MPI_Comm_rank( MPI_COMM_WORLD, &pID);
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs);
    int namelen;
    std::string name('x', 64);

    MPI_Get_processor_name( &name[0], &namelen);
    name.resize( namelen);
#ifdef __CUDACC__
    int num_devices=0;
    cudaGetDeviceCount(&num_devices);
    if(num_devices == 0)
    {
        std::cerr << "No CUDA capable devices found"<<std::endl;
        return -1;
    }
    int device = pID % num_devices; //assume # of gpus/node is fixed
    cudaSetDevice( device);
#endif//__CUDACC__
    for( int i=0; i<nprocs; i++)
    {
        if(pID==i)
        {
#ifdef _OPENMP
#pragma omp parallel
{
            int nthreads = 0, tID = 0;
            nthreads = omp_get_num_threads();
            tID = omp_get_thread_num(); 
#pragma omp critical
            std::cout << "Hello from thread "<<tID<<" out of "<<nthreads<<" from process "<<pID<<" out of "<<nprocs<<" on node "<<name<<" and id "<<sched_getcpu()<<std::endl;
}
#else
#ifdef __CUDACC__ 
            std::cout << "Hello from GPU "<< device<<" out of "<<num_devices<<" from process "<<pID<<" out of "<<nprocs<<" on node "<<name<<" and id "<<sched_getcpu()<<std::endl;
#else
            std::cout << "Hello from process "<<pID<<" out of "<<nprocs<<" on node "<<name<<" and id "<<sched_getcpu()<<std::endl;
#endif//__CUDACC__
#endif //_OPENMP

        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;

}
