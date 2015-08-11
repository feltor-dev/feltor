#include <iostream>

#include <mpi.h>
#include "mpi_collective.h"


int main( int argc, char * argv[])
{
    MPI_Init( &argc, &argv);
    unsigned Nx = 30, Ny = 30; 
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);

    if(rank==0)std::cout <<"Nx " <<Nx << " and Ny "<<Ny<<std::endl;
    if(rank==0)std::cout <<"Size =  " <<size <<std::endl;
    if(size!=4 && rank == 0){std::cerr <<"You run with "<<size<<" processes. Run with 4 processes!\n"; MPI_Finalize(); return -1;}

    if(rank==0)std::cout << "Test if scatter followed by gather leaves the input vector intact\n";
    thrust::host_vector<double> v( Nx*Ny, 3*rank);
    thrust::host_vector<double> m( Nx*Ny);
    for( int i=0; i<m.size(); i++)
    {
        m[i] = i%3 + rank/2;
        v[i] = v[i] + (double)(i + 17%(rank+1));
    }
    const thrust::host_vector<double> w(v);
    dg::BijectiveComm<thrust::host_vector<int>, thrust::host_vector<double> > c(m, MPI_COMM_WORLD);
    thrust::host_vector<double> receive(c.size());
    c.collect( v, receive);
    //for( unsigned i=0; i<receive.size(); i++)
    //{
    //    if( rank==0)
    //        std::cout << receive[i]<<std::endl;
    //}
    MPI_Barrier( MPI_COMM_WORLD);
    //for( unsigned i=0; i<receive.size(); i++)
    //{
    //    if( rank==1)
    //        std::cout << receive[i]<<std::endl;
    //}
    c.send_and_reduce( receive, v);
    bool equal = true;
    for( unsigned i=0; i<m.size(); i++)
    {
        if( v[i] != w[i])
        {
            equal = false;
        }
        //if( rank==0) std::cout << i << " "<<v[i]<<" "<<w[i]<<"\n";
    }
    if( rank==0)
    {
        if( equal) 
            std::cout <<"TEST PASSED\n";
        else
            std::cerr << "TEST FAILED\n";
    }

    MPI_Finalize();

    return 0;
}
