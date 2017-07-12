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

    if(rank==0)std::cout << "Test BijectiveComm: scatter followed by gather leaves the input vector intact\n";
    thrust::host_vector<double> v( Nx*Ny, 3*rank);
    thrust::host_vector<double> m( Nx*Ny);
    for( unsigned i=0; i<m.size(); i++)
    {
        m[i] = i%3 + rank/2;
        v[i] = v[i] + (double)(i + 17%(rank+1));
    }
    const thrust::host_vector<double> w(v);
    dg::BijectiveComm<thrust::host_vector<int> > c(m, MPI_COMM_WORLD);
    thrust::host_vector<double> receive(c.size());
    receive = c.global_gather( v);
    MPI_Barrier( MPI_COMM_WORLD);
    c.global_scatter_reduce( receive, v);
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
    if(rank==0)std::cout << "Test SurjectiveComm:\n";
    Nx = 3, Ny = 3; 
    thrust::host_vector<double> vec( Nx*Ny, rank);
    thrust::host_vector<int> idx( (Nx+1)*(Ny+1)), pids( (Nx+1)*(Ny+1));
    for( unsigned i=0; i<(Nx+1)*(Ny+1); i++)
    {
        idx[i] = i%(Nx*Ny);
        pids[i] = rank;
        if( i>=Nx*Ny) pids[i] = (rank+1)%size;
    }
    dg::SurjectiveComm<thrust::host_vector<int> > s( idx, pids, MPI_COMM_WORLD);
    //receive = s.global_gather( vec);
    //for( unsigned i=0; i<(Nx+1)*(Ny+1); i++)
    //    if(rank==0) std::cout << i << std::endl;

    MPI_Finalize();

    return 0;
}
