#include <iostream>

#include <mpi.h>
#include "mpi_collective.h"
#include "mpi_vector.h"


int main( int argc, char * argv[])
{
    MPI_Init( &argc, &argv);
    unsigned Nx = 30, Ny = 30; 
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    if(size!=4 ){std::cerr <<"You run with "<<size<<" processes. Run with 4 processes!\n"; MPI_Finalize(); return 0;}

    if(rank==0)std::cout <<"Nx " <<Nx << " and Ny "<<Ny<<std::endl;
    if(rank==0)std::cout <<"Size =  " <<size <<std::endl;

    if(rank==0)std::cout << "Test BijectiveComm: scatter followed by gather leaves the input vector intact\n";
    thrust::host_vector<double> v( Nx*Ny, 3*rank);
    thrust::host_vector<double> m( Nx*Ny);
    for( unsigned i=0; i<m.size(); i++)
    {
        m[i] = i%3 + rank/2;
        v[i] = v[i] + (double)(i + 17%(rank+1));
    }
    const thrust::host_vector<double> w(v);
    dg::BijectiveComm<thrust::host_vector<int>, thrust::host_vector<double> > c(m, MPI_COMM_WORLD);
    thrust::host_vector<double> receive = c.allocate_buffer();
    receive = c.global_gather( v);
    MPI_Barrier( MPI_COMM_WORLD);
    c.global_scatter_reduce( receive, v);
    bool equal = true;
    for( unsigned i=0; i<m.size(); i++)
    {
        if( v[i] != w[i]) { equal = false; }
        //if( rank==0) std::cout << i << " "<<v[i]<<" "<<w[i]<<"\n";
    }
    {
        if( equal) 
            std::cout <<"Rank "<<rank<<" PASSED"<<std::endl;
        else
            std::cerr <<"Rank "<<rank<<" FAILED"<<std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==0)std::cout << "Test SurjectiveComm and GeneralComm:\n";
    Nx = 5, Ny = 5; 
    thrust::host_vector<double> vec( Nx*Ny, rank), result( Nx*Ny);
    thrust::host_vector<int> idx( (Nx+1)*(Ny+1)), pids( (Nx+1)*(Ny+1));
    for( unsigned i=0; i<(Nx+1)*(Ny+1); i++)
    {
        idx[i] = i%(Nx*Ny);
        pids[i] = rank;
        if( i>=Nx*Ny) pids[i] = (rank+1)%size;
    }
    dg::GeneralComm<thrust::host_vector<int>, thrust::host_vector<double> > s( idx, pids, MPI_COMM_WORLD);
    receive = s.global_gather( vec);
    //for( unsigned i=0; i<(Nx+1)*(Ny+1); i++)
    //    if(rank==0) std::cout << i<<"\t "<< receive[i] << std::endl;
    thrust::host_vector<double> vec2(vec.size());
    s.global_scatter_reduce( receive, vec2);
    equal=true;
    for( unsigned i=0; i<(Nx)*(Ny); i++)
    {
        //if(rank==1) std::cout << i<<"\t "<< vec[i] << std::endl;
        result[i] = rank; 
        if( i < (Nx+1)*(Ny+1) - Nx*Ny) result[i] += (rank)%size;
        if( vec2[i] != result[i]) equal = false;
    }
    {
        if( equal) 
            std::cout <<"Rank "<<rank<<" PASSED"<<std::endl;
        else
            std::cerr <<"Rank "<<rank<<" FAILED"<<std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==0)std::cout<< "Test Nearest Neighbor Comm\n";
    unsigned dims[3] = {Nx, Ny, 1}; 
    int np[2] = {2,2}, periods[2] = { false, false};
    MPI_Comm comm;
    MPI_Cart_create( MPI_COMM_WORLD, 2, np, periods, true, &comm);

    dg::NearestNeighborComm<thrust::host_vector<int>, thrust::host_vector<double> > nnch( 2, dims, comm, 1);
    thrust::host_vector<double> tmp = nnch.allocate_buffer();
    nnch.global_gather( vec, tmp);
    vec2=vec;
    nnch.global_scatter_reduce( tmp, vec2);
    for( unsigned i=0; i<(Nx)*(Ny); i++)
    {
        if( vec[i] != vec2[i]) equal = false;
    }
    {
        if( equal) 
            std::cout <<"Rank "<<rank<<" PASSED"<<std::endl;
        else
            std::cerr <<"Rank "<<rank<<" FAILED"<<std::endl;
    }



    MPI_Finalize();

    return 0;
}
