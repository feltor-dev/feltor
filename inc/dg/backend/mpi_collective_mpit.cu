#include <iostream>

#include <mpi.h>
#include "../blas1.h"
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
    if(rank==0)std::cout <<"# processes =  " <<size <<std::endl;

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
    receive = c.global_gather( thrust::raw_pointer_cast(v.data()));
    MPI_Barrier( MPI_COMM_WORLD);
    c.global_scatter_reduce( receive, thrust::raw_pointer_cast(v.data()));
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
    MPI_Barrier(MPI_COMM_WORLD);
    Nx = 10, Ny = 10;
    thrust::host_vector<double> vec( Nx*Ny, rank), result( Nx*Ny,1);
    for( int i=0; i<(int)(Nx*Ny); i++)
        vec[i]*=i;
    thrust::host_vector<int> idx( (Nx+rank)*(Ny+rank)), pids( (Nx+rank)*(Ny+rank));
    for( unsigned i=0; i<(Nx+rank)*(Ny+rank); i++)
    {
        idx[i] = i%(Nx*Ny);
        if(i==10) idx[i]+=1;
        pids[i] = rank;
        if( i>=Nx*Ny) pids[i] = (rank+1)%size;
    }
    dg::GeneralComm<thrust::host_vector<int>, thrust::host_vector<double> > s2( idx, pids, MPI_COMM_WORLD), s(s2);
    receive = s.global_gather( thrust::raw_pointer_cast(vec.data()));
    /// Test if global_scatter_reduce is in fact the transpose of global_gather
    dg::MPI_Vector<thrust::host_vector<double>> mpi_receive( receive, MPI_COMM_WORLD);
    double norm1 = dg::blas1::dot( mpi_receive, mpi_receive);
    thrust::host_vector<double> vec2(vec.size());
    s.global_scatter_reduce( receive, thrust::raw_pointer_cast(vec2.data()));
    dg::MPI_Vector<thrust::host_vector<double>> mpi_vec( vec, MPI_COMM_WORLD);
    dg::MPI_Vector<thrust::host_vector<double>> mpi_vec2( vec2, MPI_COMM_WORLD);
    double norm2 = dg::blas1::dot( mpi_vec, mpi_vec2);
    //if(rank==0)std::cout << "Test Norm Gv*Gv vs G^T(Gv)*v: "<<norm1<<" ";
    //if(rank==0)std::cout << norm2<< " "<<norm1-norm2<<std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    equal=true;
    if( fabs(norm1-norm2)>1e-14) equal = false;
    {
        if( equal)
            std::cout <<"Rank "<<rank<<" PASSED "<<std::endl;
        else
            std::cerr <<"Rank "<<rank<<" FAILED "<<std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
