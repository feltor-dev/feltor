#include <iostream>
#include <cassert>
#include <mpi.h>

#include "mpi_kron.h"

int main( int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm comm;
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    if( size != 4)
    {
        std::cerr << "Please run with 4 MPI threads!\n";
        MPI_Finalize();
        return 0;
    }
    int periods[3] = {false,false, false};
    int np[3] = {1, 2, 2};
    MPI_Cart_create( MPI_COMM_WORLD, 3, np, periods, true, &comm);
    dg::register_mpi_cart_create( MPI_COMM_WORLD, 3, np, periods, true, comm);
    const auto& mm = dg::detail::mpi_cart_info_map;
    std::vector<int> remains{1,1,1};
    assert( mm.at(comm).root == comm);
    assert( mm.at(comm).remains == remains);
    MPI_Comm comm2;
    int np2[3] = {2, 2, 1};
    dg::mpi_cart_create( MPI_COMM_WORLD, 3, np2, periods, true, &comm2);
    assert( mm.at(comm2).root == comm2);
    assert( mm.at(comm2).remains == remains);

    int remain_dims[3] = {1, 0, 0};
    MPI_Comm comm_sub100;
    MPI_Cart_sub( comm, remain_dims, &comm_sub100);
    dg::register_mpi_cart_sub( comm, remain_dims, comm_sub100);
    remains = {1,0,0};
    assert( mm.at(comm_sub100).root == comm);
    assert( mm.at(comm_sub100).remains == remains);
    MPI_Comm same;
    dg::mpi_cart_sub( comm, remain_dims, &same, false);
    assert( same == comm_sub100);


    int remain_dims001[3] = {0, 0, 1};
    dg::mpi_cart_sub( comm, remain_dims001, &same);
    remains = {0,0,1};
    assert( mm.at(same).root == comm);
    assert( mm.at(same).remains == remains);

    MPI_Comm kron = dg::mpi_cart_kron( {comm_sub100, same});
    int coords[2];
    MPI_Cart_get( kron, 2, np, periods, coords);
    assert( np[0] == 1);
    assert( np[1] == 2);
    assert(mm.at(kron).root == comm);
    remains = {1,0,1};
    assert(mm.at(kron).remains == remains);
    MPI_Comm comm010;
    int remain_dims010[3] = {0, 1, 0};
    dg::mpi_cart_sub( comm, remain_dims010, &comm010);
    MPI_Comm comm111 = dg::mpi_cart_kron( {kron, comm010});
    assert( comm111 == comm);
    try{
        MPI_Comm doesNotWork =dg::mpi_cart_kron( {comm111, comm010});
    }catch( dg::Error & err)
    {
        if(rank==0)std::cout << "Expected error message\n";
        if(rank==0)std::cout << err.what()<<"\n";
    }
    if(rank==0) std::cout<< "ALL TESTS PASSED\n";
    dg::mpi_comm_free( &same);
    MPI_Finalize();

    return 0;
}

