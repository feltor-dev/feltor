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
    if( size % 2 != 0)
    {
        std::cerr << "Please run with a multiply of 2 processes!\n";
        MPI_Finalize();
        return 0;
    }
    int np3d[3] = {0,0,0};
    MPI_Dims_create( size, 3, np3d);
    int periods[3] = {false,false, false};
    MPI_Cart_create( MPI_COMM_WORLD, 3, np3d, periods, true, &comm);
    dg::register_mpi_cart_create( MPI_COMM_WORLD, 3, np3d, periods, true, comm);
    const auto& mm = dg::detail::mpi_cart_info_map;
    std::vector<int> remains{1,1,1};
    assert( mm.at(comm).root == comm);
    assert( mm.at(comm).remains == remains);

    MPI_Comm comm2;
    int np3d_2[3] = {0,0,2};
    MPI_Dims_create( size, 3, np3d_2);
    dg::mpi_cart_create( MPI_COMM_WORLD, {np3d_2[0], np3d_2[1], np3d_2[2]},
        {periods[0], periods[1], periods[2]}, true, &comm2);
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
    dg::mpi_cart_sub( comm, {1,0,0}, &same, false);
    assert( same == comm_sub100);


    dg::mpi_cart_sub( comm, {0,0,1}, &same);
    remains = {0,0,1};
    assert( mm.at(same).root == comm);
    assert( mm.at(same).remains == remains);

    MPI_Comm kron = dg::mpi_cart_kron( {comm_sub100, same}); // 100 + 001
    int coords[2], np[2] = {0,0};
    MPI_Cart_get( kron, 2, np, periods, coords);
    assert( np[0] == np3d[0]);
    assert( np[1] == np3d[2]);
    assert(mm.at(kron).root == comm);
    remains = {1,0,1};
    assert(mm.at(kron).remains == remains);
    MPI_Comm comm010;
    dg::mpi_cart_sub( comm, {0,1,0}, &comm010);
    try{
        MPI_Comm comm111 = dg::mpi_cart_kron( {kron, comm010}); // 101 + 010
        assert( comm111 == comm);
    }catch( dg::Error & err)
    {
        if(rank==0)std::cout << "Expected error message\n";
        if(rank==0)std::cout << err.what()<<"\n";
    }
    try{
        MPI_Comm doesNotWork =dg::mpi_cart_kron( {comm010, comm010});
    }catch( dg::Error & err)
    {
        if(rank==0)std::cout << "Expected error message\n";
        if(rank==0)std::cout << err.what()<<"\n";
    }
    std::array<MPI_Comm,3> axes = dg::mpi_cart_split<3>( comm);
    assert( axes[0] == comm_sub100);
    assert( axes[1] == comm010);
    assert( axes[2] == same);
    MPI_Comm joined = dg::mpi_cart_kron( axes);
    assert( joined == comm);

    if(rank==0)dg::mpi_cart_registry_display();
    // A test originating from geometry_elliptic_mpib
    auto taxe = dg::mpi_cart_kron( {axes[0], axes[1]});
    if(rank==0)dg::mpi_cart_registry_display();
    auto paxes = dg::mpi_cart_split<2>( taxe);
    if(rank==0)dg::mpi_cart_registry_display();
    auto kaxe = dg::mpi_cart_kron( paxes);
    if(rank==0)dg::mpi_cart_registry_display();
    assert( kaxe == taxe);

    if(rank==0) std::cout<< "ALL TESTS PASSED\n";
    MPI_Finalize();

    return 0;
}

