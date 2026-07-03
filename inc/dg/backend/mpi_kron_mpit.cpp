#include <iostream>
#include <mpi.h>

#include "mpi_init.h"
#include "mpi_kron.h"
#include "exblas/mpi_accumulate.h"
#include "catch2/catch_test_macros.hpp"

TEST_CASE("MPI Kron test")
{
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    const auto& mm = dg::detail::mpi_cart_registry;
    SECTION( "Check sub entry creation")
    {
        MPI_Comm comm = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0,0},
            {false,false,false}, true);

        std::vector<int> remain_dims = {1, 0, 0};
        MPI_Comm comm_sub100;
        dg::mpi_cart_sub( comm, &remain_dims[0], &comm_sub100);

        dg::register_mpi_cart_sub( comm, &remain_dims[0], comm_sub100);

        CHECK( mm.at(comm_sub100).root == comm);
        CHECK( mm.at(comm_sub100).remain_dims == remain_dims);

        MPI_Comm same = dg::mpi_cart_sub( comm, {1,0,0}, false);
        CHECK( same == comm_sub100);

        same = dg::mpi_cart_sub( comm, {0,0,1});
        remain_dims = {0,0,1};
        CHECK( mm.at(same).root == comm);
        CHECK( mm.at(same).remain_dims == remain_dims);
    }
    SECTION( "Check double sub does not segfault")
    {
        // This tests for issue in openmpi 4.0.0
        // https://github.com/open-mpi/ompi/issues/13081
        // Test passes if no segfault
        int rank, size;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
        MPI_Comm_size( MPI_COMM_WORLD, &size);

        MPI_Comm comm = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0}, {1,1});
        auto comms = dg::mpi_cart_split_as<2>(comm);
        MPI_Comm comm_join = dg::mpi_cart_kron(comms);
        MPI_Comm comm_mod, comm_red;
        dg::exblas::mpi_reduce_communicator( comm_join, &comm_mod, &comm_red);
        MPI_Comm comm2 = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0,0}, {1,1,1});
        auto comms2 = dg::mpi_cart_split_as<2>(comm2);

        CHECK( comms.size() == 2);
        CHECK( comms2.size() == 2);
    }

    SECTION( "Direct kronecker")
    {
        std::vector<int> np3d = {0,0,0};
        MPI_Dims_create( size, 3, &np3d[0]);
        MPI_Comm comm = dg::mpi_cart_create( MPI_COMM_WORLD, np3d, {0,0,0},
            true);
        MPI_Comm comm_sub100 = dg::mpi_cart_sub( comm, {1,0,0});
        MPI_Comm comm_sub001 = dg::mpi_cart_sub( comm, {0,0,1});

        MPI_Comm kron = dg::mpi_cart_kron( {comm_sub100, comm_sub001}); // 100 + 001
        int coords[2], np[2] = {0,0}, periods[2];
        dg::mpi_cart_get( kron, 2, np, periods, coords);
        CHECK( np[0] == np3d[0]);
        CHECK( np[1] == np3d[2]);
        CHECK(mm.at(kron).root == comm);
        CHECK(mm.at(kron).remain_dims == std::vector<int>{1,0,1});
        MPI_Comm comm_sub010 = dg::mpi_cart_sub( comm, {0,1,0});
        MPI_Comm comm111 = MPI_COMM_NULL;
        CHECK_THROWS_AS(
            // Cannot create kronecker in reverse order
            comm111 = dg::mpi_cart_kron( {kron, comm_sub010}), // 101 + 010
            dg::Error
        );
        CHECK( comm111 == MPI_COMM_NULL);
        MPI_Comm doesNotWork = comm111;
        CHECK_THROWS_AS(
            doesNotWork =dg::mpi_cart_kron( {comm_sub010, comm_sub010}),
            dg::Error
        );
        CHECK( doesNotWork == MPI_COMM_NULL);
        std::array<MPI_Comm,3> axes = dg::mpi_cart_split_as<3>( comm);
        CHECK( axes[0] == comm_sub100);
        CHECK( axes[1] == comm_sub010);
        CHECK( axes[2] == comm_sub001);
        MPI_Comm joined = dg::mpi_cart_kron( axes);
        CHECK( joined == comm);

        //if(rank==0)dg::mpi_cart_registry_display();
        // A test originating from geometry_elliptic_mpib
        auto taxe = dg::mpi_cart_kron( {axes[0], axes[1]});
        //if(rank==0)dg::mpi_cart_registry_display();
        auto paxes = dg::mpi_cart_split_as<2>( taxe);
        //if(rank==0)dg::mpi_cart_registry_display();
        auto kaxe = dg::mpi_cart_kron( paxes);
        //if(rank==0)dg::mpi_cart_registry_display();
        CHECK( kaxe == taxe);
    }
    SECTION( "Documentation")
    {
        //! [split and join]
        // Create a Cartesian communicator
        MPI_Comm comm = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0,0}, {0,0,0});
        // Split into 3 axes
        std::array<MPI_Comm,3> axes3d = dg::mpi_cart_split_as<3>( comm);
        // Join the first and third axis
        MPI_Comm comm_101 = dg::mpi_cart_kron( {axes3d[0], axes3d[2]});
        // Split up again
        std::array<MPI_Comm,2> axes2d = dg::mpi_cart_split_as<2>( comm_101);
        CHECK( axes2d[0] == axes3d[0]);
        CHECK( axes2d[1] == axes3d[2]);
        //! [split and join]
    }

    dg::detail::mpi_cart_registry_clear();
}

