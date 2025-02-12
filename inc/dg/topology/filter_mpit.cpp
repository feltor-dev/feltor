#include <iostream>
#include <mpi.h>

#include "dg/blas.h"
#include "dg/functors.h"
#include "dg/backend/mpi_init.h"

#include "mpi_projection.h"
#include "mpi_evaluation.h"
#include "mpi_weights.h"
#include "filter.h"

#include "catch2/catch_all.hpp"

static double function( double x, double y){return sin(x)*sin(y);}
static double function( double x, double y, double z){return sin(x)*sin(y)*sin(z);}


TEST_CASE( "mpi filter")
{
    const unsigned Nx = 8, Ny = 10, Nz = 6;
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);

    SECTION( "Test 2d exponential filter")
    {
        MPI_Comm comm = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0}, {1,1});
        dg::MPIGrid2d g3( -M_PI, 0, -5*M_PI, -4*M_PI, 3, Nx, Ny, comm);
        dg::MPIGrid2d g2( -M_PI, 0, -5*M_PI, -4*M_PI, 2, Nx, Ny, comm);

        const dg::MDVec vec = dg::evaluate( function, g3);
        const dg::MDVec weights = dg::create::weights( g3);
        dg::MDVec filtered_vec(vec), projected_vec(dg::evaluate( dg::zero, g2)), inter_vec( vec);
        auto op = dg::ExponentialFilter( 36, 0.5, 8, g3.nx());
        dg::MultiMatrix<dg::MDMatrix, dg::MDVec> filter = dg::create::fast_transform(
           dg::create::modal_filter( op, g3.nx()),
           dg::create::modal_filter( op, g3.ny()), g3);
        dg::MIDMatrix project = dg::create::projection( g2,g3);
        dg::MIDMatrix interpo = dg::create::interpolation( g3,g2);

        dg::blas2::symv( project, vec, projected_vec);
        dg::blas2::symv( interpo, projected_vec, inter_vec);
        dg::blas2::symv( filter, vec, filtered_vec);
        dg::blas1::axpby( 1., filtered_vec, -1., inter_vec);
        double error = sqrt(dg::blas2::dot( inter_vec, weights, inter_vec)/ dg::blas2::dot( vec, weights, vec));
        INFO( "Error by filtering: "<<error);
        CHECK( error < 1e-14);
    }
    SECTION( "Test 3d exponential filter")
    {
        MPI_Comm comm = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0,0}, {1,1,1});
        dg::MPIGrid3d g3( -M_PI, 0, -5*M_PI, -4*M_PI, 0., 2.*M_PI, 3, Nx, Ny,
        Nz, comm);
        dg::MPIGrid3d g2( -M_PI, 0, -5*M_PI, -4*M_PI, 0., 2.*M_PI, 2, Nx, Ny,
        Nz, comm);

        const dg::MDVec vec = dg::evaluate( function, g3);
        const dg::MDVec weights = dg::create::weights( g3);
        dg::MDVec filtered_vec(vec), projected_vec(dg::evaluate( dg::zero, g2)), inter_vec( vec);
        auto op = dg::ExponentialFilter( 36, 0.5, 8, g3.nx());
        dg::MultiMatrix<dg::MDMatrix, dg::MDVec> filter = dg::create::fast_transform(
           dg::create::modal_filter( op, g3.nx()),
           dg::create::modal_filter( op, g3.ny()), g3);
        dg::MIDMatrix project = dg::create::projection( g2,g3);
        dg::MIDMatrix interpo = dg::create::interpolation( g3,g2);

        dg::blas2::symv( project, vec, projected_vec);
        dg::blas2::symv( interpo, projected_vec, inter_vec);
        dg::blas2::symv( filter, vec, filtered_vec);
        dg::blas1::axpby( 1., filtered_vec, -1., inter_vec);
        double error = sqrt(dg::blas2::dot( inter_vec, weights, inter_vec)/
            dg::blas2::dot( vec, weights, vec));
        INFO( "Error by filtering: "<<error);
        CHECK( error < 1e-14);
    }
}
