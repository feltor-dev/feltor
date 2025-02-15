#include <iostream>
#include <sstream>
#include <mpi.h>
#include "dg/backend/mpi_init.h"
#include "dg/blas.h"
#include "mpi_projection.h"
#include "mpi_evaluation.h"

#include "catch2/catch_all.hpp"


const double shift = 0.2;
static double function( double x, double y){ return sin(M_PI*x/2.)*sin(M_PI*y/2.);}

TEST_CASE( "MPI Matrices")
{

    int rank, size;
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    unsigned n = 3, Nx = 8, Ny = 8;
    MPI_Comm comm = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0}, {1,1});
    dg::MPIGrid2d g2d( 0,1,0,1, n,Nx,Ny, comm);
    ///%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    SECTION( "NON-COMMUNICATING MPI matrix-creation!")
    {
        dg::MPIGrid2d g2d_half = g2d;
        g2d_half.multiplyCellNumbers(0.5, 0.5);

        dg::MIHMatrix converted_p = dg::create::interpolation( g2d, g2d_half);
        dg::MHVec sine = dg::evaluate( function, g2d_half);
        dg::MHVec temp = dg::evaluate( function, g2d);
        converted_p.symv( sine, temp);

        dg::IHMatrix direct_p= dg::create::interpolation( g2d.global(), g2d_half.global());
        dg::HVec global_sine = dg::evaluate( function, g2d_half.global());
        dg::HVec g_temp( g2d.global().size());
        dg::blas2::symv( direct_p, global_sine, g_temp);
        //now compare
        for( unsigned i=0; i<temp.size(); i++) //local size
        {
            int gIdx = 0;
            g2d.local2globalIdx( i, rank, gIdx);
            INFO("Rank "<< rank<<" "<<i<<" "<<gIdx<<" "<<temp.data()[i]<<" "<<g_temp[gIdx]);
            CHECK( fabs( temp.data()[i] - g_temp[gIdx]) < 1e-14);
        }
    }

    /////%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    SECTION("Now test COMMUNICATING MPI matrix-creation!")
    {
        auto x = dg::evaluate( dg::cooX2d, g2d.local());
        auto y = dg::evaluate( dg::cooY2d, g2d.local());
        for( unsigned i=0; i<x.size(); i++)
        {
            x[i] +=shift;
            y[i] +=shift;
            bool negative = false;
            dg::create::detail::shift( negative, x[i], g2d.bcx(),
                g2d.global().x0(), g2d.global().x1());
            dg::create::detail::shift( negative, y[i], g2d.bcy(),
                g2d.global().y0(), g2d.global().y1());
        }
        dg::MIHMatrix converted_i = dg::create::interpolation( x,y,g2d);
        dg::IHMatrix  direct_i = dg::create::interpolation( x,y,g2d.global());
        auto sine = dg::evaluate( function, g2d);
        auto temp = sine;
        converted_i.symv( sine, temp);
        auto global_sine = dg::evaluate( function, g2d.global());
        dg::HVec g_temp( g2d.local().size());
        dg::blas2::symv( direct_i, global_sine, g_temp);
        //now compare
        for( unsigned i=0; i<temp.size(); i++) //local size
        {
            INFO( "Rank "<< rank<<" "<<i<<" "<<temp.data()[i]<<" "<<g_temp[i]);
            CHECK(fabs(temp.data()[i] - g_temp[i]) < 1e-12);
        }
    }
    ///%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    SECTION( "COMMUNICATING MPI local2global interpolation!")
    {
        dg::MIHMatrix local2global;
        if(rank==0)
        {
            auto xx = dg::evaluate( dg::cooX2d, g2d.global());
            auto yy = dg::evaluate( dg::cooY2d, g2d.global());
            local2global = dg::create::interpolation( xx,yy,g2d);
        }
        else
        {
            auto xx = dg::HVec(); // empty for all other pids
            auto yy = dg::HVec();
            local2global = dg::create::interpolation( xx,yy,g2d);
        }
        dg::MHVec mpi_sine = dg::evaluate( function, g2d);
        dg::MHVec mpi_temp( dg::HVec(rank==0?g2d.size() : 0), g2d.communicator());
        local2global.symv( mpi_sine, mpi_temp);
        auto global_sine = dg::evaluate( function, g2d.global());
        //now compare
        if(rank==0)
        {
            for( unsigned i=0; i<mpi_temp.size(); i++)
            {
                INFO("i " << i<<" "<<mpi_temp.data()[i]<<" "<<global_sine[i]);
                CHECK( fabs(mpi_temp.data()[i] - global_sine[i]) < 1e-14);
            }
        }
        else
            CHECK( mpi_temp.data().size() == 0);

    }
}
