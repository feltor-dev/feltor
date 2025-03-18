#include <iostream>
#include <cusp/print.h>
#include "average.h"
#include "stencil.h"
#include "filter.h"
#include "../blas2.h"

#include "catch2/catch_all.hpp"

// TODO This test (and documentation organisation) needs to improve
TEST_CASE( "Stencil")
{
    std::vector<dg::bc> bcs = {dg::DIR, dg::NEU, dg::PER};
    SECTION( "Window stencil")
    {
        dg::Grid2d g2d( 0,1, 0,2, 3, 5, 3);
        auto x = dg::evaluate( dg::one, g2d), y(x);
        for( auto bc : bcs)
        {
            INFO( "Test "<<dg::bc2str( bc)<<" boundary:");
            auto stencil = dg::create::window_stencil( {3,3}, g2d, bc, bc);
            dg::blas2::symv( stencil, x, y);
            for( unsigned i=0; i<g2d.gy().size(); i++)
            {
                for( unsigned k=0; k<g2d.gx().size(); k++)
                    INFO( y[i*g2d.gx().size()+k] << " ");
            }
            INFO( "Test filtered symv");
            dg::blas2::stencil( dg::CSRSymvFilter(), (dg::IHMatrix)stencil, x, y);
            for( unsigned i=0; i<g2d.gy().size(); i++)
            {
                for( unsigned k=0; k<g2d.gx().size(); k++)
                    INFO( y[i*g2d.gx().size()+k] << " ");
            }
        }
    }

    SECTION( "dg slope limiter stencil");
    {
        dg::Grid1d g1d( 0,1, 3, 3);
        for( auto bc : bcs)
        {
            auto stencil = dg::create::limiter_stencil( g1d, bc);
            INFO( "Test "<<dg::bc2str( bc)<<" boundary");
            for( unsigned i=0; i<stencil.row_offsets.size(); i++)
                INFO( stencil.row_offsets[i]<<" ");
            for( unsigned i=0; i<stencil.column_indices.size(); i++)
                INFO( stencil.column_indices[i]<<" ");
            for( unsigned i=0; i<stencil.column_indices.size(); i++)
                INFO( stencil.values[i]<<" ");
        }
    }
    SECTION( "Test DIR boundary");
    {
        dg::Grid2d g2d( 0,1, 0,2, 3, 5, 3);
        auto stencil = dg::create::limiter_stencil( dg::coo3d::y, g2d, dg::DIR);
        auto x = dg::evaluate( [](double x, double y){return sin(x)*sin(y);}, g2d);
        auto y = x;
        dg::blas2::stencil( dg::CSRSlopeLimiter<double>(), stencil, x, y);
        //for( unsigned i=0; i<g2d.gy().size(); i++)
        //    std::cout<< y[i*g2d.gx().size()+0]<<" ";
        //std::cout << "\n";
    }


    SECTION( "Transpose")
    {
        dg::Grid2d g2d( 0,1, 0,2, 3, 5, 3);
        auto x = dg::evaluate( dg::one, g2d), y(x);
        auto xx(x), yy(x), zz(x);
        dg::Grid2d g2dT( 0,2, 0,1, 3, 3, 5);
        auto stencilX = dg::create::limiter_stencil( dg::coo3d::x, g2dT, dg::DIR);
        dg::blas2::stencil( dg::CSRSlopeLimiter<double>(), stencilX, xx, zz);
        //for( unsigned i=0; i<g2d.gy().size(); i++)
        //    std::cout<< zz[i]<<" ";
        //std::cout << "\n";
        // transpose
        unsigned nx = g2d.shape(0), ny = g2d.shape(1);
        dg::blas2::parallel_for( [ny,nx]DG_DEVICE( unsigned k, const
                    double* ii, double* oo)
        {
            unsigned i = k/nx, j =  k%nx;
            oo[j*ny+i] = ii[i*nx+j];
        }, nx*ny, zz, yy);
        dg::blas1::axpby( 1., yy, -1., y);
        double err = dg::blas1::dot( y, 1.);
        INFO( "error between transposed application and app is: " <<err<<" (0)");
        CHECK( err == 0);
    }
}
