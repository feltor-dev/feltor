#include <iostream>
#include <cusp/print.h>
#include "average.h"
#include "stencil.h"
#include "filter.h"
#include "../blas2.h"


int main()
{
    std::cout << "Test window stencil\n";
    dg::Grid2d g2d( 0,1, 0,2, 3, 5, 3);
    auto x = dg::evaluate( [](double x, double y){return 1;}, g2d), y(x);
    std::vector<dg::bc> bcs = {dg::DIR, dg::NEU, dg::PER};
    for( auto bc : bcs)
    {
        std::cout << "Test "<<dg::bc2str( bc)<<" boundary:\n";
        auto stencil = dg::create::window_stencil( {3,3}, g2d, bc, bc);
        dg::blas2::symv( stencil, x, y);
        for( unsigned i=0; i<g2d.gy().size(); i++)
        {
            for( unsigned k=0; k<g2d.gx().size(); k++)
                std::cout << y[i*g2d.gx().size()+k] << " ";
            std::cout<< std::endl;
        }
        std::cout << "Test filtered symv\n";
        dg::blas2::stencil( dg::CSRSymvFilter(), (dg::IHMatrix)stencil, x, y);
        for( unsigned i=0; i<g2d.gy().size(); i++)
        {
            for( unsigned k=0; k<g2d.gx().size(); k++)
                std::cout << y[i*g2d.gx().size()+k] << " ";
            std::cout<< std::endl;
        }
    }

    std::cout << "Test dg slope limiter stencil\n";
    dg::Grid1d g1d( 0,1, 3, 3);
    for( auto bc : bcs)
    {
        auto stencil = dg::create::limiter_stencil( g1d, bc);
        std::cout << "Test "<<dg::bc2str( bc)<<" boundary:\n";
        for( unsigned i=0; i<stencil.row_offsets.size(); i++)
            std::cout << stencil.row_offsets[i]<<" ";
        std::cout << "\n";
        for( unsigned i=0; i<stencil.column_indices.size(); i++)
            std::cout << stencil.column_indices[i]<<" ";
        std::cout << "\n";
        for( unsigned i=0; i<stencil.column_indices.size(); i++)
            std::cout << stencil.values[i]<<" ";
        std::cout << "\n";
    }
    std::cout << "Test "<<dg::bc2str( dg::DIR)<<" boundary:\n";
    auto stencil = dg::create::limiter_stencil( dg::coo3d::y, g2d, dg::DIR);
    x = dg::evaluate( [](double x, double y){return sin(x)*sin(y);}, g2d);
    dg::blas2::stencil( dg::CSRSlopeLimiter<double>(), stencil, x, y);
    //for( unsigned i=0; i<g2d.gy().size(); i++)
    //    std::cout<< y[i*g2d.gx().size()+0]<<" ";
    //std::cout << "\n";


    auto xx(x), yy(x), zz(x);
    dg::Grid2d g2dT( 0,2, 0,1, 3, 3, 5);
    auto stencilX = dg::create::limiter_stencil( dg::coo3d::x, g2dT, dg::DIR);
    dg::transpose( g2d.gx().size(), g2d.gy().size(), x, xx);
    dg::blas2::stencil( dg::CSRSlopeLimiter<double>(), stencilX, xx, zz);
    //for( unsigned i=0; i<g2d.gy().size(); i++)
    //    std::cout<< zz[i]<<" ";
    //std::cout << "\n";
    dg::transpose( g2d.gy().size(), g2d.gx().size(), zz, yy);
    dg::blas1::axpby( 1., yy, -1., y);
    double err = dg::blas1::dot( y, 1.);
    std::cout << "error between transposed application and app is: " <<err<<" (0)\n";



    return 0;
}
