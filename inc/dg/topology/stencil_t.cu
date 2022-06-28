#include <iostream>
#include <cusp/print.h>
#include "stencil.h"
#include "filter.h"
#include "../blas2.h"


int main()
{
    std::cout << "Test window stencil\n";
    dg::Grid2d g2d( 0,1, 0,2, 3, 4, 2);
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
    return 0;
}
