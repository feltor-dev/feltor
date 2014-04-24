#include <iostream>

#include <cusp/print.h>
#include "xspacelib.cuh"
#include "interpolation.cuh"
#include "typedefs.cuh"

const unsigned n = 2;
const unsigned Nx = 2; 
const unsigned Ny = 2; 


int main()
{

    dg::Grid2d<double> g( 0, 10, 0, 5, n, Nx, Ny);
    dg::Matrix A = dg::create::backscatter( g, dg::XSPACE);
    A.sort_by_row_and_column();

    std::vector<double> x( g.size()), y(x);
    for( unsigned i=0; i<g.Ny(); i++)
        for( unsigned j=0; j<g.Nx(); j++)
            for( unsigned k=0; k<g.n(); k++)
                for( unsigned l=0; l<g.n(); l++)
        {
            x[i*g.Nx()*g.n()*g.n() + j*g.n()*g.n() + k*g.n() + l] = 
                    j*g.hx() + g.hx()/(double)(2*g.n()) + l*g.hx()/(double)g.n();
            y[i*g.Nx()*g.n()*g.n() + j*g.n()*g.n() + k*g.n() + l] = 
                    i*g.hy() + g.hy()/(double)(2*g.n()) + k*g.hy()/(double)g.n();
        }
    dg::Matrix B = dg::create::interpolation( x, y, g);
    std::cout << "Note that the first is scattered!\n";
    cusp::print( A);
    cusp::print( B);


    return 0;
}
