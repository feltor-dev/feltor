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

    dg::Grid2d<double> g( -10, 10, -5, 5, n, Nx, Ny);
    dg::Matrix A = dg::create::backscatter( g, dg::XSPACE);
    A.sort_by_row_and_column();

    std::vector<double> x( g.size()), y(x);
    for( unsigned i=0; i<g.Ny()*g.n(); i++)
        for( unsigned j=0; j<g.Nx()*g.n(); j++)
        {
            x[i*g.Nx()*g.n() + j] = 
                    g.x0() + (j+0.5)*g.hx()/(double)(g.n());
            y[i*g.Nx()*g.n() + j] = 
                    g.y0() + (i+0.5)*g.hy()/(double)(g.n());
        }
    dg::Matrix B = dg::create::interpolation( x, y, g);
    bool passed = true;
    for( unsigned i=0; i<A.values.size(); i++)
    {
        if( A.values[i] != B.values[i])
        {
            std::cerr << "NOT EQUAL "<<A.row_indices[i] <<" "<<A.column_indices[i]<<" "<<A.values[i] << "\t "<<B.values[i]<<"\n";
            passed = false;
        }
    }
    if( passed)
        std::cout << "TEST PASSED!\n";


    return 0;
}
