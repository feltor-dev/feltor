#include <cusp/print.h>
#include "xspacelib.cuh"
#include "ell_interpolation.cuh"
#include "typedefs.cuh"
const unsigned n = 2;
const unsigned Nx = 2; 
const unsigned Ny = 2; 
const unsigned Nz = 2; 


int main()
{

    {
    dg::Grid2d<double> g( -10, 10, -5, 5, n, Nx, Ny);
    dg::Matrix A = dg::create::backscatter( g);
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
    thrust::device_vector<double> xd(x), yd(y);
    dg::DMatrix dB = dg::create::ell_interpolation( xd,yd, g);
    dg::Matrix B = dB;
    bool passed = true;
    for( unsigned i=0; i<A.values.size(); i++)
    {
        if( (A.values[i] - B.values[i]) > 1e-14)
        {
            std::cerr << "NOT EQUAL "<<A.row_indices[i] <<" "<<A.column_indices[i]<<" "<<A.values[i] << "\t "<<B.row_indices[i]<<" "<<B.column_indices[i]<<" "<<B.values[i]<<"\n";
            passed = false;
        }
    }
    if( passed)
        std::cout << "2D TEST PASSED!\n";
    }
    return 0;
}
