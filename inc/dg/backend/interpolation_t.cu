#include <iostream>

#include <cusp/print.h>
#include "xspacelib.cuh"
#include "interpolation.cuh"
#include "typedefs.cuh"

const unsigned n = 3;
const unsigned Nx = 3; 
const unsigned Ny = 5; 
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
    dg::Matrix B = dg::create::interpolation( x, y, g);
    bool passed = true;
    //cusp::print(A);
    //cusp::print(B);
    //ATTENTION: backscatter might delete zeroes in matrices
    for( unsigned i=0; i<A.values.size(); i++)
    {
        if( (A.values[i] - B.values[i]) > 1e-14)
        {
            std::cerr << "NOT EQUAL "<<A.row_indices[i] <<" "<<A.column_indices[i]<<" "<<A.values[i] << "\t "<<B.row_indices[i]<<" "<<B.column_indices[i]<<" "<<B.values[i]<<"\n";
            passed = false;
        }
    }
    if( A.num_entries != B.num_entries)
    {
        std::cerr << "Number of entries not equal!\n";
        passed = false;
    }
    if( passed)
        std::cout << "2D TEST PASSED!\n";
    }
    ////////////////////////////////////////////////////////////////////////////
    {
    dg::Grid3d<double> g( -10, 10, -5, 5, -7, -3, n, Nx, Ny, Nz);
    dg::Matrix A = dg::create::backscatter( g);
    A.sort_by_row_and_column();

    std::vector<double> x( g.size()), y(x), z(x);
    for( unsigned k=0; k<g.Nz(); k++)
        for( unsigned i=0; i<g.Ny()*g.n(); i++)
            for( unsigned j=0; j<g.Nx()*g.n(); j++)
            {
                x[(k*g.Ny()*g.n() + i)*g.Nx()*g.n() + j] = 
                        g.x0() + (j+0.5)*g.hx()/(double)(g.n());
                y[(k*g.Ny()*g.n() + i)*g.Nx()*g.n() + j] = 
                        g.y0() + (i+0.5)*g.hy()/(double)(g.n());
                z[(k*g.Ny()*g.n() + i)*g.Nx()*g.n() + j] = 
                        g.z0() + (k+0.5)*g.hz();
            }
    dg::Matrix B = dg::create::interpolation( x, y, z, g);
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
        std::cout << "3D TEST PASSED!\n";
    }

    return 0;
}
