#include <iostream>

#include <cusp/print.h>
#include "xspacelib.cuh"
#include "interpolation.cuh"
#include "../blas.h"
#include "evaluation.cuh"

double function( double x, double y){return sin(x)*sin(y);}
double function( double x, double y, double z){return sin(x)*sin(y)*sin(z);}

const unsigned n = 3;
const unsigned Nx = 3; 
const unsigned Ny = 5; 
const unsigned Nz = 2; 

typedef cusp::coo_matrix<int, double, cusp::host_memory> Matrix;

int main()
{

    {
    dg::Grid2d g( -10, 10, -5, 5, n, Nx, Ny);
    std::cout << "First test grid set functions: \n";
    g.display( std::cout);
    g.set(2,2,3);
    g.display( std::cout);
    g.set(n, Nx, Ny);
    g.display( std::cout);
    Matrix A = dg::create::backscatter( g);
    //A.sort_by_row_and_column();

    thrust::host_vector<double> x( g.size()), y(x);
    for( unsigned i=0; i<g.Ny()*g.n(); i++)
        for( unsigned j=0; j<g.Nx()*g.n(); j++)
        {
            x[i*g.Nx()*g.n() + j] = 
                    g.x0() + (j+0.5)*g.hx()/(double)(g.n());
            y[i*g.Nx()*g.n() + j] = 
                    g.y0() + (i+0.5)*g.hy()/(double)(g.n());
        }
    Matrix B = dg::create::interpolation( x, y, g);
    thrust::host_vector<double> vec = dg::evaluate( function, g), inter1(vec), inter2(vec);
    dg::blas2::symv( A, vec, inter1);
    dg::blas2::symv( B, vec, inter2);
    dg::blas1::axpby( 1., inter1, -1., inter2, vec);
    double error = dg::blas1::dot( vec, vec);
    std::cout << "Error is "<<error<<" (should be small)!\n";
    //cusp::print(A);
    //cusp::print(B);
    //ATTENTION: backscatter might delete zeroes in matrices
    //for( unsigned i=0; i<A.values.size(); i++)
    //{
    //    if( (A.values[i] - B.values[i]) > 1e-14)
    //    {
    //        std::cerr << "NOT EQUAL "<<A.row_indices[i] <<" "<<A.column_indices[i]<<" "<<A.values[i] << "\t "<<B.row_indices[i]<<" "<<B.column_indices[i]<<" "<<B.values[i]<<"\n";
    //        passed = false;
    //    }
    //}
    //if( A.num_entries != B.num_entries)
    //{
    //    std::cerr << "Number of entries not equal!\n";
    //    passed = false;
    //}
    if( error > 1e-14) 
        std::cout<< "2D TEST FAILED!\n";
    else
        std::cout << "2D TEST PASSED!\n";


    bool passed = true;
    thrust::host_vector<double> xs = dg::evaluate( dg::cooX2d, g); 
    thrust::host_vector<double> ys = dg::evaluate( dg::cooY2d, g); 
    thrust::host_vector<double> xF = dg::create::forward_transform( xs, g);
    thrust::host_vector<double> yF = dg::create::forward_transform( ys, g);
    for( unsigned i=0; i<x.size(); i++)
    {
        double xi = dg::interpolate(x[i],y[i], xF, g);
        double yi = dg::interpolate(x[i],y[i], yF, g);
        if( x[i] - xi > 1e-14)
        {
            std::cerr << "X NOT EQUAL "<<i<<"\t"<<x[i]<<"  \t"<<xi<<"\n";
            passed = false;
        }
        if( y[i] - yi > 1e-14)
        {
            std::cerr << "Y NOT EQUAL "<<i<<"\t"<<y[i]<<"  \t"<<yi<<"\n";
            passed = false;
        }
    }
    if( passed)
        std::cout << "2D INTERPOLATE TEST PASSED!\n";
    }
    ////////////////////////////////////////////////////////////////////////////
    {
    dg::Grid3d g( -10, 10, -5, 5, -7, -3, n, Nx, Ny, Nz);
    g.set( 2,2,2,3);
    g.set( n, Nx,Ny,Nz);
    Matrix A = dg::create::backscatter( g);
    //A.sort_by_row_and_column();

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
    Matrix B = dg::create::interpolation( x, y, z, g);
    thrust::host_vector<double> vec = dg::evaluate( function, g), inter1(vec), inter2(vec);
    dg::blas2::symv( A, vec, inter1);
    dg::blas2::symv( B, vec, inter2);
    dg::blas1::axpby( 1., inter1, -1., inter2, vec);
    double error = dg::blas1::dot( vec, vec);
    std::cout << "Error is "<<error<<" (should be small)!\n";
    if( error > 1e-14) 
        std::cout<< "3D TEST FAILED!\n";
    else
        std::cout << "3D TEST PASSED!\n";

    //bool passed = true;
    //for( unsigned i=0; i<A.values.size(); i++)
    //{
    //    if( (A.values[i] - B.values[i]) > 1e-14)
    //    {
    //        std::cerr << "NOT EQUAL "<<A.row_indices[i] <<" "<<A.column_indices[i]<<" "<<A.values[i] << "\t "<<B.row_indices[i]<<" "<<B.column_indices[i]<<" "<<B.values[i]<<"\n";
    //        passed = false;
    //    }
    //}
    //if( passed)
    //    std::cout << "3D TEST PASSED!\n";
    }

    return 0;
}
