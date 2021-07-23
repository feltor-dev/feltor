#include <iostream>

#include <cusp/print.h>
#include "xspacelib.h"
#include "interpolation.h"
#include "../blas.h"
#include "evaluation.h"

double function( double x){return sin(x);}
double function( double x, double y){return sin(x)*sin(y);}
double function( double x, double y, double z){return sin(x)*sin(y)*sin(z);}

const unsigned n = 3;
const unsigned Nx = 9;
const unsigned Ny = 5;
const unsigned Nz = 2;

typedef cusp::coo_matrix<int, double, cusp::host_memory> Matrix;

int main()
{

    dg::Grid2d g( -M_PI, 0, -5*M_PI, -4*M_PI, n, Nx, Ny);
    {
    std::cout << "First test grid set functions: \n";
    g.display( std::cout);
    g.set(2,2,3);
    g.display( std::cout);
    g.set(n, Nx, Ny);
    g.display( std::cout);
    }
    {
    bool passed = true;
    dg::Grid1d g1d( -M_PI, 0, n, Nx);
    thrust::host_vector<double> xs = dg::evaluate( dg::cooX1d, g1d);
    thrust::host_vector<double> x( g1d.size());
    for( unsigned i=0; i<x.size(); i++)
    {
        //create equidistant values
        x[i] = g1d.x0() + g1d.lx() + (i+0.5)*g1d.h()/(double)(g1d.n());
        //use DIR because the cooX1d is zero on the right boundary
        double xi = dg::interpolate( dg::xspace,xs, x[i], g1d, dg::DIR);
        if( x[i] - xi > 1e-14)
        {
            std::cerr << "X NOT EQUAL "<<i<<"\t"<<x[i]<<"  \t"<<xi<<"\n";
            passed = false;
        }
    }
    if( passed)
        std::cout << "1D INTERPOLATE TEST PASSED!\n";
    //use DIR because the coo.2d is zero on the right boundary
    Matrix B = dg::create::interpolation( x, g1d, dg::DIR, "dg");
    //values outside the grid are mirrored back in

    const thrust::host_vector<double> vec = dg::evaluate( function, g1d);
    thrust::host_vector<double> inter(vec);
    dg::blas2::symv( B, vec, inter);
    //inter now contains the values of vec interpolated at equidistant points
    Matrix A = dg::create::backscatter( g1d);
    thrust::host_vector<double> inter1(vec);
    dg::blas2::symv( A, vec, inter1);
    dg::blas1::axpby( 1., inter1, +1., inter, inter1);//the mirror makes a sign!!
    double error = dg::blas1::dot( inter1, inter1);
    std::cout << "Error for method dg is "<<error<<" (should be small)!\n";
    if( error > 1e-14)
        std::cout << "1D TEST FAILED!\n";
    else
        std::cout << "1D TEST PASSED!\n";
    g1d = dg::Grid1d ( -1.5, 6.5, 1, 8);
    x = std::vector<double>{ -1.4, -0.4, .1, 2.2, 2.8, 3, 3.1, 3.4, 4.9, 6.1};
    std::vector<std::string> methods = {"nearest", "linear", "cubic"};
    for ( auto method : methods)
    {
        //use DIR because the coo.2d is zero on the right boundary
        Matrix B = dg::create::interpolation( x, g1d, dg::DIR, method);
        //values outside the grid are mirrored back in

        const std::vector<double> vec{0,0,0,0,1,1,1,1};
        const std::vector<double> sol0{0, 0,0,0,1,1,1,1,1,1};
        const std::vector<double> sol1{0, 0,0,0.2,0.8,1,1,1,1,1};
        const std::vector<double> sol3{0, 0,0,0.184,0.816,1.,1.0285,1.064,1,1};

        thrust::host_vector<double> inter(x.size());
        dg::blas2::symv( B, (dg::HVec)vec, inter);
        if( method == "nearest") dg::blas1::axpby( 1., (dg::HVec)sol0, -1., inter);
        if( method == "linear") dg::blas1::axpby( 1., (dg::HVec)sol1, -1., inter);
        if( method == "cubic") dg::blas1::axpby( 1., (dg::HVec)sol3, -1., inter);
        //inter now contains the values of vec interpolated at equidistant points
        double error = dg::blas1::dot( inter, inter);
        std::cout << "Error for method "<<method<<" is "<<error<<" (should be small)!\n";
        if( error > 1e-14)
            std::cout << "1D TEST FAILED!\n";
        else
            std::cout << "1D TEST PASSED!\n";

    }
    }
    {


    //![doxygen]
    //create equidistant values
    thrust::host_vector<double> x( g.size()), y(x);
    for( unsigned i=0; i<g.Ny()*g.n(); i++)
        for( unsigned j=0; j<g.Nx()*g.n(); j++)
        {
            //intentionally set values outside the grid domain
            x[i*g.Nx()*g.n() + j] =
                    g.x0() + g.lx() + (j+0.5)*g.hx()/(double)(g.n());
            y[i*g.Nx()*g.n() + j] =
                    g.y0() + 2*g.ly() + (i+0.5)*g.hy()/(double)(g.n());
        }
    std::vector<std::string> methods = {"nearest", "linear", "dg", "cubic"};
    for( auto method : methods)
    {
        //use DIR because the coo.2d is zero on the right boundary
        Matrix B = dg::create::interpolation( x, y, g, dg::DIR, dg::DIR, method);
        //values outside the grid are mirrored back in

        const thrust::host_vector<double> vec = dg::evaluate( function, g);
        thrust::host_vector<double> inter(vec);
        dg::blas2::symv( B, vec, inter);
        //inter now contains the values of vec interpolated at equidistant points
        //![doxygen]
        dg::Grid2d gequi( -M_PI, 0, -5*M_PI, -4*M_PI, 1, n*Nx, n*Ny);
        thrust::host_vector<double> inter1 = dg::evaluate( function, gequi);
        dg::blas1::axpby( 1., inter1, +1., inter, inter1);//the mirror makes a sign!!
        double error = dg::blas1::dot( inter1, inter1);
        std::cout << "Error for method "<<method<<" is "<<error<<" (should be small)!\n";
        // Maybe we should get an exact test here ...
        //if( error > 1e-14)
        //    std::cout<< "2D TEST FAILED!\n";
        //else
        //    std::cout << "2D TEST PASSED!\n";
    }


    bool passed = true;
    thrust::host_vector<double> xs = dg::evaluate( dg::cooX2d, g);
    thrust::host_vector<double> ys = dg::evaluate( dg::cooY2d, g);
    thrust::host_vector<double> xF = dg::forward_transform( xs, g);
    for( unsigned i=0; i<g.size(); i++)
    {
        //use DIR because the coo.2d is zero on the right boundary
        double xi = dg::interpolate(dg::lspace, xF, x[i],y[i], g, dg::DIR, dg::DIR);
        double yi = dg::interpolate(dg::xspace, ys, x[i],y[i], g, dg::DIR, dg::DIR);
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

    //![doxygen3d]
    //create equidistant values
    thrust::host_vector<double> x( g.size()), y(x), z(x);
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
    const thrust::host_vector<double> vec = dg::evaluate( function, g);
    thrust::host_vector<double> inter(vec);
    dg::blas2::symv( B, vec, inter);
    //![doxygen3d]
    thrust::host_vector<double> inter1(vec);
    dg::blas2::symv( A, vec, inter1);
    dg::blas1::axpby( 1., inter1, -1., inter, inter1);
    double error = dg::blas1::dot( inter1, inter1);
    std::cout << "Error is "<<error<<" (should be small)!\n";
    if( error > 1e-14)
        std::cout<< "3D TEST FAILED!\n";
    else
        std::cout << "3D TEST PASSED!\n";
    }

    return 0;
}
