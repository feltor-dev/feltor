#include <iostream>

#include <cusp/print.h>
#include "xspacelib.h"
#include "interpolation.h"
#include "fast_interpolation.h"
#include "../blas.h"
#include "evaluation.h"

double function( double x){return sin(x);}
double function( double x, double y){return sin(x)*sin(y);}
double function( double x, double y, double z){return sin(x)*sin(y)*sin(z);}

const unsigned n = 3;
const unsigned Nx = 9;
const unsigned Ny = 5;
const unsigned Nz = 4;

typedef cusp::coo_matrix<int, double, cusp::host_memory> Matrix;

int main()
{

    dg::Grid2d g( -M_PI, 0, -5*M_PI, -4*M_PI, n, Nx, Ny);
    std::cout << "First test grid set functions: \n";
    {
    g.display( std::cout);
    g.set(2,2,3);
    g.display( std::cout);
    g.set(n, Nx, Ny);
    g.display( std::cout);
    }
    std::cout << "\n1D TESTS:\n";
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
    x = std::vector<double>{ -1.6, -0.4, .1, 2.2, 2.8, 3, 3.1, 3.4, 4.9, 6.1};
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

    std::cout << "\n2D TESTS:\n";
    {
    //![doxygen]
    //create equidistant values
    thrust::host_vector<double> x( g.size()), y(x);
    for( unsigned i=0; i<g.Ny()*g.ny(); i++)
        for( unsigned j=0; j<g.Nx()*g.nx(); j++)
        {
            //intentionally set values outside the grid domain
            x[i*g.Nx()*g.nx() + j] =
                    g.x0() + g.lx() + (j+0.5)*g.hx()/(double)(g.nx());
            y[i*g.Nx()*g.ny() + j] =
                    g.y0() + 2*g.ly() + (i+0.5)*g.hy()/(double)(g.ny());
        }
    //use DIR because the coo.2d is zero on the right boundary
    Matrix B = dg::create::interpolation( x, y, g, dg::DIR, dg::DIR, "dg");
    //values outside the grid are mirrored back in

    const thrust::host_vector<double> vec = dg::evaluate( function, g);
    thrust::host_vector<double> inter(vec);
    dg::blas2::symv( B, vec, inter);
    //![doxygen]
    //inter now contains the values of vec interpolated at equidistant points
    Matrix A = dg::create::backscatter( g);
    thrust::host_vector<double> inter1(vec);
    dg::blas2::symv( A, vec, inter1);
    dg::blas1::axpby( 1., inter1, +1., inter, inter1);//the mirror makes a sign!!
    double error = dg::blas1::dot( inter1, inter1);
    std::cout << "Error for method dg is "<<error<<" (should be small)!\n";
    if( error > 1e-14)
        std::cout<< "2D TEST FAILED!\n";
    else
        std::cout << "2D TEST PASSED!\n";
    std::vector<std::string> methods = {"nearest", "linear", "cubic"};
    dg::Grid2d g2d ( -1.5, 6.5, -1.5, 6.5, 1, 8, 8);
    std::vector<double> x1d{ -1.4, -0.4, .1, 2.2, 2.8, 3, 3.1, 3.4, 4.9, 6.1};
    std::vector<double> y1d{ -1.4, -0.4, .1, 2.2, 2.8, 3, 3.1, 3.4, 4.9, 6.1};
    dg::HVec x2d( x1d.size()*y1d.size()), y2d(x2d);
    for( unsigned k=0; k<y1d.size(); k++)
    for( unsigned i=0; i<x1d.size(); i++)
    {
        x2d[k*x1d.size()+i] = x1d[i];
        y2d[k*x1d.size()+i] = y1d[k];
    }
    for( auto method : methods)
    {
        //use DIR because the coo.2d is zero on the right boundary
        Matrix B = dg::create::interpolation( x2d, y2d, g2d, dg::DIR, dg::DIR,
                method);
        const std::vector<double> v1ec{0,0,0,0,1,1,1,1};
        dg::HVec vec(v1ec.size()*v1ec.size());
        for( unsigned k=0; k<v1ec.size(); k++)
        for( unsigned i=0; i<v1ec.size(); i++)
            vec[k*v1ec.size()+i] = v1ec[i]*v1ec[k];
        const std::vector<double> s1ol0{0, 0,0,0,1,1,1,1,1,1};
        const std::vector<double> s1ol1{0, 0,0,0.2,0.8,1,1,1,1,1};
        const std::vector<double> s1ol3{0, 0,0,0.184,0.816,1.,1.0285,1.064,1,1};
        dg::HVec sol0(x2d.size()), sol1(sol0), sol3(sol0);
        for( unsigned k=0; k<s1ol0.size(); k++)
        for( unsigned i=0; i<s1ol0.size(); i++)
        {
            sol0[k*s1ol0.size()+i] = s1ol0[k]*s1ol0[i];
            sol1[k*s1ol0.size()+i] = s1ol1[k]*s1ol1[i];
            sol3[k*s1ol0.size()+i] = s1ol3[k]*s1ol3[i];
        }

        thrust::host_vector<double> inter(x2d.size());
        dg::blas2::symv( B, vec, inter);
        if( method == "nearest") dg::blas1::axpby( 1., sol0, -1., inter);
        if( method == "linear") dg::blas1::axpby( 1., sol1, -1., inter);
        if( method == "cubic") dg::blas1::axpby( 1., sol3, -1., inter);
        double error = dg::blas1::dot( inter, inter);
        std::cout << "Error for method "<<method<<" is "<<error<<" (should be small)!\n";
        if( error > 1e-14)
            std::cout<< "2D TEST FAILED!\n";
        else
            std::cout << "2D TEST PASSED!\n";
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
    std::cout << "\n3D TESTS:\n";
    {
    dg::Grid3d g( -10, 10, -5, 5, -7, -3, n, Nx, Ny, Nz);
    g.set( 2,2,2,3);
    g.set( {n,n,1}, {Nx, Ny,Nz});
    //g.set( n, 3*Nx,n, 4*Ny,n,4*Nz);
    //g.display();

    //![doxygen3d]
    //create equidistant values
    thrust::host_vector<double> x( g.size()), y(x), z(x);
    for( unsigned k=0; k<g.nz()*g.Nz(); k++)
        for( unsigned i=0; i<g.Ny()*g.ny(); i++)
            for( unsigned j=0; j<g.Nx()*g.nx(); j++)
            {
                x[(k*g.Ny()*g.ny() + i)*g.Nx()*g.nx() + j] =
                        g.x0() + (j+0.5)*g.hx()/(double)(g.nx());
                y[(k*g.Ny()*g.ny() + i)*g.Nx()*g.nx() + j] =
                        g.y0() + (i+0.5)*g.hy()/(double)(g.ny());
                z[(k*g.Ny()*g.ny() + i)*g.Nx()*g.nx() + j] =
                        g.z0() + (k+0.5)*g.hz()/(double)(g.nz());
            }
    std::vector<std::string> methods = {"nearest", "linear", "dg", "cubic"};
    for ( auto method : methods)
    {
        Matrix B = dg::create::interpolation( x, y, z, g, dg::DIR, dg::DIR, dg::PER,
                method);
        const thrust::host_vector<double> vec = dg::evaluate( function, g);
        thrust::host_vector<double> inter(vec);
        dg::blas2::symv( B, vec, inter);
        //![doxygen3d]
        dg::Grid3d gequi( -10, 10, -5, 5, -7, -3, 1, g.nx()*g.Nx(),
                g.ny()*g.Ny(), g.nz()*g.Nz());
        thrust::host_vector<double> inter1 = dg::evaluate( function, gequi);
        dg::blas1::axpby( 1., inter1, -1., inter, inter1);
        double error = dg::blas1::dot( inter1, inter1)/ dg::blas1::dot( inter,inter);
        std::cout << "Error for method "<<method<<" is "<<sqrt(error)<<" (should be small)!\n";
        //if( error > 1e-14)
        //    std::cout<< "3D TEST FAILED!\n";
        //else
        //    std::cout << "3D TEST PASSED!\n";
    }
    }

    return 0;
}
