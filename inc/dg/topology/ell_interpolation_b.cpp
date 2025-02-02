#include <iostream>
#ifndef __NVCC__
#warning "This program has to be compiled with nvcc!"
int main(){
    std::cout << "This program has to be compiled with nvcc!\n";
    return 0;
}
#else
#include <cusp/print.h>
#include "dg/backend/timer.h"
#include "xspacelib.h"
#include "ell_interpolation.cuh"
#include "interpolation.h"

double sinus( double x, double y) {return sin(x)*sin(y);}
double sinus( double x, double y, double z) {return sin(x)*sin(y)*sin(z);}

int main()
{

    {
    unsigned n, Nx, Ny;
    std::cout << "Type n, Nx, Ny:\n";
    std::cin >> n >> Nx >> Ny;
    dg::Grid2d g( -10, 10, -5, 5, n, Nx, Ny);

    thrust::host_vector<double> x( g.size()), y(x);
    for( unsigned i=0; i<g.Ny()*g.n(); i++)
        for( unsigned j=0; j<g.Nx()*g.n(); j++)
        {
            x[i*g.Nx()*g.n() + j] =
                    g.x0() + (j+0.5)*g.hx()/(double)(g.n());
            y[i*g.Nx()*g.n() + j] =
                    g.y0() + (i+0.5)*g.hy()/(double)(g.n());
        }
    thrust::device_vector<double> xd(x), yd(y);
    dg::Timer t;
    t.tic();
    cusp::ell_matrix<int, double, cusp::device_memory> A = dg::create::ell_interpolation( xd, yd, g);
    t.toc();
    std::cout << "Ell  Interpolation matrix creation took: "<<t.diff()<<"s\n";
    t.tic();
    cusp::ell_matrix<int, double, cusp::device_memory> B = dg::create::interpolation( x, y, g);
    t.toc();
    std::cout << "Host Interpolation matrix creation took: "<<t.diff()<<"s\n";
    dg::DVec vector = dg::evaluate( sinus, g);
    dg::DVec w2( vector);
    dg::DVec w(vector);
    t.tic();
    dg::blas2::symv( B, vector, w2);
    t.toc();
    std::cout << "Application of interpolation matrix took: "<<t.diff()<<"s\n";

    dg::blas2::symv( A, vector, w);
    t.tic();
    dg::blas1::axpby( 1., w, -1., w2, w2);
    t.toc();
    std::cout << "Axpby took "<<t.diff()<<"s\n";
    std::cout << "Error is: "<<dg::blas1::dot( w2, w2)<<std::endl;
    }
    {
    unsigned n, Nx, Ny, Nz;
    std::cout << "Type n, Nx, Ny, Nz:\n";
    std::cin >> n >> Nx >> Ny >> Nz;
    dg::Grid3d g( -10, 10, -5, 5, -M_PI, M_PI, n, Nx, Ny, Nz);

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
    thrust::device_vector<double> xd(x), yd(y), zd(z);
    dg::Timer t;
    t.tic();
    cusp::ell_matrix<int,double, cusp::device_memory> A = dg::create::interpolation( x, y, z, g);
    t.toc();
    std::cout << "3D Host   Interpolation matrix creation took: "<<t.diff()<<"s\n";
    t.tic();
    cusp::ell_matrix<int,double, cusp::device_memory> dB = dg::create::ell_interpolation( xd, yd, zd, g);
    t.toc();
    std::cout << "3D Device Interpolation matrix creation took: "<<t.diff()<<"s\n";
    dg::DVec vector = dg::evaluate( sinus, g);
    dg::DVec dv( vector), w2( vector);
    dg::DVec w(vector);
    dg::blas2::symv( dB, dv, w2);
    t.tic();
    dg::blas2::symv( A, vector, w);
    t.toc();
    std::cout << "Application of matrix took "<<t.diff()<<"s\n";
    dg::blas1::axpby( 1., w, -1., w2, w2);
    std::cout << "3D Error is: "<<dg::blas1::dot( w2, w2)<<std::endl;
    }

    return 0;
}
#endif //__NVCC__
