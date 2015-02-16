#include <cusp/print.h>
#include "timer.cuh"
#include "xspacelib.cuh"
#include "ell_interpolation.cuh"
#include "interpolation.cuh"
#include "typedefs.cuh"

double sinus( double x, double y) {return sin(x)*sin(y);}

int main()
{

    unsigned n, Nx, Ny;
    std::cout << "Type n, Nx, Ny:\n";
    std::cin >> n >> Nx >> Ny;
    dg::Grid2d<double> g( -10, 10, -5, 5, n, Nx, Ny);

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
    dg::blas1::axpby( 1., w, -1., w2, w2);
    std::cout << "Error is: "<<dg::blas1::dot( w2, w2)<<std::endl;
    
    return 0;
}
