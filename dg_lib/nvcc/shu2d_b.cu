#include <iostream>
#include <iomanip>
#include <thrust/host_vector.h>

#include "timer.cuh"

#include "functors.cuh"
#include "arrvec2d.cuh"
#include "evaluation.cuh"
#include "shu.cuh"
#include "rk.cuh"



using namespace std;
using namespace dg;

const unsigned n = 3;
const unsigned Nx = 25;
const unsigned Ny = 25;
const double lx = 1.;
const double ly = 1.;

const unsigned k = 2;
const double D = 0.;
const double U = 1.;
const double T = 1.;
const unsigned NT = (unsigned)(U*n*Nx/0.1/lx);

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;
typedef dg::ArrVec2d< double, n, HVec>  HArrVec;
typedef dg::ArrVec2d< double, n, DVec>  DArrVec;
typedef cusp::ell_matrix<int, double, cusp::host_memory> HMatrix;
typedef cusp::ell_matrix<int, double, cusp::device_memory> DMatrix;


double initial( double x, double y){return 2.*sin(x)*sin(y);}
double solution( double x, double y) {return 2.*sin(x)*sin(y)*exp( -2.*T*D);}


using namespace std;

int main()
{
    Timer t;
    const double hx = lx/ (double)Nx;
    const double hy = ly/ (double)Ny;
    const double dt = T/(double)NT;
    cout << "# of Legendre coefficients: " << n<<endl;
    cout << "# of grid cells:            " << Nx*Ny<<endl;
    cout << "Timestep                    " << dt << endl;
    //cout << "# of timesteps              " << NT << endl;
    cout << "Diffusion                   " << D <<endl;
    dg::Lamb lamb( 0.5*lx, 0.5*ly, 0.2*lx, U);
    HArrVec omega = expand< dg::Lamb, n> ( lamb, 0, lx, 0, ly, Nx, Ny);
    DArrVec stencil = expand< double(&)(double, double), n> ( one, 0, lx, 0, ly, Nx, Ny);
    DArrVec solution = expand< dg::Lamb, n> ( lamb, 0, lx, 0, ly, Nx, Ny);
    DVec y0( omega.data()), y1( y0);
    Shu<double, n, DVec, cusp::device_memory> test( Nx, Ny, hx, hy, D);
    RK< 3, Shu<double, n, DVec, cusp::device_memory> > rk( y0);

    t.tic();
    test( y0, y1);
    t.toc();
    cout << "Time for one rhs evaluation: "<<t.diff()<<"s\n";
    double vorticity1 = blas2::dot( stencil.data(), S2D<double, n>(hx, hy), y0);
    double enstrophy1 = blas2::dot( y0, S2D<double, n>(hx, hy), y0);
    double energy1 =    blas2::dot( y0, S2D<double, n>(hx, hy), test.potential()) ;
    cout << "Total vorticity is: "<<vorticity1<<std::endl;
    cout << "Total enstrophy is: "<<enstrophy1<<std::endl;
    cout << "Total energy    is: "<<energy1<<std::endl;

    for( unsigned i=0; i<NT; i++)
    {
        rk( test, y0, y1, dt);
        thrust::swap(y0, y1);
    }
    double vorticity2 = blas2::dot( stencil.data(), S2D<double, n>(hx, hy), y0);
    double enstrophy2 = blas2::dot( y0, S2D<double, n>(hx, hy), y0);
    test( y0, y1); //get the potential ready
    double energy2 =    blas2::dot( y0, S2D<double, n>(hx, hy), test.potential()) ;
    cout << "Total vorticity is: "<<vorticity2<<std::endl;
    cout << "Total enstrophy is: "<<enstrophy2<<std::endl;
    cout << "Total energy    is: "<<energy2<<std::endl;
    cout << "Relative error enstrophy: "<<(enstrophy2-enstrophy1)/enstrophy1<<std::endl;
    cout << "Relative error energy:    "<<(energy2-energy1)/energy1<<std::endl;
    ////////////////////////////////////////////////////////////////////
    blas1::axpby( 1., solution.data(), -1., y0);
    cout << "Distance to solution "<<sqrt( blas2::dot( S2D<double, n>(hx, hy), y0))<<endl; //don't forget sqrt when comuting errors

    return 0;

}
