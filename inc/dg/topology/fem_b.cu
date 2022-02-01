#include <iostream>
#include <cmath>
#include <cusp/print.h>
#include "dg/algorithm.h"
#include "fem.h"
#include "fem_weights.h"

double function( double x, double y){return sin(x)*cos(y);}
using IDMatrix = cusp::ell_matrix<int, double, cusp::device_memory>;

int main ()
{
    unsigned n = 3, Nx = 20, Ny = 20;
    std::cout << "# Type in n Nx Ny mx eps!\n";
    std::cin >> n >> Nx >> Ny;
    std::cout << "# on grid " << n <<" x "<<Nx<<" x "<<Ny<<"\n";
    dg::CartesianGrid2d gDIR( 0, 2.*M_PI, M_PI/2., 5*M_PI/2., n, Nx, Ny, dg::DIR,
            dg::DIR);
    const dg::DVec x = dg::evaluate( function, gDIR);
    dg::DVec y(x);
    IDMatrix fem_mass = dg::create::fem_mass( gDIR);
    double gbytes=(double)x.size()*sizeof(double)/1e9;
    unsigned multi=100;
    dg::Timer t;
    dg::blas2::symv( fem_mass, x, y);
    t.tic();
    for( unsigned i=0; i<multi; i++)
        dg::blas2::symv( fem_mass, x, y);
    t.toc();
    std::cout<<"SYMV (y=Sx)                      "
             <<t.diff()/multi<<"s\t"<<8*gbytes*multi/t.diff()<<"GB/s\n";
    return 0;
}
