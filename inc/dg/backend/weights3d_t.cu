#include <iostream>

#include <cusp/print.h>

#include "file/read_input.h"
#include "xspacelib.cuh"
#include "../blas2.h"

double R_0 = 4.*M_PI;


double sine( double R, double Z,double phi){ return sin(R-R_0)*sin(Z)*sin(phi)/sqrt(R);}

int main()
{
    std::cout << "Type n, Nx, Ny, Nz\n";
    //std::cout << "Note, that function is resolved exactly in R,Z for n > 2\n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;
    dg::Grid3d<double> grid3d( R_0 , R_0+ 2.*M_PI, 0.,2.*M_PI, 0., 2.*M_PI,  n, Nx, Ny, Nz,dg::DIR, dg::DIR, dg::PER,dg::cylindrical);

    dg::DVec b = dg::evaluate( sine, grid3d);
    dg::DVec w3d = dg::create::weights( grid3d);
    dg::DVec v3d = dg::create::inv_weights( grid3d);

    std::cout << "Test of w3d: "<<dg::blas2::dot(b, w3d, b)<< " sol = " << M_PI*M_PI*M_PI<< std::endl;
    std::cout << "rel diff = " <<( dg::blas2::dot(b, w3d, b) -  M_PI*M_PI*M_PI)/ M_PI*M_PI*M_PI<<std::endl;

    return 0;
}
