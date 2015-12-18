#include <iostream>

#include <cusp/print.h>

#include "backend/evaluation.cuh"
#include "geometry.h"

#include "blas2.h"

double R_0 = 4.*M_PI;


double sine( double R, double Z,double phi){ return sin(R-R_0)*sin(Z)*sin(phi)/sqrt(R);}

namespace dg
{
typedef thrust::device_vector<double> DVec;
}

//TEST geometry.h for every container and geometry that you want to use
int main()
{
    std::cout << "Type n, Nx, Ny, Nz\n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;
    dg::CylindricalGrid<dg::DVec> grid( R_0 , R_0+ 2.*M_PI, 0.,2.*M_PI, 0., 2.*M_PI,  n, Nx, Ny, Nz, dg::DIR, dg::DIR, dg::PER);

    dg::DVec b = dg::evaluate( sine, grid);
    dg::DVec vol3d = dg::create::volume( grid);
    double test = dg::blas2::dot( b, vol3d, b);
    double sol = M_PI*M_PI*M_PI;
    std::cout << "Test of volume:         "<<test<< " sol = " << sol<< "\t";
    std::cout << "rel diff = " <<( test -  sol)/ sol<<"\n";
    dg::DVec temp = dg::create::weights( grid);
    dg::geo::multiplyVolume( temp, grid);
    test = dg::blas2::dot( b, temp, b);
    std::cout << "Test of multiplyVolume: "<<test<< " sol = " << sol<< "\t";
    std::cout << "rel diff = " <<( test -  sol)/ sol<<"\n";

    dg::DVec inv3d = dg::create::inv_volume( grid);
    dg::blas1::pointwiseDot( vol3d, b, b);
    test = dg::blas2::dot( b, inv3d, b);
    std::cout << "Test of inv_volume:     "<<test<< " sol = " << sol<< "\t";
    std::cout << "rel diff = " <<( test -  sol)/ sol<<"\n";
    temp = dg::create::inv_weights( grid);
    dg::geo::divideVolume( temp, grid);
    test = dg::blas2::dot( b, temp, b);
    std::cout << "Test of divideVolume:   "<<test<< " sol = " << sol<< "\t";
    std::cout << "rel diff = " <<( test -  sol)/ sol<<"\n";


    return 0;
}
