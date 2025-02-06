#include <iostream>
#ifdef WITH_MPI
#include <mpi.h>
#include "../backend/mpi_init.h"
#include "../backend/typedefs.h"
#endif

#include "geometry.h"
#include "evaluation.h"

#include "../blas2.h"
#include "catch2/catch_all.hpp"

inline double R_0 = 4.*M_PI;

inline double sine( double R, double Z,double phi){ return
    sin(R-R_0)*sin(Z)*sin(phi)/sqrt(R);}

//TEST geometry.h for every container and geometry that you want to use
TEST_CASE("Tensor volume")
{
#ifdef WITH_MPI
    MPI_Comm comm3d = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0,0}, {0,0,1});
    dg::CylindricalMPIGrid3d grid( R_0 , R_0+ 2.*M_PI, 0.,2.*M_PI, 0., 2.*M_PI,
            3,32,24,16, dg::DIR, dg::DIR, dg::PER, comm3d);
#else
    dg::CylindricalGrid3d grid( R_0 , R_0+ 2.*M_PI, 0.,2.*M_PI, 0., 2.*M_PI,
            3,32,24,16, dg::DIR, dg::DIR, dg::PER);
#endif
    dg::x::DVec vol = dg::tensor::volume(grid.metric());

    dg::x::DVec b = dg::evaluate( sine, grid);
    dg::x::DVec vol3d = dg::create::volume( grid);
    double test = dg::blas2::dot( b, vol3d, b);
    double sol = M_PI*M_PI*M_PI;
    INFO( "Test of volume:         "<<test<< " sol = " << sol<< "\t");
    INFO( "rel diff = " <<( test -  sol)/ sol);
    CHECK ( fabs( test-sol)/sol < 1e-15);
    dg::x::DVec temp = dg::create::weights( grid);
    dg::blas1::pointwiseDot( vol, temp, temp);
    test = dg::blas2::dot( b, temp, b);
    INFO( "Test of multiplyVolume: "<<test<< " sol = " << sol<< "\t");
    INFO( "rel diff = " <<( test -  sol)/ sol);
    CHECK ( fabs( test-sol)/sol < 1e-15);

    dg::x::DVec inv3d = dg::create::inv_volume( grid);
    dg::blas1::pointwiseDot( vol3d, b, b);
    test = dg::blas2::dot( b, inv3d, b);
    INFO( "Test of inv_volume:     "<<test<< " sol = " << sol<< "\t");
    INFO( "rel diff = " <<( test -  sol)/ sol);
    CHECK ( fabs( test-sol)/sol < 1e-15);
    temp = dg::create::inv_weights( grid);
    dg::blas1::pointwiseDivide(temp, vol, temp );
    test = dg::blas2::dot( b, temp, b);
    INFO( "Test of divideVolume:   "<<test<< " sol = " << sol<< "\t");
    INFO( "rel diff = " <<( test -  sol)/ sol);
    CHECK ( fabs( test-sol)/sol < 1e-15);

}
