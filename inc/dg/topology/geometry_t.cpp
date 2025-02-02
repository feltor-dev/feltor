#include <iostream>

#include <cusp/print.h>

#include "geometry.h"
#include "evaluation.h"

#include "../blas2.h"
#include "catch2/catch.hpp"

inline double R_0 = 4.*M_PI;

inline double sine( double R, double Z,double phi){ return
    sin(R-R_0)*sin(Z)*sin(phi)/sqrt(R);}

namespace dg { typedef thrust::device_vector<double> DVec; }
namespace dg { typedef thrust::host_vector<double> HVec; }

//TEST geometry.h for every container and geometry that you want to use
TEST_CASE("Geometry")
{
    dg::CylindricalGrid3d grid( R_0 , R_0+ 2.*M_PI, 0.,2.*M_PI, 0., 2.*M_PI,
            3,32,24,16, dg::DIR, dg::DIR, dg::PER);
    dg::DVec vol = dg::tensor::volume(grid.metric());

    dg::DVec b = dg::evaluate( sine, grid);
    dg::DVec vol3d = dg::create::volume( grid);
    double test = dg::blas2::dot( b, vol3d, b);
    double sol = M_PI*M_PI*M_PI;
    INFO( "Test of volume:         "<<test<< " sol = " << sol<< "\t");
    INFO( "rel diff = " <<( test -  sol)/ sol);
    CHECK ( fabs( test-sol)/sol < 1e-15);
    dg::DVec temp = dg::create::weights( grid);
    dg::blas1::pointwiseDot( vol, temp, temp);
    test = dg::blas2::dot( b, temp, b);
    INFO( "Test of multiplyVolume: "<<test<< " sol = " << sol<< "\t");
    INFO( "rel diff = " <<( test -  sol)/ sol);
    CHECK ( fabs( test-sol)/sol < 1e-15);

    dg::DVec inv3d = dg::create::inv_volume( grid);
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
