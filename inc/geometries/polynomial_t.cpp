
#include <iostream>
#include <iomanip>

#include "dg/algorithm.h"

#include "polynomial.h"
#include "magnetic_field.h"
#include "catch2/catch_all.hpp"


const std::string geometry_params = R"asdf({
    "PP": 1,
    "PI": 1,
    "c" :[  1,2,3,4],
    "R_0"                : 4,
    "inverseaspectratio" : 0.25,
    "elongation"         : 1,
    "triangularity"      : 1,
    "M" : 2,
    "N" : 2,
    "equilibrium"  : "polynomial",
    "description" : "standardX"
})asdf";

TEST_CASE("Polynomial")
{
    dg::file::WrappedJsonValue js( dg::file::error::is_warning);
    js.asJson() = dg::file::string2Json( geometry_params);

    dg::geo::polynomial::Parameters gp(js);
    dg::geo::TokamakMagneticField mag = dg::geo::createPolynomialField(gp);
    REQUIRE( gp.a == 1);
    SECTION( "Pointwise evaluation")
    {
        double result;
        result = mag.psip()(2.,3.);
        CHECK( result == 22);
        result = mag.psipR()(2.,3.);
        CHECK( result == 6);
        result = mag.psipZ()(2.,3.);
        CHECK( result == 4);
        result = mag.psipRR()(2.,3.);
        CHECK( result == 0);
        result = mag.psipZZ()(2.,3.);
        CHECK( result == 0);
        result = mag.psipRZ()(2.,3.);
        CHECK( result == 1);
    }
    SECTION( "Polynomial integration")
    {
        double Rmin=gp.R_0-gp.a;
        double Zmin=-gp.a*gp.elongation;
        double Rmax=gp.R_0+gp.a;
        double Zmax=gp.a*gp.elongation;
        dg::Grid2d grid(Rmin,Rmax,Zmin,Zmax, 3,100,100);
        auto weights = dg::create::weights( grid);
        auto psi = dg::evaluate( mag.psip(), grid);
        double result = dg::blas1::dot( weights, psi);
        CHECK( result == 64);
    }
    SECTION( "Accuracy of bhat")
    {
        double Rmin=gp.R_0-gp.a;
        double Zmin=-gp.a*gp.elongation;
        double Rmax=gp.R_0+gp.a;
        double Zmax=gp.a*gp.elongation;
        dg::CylindricalGrid3d grid3d(Rmin,Rmax,Zmin,Zmax, 0, 2.*M_PI, 3,100,100,32);
        dg::geo::CylindricalVectorLvl0 bhat_ = dg::geo::createBHat( mag);
        std::array<dg::HVec, 3> bhat;
        dg::pushForward( bhat_.x(), bhat_.y(), bhat_.z(),
                bhat[0], bhat[1], bhat[2], grid3d);
        std::array<dg::HVec, 3> bhat_covariant(bhat);
        dg::tensor::inv_multiply3d( grid3d.metric(), bhat[0], bhat[1], bhat[2],
                bhat_covariant[0], bhat_covariant[1], bhat_covariant[2]);
        dg::HVec normb( bhat[0]), one3d = dg::evaluate( dg::one, grid3d);
        dg::blas1::pointwiseDot( 1., bhat[0], bhat_covariant[0],
                                 1., bhat[1], bhat_covariant[1],
                                 0., normb);
        dg::blas1::pointwiseDot( 1., bhat[2], bhat_covariant[2],
                                 1., normb);
        dg::blas1::axpby( 1., one3d, -1, normb);
        double error = sqrt(dg::blas1::dot( normb, normb));
        INFO( "Error in norm b == 1 :  "<<error);
        CHECK( error < 1e-12);
    }
}
