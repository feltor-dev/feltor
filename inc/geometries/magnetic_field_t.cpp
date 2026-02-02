
#include <iostream>
#include <iomanip>

#include "dg/algorithm.h"

#include "solovev.h"
#include "magnetic_field.h"
#include "catch2/catch_all.hpp"

// We correct PI so that B(R_O, 0) = 1 exactly
const std::string geometry_params_Xpoint = R"asdf({
    "A": 0,
    "R_0": 0.545,
    "PP": 1,
    "PI": 1.042138271304694,
    "c":
    [
        0.085858385692821789,
        0.25157608973291837,
        -0.38061591778794789,
        -0.10218614802095409,
        0.36709261273307433,
        -0.39303943992277839,
        -0.016931807751497822,
        0,
        0,
        0,
        0,
        0
    ],
    "equilibrium": "solovev",
    "description" : "doubleX",
    "inverseaspectratio": 0.3211009174311926,
    "triangularity": 0.3,
    "elongation": 1.44
})asdf";

TEST_CASE("Magnetic field")
{
    dg::file::WrappedJsonValue js( dg::file::error::is_warning);
    js.asJson() = dg::file::string2Json( geometry_params_Xpoint);

    dg::geo::solovev::Parameters gp(js);
    int sign = GENERATE( +1, -1);
    gp.pi = sign*gp.pi;

    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField(gp);

    double R_O = gp.R_0;
    double Z_O = 0.;
    dg::geo::findOpoint( mag.get_psip(), R_O, Z_O);
    SECTION( "O-point")
    {
        CHECK( fabs(R_O - 0.567965) < 1e-6);
        CHECK( fabs(Z_O ) < 1e-6);
    }

    SECTION( "Toroidal field")
    {
        // Magnetic field strength at O-point is purely toroidal
        dg::geo::Bmodule bmod( mag);
        dg::geo::Btor btor(mag);
        dg::geo::InvB invB(mag);
        dg::geo::LnB lnB(mag);
        CHECK ( fabs( bmod( R_O, 0) - 1 ) < 1e-12);
        CHECK ( fabs( btor( R_O, 0) - sign ) < 1e-12);
        CHECK ( fabs( invB( R_O, 0) - 1 ) < 1e-12);
        CHECK ( fabs( lnB(  R_O, 0) ) < 1e-12);
    }
    SECTION( "Curvature")
    {
        // Curvatures at O-point are the same for all
        dg::geo::CurvatureNablaBR  curvKBR( mag, sign);
        dg::geo::CurvatureNablaBZ  curvKBZ( mag, sign);
        dg::geo::CurvatureKappaR   curvKKR( mag, sign);
        dg::geo::CurvatureKappaZ   curvKKZ( mag, sign);
        dg::geo::DivCurvatureKappa divCurvKappa( mag, sign);

        dg::geo::ToroidalCurvatureNablaBR  torcurvKBR( mag);
        dg::geo::ToroidalCurvatureNablaBZ  torcurvKBZ( mag);
        dg::geo::ToroidalCurvatureKappaR   torcurvKKR( mag);
        dg::geo::ToroidalCurvatureKappaZ   torcurvKKZ( mag);
        dg::geo::ToroidalDivCurvatureKappa tordivCurvKappa( mag);

        dg::geo::TrueCurvatureNablaBR  truecurvKBR( mag);
        dg::geo::TrueCurvatureNablaBZ  truecurvKBZ( mag);
        dg::geo::TrueCurvatureKappaR   truecurvKKR( mag);
        dg::geo::TrueCurvatureKappaZ   truecurvKKZ( mag);
        dg::geo::TrueDivCurvatureKappa truedivCurvKappa( mag);

        CHECK( fabs(curvKBR( R_O, Z_O) - torcurvKBR( R_O, Z_O)) < 1e-12);
        CHECK( fabs(curvKBZ( R_O, Z_O) - torcurvKBZ( R_O, Z_O)) < 1e-12);
        CHECK( fabs(curvKKR( R_O, Z_O) - torcurvKKR( R_O, Z_O)) < 1e-12);
        CHECK( fabs(curvKKZ( R_O, Z_O) - torcurvKKZ( R_O, Z_O)) < 1e-12);
        CHECK( fabs(divCurvKappa( R_O, Z_O) - tordivCurvKappa( R_O, Z_O)) < 1e-12);

        CHECK( fabs(curvKBR( R_O, Z_O) - truecurvKBR( R_O, Z_O)) < 1e-12);
        CHECK( fabs(curvKBZ( R_O, Z_O) - truecurvKBZ( R_O, Z_O)) < 1e-12);
        CHECK( fabs(curvKKR( R_O, Z_O) - truecurvKKR( R_O, Z_O)) < 1e-12);
        CHECK( fabs(curvKKZ( R_O, Z_O) - truecurvKKZ( R_O, Z_O)) < 1e-12);
        CHECK( fabs(divCurvKappa( R_O, Z_O) - truedivCurvKappa( R_O, Z_O)) < 1e-12);

    }
    SECTION( "BHat")
    {
        auto bhat = dg::geo::createBHat( mag);
        auto torbhat = dg::geo::createToroidalBHat( mag);
        double R = R_O + 0.1, Z = 0.1;

        CHECK( fabs(bhat.x()(R,Z)/bhat.z()(R,Z)   - R*torbhat.x()( R,Z)) < 1e-12);
        CHECK( fabs(bhat.y()(R,Z)/bhat.z()(R,Z)   - R*torbhat.y()( R,Z)) < 1e-12);
        CHECK( fabs(1./R   - torbhat.z()( R,Z)) < 1e-12);
        CHECK ( fabs( torbhat.div()(R,Z) - ( bhat.divvvz()(R,Z)/R - bhat.x()(R,Z)/bhat.z()(R,Z)/R/R )) < 1e-12);
    }

    SECTION( "Divb")
    {
        dg::geo::Divb div(mag);
        dg::geo::GradLnB gradLnB(mag);
        dg::geo::ToroidalDivb tordiv(mag);
        dg::geo::ToroidalGradLnB torgradLnB(mag);
        dg::geo::DivVVP divvvp(mag);
        CHECK( fabs(div( R_O, 0)  - tordiv( R_O,0))< 1e-12);
        double R = R_O + 0.1, Z = 0.1;
        CHECK( fabs( div( R,Z) + gradLnB(  R,Z)) < 1e-12);

        CHECK( fabs( tordiv( R,Z) + torgradLnB(  R,Z)) < 1e-12);

        dg::geo::BFieldR BR(mag);
        dg::geo::BFieldP Bphi(mag);
        CHECK ( fabs( tordiv(R,Z) - ( divvvp(R,Z)/R - BR(R,Z)/Bphi(R,Z)/R/R )) < 1e-12);

    }
}
