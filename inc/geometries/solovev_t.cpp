#include <iostream>
#include <iomanip>

#include "dg/algorithm.h"

#include "solovev.h"
#include "taylor.h"
#include "magnetic_field.h"
#include "catch2/catch_all.hpp"

struct JPhi
{
    JPhi( dg::geo::solovev::Parameters gp): R_0(gp.R_0), A(gp.A){}
    double operator()(double R, double, double)const
    {
        return ((A-1.)*R - A*R_0*R_0/R)/R_0/R_0/R_0;
    }
    private:
    double R_0, A;
};
const std::string geometry_params_Xpoint = R"asdf({
    "A" : 0.0,
    "PP": 1,
    "PI": 1,
    "c" :[  0.07350114445500399706283007092406934834526,
           -0.08662417436317227513877947632069712210813,
           -0.1463931543401102620740934776490506239925,
           -0.07631237100536276213126232216649739043965,
            0.09031790113794227394476271394334515457567,
           -0.09157541239018724584036670247895160625891,
           -0.003892282979837564486424586266476650443202,
            0.04271891225076417603805495295590637082745,
            0.2275545646002791311716859154040182853650,
           -0.1304724136017769544849838714185080700328,
           -0.03006974108476955225335835678782676287818,
            0.004212671892103931178531621087067962015783 ],
    "R_0"                : 547.891714877869,
    "inverseaspectratio" : 0.41071428571428575,
    "elongation"         : 1.75,
    "triangularity"      : 0.47,
    "equilibrium"  : "solovev",
    "description" : "standardX"
})asdf";

TEST_CASE("Solovev")
{
    dg::file::WrappedJsonValue js( dg::file::error::is_warning);
    js.asJson() = dg::file::string2Json( geometry_params_Xpoint);

    dg::geo::solovev::Parameters gp(js);
    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField(gp);

    SECTION( "X-point")
    {
        double R_X = gp.R_0-1.1*gp.triangularity*gp.a;
        double Z_X = -1.1*gp.elongation*gp.a;
        REQUIRE( gp.hasXpoint());

        dg::geo::findXpoint( mag.get_psip(), R_X, Z_X);
        INFO(  "X-point found at "<<R_X << " "<<Z_X<<" with Psip "<<mag.psip()(R_X, Z_X));
        INFO("     R - Factor "<<(gp.R_0-R_X)/gp.triangularity/gp.a
              << " Z - factor "<<-(Z_X/gp.elongation/gp.a));
        CHECK( fabs(R_X - (gp.R_0-1.1*gp.triangularity*gp.a) ) < 1e-10);
        CHECK( fabs(Z_X + 1.1*gp.elongation*gp.a ) < 1e-10);
    }

    SECTION( "Accuracy of psi")
    {
        const double R_H = gp.R_0-gp.triangularity*gp.a;
        const double Z_H = gp.elongation*gp.a;
        const double alpha_ = asin(gp.triangularity);
        const double N1 = -(1.+alpha_)/(gp.a*gp.elongation*gp.elongation)*(1.+alpha_);
        const double N2 =  (1.-alpha_)/(gp.a*gp.elongation*gp.elongation)*(1.-alpha_);
        const double N3 = -gp.elongation/(gp.a*cos(alpha_)*cos(alpha_));
        const double R_X = gp.R_0-1.1*gp.triangularity*gp.a;
        const double Z_X = -1.1*gp.elongation*gp.a;

        double result;
        result = mag.psip()(gp.R_0 + gp.a, 0.);
        INFO( "psip( 1+e,0)           "<<result);
        CHECK( fabs( result) < 1e-13);
        result = mag.psip()(gp.R_0 - gp.a, 0.);
        INFO( "psip( 1-e,0)           "<<result);
        CHECK( fabs( result) < 1e-13);
        result = mag.psip()(R_H, Z_H);
        INFO( "psip( 1-de,ke)         "<<result);
        CHECK( fabs( result) < 1e-13);
        result = mag.psip()(R_X, Z_X);
        INFO( "psip( 1-1.1de,-1.1ke)  "<<result);
        CHECK( fabs( result) < 1e-13);
        result = mag.psipZ()(gp.R_0 + gp.a, 0.);
        INFO( "psipZ( 1+e,0)          "<<result);
        CHECK( fabs( result) < 1e-13);
        result = mag.psipZ()(gp.R_0 - gp.a, 0.);
        INFO( "psipZ( 1-e,0)          "<<result);
        CHECK( fabs( result) < 1e-13);
        result = mag.psipR()(R_H,Z_H);
        INFO( "psipR( 1-de,ke)        "<<result);
        CHECK( fabs( result) < 1e-13);
        result = mag.psipR()(R_X,Z_X);
        INFO( "psipR( 1-1.1de,-1.1ke) "<<result);
        CHECK( fabs( result) < 1e-13);
        result = mag.psipZ()(R_X,Z_X);
        INFO( "psipZ( 1-1.1de,-1.1ke) "<<result);
        CHECK( fabs( result) < 1e-13);
        result = mag.psipZZ()(gp.R_0+gp.a,0.)+N1*mag.psipR()(gp.R_0+gp.a,0);
        INFO( "psipZZ( 1+e,0)         "<<result);
        CHECK( fabs( result) < 1e-13);
        result = mag.psipZZ()(gp.R_0-gp.a,0.)+N2*mag.psipR()(gp.R_0-gp.a,0);
        INFO( "psipZZ( 1-e,0)         "<<result);
        CHECK( fabs( result) < 1e-13);
        result = mag.psipRR()(R_H,Z_H)+N3*mag.psipZ()(R_H,Z_H);
        INFO( "psipRR( 1-de,ke)       "<<result);
        CHECK( fabs( result) < 1e-13);
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

    SECTION( "Push Forward curvatures")
    {
        double Rmin=gp.R_0-gp.a;
        double Zmin=-gp.a*gp.elongation;
        double Rmax=gp.R_0+gp.a;
        double Zmax=gp.a*gp.elongation;
        dg::CylindricalGrid3d grid3d(Rmin,Rmax,Zmin,Zmax, 0, 2.*M_PI, 3,100,100,32);
        dg::geo::CylindricalVectorLvl0 curvB_ = dg::geo::createTrueCurvatureNablaB( mag);
        dg::geo::CylindricalVectorLvl0 curvK_ = dg::geo::createTrueCurvatureKappa( mag);
        std::array<dg::HVec, 3> curvB, curvK;
        //Test NablaTimes B = B^2( curvK - curvB)
        dg::pushForward( curvB_.x(), curvB_.y(), curvB_.z(),
                curvB[0], curvB[1], curvB[2], grid3d);
        dg::pushForward( curvK_.x(), curvK_.y(), curvK_.z(),
                curvK[0], curvK[1], curvK[2], grid3d);
        //Test NablaTimes B = B^2( curvK - curvB)
        dg::blas1::axpby( 1., curvK, -1., curvB);
        dg::HVec Bmodule = dg::pullback( dg::geo::Bmodule(mag), grid3d);
        dg::blas1::pointwiseDot( Bmodule, Bmodule, Bmodule);
        for( int i=0; i<3; i++)
            dg::blas1::pointwiseDot( Bmodule, curvB[i], curvB[i]);
        dg::HVec R = dg::pullback( dg::cooX3d, grid3d);
        dg::HVec IR =  dg::pullback( mag.ipolR(), grid3d);
        dg::blas1::pointwiseDivide( gp.R_0, IR, R, 0., IR);
        dg::HVec IZ =  dg::pullback( mag.ipolZ(), grid3d);
        dg::blas1::pointwiseDivide( gp.R_0, IZ, R, 0., IZ);
        dg::HVec IP =  dg::pullback( JPhi( gp), grid3d);
        dg::blas1::pointwiseDivide( gp.R_0, IP, R, 0., IP);
        dg::blas1::axpby( 1., IZ, -1., curvB[0]);
        dg::blas1::axpby( 1., IR, +1., curvB[1]);
        dg::blas1::axpby( 1., IP, -1., curvB[2]);
        for( int i=0; i<3; i++)
        {
            double error = sqrt(dg::blas1::dot( curvB[i], curvB[i] ) );
            INFO( "Error in curv "<<i<<" :   "<<error);
            CHECK( error < 1e-13);
        }
    }
}

