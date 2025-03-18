#include <iostream>
#include <iomanip>

#include "dg/algorithm.h"

#include "solovev.h"
//#include "taylor.h"
#include "magnetic_field.h"

struct JPhi
{
    JPhi( dg::geo::solovev::Parameters gp): R_0(gp.R_0), A(gp.A){}
    double operator()(double R, double Z, double phi)const
    {
        return ((A-1.)*R - A*R_0*R_0/R)/R_0/R_0/R_0;
    }
    private:
    double R_0, A;
};

int main( int argc, char* argv[])
{
    std::string input = argc==1 ? "geometry_params_Xpoint.json" : argv[1];
    dg::file::WrappedJsonValue js = dg::file::file2Json( input);

    std::string e = js.get( "equilibrium", "solovev" ).asString();
    dg::geo::equilibrium equi = dg::geo::str2equilibrium.at( e);
    if( equi != dg::geo::equilibrium::solovev)
    {
        std::cerr << "ERROR: This test only works for solovev equilibrium\n";
        return -1;
    }
    dg::geo::solovev::Parameters gp(js);
    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField(gp);

    double R_X = gp.R_0-1.1*gp.triangularity*gp.a;
    double Z_X = -1.1*gp.elongation*gp.a;
    if( gp.hasXpoint())
    {
        dg::geo::findXpoint( mag.get_psip(), R_X, Z_X);
        std::cout <<  "X-point found at "<<R_X << " "<<Z_X<<" with Psip "<<mag.psip()(R_X, Z_X)<<"\n";
        std::cout <<  "     R - Factor "<<(gp.R_0-R_X)/gp.triangularity/gp.a << "   Z - factor "<<-(Z_X/gp.elongation/gp.a)<<std::endl;

    }

    std::cout << "test accuracy of psi (values must be close to 0!)\n";
    const double R_H = gp.R_0-gp.triangularity*gp.a;
    const double Z_H = gp.elongation*gp.a;
    const double alpha_ = asin(gp.triangularity);
    const double N1 = -(1.+alpha_)/(gp.a*gp.elongation*gp.elongation)*(1.+alpha_);
    const double N2 =  (1.-alpha_)/(gp.a*gp.elongation*gp.elongation)*(1.-alpha_);
    const double N3 = -gp.elongation/(gp.a*cos(alpha_)*cos(alpha_));

    if( gp.hasXpoint())
        std::cout << "    Equilibrium with X-point!\n";
    else
        std::cout << "    NO X-point in flux function!\n";
    std::cout << "psip( 1+e,0)           "<<mag.psip()(gp.R_0 + gp.a, 0.)<<"\n";
    std::cout << "psip( 1-e,0)           "<<mag.psip()(gp.R_0 - gp.a, 0.)<<"\n";
    std::cout << "psip( 1-de,ke)         "<<mag.psip()(R_H, Z_H)<<"\n";
    if( !gp.hasXpoint())
        std::cout << "psipR( 1-de,ke)        "<<mag.psipR()(R_H, Z_H)<<"\n";
    else
    {
        std::cout << "psip( 1-1.1de,-1.1ke)  "<<mag.psip()(R_X, Z_X)<<"\n";
        std::cout << "psipZ( 1+e,0)          "<<mag.psipZ()(gp.R_0 + gp.a, 0.)<<"\n";
        std::cout << "psipZ( 1-e,0)          "<<mag.psipZ()(gp.R_0 - gp.a, 0.)<<"\n";
        std::cout << "psipR( 1-de,ke)        "<<mag.psipR()(R_H,Z_H)<<"\n";
        std::cout << "psipR( 1-1.1de,-1.1ke) "<<mag.psipR()(R_X,Z_X)<<"\n";
        std::cout << "psipZ( 1-1.1de,-1.1ke) "<<mag.psipZ()(R_X,Z_X)<<"\n";
    }
    std::cout << "psipZZ( 1+e,0)         "<<mag.psipZZ()(gp.R_0+gp.a,0.)+N1*mag.psipR()(gp.R_0+gp.a,0)<<"\n";
    std::cout << "psipZZ( 1-e,0)         "<<mag.psipZZ()(gp.R_0-gp.a,0.)+N2*mag.psipR()(gp.R_0-gp.a,0)<<"\n";
    std::cout << "psipRR( 1-de,ke)       "<<mag.psipRR()(R_H,Z_H)+N3*mag.psipZ()(R_H,Z_H)<<"\n";

    std::cout << "Test accuracy of curvatures (values must be close to 0)\n";
    dg::geo::CylindricalVectorLvl0 bhat_ = dg::geo::createBHat( mag);
    dg::geo::CylindricalVectorLvl0 curvB_ = dg::geo::createTrueCurvatureNablaB( mag);
    dg::geo::CylindricalVectorLvl0 curvK_ = dg::geo::createTrueCurvatureKappa( mag);
    //Test NablaTimes B = B^2( curvK - curvB)
    std::array<dg::HVec, 3> bhat, curvB, curvK;
    double Rmin=gp.R_0-gp.a;
    double Zmin=-gp.a*gp.elongation;
    double Rmax=gp.R_0+gp.a;
    double Zmax=gp.a*gp.elongation;
    dg::CylindricalGrid3d grid3d(Rmin,Rmax,Zmin,Zmax, 0, 2.*M_PI, 3,100,100,32);
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
    std::cout << "Error in norm b == 1 :  "<<error<<std::endl;

    std::cout << "Push Forward curvatures ...\n";
    dg::pushForward( curvB_.x(), curvB_.y(), curvB_.z(),
            curvB[0], curvB[1], curvB[2], grid3d);
    dg::pushForward( curvK_.x(), curvK_.y(), curvK_.z(),
            curvK[0], curvK[1], curvK[2], grid3d);
    std::cout << "Done!\n";
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
        error = sqrt(dg::blas1::dot( curvB[i], curvB[i] ) );
        std::cout << "Error in curv "<<i<<" :   "<<error<<"\n";
    }
}

