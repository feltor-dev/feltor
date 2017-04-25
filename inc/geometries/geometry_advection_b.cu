#include <iostream>
#include <iomanip>

#include "dg/backend/timer.cuh"
#include "dg/arakawa.h"
#include "dg/poisson.h"
#include "dg/geometry.h"

#include "conformal.h"
#include "orthogonal.h"
#include "curvilinear.h"

#include "flux.h"
#include "simple_orthogonal.h"

#include "solovev.h"
#include "magnetic_field.h"
#include "testfunctors.h"

using namespace dg::geo::solovev;

template< class MagneticField>
struct FuncDirPer2
{
    FuncDirPer2( MagneticField c, double R0, double psi_0, double psi_1):
        R_0_(R0), psi0_(psi_0), psi1_(psi_1), psip_(c.psip), psipR_(c.psipR), psipZ_(c.psipZ){}
    double operator()(double R, double Z, double phi) const {
        return this->operator()(R,Z);
    }
    double operator()(double R, double Z) const {
        double psip = psip_(R,Z);
        return (psip-psi0_)*(psip-psi1_)*cos(theta(R,Z));
    }
    double dR( double R, double Z)const
    {
        double psip = psip_(R,Z), psipR = psipR_(R,Z), theta_ = theta(R,Z);
        return (2.*psip*psipR - (psi0_+psi1_)*psipR)*cos(theta_) 
            - (psip-psi0_)*(psip-psi1_)*sin(theta_)*thetaR(R,Z);
    }
    double dZ( double R, double Z)const
    {
        double psip = psip_(R,Z), psipZ = psipZ_(R,Z), theta_=theta(R,Z);
        return (2*psip*psipZ - (psi0_+psi1_)*psipZ)*cos(theta_) 
            - (psip-psi0_)*(psip-psi1_)*sin(theta_)*thetaZ(R,Z);
    }
    private:
    double theta( double R, double Z) const {
        double dR = R-R_0_;
        if( Z >= 0)
            return acos( dR/sqrt( dR*dR + Z*Z));
        else
            return 2.*M_PI-acos( dR/sqrt( dR*dR + Z*Z));
    }
    double thetaR( double R, double Z) const {
        double dR = R-R_0_;
        return -Z/(dR*dR+Z*Z);
    }
    double thetaZ( double R, double Z) const {
        double dR = R-R_0_;
        return dR/(dR*dR+Z*Z);
    }
    double R_0_;
    double psi0_, psi1_;
    Psip psip_;
    PsipR psipR_;
    PsipZ psipZ_;
};

template<class MagneticField>
struct ArakawaDirPer
{
    ArakawaDirPer( MagneticField c, double R0, double psi_0, double psi_1): 
        f_(c, R0, psi_0, psi_1, 4), g_(c, R0, psi_0, psi_1){ }
    double operator()(double R, double Z, double phi) const {
        return this->operator()(R,Z);
    }
    double operator()(double R, double Z) const {
        return f_.dR( R,Z)*g_.dZ(R,Z) - f_.dZ(R,Z)*g_.dR(R,Z);
    }
    private:
    dg::geo::FuncDirPer<MagneticField> f_;
    FuncDirPer2<MagneticField> g_;
};

template<class MagneticField>
struct VariationDirPer
{
    VariationDirPer( MagneticField c, double R0, double psi_0, double psi_1): f_(c, R0, psi_0, psi_1,4. ){}
    double operator()(double R, double Z, double phi) const {
        return this->operator()(R,Z);}

    double operator()(double R, double Z) const {
        return f_.dR( R,Z)*f_.dR(R,Z) + f_.dZ(R,Z)*f_.dZ(R,Z);
    }
    private:
    dg::geo::FuncDirPer<MagneticField> f_;
};

template< class MagneticField>
struct CurvatureDirPer
{
    CurvatureDirPer( MagneticField c, double R0, double psi_0, double psi_1): f_(c, R0, psi_0, psi_1,4.), curvR(c, R0), curvZ(c, R0){}
    double operator()(double R, double Z, double phi) const {
        return this->operator()(R,Z);}
    double operator()(double R, double Z) const {
        return curvR( R,Z)*f_.dR(R,Z) + curvZ(R,Z)*f_.dZ(R,Z);
    }
    private:
    dg::geo::FuncDirPer<MagneticField> f_;
    dg::geo::CurvatureNablaBR<MagneticField> curvR;
    dg::geo::CurvatureNablaBZ<MagneticField> curvZ;
};


typedef dg::CurvilinearGrid2d<dg::DVec> Geometry;

int main(int argc, char** argv)
{
    std::cout << "Type n, Nx, Ny, Nz\n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;   
    Json::Reader reader;
    Json::Value js;
    if( argc==1)
    {
        std::ifstream is("geometry_params_Xpoint.js");
        reader.parse(is,js,false);
    }
    else
    {
        std::ifstream is(argv[1]);
        reader.parse(is,js,false);
    }
    GeomParameters gp(js);
    Psip psip( gp); 
    std::cout << "Psi min "<<psip(gp.R_0, 0)<<"\n";
    std::cout << "Type psi_0 (-20) and psi_1 (-4)\n";
    double psi_0, psi_1;
    std::cin >> psi_0>> psi_1;
    std::cout << "Psi_0 = "<<psi_0<<" psi_1 = "<<psi_1<<std::endl;
    //gp.display( std::cout);
    dg::Timer t;
    //solovev::detail::Fpsi fpsi( gp, -10);
    std::cout << "Constructing grid ... \n";
    t.tic();
    MagneticField c( gp);
    dg::geo::RibeiroFluxGenerator<Psip, PsipR, PsipZ, PsipRR, PsipRZ, PsipZZ>
        ribeiro( c.psip, c.psipR, c.psipZ, c.psipRR, c.psipRZ, c.psipZZ, psi_0, psi_1, gp.R_0, 0., 1);
    //dg::geo::SimpleOrthogonal<Psip, PsipR, PsipZ, LaplacePsip>
    //    ribeiro( c.psip, c.psipR, c.psipZ, c.laplacePsip, psi_0, psi_1, gp.R_0, 0., 1);
    Geometry grid(ribeiro, n, Nx, Ny, dg::DIR); //2d
    t.toc();
    std::cout << "Construction took "<<t.diff()<<"s"<<std::endl;
    grid.display();

    const dg::DVec vol = dg::create::volume( grid);
    std::cout <<std::fixed<< std::setprecision(6)<<std::endl;


    dg::geo::FuncDirPer<MagneticField> left(c, gp.R_0, psi_0, psi_1, 4);
    FuncDirPer2<MagneticField> right( c, gp.R_0, psi_0, psi_1);
    ArakawaDirPer<MagneticField> jacobian( c, gp.R_0, psi_0, psi_1);
    VariationDirPer<MagneticField> variationLHS(c, gp.R_0, psi_0, psi_1);

    const dg::DVec lhs = dg::pullback( left, grid);
    dg::DVec jac(lhs);
    const dg::DVec rhs = dg::pullback( right, grid);
    const dg::DVec sol = dg::pullback ( jacobian, grid);
    const dg::DVec variation = dg::pullback ( variationLHS, grid);
    dg::DVec eins = dg::evaluate( dg::one, grid);

    ///////////////////////////////////////////////////////////////////////
    std::cout << "TESTING ARAKAWA\n";
    dg::ArakawaX<Geometry, dg::DMatrix, dg::DVec> arakawa( grid);
    arakawa( lhs, rhs, jac);
    const double norm = dg::blas2::dot( sol, vol, sol);
    std::cout << std::scientific;
    double result = dg::blas2::dot( eins, vol, jac);
    std::cout << "Mean     Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( rhs, vol, jac);
    std::cout << "Mean rhs*Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( lhs, vol, jac);
    std::cout << "Mean lhs*Jacobian is "<<result<<"\n";
    //std::cout << "norm of solution "<<norm<<"\n";
    //std::cout << "norm of Jacobian "<<dg::blas2::dot( jac, vol, jac)<<"\n";
    //std::cout << "norm of lhs      "<<dg::blas2::dot( lhs, vol, lhs)<<"\n";
    //std::cout << "norm of rhs      "<<dg::blas2::dot( rhs, vol, rhs)<<"\n";
    dg::blas1::axpby( 1., sol, -1., jac);
    result = dg::blas2::dot( jac, vol, jac);
    std::cout << "          Rel. distance to solution "<<sqrt( result/norm)<<std::endl; //don't forget sqrt when comuting errors
    arakawa.variation( lhs, jac);
    const double normVar = dg::blas2::dot( vol, variation);
    //std::cout << "norm of variation "<<normVar<<"\n";
    dg::blas1::axpby( 1., variation, -1., jac);
    result = dg::blas2::dot( jac, vol, jac);
    std::cout << "Variation rel. distance to solution "<<sqrt( result/normVar)<<std::endl; //don't forget sqrt when comuting errors
    ///////////////////////////////////////////////////////////////////////
    std::cout << "TESTING POISSON\n";
    dg::Poisson<Geometry, dg::DMatrix, dg::DVec> poisson( grid);
    poisson( lhs, rhs, jac);
    result = dg::blas2::dot( eins, vol, jac);
    std::cout << "Mean     Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( rhs, vol, jac);
    std::cout << "Mean rhs*Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( lhs, vol, jac);
    std::cout << "Mean lhs*Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( jac, vol, jac);
    //std::cout << "norm of solution "<<norm<<"\n";
    //std::cout << "norm of Jacobian "<<result<<"\n";
    dg::blas1::axpby( 1., sol, -1., jac);
    result = dg::blas2::dot( jac, vol, jac);
    std::cout << "          Rel. distance to solution "<<sqrt( result/norm)<<std::endl; //don't forget sqrt when comuting errors
    poisson.variationRHS( lhs, jac);
    dg::blas1::axpby( 1., variation, -1., jac);
    result = dg::blas2::dot( jac, vol, jac);
    std::cout << "Variation rel. distance to solution "<<sqrt( result/normVar)<<std::endl; //don't forget sqrt when comuting errors

    ////////////////////////////transform curvature components////////
    std::cout << "TESTING CURVATURE 3D\n";
    dg::DVec curvX, curvY;
    dg::HVec tempX, tempY;
    dg::geo::pushForwardPerp(dg::geo::CurvatureNablaBR<MagneticField>(c, gp.R_0), dg::geo::CurvatureNablaBZ<MagneticField>(c, gp.R_0), tempX, tempY, grid);
    dg::blas1::transfer(  tempX, curvX);
    dg::blas1::transfer(  tempY, curvY);
    dg::DMatrix dx, dy;
    dg::blas2::transfer( dg::create::dx(grid), dx);
    dg::blas2::transfer( dg::create::dy(grid), dy);
    dg::DVec tempx(curvX), tempy(curvX);
    dg::blas2::symv( dx, lhs, tempx);
    dg::blas2::symv( dy, lhs, tempy);
    dg::blas1::pointwiseDot( tempx, curvX, tempx);
    dg::blas1::pointwiseDot( 1., tempy, curvY, 1.,  tempx);
    const double normCurv = dg::blas2::dot( tempx, vol, tempx);

    CurvatureDirPer<MagneticField> curv(c, gp.R_0, psi_0, psi_1);
    dg::DVec curvature;
    dg::blas1::transfer( dg::pullback(curv, grid), curvature);

    dg::blas1::axpby( 1., tempx, -1., curvature, tempx);
    result = dg::blas2::dot( vol, tempx);
    std::cout << "Curvature rel. distance to solution "<<sqrt( result/normCurv)<<std::endl; //don't forget sqrt when comuting errors



    return 0;
}
