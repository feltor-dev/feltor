#include <iostream>
#include <iomanip>

#include "file/read_input.h"

#include "dg/backend/timer.cuh"
#include "dg/arakawa.h"
#include "dg/poisson.h"
#include "dg/geometry.h"

#include "solovev.h"
#include "fields.h"
#include "conformal.h"
#include "orthogonal.h"

struct FuncDirPer2
{
    FuncDirPer2( solovev::GeomParameters gp, double psi_0, double psi_1):
        R_0_(gp.R_0), psi0_(psi_0), psi1_(psi_1), psip_(gp), psipR_(gp), psipZ_(gp){}
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
    solovev::Psip psip_;
    solovev::PsipR psipR_;
    solovev::PsipZ psipZ_;
};

struct ArakawaDirPer
{
    ArakawaDirPer( solovev::GeomParameters gp, double psi_0, double psi_1): f_(gp, psi_0, psi_1), g_(gp, psi_0, psi_1){}
    double operator()(double R, double Z, double phi) const {
        return this->operator()(R,Z);
    }
    double operator()(double R, double Z) const {
        return f_.dR( R,Z)*g_.dZ(R,Z) - f_.dZ(R,Z)*g_.dR(R,Z);
    }
    private:
    solovev::FuncDirPer f_;
    FuncDirPer2 g_;
};
struct VariationDirPer
{
    VariationDirPer( solovev::GeomParameters gp, double psi_0, double psi_1): f_(gp, psi_0, psi_1){}
    double operator()(double R, double Z, double phi) const {
        return this->operator()(R,Z);}

    double operator()(double R, double Z) const {
        return f_.dR( R,Z)*f_.dR(R,Z) + f_.dZ(R,Z)*f_.dZ(R,Z);
    }
    private:
    solovev::FuncDirPer f_;
};

struct CurvatureDirPer
{
    CurvatureDirPer( solovev::GeomParameters gp, double psi_0, double psi_1): f_(gp, psi_0, psi_1), curvR(gp), curvZ(gp){}
    double operator()(double R, double Z, double phi) const {
        return this->operator()(R,Z);}
    double operator()(double R, double Z) const {
        return curvR( R,Z)*f_.dR(R,Z) + curvZ(R,Z)*f_.dZ(R,Z);
    }
    private:
    solovev::FuncDirPer f_;
    solovev::CurvatureR curvR;
    solovev::CurvatureZ curvZ;
};


//typedef ConformalGrid3d<dg::DVec> Geometry;
//typedef OrthogonalGrid3d<dg::DVec> Geometry;
typedef ConformalGrid2d<dg::DVec> Geometry;
//typedef OrthogonalGrid2d<dg::DVec> Geometry;

int main(int argc, char** argv)
{
    std::cout << "Type n, Nx, Ny, Nz\n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;   
    std::vector<double> v, v2;
    try{ 
        if( argc==1)
        {
            v = file::read_input( "geometry_params_Xpoint.txt"); 
        }
        else
        {
            v = file::read_input( argv[1]); 
        }
    }
    catch (toefl::Message& m) {  
        m.display(); 
        for( unsigned i = 0; i<v.size(); i++)
            std::cout << v[i] << " ";
            std::cout << std::endl;
        return -1;}
    //write parameters from file into variables
    solovev::GeomParameters gp(v);
    solovev::Psip psip( gp); 
    std::cout << "Psi min "<<psip(gp.R_0, 0)<<"\n";
    std::cout << "Type psi_0 and psi_1\n";
    double psi_0, psi_1;
    std::cin >> psi_0>> psi_1;
    //gp.display( std::cout);
    dg::Timer t;
    //solovev::detail::Fpsi fpsi( gp, -10);
    std::cout << "Constructing conformal grid ... \n";
    t.tic();
    //Geometry grid(gp, psi_0, psi_1, n, Nx, Ny,Nz, dg::DIR);//3d
    Geometry grid(gp, psi_0, psi_1, n, Nx, Ny, dg::DIR); //2d
    t.toc();
    std::cout << "Construction took "<<t.diff()<<"s"<<std::endl;

    dg::DVec vol = dg::create::volume( grid);
    std::cout <<std::fixed<< std::setprecision(2)<<std::endl;

    solovev::FuncDirPer left( gp, psi_0, psi_1);
    FuncDirPer2 right( gp, psi_0, psi_1);
    ArakawaDirPer jacobian( gp, psi_0, psi_1);
    VariationDirPer variationLHS( gp, psi_0, psi_1);

    const dg::DVec lhs = dg::pullback( left, grid);
    dg::DVec jac(lhs);
    const dg::DVec rhs = dg::pullback( right, grid);
    const dg::DVec sol = dg::pullback ( jacobian, grid);
    const dg::DVec variation = dg::pullback ( variationLHS, grid);
    dg::DVec eins = dg::evaluate( dg::one, grid);

    ///////////////////////////////////////////////////////////////////////
    std::cout << "TESTING ARAKAWA 3D\n";
    dg::ArakawaX<Geometry, dg::DMatrix, dg::DVec> arakawa( grid);
    arakawa( lhs, rhs, jac);
    double norm = dg::blas2::dot( vol, jac);
    std::cout << std::scientific;
    double result = dg::blas2::dot( eins, vol, jac);
    std::cout << "Mean     Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( rhs, vol, jac);
    std::cout << "Mean rhs*Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( lhs, vol, jac);
    std::cout << "Mean lhs*Jacobian is "<<result<<"\n";
    dg::blas1::axpby( 1., sol, -1., jac);
    result = dg::blas2::dot( jac, vol, jac);
    std::cout << "          Rel. distance to solution "<<sqrt( result/norm)<<std::endl; //don't forget sqrt when comuting errors
    arakawa.variation( lhs, jac);
    norm = dg::blas2::dot( vol, jac);
    dg::blas1::axpby( 1., variation, -1., jac);
    result = dg::blas2::dot( jac, vol, jac);
    std::cout << "Variation rel. distance to solution "<<sqrt( dg::blas2::dot( vol, jac)/norm)<<std::endl; //don't forget sqrt when comuting errors
    ///////////////////////////////////////////////////////////////////////
    std::cout << "TESTING POISSON 3D\n";
    dg::Poisson<Geometry, dg::DMatrix, dg::DVec> poisson( grid);
    poisson( lhs, rhs, jac);
    norm = dg::blas2::dot( vol, jac);
    result = dg::blas2::dot( eins, vol, jac);
    std::cout << "Mean     Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( rhs, vol, jac);
    std::cout << "Mean rhs*Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( lhs, vol, jac);
    std::cout << "Mean lhs*Jacobian is "<<result<<"\n";
    dg::blas1::axpby( 1., sol, -1., jac);
    result = dg::blas2::dot( jac, vol, jac);
    std::cout << "          Rel. distance to solution "<<sqrt( result/norm)<<std::endl; //don't forget sqrt when comuting errors
    poisson.variationRHS( lhs, jac);
    norm = dg::blas2::dot( vol, jac);
    dg::blas1::axpby( 1., variation, -1., jac);
    result = dg::blas2::dot( jac, vol, jac);
    std::cout << "Variation rel. distance to solution "<<sqrt( dg::blas2::dot( vol, jac)/norm)<<std::endl; //don't forget sqrt when comuting errors

    ////////////////////////////transform curvature components////////
    std::cout << "TESTING CURVATURE 3D\n";
    dg::DVec curvX, curvY;
    dg::HVec tempX, tempY;
    dg::geo::pushForwardPerp(solovev::CurvatureR(gp), solovev::CurvatureZ(gp), tempX, tempY, grid);
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
    norm = dg::blas2::dot( tempx, vol, tempx);

    CurvatureDirPer curv(gp, psi_0, psi_1);
    dg::DVec curvature;
    dg::blas1::transfer( dg::pullback(curv, grid), curvature);

    dg::blas1::axpby( 1., tempx, -1., curvature, tempx);
    result = dg::blas2::dot( vol, tempx);
    std::cout << "Curvature rel. distance to solution "<<sqrt( result/norm)<<std::endl; //don't forget sqrt when comuting errors



    return 0;
}
