#include <iostream>
#include <iomanip>

#include <mpi.h>

#include "file/read_input.h"

#include "dg/arakawa.h"
#include "dg/poisson.h"
#include "dg/geometry.h"
#include "dg/backend/mpi_init.h"
#include "dg/backend/timer.cuh"

#include "solovev.h"
#include "testfunctors.h"
#include "flux.h"
#include "simple_orthogonal.h"
#include "mpi_curvilinear.h"
#include "mpi_orthogonal.h"

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
        f_(c, R0, psi_0, psi_1, 4), g_(c, R0, psi_0, psi_1){}
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


typedef  dg::CurvilinearMPIGrid2d<dg::DVec> Geometry;

int main(int argc, char** argv)
{
    MPI_Init( &argc, &argv);
    int rank;
    unsigned n, Nx, Ny, Nz; 
    MPI_Comm comm;
    mpi_init3d( dg::DIR, dg::PER, dg::PER, n, Nx, Ny, Nz, comm);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
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
    if(rank==0)std::cout << "Psi min "<<psip(gp.R_0, 0)<<"\n";
    if(rank==0)std::cout << "Type psi_0 and psi_1\n";
    double psi_0, psi_1;
    if(rank==0)std::cin >> psi_0>> psi_1;
    MPI_Bcast( &psi_0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast( &psi_1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //if(rank==0)gp.display( std::cout);
    if(rank==0)std::cout << "Constructing grid ... \n";
    dg::Timer t;
    t.tic();
    //Geometry grid(gp, psi_0, psi_1, n, Nx, Ny,Nz, dg::DIR, comm);//3d
        MPI_Comm planeComm;
        int remain_dims[] = {true,true,false}; //true true false
        MPI_Cart_sub( comm, remain_dims, &planeComm);
    MagneticField c( gp);
    dg::geo::RibeiroFluxGenerator<Psip, PsipR, PsipZ, PsipRR, PsipRZ, PsipZZ>
        generator( c.psip, c.psipR, c.psipZ, c.psipRR, c.psipRZ, c.psipZZ, psi_0, psi_1, gp.R_0, 0., 1);
    //dg::geo::SimpleOrthogonal<Psip, PsipR, PsipZ, LaplacePsip> 
        //generator( c.psip, c.psipR, c.psipZ, c.laplacePsip, psi_0, psi_1, gp.R_0, 0., 1);
    Geometry grid(generator, n, Nx, Ny,dg::DIR, planeComm); //2d
    t.toc();
    if(rank==0)std::cout << "Construction took "<<t.diff()<<"s\n";

    dg::MDVec vol = dg::create::volume( grid);
    if(rank==0)std::cout <<std::fixed<< std::setprecision(2)<<std::endl;

    dg::geo::FuncDirPer<MagneticField> left(c, gp.R_0, psi_0, psi_1, 4);
    FuncDirPer2<MagneticField> right( c, gp.R_0, psi_0, psi_1);
    ArakawaDirPer<MagneticField> jacobian( c, gp.R_0, psi_0, psi_1);
    VariationDirPer<MagneticField> variationLHS(c, gp.R_0, psi_0, psi_1);

    const dg::MDVec lhs = dg::pullback( left, grid);
    dg::MDVec jac(lhs);
    const dg::MDVec rhs = dg::pullback( right, grid);
    const dg::MDVec sol = dg::pullback ( jacobian, grid);
    const dg::MDVec variation = dg::pullback ( variationLHS, grid);
    dg::MDVec eins = dg::evaluate( dg::one, grid);

    ///////////////////////////////////////////////////////////////////////
    if(rank==0)std::cout << "TESTING ARAKAWA 3D\n";
    dg::ArakawaX<Geometry, dg::MDMatrix, dg::MDVec> arakawa( grid);
    arakawa( lhs, rhs, jac);
    double norm = dg::blas2::dot( vol, jac);
    if(rank==0)std::cout << std::scientific;
    double result = dg::blas2::dot( eins, vol, jac);
    if(rank==0)std::cout << "Mean     Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( rhs, vol, jac);
    if(rank==0)std::cout << "Mean rhs*Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( lhs, vol, jac);
    if(rank==0)std::cout << "Mean lhs*Jacobian is "<<result<<"\n";
    dg::blas1::axpby( 1., sol, -1., jac);
    result = dg::blas2::dot( jac, vol, jac);
    if(rank==0)std::cout << "          Rel. distance to solution "<<sqrt( result/norm)<<std::endl; //don't forget sqrt when comuting errors
    arakawa.variation( lhs, jac);
    norm = dg::blas2::dot( vol, jac);
    dg::blas1::axpby( 1., variation, -1., jac);
    result = dg::blas2::dot( jac, vol, jac);
    if(rank==0)std::cout << "Variation rel. distance to solution "<<sqrt( result/norm)<<std::endl; //don't forget sqrt when comuting errors
    ///////////////////////////////////////////////////////////////////////
    if(rank==0)std::cout << "TESTING POISSON 3D\n";
    dg::Poisson<Geometry, dg::MDMatrix, dg::MDVec> poisson( grid);
    poisson( lhs, rhs, jac);
    norm = dg::blas2::dot( vol, jac);
    result = dg::blas2::dot( eins, vol, jac);
    if(rank==0)std::cout << "Mean     Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( rhs, vol, jac);
    if(rank==0)std::cout << "Mean rhs*Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( lhs, vol, jac);
    if(rank==0)std::cout << "Mean lhs*Jacobian is "<<result<<"\n";
    dg::blas1::axpby( 1., sol, -1., jac);
    result = dg::blas2::dot( jac, vol, jac);
    if(rank==0)std::cout << "          Rel. distance to solution "<<sqrt( result/norm)<<std::endl; //don't forget sqrt when comuting errors
    poisson.variationRHS( lhs, jac);
    norm = dg::blas2::dot( vol, jac);
    dg::blas1::axpby( 1., variation, -1., jac);
    result = dg::blas2::dot( jac, vol, jac);
    if(rank==0)std::cout << "Variation rel. distance to solution "<<sqrt( result/norm)<<std::endl; //don't forget sqrt when comuting errors

    ////////////////////////////transform curvature components////////
    if(rank==0)std::cout << "TESTING CURVATURE 3D\n";
    dg::MDVec curvX, curvY;
    dg::MHVec tempX, tempY;
    dg::geo::pushForwardPerp(dg::geo::CurvatureNablaBR<MagneticField>(c, gp.R_0), dg::geo::CurvatureNablaBZ<MagneticField>(c, gp.R_0), tempX, tempY, grid);
    dg::blas1::transfer(  tempX, curvX);
    dg::blas1::transfer(  tempY, curvY);
    dg::MDMatrix dx, dy;
    dg::blas2::transfer( dg::create::dx(grid), dx);
    dg::blas2::transfer( dg::create::dy(grid), dy);
    dg::MDVec tempx(curvX), tempy(curvX);
    dg::blas2::symv( dx, lhs, tempx);
    dg::blas2::symv( dy, lhs, tempy);
    dg::blas1::pointwiseDot( tempx, curvX, tempx);
    dg::blas1::pointwiseDot( 1., tempy, curvY, 1.,  tempx);
    norm = dg::blas2::dot( tempx, vol, tempx);

    CurvatureDirPer<MagneticField> curv(c, gp.R_0, psi_0, psi_1);
    dg::MDVec curvature;
    dg::blas1::transfer( dg::pullback(curv, grid), curvature);

    dg::blas1::axpby( 1., tempx, -1., curvature, tempx);
    result = dg::blas2::dot( vol, tempx);
    if(rank==0)std::cout << "Curvature rel. distance to solution "<<sqrt( result/norm)<<std::endl; //don't forget sqrt when comuting errors

    MPI_Finalize();


    return 0;
}
