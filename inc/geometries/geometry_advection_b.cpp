#include <iostream>
#include <iomanip>

#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "dg/algorithm.h"
#include "dg/file/json_utilities.h"

#include "curvilinear.h"
#ifdef WITH_MPI
#include "mpi_curvilinear.h"
#endif

#include "flux.h"
#include "simple_orthogonal.h"

#include "solovev.h"
#include "magnetic_field.h"
#include "testfunctors.h"

struct FuncDirPer2
{
    FuncDirPer2( dg::geo::TokamakMagneticField c, double psi_0, double psi_1):
        R_0_(c.R0()), psi0_(psi_0), psi1_(psi_1), psip_(c.psip()), psipR_(c.psipR()), psipZ_(c.psipZ()){}
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
    dg::geo::CylindricalFunctor psip_, psipR_,  psipZ_;
};

struct ArakawaDirPer
{
    ArakawaDirPer( dg::geo::TokamakMagneticField c, double psi_0, double psi_1):
        f_(c, psi_0, psi_1, 4), g_(c, psi_0, psi_1){ }
    double operator()(double R, double Z, double phi) const {
        return this->operator()(R,Z);
    }
    double operator()(double R, double Z) const {
        return f_.dR( R,Z)*g_.dZ(R,Z) - f_.dZ(R,Z)*g_.dR(R,Z);
    }
    private:
    dg::geo::FuncDirPer f_;
    FuncDirPer2 g_;
};

struct CurvatureDirPer
{
    CurvatureDirPer( dg::geo::TokamakMagneticField c, double psi_0, double psi_1): f_(c, psi_0, psi_1,4.), curvR(c,+1), curvZ(c,+1){}
    double operator()(double R, double Z, double phi) const {
        return this->operator()(R,Z);}
    double operator()(double R, double Z) const {
        return curvR( R,Z)*f_.dR(R,Z) + curvZ(R,Z)*f_.dZ(R,Z);
    }
    private:
    dg::geo::FuncDirPer f_;
    dg::geo::CurvatureNablaBR curvR;
    dg::geo::CurvatureNablaBZ curvZ;
};



int main(int argc, char** argv)
{
#ifdef WITH_MPI
    MPI_Init( &argc, &argv);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif
    unsigned n, Nx, Ny;
    double psi_0, psi_1;
#ifdef WITH_MPI
    MPI_Comm comm;
    dg::mpi_init2d( dg::DIR, dg::PER, n, Nx, Ny, comm);
    if(rank==0)std::cout << "Type psi_0 (-20) and psi_1 (-4)\n";
    if(rank==0)std::cin >> psi_0>> psi_1;
    MPI_Bcast( &psi_0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast( &psi_1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else
    std::cout << "Type n (5), Nx (8), Ny (80)\n";
    std::cin >> n>> Nx>>Ny;
    std::cout << "Type psi_0 (-20) and psi_1 (-4)\n";
    std::cin >> psi_0>> psi_1;
#endif
    auto js = dg::file::file2Json( argc == 1 ? "geometry_params_Xpoint.json" : argv[1]);
    dg::geo::solovev::Parameters gp(js);
    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField( gp);

    DG_RANK0 std::cout << "Psi min "<<mag.psip()(gp.R_0, 0)<<"\n";
    DG_RANK0 std::cout << "Psi_0 = "<<psi_0<<" psi_1 = "<<psi_1<<std::endl;
    //gp.display( std::cout);
    dg::Timer t;
    DG_RANK0 std::cout << "Constructing grid ... \n";
    t.tic();
    //dg::geo::RibeiroFluxGenerator generator( mag.get_psip(), psi_0, psi_1, gp.R_0, 0., 1);
    dg::geo::FluxGenerator generator( mag.get_psip(), mag.get_ipol(), psi_0, psi_1, gp.R_0, 0., 1);
    //dg::geo::SimpleOrthogonal generator( mag.get_psip(), psi_0, psi_1, gp.R_0, 0., 1);
    dg::geo::x::CurvilinearGrid2d grid(generator, n, Nx, Ny, dg::DIR, dg::PER
#ifdef WITH_MPI
    , comm
#endif
    ); //2d
    t.toc();
    DG_RANK0 std::cout << "Construction took "<<t.diff()<<"s"<<std::endl;
    DG_RANK0 grid.display();

    const dg::x::DVec vol = dg::create::volume( grid);
    DG_RANK0 std::cout <<std::fixed<< std::setprecision(6)<<std::endl;


    const dg::x::DVec lhs = dg::pullback( dg::geo::FuncDirPer(mag, psi_0, psi_1, 4), grid);
    dg::x::DVec jac(lhs);
    const dg::x::DVec rhs = dg::pullback( FuncDirPer2( mag, psi_0, psi_1), grid);
    const dg::x::DVec sol = dg::pullback ( ArakawaDirPer( mag, psi_0, psi_1), grid);
    dg::x::DVec eins = dg::evaluate( dg::one, grid);

    ///////////////////////////////////////////////////////////////////////
    DG_RANK0 std::cout << "TESTING ARAKAWA\n";
    dg::ArakawaX<dg::x::aGeometry2d, dg::x::DMatrix, dg::x::DVec> arakawa( grid);
    arakawa( lhs, rhs, jac);
    const double norm = dg::blas2::dot( sol, vol, sol);
    DG_RANK0 std::cout << std::scientific;
    double result = dg::blas2::dot( eins, vol, jac);
    DG_RANK0 std::cout << "Mean     Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( rhs, vol, jac);
    DG_RANK0 std::cout << "Mean rhs*Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( lhs, vol, jac);
    DG_RANK0 std::cout << "Mean lhs*Jacobian is "<<result<<"\n";
    dg::blas1::axpby( 1., sol, -1., jac);
    result = dg::blas2::dot( jac, vol, jac);
    DG_RANK0 std::cout << "          Rel. distance to solution "<<sqrt( result/norm)<<std::endl; //don't forget sqrt when comuting errors
    ///////////////////////////////////////////////////////////////////////
    DG_RANK0 std::cout << "TESTING POISSON\n";
    dg::Poisson<dg::x::aGeometry2d, dg::x::DMatrix, dg::x::DVec> poisson( grid);
    poisson( lhs, rhs, jac);
    result = dg::blas2::dot( eins, vol, jac);
    DG_RANK0 std::cout << "Mean     Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( rhs, vol, jac);
    DG_RANK0 std::cout << "Mean rhs*Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( lhs, vol, jac);
    DG_RANK0 std::cout << "Mean lhs*Jacobian is "<<result<<"\n";
    result = dg::blas2::dot( jac, vol, jac);
    dg::blas1::axpby( 1., sol, -1., jac);
    result = dg::blas2::dot( jac, vol, jac);
    DG_RANK0 std::cout << "          Rel. distance to solution "<<sqrt( result/norm)<<std::endl; //don't forget sqrt when comuting errors

    ////////////////////////////transform curvature components////////
    DG_RANK0 std::cout << "TESTING CURVATURE 3D\n";
    dg::x::DVec curvX, curvY;
    dg::x::HVec tempX, tempY;
    dg::pushForwardPerp(dg::geo::CurvatureNablaBR(mag,+1), dg::geo::CurvatureNablaBZ(mag,+1), tempX, tempY, grid);
    dg::assign(  tempX, curvX);
    dg::assign(  tempY, curvY);
    dg::x::DMatrix dx, dy;
    dg::blas2::transfer( dg::create::dx(grid), dx);
    dg::blas2::transfer( dg::create::dy(grid), dy);
    dg::x::DVec tempx(curvX), tempy(curvX);
    dg::blas2::symv( dx, lhs, tempx);
    dg::blas2::symv( dy, lhs, tempy);
    dg::blas1::pointwiseDot( tempx, curvX, tempx);
    dg::blas1::pointwiseDot( 1., tempy, curvY, 1.,  tempx);
    const double normCurv = dg::blas2::dot( tempx, vol, tempx);

    CurvatureDirPer curv(mag, psi_0, psi_1);
    dg::x::DVec curvature;
    dg::assign( dg::pullback(curv, grid), curvature);

    dg::blas1::axpby( 1., tempx, -1., curvature, tempx);
    result = dg::blas2::dot( vol, tempx);
    DG_RANK0 std::cout << "Curvature rel. distance to solution "<<sqrt( result/normCurv)<<std::endl; //don't forget sqrt when comuting errors

#ifdef WITH_MPI
    MPI_Finalize();
#endif
    return 0;
}
