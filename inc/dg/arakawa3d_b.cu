#include <iostream>
#include <iomanip>

#include "arakawa.h"
#include "blas.h"

#include "backend/timer.h"

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
double left( double x, double y) {return sin(x)*sin(y);}
double right( double x, double y) {return sin(2*x)*sin(2*y);}
//double right2( double x, double y) {return sin(y);}
double jacobian( double x, double y)
{
    return cos(x)*sin(y)*2*sin(2*x)*cos(2*y)-sin(x)*cos(y)*2*cos(2*x)*sin(2*y);
}

const double lz = 1.;
double left( double x, double y, double z) {return left(x,y)*z;}
double right( double x, double y, double z) {return right(x,y)*z;}
//double right2( double x, double y) {return sin(y);}
double jacobian( double x, double y, double z)
{
    return jacobian(x,y)*z*z;
}

int main()
{

    std::cout << std::fixed<<"\nTEST 3D VERSION!!\n";
    unsigned n, Nx, Ny, Nz;
    std::cout << "Type n, Nx, Ny and Nz! \n";
    std::cin >> n >> Nx >> Ny >> Nz;
    dg::Grid3d grid( 0, lx, 0, ly,0,lz, n, Nx, Ny,Nz, dg::PER, dg::PER);
    dg::DVec w3d = dg::create::weights( grid);
    std::cout << "# of 2d cells                     " << Nx*Ny <<std::endl;
    std::cout << "# of z  planes                    " << Nz <<std::endl;
    std::cout << "# of Legendre nodes per dimension "<< n <<std::endl;
    dg::DVec lhs = dg::evaluate ( left, grid), jac(lhs);
    dg::DVec rhs = dg::evaluate ( right,grid);
    const dg::DVec sol = dg::evaluate( jacobian, grid );
    dg::DVec eins( grid.size(), 1.);
    dg::Timer t;

    dg::ArakawaX<dg::CartesianGrid3d, dg::DMatrix, dg::DVec> arakawa( grid);
    t.tic();
    for( unsigned i=0; i<20; i++)
        arakawa( lhs, rhs, jac);
    t.toc();
    std::cout << "\nArakawa took "<<t.diff()/0.02<<"ms\n";
    std::cout <<   "which is     "<<t.diff()/0.02/Nz<<"ms per z plane \n\n";

    std::cout << std::scientific;
    std::cout << "Mean     Jacobian is "<<dg::blas2::dot( eins, w3d, jac)<<"\n";
    std::cout << "Mean rhs*Jacobian is "<<dg::blas2::dot( rhs, w3d, jac)<<"\n";
    std::cout << "Mean   n*Jacobian is "<<dg::blas2::dot( lhs, w3d, jac)<<"\n";
    dg::blas1::axpby( 1., sol, -1., jac);
    std::cout << "Distance to solution "<<sqrt(dg::blas2::dot( w3d, jac))<<std::endl;
    return 0;
}

