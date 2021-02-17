#include <iostream>
#include <iomanip>

#include "gradient.h"

const double lx = 2*M_PI;
const double ly = 2*M_PI;
dg::bc bcx = dg::PER;
dg::bc bcy = dg::PER;

double phi( double x, double y) {
    return sin(x)*cos(y);
}
double dxphi( double x, double y) {
    return cos(x)*cos(y);
}
double dyphi( double x, double y) {
    return -sin(x)*sin(y);
}
double phi3d( double x, double y,double z) {
    return sin(x)*cos(y)*cos(z);
}
double dxphi3d( double x, double y, double z) {
    return cos(x)*cos(y)*cos(z);
}
double dyphi3d( double x, double y, double z) {
    return -sin(x)*sin(y)*cos(z);
}
double dzphi3d( double x, double y, double z) {
    return -sin(x)*cos(y)*sin(z)/x/x;
}

// There are more tests in geometries/geometry_advection_(mpi)b.cu
int main()
{
    std::cout<<"This program tests the execution of the gradient! A test is passed if the number in the second column shows exactly zero!\n";
    unsigned n = 5, Nx = 32, Ny = 48;
    std::cout << "TEST 2D\n";
    std::cout << "Computing on the Grid " <<n<<" x "<<Nx<<" x "<<Ny <<std::endl;

    // create a Cartesian grid on the domain [0,lx]x[0,ly]
    const dg::CartesianGrid2d grid( 0, lx, 0, ly, n, Nx, Ny, bcx, bcy);

    // evaluate left and right hand side on the grid
    const dg::DVec ph = dg::construct<dg::DVec>( dg::evaluate( phi, grid));
    const dg::DVec phx = dg::construct<dg::DVec>( dg::evaluate( dxphi, grid));
    const dg::DVec phy = dg::construct<dg::DVec>( dg::evaluate( dyphi, grid));
    dg::DVec dxph(ph), dyph(ph);

    // create a Gradient object
    dg::Gradient<dg::aGeometry2d, dg::DMatrix, dg::DVec> gradient( grid);

    //apply arakawa scheme
    gradient.gradient( ph, dxph, dyph);

    int64_t binary[] = {4500635718861276907,4487444521638156650};
    dg::exblas::udouble res;
    dg::DVec w2d = dg::create::weights( grid);

    dg::blas1::axpby( 1., phx, -1., dxph);
    res.d = sqrt(dg::blas2::dot( w2d, dxph)); //don't forget sqrt when computing errors
    std::cout << "Gx Distance to solution "<<res.d<<"\t\t"<<res.i-binary[0]<<std::endl;
    dg::blas1::axpby( 1., phy, -1., dyph);
    res.d = sqrt(dg::blas2::dot( w2d, dyph)); //don't forget sqrt when computing errors
    std::cout << "Gy Distance to solution "<<res.d<<"\t\t"<<res.i-binary[1]<<std::endl;
    //periocid bc       |  dirichlet bc
    //n = 1 -> p = 2    |
    //n = 2 -> p = 1    |
    //n = 3 -> p = 3    |        3
    //n = 4 -> p = 3    |
    //n = 5 -> p = 5    |
    std::cout << "TEST 3D\n";
    unsigned Nz = 100;
    std::cout << "Computing on the Grid " <<n<<" x "<<Nx<<" x "<<Ny <<" x "<<Nz<<std::endl;

    // create a Cylindrical grid on the domain [0,lx]x[0,ly]
    const dg::CylindricalGrid3d grid3d( M_PI, 3*M_PI, -M_PI, M_PI, 0., 2*M_PI, n, Nx, Ny, Nz, bcx, bcy, dg::PER);

    // evaluate left and right hand side on the grid
    const dg::DVec ph3d = dg::construct<dg::DVec>( dg::evaluate( phi3d, grid3d));
    const dg::DVec phx3d = dg::construct<dg::DVec>( dg::evaluate( dxphi3d, grid3d));
    const dg::DVec phy3d = dg::construct<dg::DVec>( dg::evaluate( dyphi3d, grid3d));
    const dg::DVec phz3d = dg::construct<dg::DVec>( dg::evaluate( dzphi3d, grid3d));
    dg::DVec dxph3d(ph3d), dyph3d(ph3d), dzph3d(ph3d);

    // create a Gradient object
    dg::Gradient3d<dg::aGeometry3d, dg::DMatrix, dg::DVec> gradient3d( grid3d);

    //apply arakawa scheme
    gradient3d.gradient( ph3d, dxph3d, dyph3d, dzph3d);

    int64_t binary3d[] = {4504451755369532568,4491224193368827475,4549042274897523598};
    dg::exblas::udouble res3d;
    dg::DVec w3d = dg::create::weights( grid3d);

    dg::blas1::axpby( 1., phx3d, -1., dxph3d);
    res3d.d = sqrt(dg::blas2::dot( w3d, dxph3d)); //don't forget sqrt when computing errors
    std::cout << "Gx Distance to solution "<<res3d.d<<"\t\t"<<res3d.i-binary3d[0]<<std::endl;
    dg::blas1::axpby( 1., phy3d, -1., dyph3d);
    res3d.d = sqrt(dg::blas2::dot( w3d, dyph3d)); //don't forget sqrt when computing errors
    std::cout << "Gy Distance to solution "<<res3d.d<<"\t\t"<<res3d.i-binary3d[1]<<std::endl;
    dg::blas1::axpby( 1., phz3d, -1., dzph3d);
    res3d.d = sqrt(dg::blas2::dot( w3d, dzph3d)); //don't forget sqrt when computing errors
    std::cout << "Gz Distance to solution "<<res3d.d<<"\t\t"<<res3d.i-binary3d[2]<<std::endl;
    //periocid bc       |  dirichlet bc
    //n = 1 -> p = 2    |
    //n = 2 -> p = 1    |
    //n = 3 -> p = 3    |        3
    //n = 4 -> p = 3    |
    //n = 5 -> p = 5    |
    std::cout << "\nContinue with topology/average_t.cu !\n\n";
    return 0;
}
