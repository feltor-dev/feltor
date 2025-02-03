#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cusp/print.h>

#include "backend/timer.h"
#include "topology/evaluation.h"
#include "topology/split_and_join.h"

#include "pcg.h"
#include "elliptic.h"


const double R_0 = 10;
const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
const double lz = 2.*M_PI;
double fct(double x, double y, double z){ return sin(x-R_0)*sin(y)*sin(z);}
double fctX( double x, double y, double z){return cos(x-R_0)*sin(y)*sin(z);}
double fctY(double x, double y, double z){ return sin(x-R_0)*cos(y)*sin(z);}
double fctZ(double x, double y, double z){ return sin(x-R_0)*sin(y)*cos(z);}
double laplace2d_fct( double x, double y, double z) { return -1./x*cos(x-R_0)*sin(y)*sin(z) + 2.*fct(x,y,z);}
double laplace3d_fct( double x, double y, double z) { return -1./x*cos(x-R_0)*sin(y)*sin(z) + 2.*fct(x,y,z) + 1./x/x*fct(x,y,z);}
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::DIR;
dg::bc bcz = dg::PER;
double initial( double x, double y, double z) {return sin(0);}
double variation3d( double x, double y, double z) {
    return (fctX(x,y,z)*fctX(x,y,z)
        + fctY(x,y,z)*fctY(x,y,z)
        + fctZ(x,y,z)*fctZ(x,y,z)/x/x)*fct(x,y,z)*fct(x,y,z);
}


int main()
{
    dg::Timer t;
    unsigned n, Nx, Ny, Nz;
    std::cout << "Type n, Nx, Ny and Nz\n";
    std::cin >> n >> Nx >> Ny >> Nz;
    double eps;
    std::cout << "Type epsilon! \n";
    std::cin >> eps;
    bool jump_weight;
    std::cout << "Jump weighting on or off? Type 1 for true or 0 for false (default): \n";
    std::cin >> jump_weight;
    std::cout << "TEST CYLINDRICAL LAPLACIAN\n";
    //std::cout << "Create Laplacian\n";
    //! [invert]
    dg::CylindricalGrid3d grid( R_0, R_0+lx, 0, ly, 0,lz, n, Nx, Ny,Nz, bcx, bcy, bcz);
    dg::DVec w3d = dg::create::volume( grid);
    dg::DVec x = dg::evaluate( initial, grid);

    dg::Elliptic3d<dg::aGeometry3d, dg::DMatrix, dg::DVec> laplace(grid, dg::centered);

    laplace.set_jump_weighting(jump_weight);

    dg::PCG< dg::DVec > pcg( x, n*n*Nx*Ny*Nz);

    const dg::DVec solution = dg::evaluate ( fct, grid);
    dg::DVec b = dg::evaluate ( laplace3d_fct, grid);

    std::cout << "For a precision of "<< eps<<" ..."<<std::endl;
    unsigned num;
    t.tic();
    num = pcg.solve( laplace, x, b, 1., w3d, eps);
    t.toc();
    std::cout << "Number of pcg iterations "<<num<<std::endl;
    std::cout << "... on the device took   "<< t.diff()<<"s\n";
    //! [invert]
    dg::DVec  error(  solution);
    dg::blas1::axpby( 1., x,-1., error);

    double normerr = dg::blas2::dot( w3d, error);
    double norm = dg::blas2::dot( w3d, solution);
    dg::exblas::udouble res;
    norm = sqrt(normerr/norm); res.d = norm;
    std::cout << "L2 Norm of relative error is:               " <<res.d<<"\t"<<res.i<<std::endl;
    const dg::DVec deriv = dg::evaluate( fctX, grid);
    dg::DMatrix DX = dg::create::dx( grid);
    dg::blas2::gemv( DX, x, error);
    dg::blas1::axpby( 1., deriv, -1., error);
    normerr = dg::blas2::dot( w3d, error);
    norm = dg::blas2::dot( w3d, deriv);
    std::cout << "L2 Norm of relative error in derivative is: " <<sqrt( normerr/norm)<<std::endl;
    std::cout << "Compute variation in Elliptic               ";
    const dg::DVec variatio = dg::evaluate ( variation3d, grid);
    laplace.variation( solution, x, error);
    dg::blas1::axpby( 1., variatio, -1., error);
    norm = dg::blas2::dot( w3d, variatio);
    normerr = dg::blas2::dot( w3d, error);
    std::cout <<sqrt( normerr/norm) << "\n";


    std::cout << "TEST SPLIT SOLUTION\n";
    x = dg::evaluate( initial, grid);
    b = dg::evaluate ( laplace2d_fct, grid);
    //create grid and perp and parallel volume
    dg::ClonePtr<dg::aGeometry2d> grid_perp = grid.perp_grid();
    dg::DVec w2d = dg::create::volume( *grid_perp);
    dg::DVec g_parallel = grid.metric().value(2,2);
    dg::blas1::transform( g_parallel, g_parallel, dg::SQRT<>());
    dg::DVec chi = dg::evaluate( dg::one, grid);
    dg::blas1::pointwiseDivide( chi, g_parallel, chi);
    //create split Laplacian
    std::vector< dg::Elliptic<dg::aGeometry2d, dg::DMatrix, dg::DVec> > laplace_split(
            grid.Nz(), {*grid_perp, dg::centered});
    // create split  vectors and solve
    std::vector<dg::View<dg::DVec>> b_split, x_split, chi_split;
    pcg.construct( w2d, w2d.size());
    std::vector<unsigned>  number(grid.Nz());
    t.tic();
    dg::blas1::pointwiseDivide( b, g_parallel, b);
    b_split = dg::split( b, grid);
    chi_split = dg::split( chi, grid);
    x_split = dg::split( x, grid);
    for( unsigned i=0; i<grid.Nz(); i++)
    {
        laplace_split[i].set_chi( chi_split[i]);
        number[i] = pcg.solve( laplace_split[i], x_split[i], b_split[i], 1., w2d, eps);
    }
    t.toc();
    std::cout << "Number of iterations in split     "<< number[0]<<"\n";
    std::cout << "Split solution on the device took "<< t.diff()<<"s\n";
    dg::blas1::axpby( 1., x,-1., solution, error);
    normerr = dg::blas2::dot( w3d, error);
    norm = dg::blas2::dot( w3d, solution);
    std::cout << "L2 Norm of relative error is:     " <<sqrt( normerr/norm)<<std::endl;

    //both function and derivative converge with order P

    return 0;
}
