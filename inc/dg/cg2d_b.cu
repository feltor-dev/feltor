#include <iostream>
#include <iomanip>

#include "cg.h"
#include "elliptic.h"

#include "backend/timer.cuh"

const double lx = M_PI;
const double ly = 2.*M_PI;

double fct(double x, double y){ return sin(y)*sin(x+M_PI/2.);}
double derivative( double x, double y){return cos(x+M_PI/2.)*sin(y);}
double laplace_fct( double x, double y) { return 2*sin(y)*sin(x+M_PI/2.);}
dg::bc bcx = dg::NEU;
//double fct(double x, double y){ return sin(x);}
//double derivative( double x, double y){return cos(x);}
//double laplace_fct( double x, double y) { return sin(x);}
//const double lx = 2./3.*M_PI;
//double fct(double x, double y){ return sin(y)*sin(3.*x/4.);}
//double laplace_fct( double x, double y) { return 25./16.*sin(y)*sin(3.*x/4.);}
//dg::bc bcx = dg::DIR_NEU;
double initial( double x, double y) {return sin(0);}


int main()
{
    dg::Timer t;
    unsigned n, Nx, Ny; 
    std::cout << "Type n, Nx and Ny\n";
    std::cin >> n >> Nx >> Ny;
    std::cout << "Type in eps\n";
    double eps = 1e-6; //# of pcg iterations increases very much if 
    std::cin >> eps;

    dg::Grid2d grid( 0., lx, 0, ly, n, Nx, Ny, bcx, dg::PER);
    const dg::DVec w2d = dg::create::weights( grid);
    const dg::DVec v2d = dg::create::inv_weights( grid);
    std::cout<<"Evaluate initial condition...\n";
    dg::DVec x = dg::evaluate( initial, grid);

    std::cout << "Create Laplacian...\n";
    t.tic();
    dg::DMatrix DX = dg::create::dx( grid);
    dg::Elliptic<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> lap(grid, dg::not_normed, dg::forward );
    dg::Elliptic<dg::CartesianGrid2d, dg::fDMatrix, dg::fDVec> flap(grid, dg::not_normed, dg::forward );
    t.toc();
    std::cout<< "Creation took "<<t.diff()<<"s\n";

    dg::CG< dg::DVec > pcg( x, n*n*Nx*Ny);

    std::cout<<"Expand right hand side\n";
    const dg::DVec solution = dg::evaluate ( fct, grid);
    const dg::DVec deriv = dg::evaluate( derivative, grid);
    dg::DVec b = dg::evaluate ( laplace_fct, grid);
    //compute S b
    dg::blas1::pointwiseDivide( b, lap.inv_weights(), b);
    //////////////////////////////////////////////////////////////////////
    std::cout << "Computing on the Grid " <<n<<" x "<<Nx<<" x "<<Ny <<std::endl;

    std::cout << "... for a precision of "<< eps<<std::endl;
    x = dg::evaluate( initial, grid);
    t.tic();
    std::cout << "Number of pcg iterations "<< pcg( lap, x, b, v2d, eps)<<std::endl;
    t.toc();
    std::cout << "... on the device took "<< t.diff()<<"s\n";

    dg::DVec error( solution);
    dg::blas1::axpby( 1., x,-1., error);

    double normerr = dg::blas2::dot( w2d, error);
    double norm = dg::blas2::dot( w2d, solution);
    std::cout << "L2 Norm of relative error is: " <<sqrt( normerr/norm)<<std::endl;
    dg::blas2::gemv( DX, x, error);
    dg::blas1::axpby( 1., deriv, -1., error);
    normerr = dg::blas2::dot( w2d, error); 
    norm = dg::blas2::dot( w2d, deriv);
    std::cout << "L2 Norm of relative error in derivative is: " <<sqrt( normerr/norm)<<std::endl;
    //both functiona and derivative converge with order P 

    return 0;
}
