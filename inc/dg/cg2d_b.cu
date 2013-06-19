#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "timer.cuh"
#include "cusp_eigen.h"
#include "evaluation.cuh"
#include "cg.cuh"
#include "derivatives.cuh"
#include "preconditioner.cuh"

#include "typedefs.cuh"

const unsigned n = 3; //global relative error in L2 norm is O(h^P)

//leo3 can do 350 x 350 but not 375 x 375
const double lx = 2.*M_PI;
const double ly = 2.*M_PI;

const double eps = 1e-6; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

typedef dg::T2D<double, n> Preconditioner;
typedef dg::S2D<double, n> Postconditioner;

double fct(double x, double y){ return sin(y)*sin(x);}
double laplace_fct( double x, double y) { return 2*sin(y)*sin(x);}
double initial( double x, double y) {return sin(0);}

using namespace std;

int main()
{
    dg::Timer t;
    unsigned Nx, Ny; 
    std::cout << "Type Nx and Ny\n";
    std::cin >> Nx >> Ny;
    dg::Grid<double, n> grid( 0, lx, 0, ly, Nx, Ny, dg::DIR, dg::DIR);
    dg::S2D<double,n > s2d( grid);
    cout<<"Expand initial condition\n";
    dg::HVec x = dg::expand( initial, grid);

    cout << "Create Laplacian\n";
    t.tic();
    dg::DMatrix dA = dg::create::laplacianM( grid, dg::not_normed, dg::LSPACE); 
    dg::HMatrix A = dA;
    /*
    dg::dgtensor<double, n>( dg::create::laplace1d_dir<double, n>( Ny, hy), 
                               dg::S1D<double, n>( hx),
                               dg::S1D<double, n>( hy),
                               dg::create::laplace1d_per<double, n>( Nx, hx)); //dir does also work but is slow
                               */
    t.toc();
    cout<< "Creation took "<<t.diff()<<"s\n";

    //create conjugate gradient and one eigen Cholesky
    dg::CG< dg::DVec > pcg( x, n*n*Nx*Ny);
    dg::CG< dg::HVec > pcg_host( x, n*n*Nx*Ny);
    //dg::SimplicialCholesky sol;
    //sol.compute( A);

    cout<<"Expand right hand side\n";
    const dg::HVec solution = dg::expand ( fct, grid);
    dg::HVec b = dg::expand ( laplace_fct, grid);
    //compute S b
    dg::blas2::symv( s2d, b, b);
    cudaThreadSynchronize();

    //copy data to device memory
    t.tic();
    const dg::DVec dsolution( solution);
    dg::DVec db( b), dx( x);
    t.toc();
    cout << "Allocation and copy to device "<<t.diff()<<"s\n";
    //////////////////////////////////////////////////////////////////////
    cout << "# of polynomial coefficients: "<< n <<endl;
    cout << "# of 2d cells                 "<< Nx*Ny <<endl;
    
    t.tic();
    cout << "Number of pcg iterations "<< pcg( dA, dx, db, Preconditioner(grid), eps)<<endl;
    t.toc();
    cout << "... for a precision of "<< eps<<endl;
    cout << "... on the device took "<< t.diff()<<"s\n";
    t.tic();
    cout << "Number of pcg iterations "<< pcg_host( A, x, b, Preconditioner(grid.hx(), grid.hy()), eps)<<endl;
    t.toc();
    cout << "... for a precision of "<< eps<<endl;
    cout << "... on the host took   "<< t.diff()<<"s\n";
    //t.tic();
    //cout << "Success (1) "<< sol.solve( x.data().data(), b.data().data(), n*n*Nx*Ny)<<endl;
    //t.toc();
    //cout << "Cholesky took          "<< t.diff()<<"s\n";
    //compute error
    dg::DVec derror( dsolution);
    dg::HVec  error(  solution);
    dg::blas1::axpby( 1.,dx,-1.,derror);
    dg::blas1::axpby( 1., x,-1., error);

    double normerr = dg::blas2::dot( s2d, derror);
    cout << "L2 Norm2 of CG Error is        " << normerr << endl;
    double normerr2= dg::blas2::dot( s2d,  error);
    cout << "L2 Norm2 of Cholesky Error is  " << normerr2 << endl;
    double norm = dg::blas2::dot( s2d, dsolution);
    cout << "L2 Norm of relative error is   " <<sqrt( normerr/norm)<<endl;
    cout << "L2 Norm of relative error is   " <<sqrt( normerr2/norm)<<endl;

    return 0;
}
