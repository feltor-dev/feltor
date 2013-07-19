#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "evaluation.cuh"
#include "cg.cuh"
#include "tensor.cuh"
#include "derivatives.cuh"
#include "preconditioner.cuh"
#include "typedefs.cuh"

const unsigned n = 5; //global relative error in L2 norm is O(h^P)

const unsigned Nx = 40;  //more N means less iterations for same error
const unsigned Ny = 40;  //more N means less iterations for same error
const double lx = 2.*M_PI;
const double ly = 2.*M_PI;

const double eps = 1e-9; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus


typedef dg::T2D<double, n> Preconditioner;
typedef dg::S2D<double, n> Postconditioner;


double fct(double x, double y){ return sin(y)*sin(x);}
double laplace_fct( double x, double y) { return 2*sin(y)*sin(x);}
double initial( double x, double y) {return sin(0);}
using namespace std;
int main()
{
    dg::Grid<double, n> grid( 0, lx, 0, ly, Nx, Ny, dg::PER, dg::PER);
    dg::S2D<double,n > s2d( grid.hx(), grid.hy());
    cout<<"Expand initial condition\n";
    dg::HVec x = dg::expand( initial, grid);

    cout << "Create Laplacian\n";
    dg::DMatrix A = dg::create::laplacianM( grid, dg::not_normed, dg::LSPACE); 

    dg::CG<dg::DVec > pcg( x, n*n*Nx*Ny);
    //dg::CG<DMatrix, DVec> cg( x.data(), n*N);
    cout<<"Expand right hand side\n";
    dg::HVec b = dg::expand ( laplace_fct, grid);
    const dg::HVec solution = dg::expand ( fct, grid);

    //copy data to device memory
    const dg::DVec dsolution( solution);
    dg::DVec db( b), dx( x);
    //////////////////////////////////////////////////////////////////////
    cout << "# of polynomial coefficients: "<< n <<endl;
    cout << "# of 2d cells                 "<< Nx*Ny <<endl;
    //compute S b
    dg::blas2::symv( s2d, db, db);
    cudaThreadSynchronize();
    cout << "Number of pcg iterations "<< pcg( A, dx, db, Preconditioner(grid.hx(), grid.hy()), eps)<<endl;
    cudaThreadSynchronize();
    //std::cout << "Number of cg iterations "<< cg( A, dx.data(), db.data(), dg::Identity<double>(), eps)<<endl;
    cout << "For a precision of "<< eps<<endl;
    //compute error
    dg::DVec derror( dsolution);
    dg::blas1::axpby( 1.,dx,-1.,derror);

    dg::DVec dAx(dx), res( db);
    dg::blas2::symv(  A, dx, dAx);
    dg::blas1::axpby( 1.,dAx,-1.,res);
    cudaThreadSynchronize();

    double xnorm = dg::blas2::dot( s2d, dx);
    cout << "L2 Norm2 of x0 is              " << xnorm << endl;
    double eps = dg::blas2::dot(s2d , derror);
    cout << "L2 Norm2 of Error is           " << eps << endl;
    double norm = dg::blas2::dot(s2d , dsolution);
    cout << "L2 Norm2 of Solution is        " << norm << endl;
    double normres = dg::blas2::dot( s2d, res);
    cout << "L2 Norm2 of Residuum is        " << normres << endl;
    cout << "L2 Norm of relative error is   " <<sqrt( eps/norm)<<endl;
    //Fehler der Integration des Sinus ist vernachlässigbar (vgl. evaluation_t)

    return 0;
}
