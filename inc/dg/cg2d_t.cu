#include <iostream>
#include <iomanip>

#include <cusp/print.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "backend/evaluation.cuh"
#include "cg.h"
#include "backend/tensor.cuh"
#include "backend/derivatives.cuh"
#include "backend/typedefs.cuh"

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;

const double eps_ = 1e-6; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

double fct(double x, double y){ return sin(y)*sin(x);}
double laplace_fct( double x, double y) { return 2*sin(y)*sin(x);}
double initial( double x, double y) {return sin(0);}

using namespace std;

int main()
{
    //global relative error in L2 norm is O(h^P)
    //more N means less iterations for same error
    unsigned n, Nx, Ny;
    cout << "Type n, Nx and Ny! \n";
    cin >> n >> Nx >> Ny;
    cout << "Computing on the Grid " <<n<<" x "<<Nx<<" x "<<Ny <<endl;
    dg::Grid2d<double> grid( 0, lx, 0, ly,n, Nx, Ny, dg::PER, dg::PER);
    dg::HVec w2d = dg::create::weights( grid);
    dg::HVec v2d = dg::create::inv_weights( grid);
    cout<<"Evaluate initial condition\n";
    dg::HVec x = dg::evaluate( initial, grid);

    cout << "Create Laplacian\n";
    //Note that this function is deprecated (use the elliptic class instead)
    dg::HMatrix A = dg::create::laplacianM( grid, dg::not_normed, dg::forward); 
    //cusp::print(A);
    dg::CG<dg::HVec > pcg( x, n*n*Nx*Ny);
    cout<<"Evaluate right hand side\n";
    dg::HVec b = dg::evaluate ( laplace_fct, grid);
    const dg::HVec solution = dg::evaluate ( fct, grid);
    //////////////////////////////////////////////////////////////////////
    //compute S b
    dg::blas2::symv( w2d, b, b);
    cout << "Number of pcg iterations "<< pcg( A, x, b, v2d, eps_)<<endl;
    //std::cout << "Number of cg iterations "<< pcg( A, x, b, dg::Identity<double>(), eps)<<endl;
    cout << "For a precision of "<< eps_<<endl;
    //compute error
    dg::HVec error( solution);
    dg::blas1::axpby( 1.,x,-1.,error);

    dg::HVec Ax(x), res( b);
    dg::blas2::symv(  A, x, Ax);
    dg::blas1::axpby( 1.,Ax,-1.,res);

    double xnorm = sqrt(dg::blas2::dot( w2d, x));
    cout << "L2 Norm of x0 is              " << xnorm << endl;
    double norm = sqrt(dg::blas2::dot(w2d , solution));
    cout << "L2 Norm of Solution is        " << norm << endl;
    double eps = sqrt(dg::blas2::dot(w2d , error));
    cout << "L2 Norm of Error is           " << eps << endl;
    double normres = sqrt(dg::blas2::dot( w2d, res));
    cout << "L2 Norm of Residuum is        " << normres << endl;
    cout << "L2 Norm of relative error is  " << eps/norm<<endl;
    //Fehler der Integration des Sinus ist vernachlässigbar (vgl. evaluation_t)

    return 0;
}
