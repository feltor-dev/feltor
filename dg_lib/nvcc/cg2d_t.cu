#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "evaluation.cuh"
#include "cg.cuh"
#include "arrvec2d.cuh"
#include "laplace.cuh"
#include "tensor.cuh"
#include "preconditioner.cuh"

const unsigned n = 3; //global relative error in L2 norm is O(h^P)

const unsigned Nx = 20;  //more N means less iterations for same error
const unsigned Ny = 20;  //more N means less iterations for same error
const double lx = 2.*M_PI;
const double ly = 2.*M_PI;

const double eps = 1e-9; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;
typedef dg::ArrVec2d< double, n, HVec>  HArrVec;
typedef dg::ArrVec2d< double, n, DVec>  DArrVec;

typedef dg::T2D<double, n> Preconditioner;
typedef dg::S2D<double, n> Postconditioner;

typedef cusp::ell_matrix<int, double, cusp::host_memory> HMatrix;
typedef cusp::ell_matrix<int, double, cusp::device_memory> DMatrix;

double fct(double x, double y){ return sin(y)*sin(x);}
double laplace_fct( double x, double y) { return 2*sin(y)*sin(x);}
double initial( double x, double y) {return sin(0);}
using namespace std;
int main()
{
    const double hx = lx/(double)Nx;
    const double hy = ly/(double)Ny;
    cout<<"Expand initial condition\n";
    HArrVec x = dg::expand<double (&)(double, double), n> ( initial, 0,lx, 0, ly, Nx, Ny);

    cout << "Create Laplacian\n";
    DMatrix A = dg::dgtensor<double, n>( 
                               dg::create::laplace1d_per<double, n>( Ny, hy), 
                               dg::S1D<double, n>( hx),
                               dg::S1D<double, n>( hy),
                               dg::create::laplace1d_per<double, n>( Nx, hx)); 
    dg::CG<DMatrix, DVec, Preconditioner > pcg( x.data(), n*n*Nx*Ny);
    //dg::CG<DMatrix, DVec> cg( x.data(), n*N);
    cout<<"Expand right hand side\n";
    HArrVec b = dg::expand<double (&)(double, double), n> ( laplace_fct, 0,lx, 0,ly, Nx, Ny);
    const HArrVec solution = dg::expand<double (&)(double, double), n> ( fct, 0,lx, 0,ly, Nx, Ny);

    //copy data to device memory
    const DArrVec dsolution( solution);
    DArrVec db( b), dx( x);
    //////////////////////////////////////////////////////////////////////
    cout << "# of polynomial coefficients: "<< n <<endl;
    cout << "# of 2d cells                 "<< Nx*Ny <<endl;
    //compute S b
    dg::blas2::symv( Postconditioner(hx, hy), db.data(), db.data());
    cudaThreadSynchronize();
    cout << "Number of pcg iterations "<< pcg( A, dx.data(), db.data(), Preconditioner(hx, hy), eps)<<endl;
    cudaThreadSynchronize();
    //std::cout << "Number of cg iterations "<< cg( A, dx.data(), db.data(), dg::Identity<double>(), eps)<<endl;
    cout << "For a precision of "<< eps<<endl;
    //compute error
    DArrVec derror( dsolution);
    dg::blas1::axpby( 1.,dx.data(),-1.,derror.data());

    DArrVec dAx(dx), res( db);
    dg::blas2::symv(  A, dx.data(), dAx.data());
    dg::blas1::axpby( 1.,dAx.data(),-1.,res.data());
    cudaThreadSynchronize();

    double xnorm = dg::blas2::dot( Postconditioner(hx, hy), dx.data());
    cout << "L2 Norm2 of x0 is              " << xnorm << endl;
    double eps = dg::blas2::dot( Postconditioner(hx, hy), derror.data());
    cout << "L2 Norm2 of Error is           " << eps << endl;
    double norm = dg::blas2::dot( Postconditioner(hx, hy), dsolution.data());
    cout << "L2 Norm2 of Solution is        " << norm << endl;
    double normres = dg::blas2::dot( Postconditioner(hx, hy), res.data());
    cout << "L2 Norm2 of Residuum is        " << normres << endl;
    cout << "L2 Norm of relative error is   " <<sqrt( eps/norm)<<endl;
    //Fehler der Integration des Sinus ist vernachlässigbar (vgl. evaluation_t)



    return 0;
}
