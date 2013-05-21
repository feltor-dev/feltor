#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cusp/print.h>

#include "polarisation.cuh"
#include "evaluation.cuh"
#include "cg.cuh"
#include "laplace.cuh"
#include "preconditioner.cuh"
#include "functions.h"

const unsigned n = 3; //global relative error in L2 norm is O(h^P)
const unsigned Nx = 80;  //more N means less iterations for same error
const unsigned Ny = 80;  //more N means less iterations for same error

const double lx = M_PI;
const double ly = M_PI;
const double hx = lx/(double)Nx;
const double hy = ly/(double)Ny;
const double eps = 1e-3; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;
typedef dg::ArrVec2d< double, n, HVec>  HArrVec;
typedef dg::ArrVec2d< double, n, DVec>  DArrVec;

typedef dg::T2D<double, n> Preconditioner;

typedef cusp::ell_matrix<int, double, cusp::host_memory> HMatrix;
typedef cusp::ell_matrix<int, double, cusp::device_memory> DMatrix;
typedef cusp::device_memory Memory;

double initial( double x, double y) {return sin(0);}
//double pol( double x, double y) {return 1. + sin(x); } //must be strictly positive
double pol( double x, double y) {return 1.; }

//double rhs( double x) { return sin(x) + 1.-2.*cos(x)*cos(x);}
double rhs( double x, double y) { return 2.*sin( x)*sin(y);}
double sol(double x, double y)  { return sin( x)*sin(y);}

using namespace std;

int main()
{
    //create functions A(chi) x = b
    HArrVec x = dg::expand<double (&)(double, double), n> ( initial, 0,lx, 0, ly, Nx, Ny);
    HArrVec b = dg::expand<double (&)(double, double), n> ( rhs, 0,lx, 0,ly, Nx, Ny);
    HArrVec chi = dg::expand<double (&)(double, double), n> ( pol, 0,lx,0, ly, Nx, Ny);
    const HArrVec solution = dg::expand<double (&)(double, double), n> (sol, 0, lx, 0 ,ly, Nx, Ny);
    HArrVec error(solution);

    //copy data to device memory
    DArrVec dx( x.data(), Nx), db( b.data(), Nx), derror( error.data(), Nx), dchi( chi.data(), Nx);
    const DArrVec dsolution( solution.data(), Nx);
    cusp::array1d_view<DVec::iterator> dchi_view( dchi.data().begin(), dchi.data().end());

    cout << "Create Polarisation object!\n";
    dg::Polarisation2d<double, n, Memory> pol( Nx, Ny, hx, hy, 0, 0);
    cout << "Create Polarisation matrix!\n";
    DMatrix A = pol.create( dchi_view ); 
    //DMatrix B = dg::create::laplace1d_dir<double, n>( N, h); 
    cout << "Create conjugate gradient!\n";
    dg::CG<DMatrix, DVec, Preconditioner > pcg( dx.data(), n*n*Nx*Ny);

    cout << "# of polynomial coefficients: "<< n <<endl;
    cout << "# of 2d cells                 "<< Nx*Ny <<endl;
    //compute S b
    dg::blas2::symv( dg::S2D<double, n>(hx, hy), db.data(), db.data());
    cudaThreadSynchronize();
    std::cout << "Number of pcg iterations "<< pcg( A, dx.data(), db.data(), Preconditioner(hx, hy), eps)<<endl;
    cout << "For a precision of "<< eps<<endl;
    //compute error
    dg::blas1::axpby( 1.,dx.data(),-1.,derror.data());

    double eps = dg::blas2::dot( dg::S2D<double, n>(hx, hy), derror.data());
    cout << "L2 Norm2 of Error is " << eps << endl;
    double norm = dg::blas2::dot( dg::S2D<double, n>(hx, hy), dsolution.data());
    std::cout << "L2 Norm of relative error is "<<sqrt( eps/norm)<<std::endl;
    //Fehler der Integration des Sinus ist vernachlässigbar (vgl. evaluation_t)

    return 0;
}

