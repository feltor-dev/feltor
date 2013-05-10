#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "polarisation.cuh"
#include "evaluation.cuh"
#include "cg.cuh"
#include "laplace.cuh"
#include "preconditioner.cuh"
#include "functions.h"

const unsigned n = 3; //global relative error in L2 norm is O(h^P)
const unsigned N = 100;  //more N means less iterations for same error

const double lx = 2.*M_PI;
const double h = lx/(double)N;
const double eps = 1e-7; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;
typedef dg::ArrVec1d< double, n, HVec>  HArrVec;
typedef dg::ArrVec1d< double, n, DVec>  DArrVec;

typedef dg::T1D<double, n> Preconditioner;

typedef cusp::ell_matrix<int, double, cusp::host_memory> HMatrix;
typedef cusp::ell_matrix<int, double, cusp::device_memory> DMatrix;
typedef cusp::device_memory Memory;

double initial( double x) {return sin(0);}
//double pol( double x) {return sin(x); }
double pol( double x) {return 1.; }

//double rhs( double x) { return 1.-2.*cos(x)*cos(x);}
double rhs( double x) { return sin(x);}
double sol(double x){ return sin( x);}

using namespace std;

int main()
{

    //create functions A(chi) x = b
    HArrVec x = dg::expand<double (&)(double), n> ( initial, 0,lx, N);
    HArrVec b = dg::expand<double (&)(double), n> ( rhs, 0,lx, N);
    HArrVec chi = dg::expand<double (&)(double), n> ( pol, 0,lx, N);
    const HArrVec solution = dg::expand<double (&)(double), n> (sol, 0 ,lx, N);
    HArrVec error(solution);

    //copy data to device memory
    DArrVec dx( x.data()), db( b.data()), derror( error.data()), dchi( chi.data());
    const DArrVec dsolution( solution.data());
    cusp::array1d_view<DVec::iterator> dchi_view( dchi.data().begin(), dchi.data().end());

    cout << "Create Polarisation object!\n";
    dg::Polarisation<double, n, Memory> pol( N, h, 0);

    cout << "Create Polarisation matrix!\n";

    DMatrix A = pol.create( dchi_view ); 
    cout << "Create conjugate gradient!\n";
    dg::CG<DMatrix, DVec, Preconditioner > pcg( dx.data(), n*N);

    cout << "# of polynomial coefficients: "<< n <<endl;
    cout << "# of intervals                "<< N <<endl;
    //compute S b
    dg::blas2::symv( dg::S1D<double, n>(h), db.data(), db.data());
    cudaThreadSynchronize();
    std::cout << "Number of pcg iterations "<< pcg( A, dx.data(), db.data(), Preconditioner(h), eps)<<endl;
    cout << "For a precision of "<< eps<<endl;
    //compute error
    dg::blas1::axpby( 1.,dx.data(),-1.,derror.data());

    double eps = dg::blas2::dot( dg::S1D<double, n>(h), derror.data());
    cout << "L2 Norm2 of Error is " << eps << endl;
    double norm = dg::blas2::dot( dg::S1D<double, n>(h), dsolution.data());
    std::cout << "L2 Norm of relative error is "<<sqrt( eps/norm)<<std::endl;
    //Fehler der Integration des Sinus ist vernachlässigbar (vgl. evaluation_t)



    return 0;
}

