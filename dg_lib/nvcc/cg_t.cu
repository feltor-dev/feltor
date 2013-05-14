#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "evaluation.cuh"
#include "cg.cuh"
#include "laplace.cuh"
#include "preconditioner.cuh"

const unsigned n = 1; //global relative error in L2 norm is O(h^P)
const unsigned N = 200;  //more N means less iterations for same error

const double lx = 2.*M_PI;
const double h = lx/(double)N;
const double eps = 1e-7; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

typedef thrust::device_vector< double>   DVec;
//typedef thrust::host_vector< double>     DVec;
typedef thrust::host_vector< double>     HVec;
typedef dg::ArrVec1d< double, n, HVec>  HArrVec;
//typedef dg::ArrVec1d< double, n, HVec>  DArrVec;
typedef dg::ArrVec1d< double, n, DVec>  DArrVec;

typedef dg::T1D<double, n> Preconditioner;

typedef cusp::ell_matrix<int, double, cusp::host_memory> HMatrix;
//typedef cusp::ell_matrix<int, double, cusp::host_memory> DMatrix;
typedef cusp::ell_matrix<int, double, cusp::device_memory> DMatrix;

double sine(double x){ return sin( x);}
double initial( double x) {return sin(0);}
using namespace std;
int main()
{
    HArrVec x = dg::expand<double (&)(double), n> ( initial, 0,lx, N);
    DMatrix A = dg::create::laplace1d_per<double, n>( N, h); 
    dg::CG<DMatrix, DVec, Preconditioner > pcg( x.data(), n*N);
    dg::CG<DMatrix, DVec> cg( x.data(), n*N);
    HArrVec b = dg::expand<double (&)(double), n> ( sine, 0,lx, N);
    HArrVec error(b);
    const HArrVec solution(b);

    //copy data to device memory
    DArrVec dx( x.data()), db( b.data()), derror( error.data());
    const DArrVec dsolution( solution.data());

    cout << "# of polynomial coefficients: "<< n <<endl;
    cout << "# of intervals                "<< N <<endl;
    //compute S b
    dg::blas2::symv( dg::S1D<double, n>(h), db.data(), db.data());
    cudaThreadSynchronize();
    std::cout << "Number of pcg iterations "<< pcg( A, dx.data(), db.data(), Preconditioner(h), eps)<<endl;
    //std::cout << "Number of cg iterations "<< cg( A, dx.data(), db.data(), dg::Identity<double>(), eps)<<endl;
    cout << "For a precision of "<< eps<<endl;
    //compute error
    dg::blas1::axpby( 1.,dx.data(),-1.,derror.data());
    /*
    //and Ax
    DArrVec dbx(dx);
    dg::blas2::symv(  A, dx.data(), dbx.data());
    cudaThreadSynchronize();

    cout<< dx <<endl;
    */

    double eps = dg::blas2::dot( dg::S1D<double, n>(h), derror.data());
    cout << "L2 Norm2 of Error is " << eps << endl;
    double norm = dg::blas2::dot( dg::S1D<double, n>(h), dsolution.data());
    std::cout << "L2 Norm of relative error is "<<sqrt( eps/norm)<<std::endl;
    //Fehler der Integration des Sinus ist vernachlässigbar (vgl. evaluation_t)



    return 0;
}
