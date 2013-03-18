#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "evaluation.cuh"
#include "cg.cuh"
#include "laplace.cuh"
#include "preconditioner.cuh"

const unsigned n = 3; //global relative error in L2 norm is O(h^P)
const unsigned N = 10;  //more N means less iterations for same error

const double lx = 2*M_PI;
const double h = lx/(double)N;
const double eps = 1e-4; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;
typedef dg::ArrVec1d< double, n, HVec>  HArrVec;
typedef dg::ArrVec1d< double, n, DVec>  DArrVec;

typedef dg::Laplace<n> Matrix;
typedef dg::T<n> Preconditioner;

double sine(double x){ return sin( x);}
double initial( double x) {return sin(0);}
using namespace std;
int main()
{
    HArrVec x = dg::expand<double (&)(double), n> ( initial, 0,lx, N);
    Matrix A( N, h); //A(h) does  not even warn me
    dg::PCG<Matrix, DVec, dg::T<n> > pcg( x.data(), n*N);
    dg::CG<Matrix, DVec> cg( x.data(), n*N);
    HArrVec b = dg::expand<double (&)(double), n> ( sine, 0,lx, N);
    HArrVec error(b);
    const HArrVec solution(b);

    //copy data to device memory
    DArrVec dx( x.data()), db( b.data()), derror( error.data());
    const DArrVec dsolution( solution.data());

    cout << "# of polynomial coefficients: "<< n <<endl;
    cout << "# of intervals                "<< N <<endl;
    //compute S b
    dg::BLAS2<dg::S<n>, DVec>::dsymv( dg::S<n>(h), db.data(), db.data());
    std::cout << "Number of pcg iterations "<< pcg( A, dx.data(), db.data(), dg::T<n>(h), eps)<<endl;
    cout << "For a precision of "<< eps<<endl;
    //compute error
    dg::BLAS1<DVec>::daxpby( 1.,dx.data(),-1.,derror.data());
    //and Ax
    DArrVec dbx(dx);
    dg::BLAS2<Matrix, DVec>::dsymv(  A, dx.data(), dbx.data());

    double eps = dg::BLAS2<dg::S<n>, DVec>::ddot( dg::S<n>(h), derror.data());
    cout << "L2 Norm2 of Error is " << eps << endl;
    double norm = dg::BLAS2<dg::S<n>, DVec>::ddot( dg::S<n>(h), dsolution.data());
    std::cout << "L2 Norm of relative error is "<<sqrt( eps/norm)<<std::endl;
    //Fehler der Integration des Sinus ist vernachlässigbar (vgl. evaluation_t)



    return 0;
}
