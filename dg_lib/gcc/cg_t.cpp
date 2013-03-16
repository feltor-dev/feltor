#include <iostream>

#include "evaluation.h"

#include "cg.h"
#include "laplace.h"
#include "preconditioner.h"

#define P 5//global relative error in L2 norm is O(h^P)
const unsigned N = 10;  //more N means less iterations for same error
const double lx = 2*M_PI;
const double h = lx/(double)N;
const double eps = 1e-4; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

typedef std::vector<dg::Array<double,P>> ArrVec;
typedef dg::Laplace_Dir<P> Matrix;
double sinus(double x){ return sin( x);}
double initial( double x) {return sin(0);}
using namespace std;
int main()
{
    auto x = dg::expand<double (&)(double), P> ( initial, 0,lx, N);
    Matrix A( h);
    dg::PCG<Matrix, ArrVec, dg::T> pcg( x, P*x.size());
    dg::CG<Matrix, ArrVec> cg( x, P*x.size());
    auto b = dg::expand<double (&)(double), P> ( sinus, 0,lx, N);
    auto error(b);
    const auto solution(b);

    cout << "Polynomial order (P-1): "<< P-1 <<endl;
    cout << "Vector size: P*N "<< P*N <<endl;
    //compute S b
    dg::BLAS2<dg::S, ArrVec>::dsymv( dg::S(h), b, b);
    std::cout << "Number of pcg iterations "<< pcg( A, x, b, dg::T(h), eps)<<endl;
    cout << "For a precision of "<< eps<<endl;
    //compute error
    dg::BLAS1<ArrVec>::daxpby( 1.,x,-1.,error);
    //and Ax
    auto bx(x);
    dg::BLAS2<Matrix, ArrVec>::dsymv(  A, x, bx);

    double eps = dg::BLAS2<dg::S, ArrVec>::ddot( dg::S(h), error);
    cout << "L2 Norm2 of Error is " << eps << endl;
    double norm = dg::BLAS2<dg::S, ArrVec>::ddot( dg::S(h), solution);
    std::cout << "L2 Norm of relative error is "<<sqrt( eps/norm)<<std::endl;
    //Fehler der Integration des Sinus ist vernachlässigbar (vgl. evaluation_t)



    return 0;
}
