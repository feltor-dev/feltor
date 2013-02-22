#include <iostream>

#include "cg.h"
#include "evaluation.h"
#include "laplace.h"

#define P 5
const unsigned N = 1000;
const double lx = 2*M_PI;

typedef std::vector<std::array<double,P>> ArrVec;
double sinus(double x){ return sin( x);}
double secondsinus( double x) {return sin(x);}
using namespace std;
int main()
{
    std::array<double,P> arr{{0}};
    ArrVec x(N, arr);
    dg::PCG<dg::Laplace<P>, ArrVec, dg::T> pcg( x, x.size(), 1e-10);
    auto b = dg::expand<double (&)(double), P> ( sinus, 0,lx, N);
    auto error(b);
    const auto solution(b);
    const double h = lx/(double)N;
    dg::Laplace<P> A( h);

    //compute S b
    dg::BLAS2<dg::S, ArrVec>::dsymv( 1., dg::S(h), b, 0, b);
    std::cout << "Number of cg iterations "<< pcg( A, x, b, dg::T(h))<<endl;
    //cout << "b is \n" << b<<endl;
    //cout << "x is \n" << x<<endl;
    //compute error
    dg::BLAS1<ArrVec>::daxpby( 1.,x,-1.,error);
    //and Ax
    auto bx(x);
    dg::BLAS2<dg::Laplace<P>, ArrVec>::dsymv( 1., A, x, 0, bx);
    //cout << "Ax is \n" << bx <<endl;

    double eps = dg::BLAS2<dg::S, ArrVec>::ddot( error, dg::S(h), error);
    cout << "L2 Norm2 of Error is " << eps << endl;
    double norm = dg::BLAS2<dg::S, ArrVec>::ddot( solution, dg::S(h), solution);
    std::cout << "L2 Norm of relative error is "<<sqrt( eps/norm)<<std::endl;



    return 0;
}
