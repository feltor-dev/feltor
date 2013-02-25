#include <iostream>

#include "cg.h"
#include "evaluation.h"
#include "laplace.h"

#define P 2
const unsigned N = 100;
const double lx = 2*M_PI;
const double h = lx/(double)N;

typedef std::vector<std::array<double,P>> ArrVec;
typedef dg::Laplace_Dir<P> Matrix;
double sinus(double x){ return sin( x);}
double secondsinus( double x) {return sin(x);}
using namespace std;
int main()
{
    std::array<double,P> arr{{0}};
    ArrVec x(N, arr);
    Matrix A( h);
    dg::PCG<Matrix, ArrVec, dg::T> pcg( x, x.size());
    dg::CG<Matrix, ArrVec> cg( x, x.size());
    auto b = dg::expand<double (&)(double), P> ( sinus, 0,lx, N);
    auto error(b);
    const auto solution(b);

    cout << "Polynomial order (P-1): "<< P-1 <<endl;
    cout << "Vector size: P*N "<< P*N <<endl;
    //compute S b
    dg::BLAS2<dg::S, ArrVec>::dsymv( 1., dg::S(h), b, 0, b);
    std::cout << "Number of pcg iterations "<< pcg( A, x, b, dg::T(h), 1e-10)<<endl;
    //std::cout << "Number of cg iterations "<< cg( A, x, b, 1e-11)<<endl;
    //cout << "b is \n" << b<<endl;
    //cout << "x is \n" << x<<endl;
    //compute error
    dg::BLAS1<ArrVec>::daxpby( 1.,x,-1.,error);
    //and Ax
    auto bx(x);
    dg::BLAS2<Matrix, ArrVec>::dsymv( 1., A, x, 0, bx);
    //cout << "Ax is \n" << bx <<endl;

    double eps = dg::BLAS2<dg::S, ArrVec>::ddot( error, dg::S(h), error);
    cout << "L2 Norm2 of Error is " << eps << endl;
    double norm = dg::BLAS2<dg::S, ArrVec>::ddot( solution, dg::S(h), solution);
    std::cout << "L2 Norm of relative error is "<<sqrt( eps/norm)<<std::endl;



    return 0;
}
