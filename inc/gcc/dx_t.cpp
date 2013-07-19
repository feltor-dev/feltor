#include <iostream>
#include <cmath>

#include "dx.h"
#include "evaluation.h"

//is derivative consistent for k even?
#define P 5
const unsigned N = 10;
const double lx = 1;
typedef std::vector<std::array<double,P>> ArrVec;
double function( double x)
{
    return exp(x);
}

using namespace std;
int main()
{
    double  h = lx/ (double)N; 
    dg::DX<P> dx(h);
    std::cout << "A is \n" <<dx.get_a();
    //evaluate exp function
    auto v = dg::evaluate< double(&)(double), P>( function,0.,lx, N);
    ArrVec w(v), der(v), error( v);
    dg::Operator<double,P> forward( dg::DLT<P>::forward);
    for( unsigned i=0; i<N; i++)
        w[i] = forward*v[i];
    cout << w << endl;

    //compute the square norm 
    double norm = dg::BLAS2<dg::S, ArrVec>::ddot( w, dg::S(h), w);
    cout << "Norm2 is: " << norm <<endl;
    //compute first derivative
    dg::BLAS2<dg::DX<P>, ArrVec>::dsymv( 1., dx, w, 0, der);
    cout << "First derivative is: \n";
    cout << der << endl;
    // and the norm of it
    double norm_d = dg::BLAS2<dg::S, ArrVec>::ddot( der, dg::S(h), der);
    cout << "Derivative Norm2 is: " << norm_d <<endl;
    //now compute error 
    dg::BLAS1<ArrVec>::daxpby( 1., der, -1, w);
    double norm_e = dg::BLAS2<dg::S, ArrVec>::ddot( der, dg::S(h), der);
    cout << "Relative Error Norm is: " << sqrt( norm_e/norm) <<endl;





    return 0;
}
