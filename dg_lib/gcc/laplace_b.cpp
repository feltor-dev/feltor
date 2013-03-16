#include <iostream>
#include <vector>
#include <cmath>

#include "timer.h"
#include "operators.h"
#include "evaluation.h"
#include "laplace.h"
#include "preconditioner.h"
#include "dlt.h"

using namespace std;
using namespace dg;

double sinus(double x){ return /*x*x*x*/sin(2*M_PI*x);}
double secondsinus(double x){ return /*-6*x*/4.*M_PI*M_PI*sin(2*M_PI*x);}

#define P 3
const unsigned num_int = 1e5;
const double lx = 1.;
typedef std::vector<Array<double, P>> ArrVec;
typedef dg::Laplace<P> Matrix;

int main()
{
    Timer t;
    cout << "Test and see the supraconvergence phenomenon!\n";
    cout << "Order is (P-1): "<<P<<endl;
    cout << "# of intervals is: "<< num_int<<"\n";
    const double h = 1./(double)num_int;
    Matrix l(h); 
    /*
    cout << " a and b: \n";
    cout << l.get_a()<<endl;
    cout << l.get_b()<<endl;
    */

    ArrVec x = dg::expand< double(&)(double), P>( sinus, 0,lx, num_int);
    ArrVec solution = dg::expand< double(&)(double), P>( secondsinus, 0,lx, num_int);
    
    double s_norm2 = BLAS2<S,ArrVec>::ddot(S(h),x); 

    cout << "Square norm of sine is : "<<s_norm2 <<endl;    
    cout << "Square norm of solution is : "<<BLAS2<S, ArrVec>::ddot( S(h), solution) <<endl;
    ArrVec w( num_int);
    t.tic();
    dg::BLAS2<Matrix, ArrVec>::dsymv( l, x, w);
    t.toc();
    cout << "Multiplication with laplace took: "<<t.diff()<<"s\n";
    t.tic();
    dg::BLAS2<T, ArrVec>::dsymv(  T(h), w, w);
    t.toc();
    cout << "Multiplication with T(h) took: "<<t.diff()<<"s\n";
    double w_norm2 = BLAS2<S, ArrVec>::ddot( S(h), w);
    cout << "Square norm of w is: "<< w_norm2 << endl;
    dg::BLAS1<ArrVec>::daxpby( 1., solution, -1., w);
    cout << "Relative error in L2 norm is \n";
    cout << sqrt(BLAS2<S, ArrVec>::ddot(S(h), w)/s_norm2)<<endl;

    return 0;
}

