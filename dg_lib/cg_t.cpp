#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cmath>

#include "operators.h"
#include "evaluation.h"
#include "laplace.h"
#include "dlt.h"
#include "cg.h"

using namespace std;
using namespace dg;

double sinus(double x){ return /*x*x*x*/sin(2*M_PI*x);}
double secondsinus(double x){ return /*-6*x*/4.*M_PI*M_PI*sin(2*M_PI*x);}

#define P 5
const unsigned num_int = 10000;
const double lx = 1.;
typedef std::vector<std::array<double, P>> ArrVec;
typedef dg::Laplace<P> Matrix;

int main()
{
    cout << "Order is (P-1): "<<P<<endl;
    const double h = 1./(double)num_int;
    Matrix l(h, 1.); //the constant makes all projection operators correct
    cout << l.get_a()<<endl;
    cout << l.get_b()<<endl;
    Operator<double,P> forward( DLT<P>::forward);

    ArrVec x = evaluate< double(&)(double), P>( sinus, 0,lx, num_int);
    ArrVec solution = evaluate< double(&)(double), P>( secondsinus, 0,lx, num_int);
    cout << "Square norm of sine is : "<<square_norm( x, XSPACE)*h/2.<<endl;
    cout << "Square norm of solution (779.273) is : "<<square_norm( solution, XSPACE)*h/2.<<endl;
    
    for( unsigned i=0; i<num_int; i++)
        x[i] = forward*x[i];
    for( unsigned i=0; i<num_int; i++)
        solution[i] = forward*solution[i];
    double s_norm2 = BLAS2<S,ArrVec>::ddot(x,S(h),x); 

    cout << "Square norm of sine is : "<<s_norm2 <<endl;    cout << "Square norm of solution is : "<<BLAS2<S, ArrVec>::ddot( solution, S(h), solution) <<endl;
    cout << "Sine approximation\n";
    //cout << x <<endl;
    cout << "Solution: \n";
    //cout << solution << endl;
    ArrVec w(num_int);
    dg::BLAS2<Matrix, ArrVec>::dsymv( 1., l, x, 0, w);
    dg::BLAS2<T, ArrVec>::dsymv( 1., T(h), w, 0, w);
    double w_norm2 = BLAS2<S, ArrVec>::ddot(w, S(h), w);
    cout << "Approximation: \n";
    //cout << w <<endl;
    cout << "Square norm of w is: "<< w_norm2 << endl;
    dg::BLAS1<ArrVec>::daxpby( 1., solution, -1., w);
    cout << "Relative error in L2 norm is \n";
    cout << sqrt(BLAS2<S, ArrVec>::ddot(w, S(h), w)/s_norm2)<<endl;
    //compute jumps in w
    auto jump = dg::evaluate_jump(w);
    unsigned interior = jump.size();
    //cout << "Jumps of approximation \n";
    //for( unsigned i=0; i<interior; i++)
    //    cout << jump[i] <<endl;

    /*
    ofstream os( "error.dat");
    for( unsigned i=0; i<num_int; i++)
        os << (double)i*h << " "<< w[i][0] << " "<<solution[i][0]<<" "
           << w[i][1]<< " "<<solution[i][1]<<"\n";
           */
    


    return 0;
}

