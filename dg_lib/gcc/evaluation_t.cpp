#include <iostream>
#include <vector>
#include <array>
#include <cmath>

#include "evaluation.h"
#include "operators.h"
#include "timer.h"

using namespace dg;
using namespace std;

unsigned counter =0;
double function( double x)
{
    counter++;
    return exp(x);
}
#define P 1
const unsigned num_int = 20;
typedef std::vector<std::array<double,P>> ArrVec;
int main()
{

    cout << "Polynomial order is (P-1): "<<P-1<<endl; 
    double h = 1./(double)num_int;
    //evaluate exp function and transform to l space using various methods
    auto v = evaluate< double(&)(double), P>( function,0.,1., num_int);
    auto w(v);
    Operator<double,P> forward( DLT<P>::forward);
    for( unsigned i=0; i<num_int; i++)
        w[i] = forward*v[i];
    auto x = expand< double(&)(double),P> ( function, 0.,1., num_int);
    BLAS1<ArrVec>::daxpby( 1., w, -1., x);
    cout << "Test of the expand function\n";
    cout << x <<endl;
    cout << "Should be only zeros\n";
    //compute the square norm with various methods
    double norm = BLAS2<Space, ArrVec>::ddot( w, LSPACE, w);
    cout << "Square norm in x "<<norm <<endl;
    cout << "Square norm in x "<<square_norm<P>( v, XSPACE)<<endl;
    cout << "Square norm in l "<<square_norm<P>( w, LSPACE)<<endl;
    //compute normalized norm using various methods
    BLAS2<S, ArrVec>::dsymv( 1, S(h), w, 0, w);
    cout << "Square normalized norm "<<BLAS2<T, ArrVec>::ddot( w, T(h), w)<<endl;
    double normalized = norm/2./(double)num_int;
    cout << "Square normalized norm "<<normalized<<endl;
    cout << "Square norm normalized to [0,1] "<< norm/2./(double)num_int<<endl;
    double solution = (exp(2.)-exp(0))/2.;
    cout << "Correct square norm of exp(x) is "<<solution<<endl;
    cout << "relative  error is "<< (normalized-solution)/solution <<endl;
    cout << "With "<<counter<<" function calls!\n";

    //compute jumps in w
    auto jump = dg::evaluate_jump(w);
    unsigned interior = jump.size();
    cout << "Jumps of approximation \n";
    for( unsigned i=0; i<interior; i++)
        cout << jump[i] <<endl;




    return 0;

}
