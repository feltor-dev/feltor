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
#define P 2
typedef std::vector<std::array<double,P>> ArrVec;
int main()
{

    cout << "Polynomial order is (P-1): "<<P-1<<endl; 
    unsigned num_int = 4;
    Timer t;
    t.tic();
    auto v = evaluate< double(&)(double), P>( function,0.,1., num_int);
    t.toc();
    cout << "Evaluation took "<<t.diff()<<"s\n";
    auto w(v);
    Operator<double,P> forward( DLT<P>::forward);
    for( unsigned i=0; i<num_int; i++)
        w[i] = forward*v[i];
    t.tic(); 
    double norm = BLAS2<Space, ArrVec>::ddot( w, LSPACE, w);
    t.toc();
    
    cout << "Square norm in x "<<norm <<endl;
    cout << "Took "<<t.diff()<<"s\n";

    cout << "Square norm in x "<<square_norm<P>( v, XSPACE)<<endl;
    cout << "Square norm in l "<<square_norm<P>( w, LSPACE)<<endl;
    //BLAS2<T, ArrVec>::dsymv( 1, T(), w, 0, w);
    BLAS2<S, ArrVec>::dsymv( 1, S(), w, 0, w);
    cout << "Square norm      "<<BLAS2<T, ArrVec>::ddot( w, T(), w)<<endl;
    t.tic();
    square_norm<P>( w, LSPACE);
    t.toc();
    cout << "Took "<<t.diff()<<" s\n";
     
    double normalized = norm/2./(double)num_int;
    double solution = (exp(2.)-exp(0))/2.;
    //double solution = 0.5;
    cout << "Square norm normalized to [0,1] "<< norm/2./(double)num_int<<endl;
    cout << "Correct square norm of exp(x) is "<<solution<<endl;
    cout << "relative  error is "<< (normalized-solution)/solution <<endl;
    cout << "With "<<counter<<" function calls!\n";




    return 0;

}
