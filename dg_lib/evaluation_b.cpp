#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include "evaluation.h"
#include "operators.h"
#include "timer.h"

using namespace dg;
using namespace std;

double function( double x)
{
    return 3*x*x-x+2+sqrt(x);
}
class Functor
{
    public:
    double operator()(double x)
    {
        return x;
    }
};
#define P 4
typedef std::vector<std::array<double,P>> ArrVec;
int main()
{
    unsigned num_int = 1000000;
    Timer t;
    t.tic();
    auto v = evaluate< double(&)(double), P>( function,0.,1., num_int);
    t.toc();
    cout << "Evaluation took "<<t.diff()<<"s\n";
    auto w(v);
    Operator<double,P> forward( DLT<P>::forward);

    for( unsigned i=0; i<num_int; i++)
        w[i] = forward*v[i];
    double norm;

    //cout << "Square norm in x "<<square_norm<P>( v, XSPACE)<<endl;
    //cout << "Square norm in l "<<square_norm<P>( w, LSPACE)<<endl;
    //t.tic();
    //norm = square_norm<P>( w, LSPACE);
    //t.toc();
    //cout << "Square norm in x "<<norm <<endl;
    //cout << "Took "<<t.diff()<<" seconds\n";
    t.tic(); 
    norm = CG_BLAS2<Space, ArrVec>::ddot( w, LSPACE, w);
    t.toc();
    
    cout << "Square norm in x "<<norm <<endl;
    cout << "Took "<<t.diff()<<"s\n";




    return 0;

}
