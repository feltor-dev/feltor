#include <iostream>
#include <vector>
#include <array>
#include <cmath>

#include "evaluation.h"
#include "operators.h"
#include "blas/vector.h"
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
#define P 3
typedef std::vector<std::array<double,P>> ArrVec;
int main()
{
    unsigned num_int = 1e5;
    cout << "Order is (P): "<<P<<endl;
    cout << "# of intervals is: "<< num_int<<"\n";
    Timer t;
    t.tic();
    auto v = evaluate< double(&)(double), P>( function,0.,1., num_int);
    t.toc();
    cout << "Evaluation took "<<t.diff()<<"s\n";
    auto w = expand< double(&) (double), P> (function, 0., 1., num_int);
    double norm;

    //cout << "Square norm in x "<<square_norm<P>( v, XSPACE)<<endl;
    //cout << "Square norm in l "<<square_norm<P>( w, LSPACE)<<endl;
    //t.tic();
    //norm = square_norm<P>( w, LSPACE);
    //t.toc();
    //cout << "Square norm in x "<<norm <<endl;
    //cout << "Took "<<t.diff()<<" seconds\n";
    t.tic(); 
    norm = BLAS2<Space, ArrVec>::ddot( w, LSPACE, w);
    t.toc();
    
    cout << "Square norm in x "<<norm <<endl;
    cout << "Took "<<t.diff()<<"s\n";




    return 0;

}
