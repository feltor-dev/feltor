#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include "evaluation.h"
#include "operators.h"

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
int main()
{

    //Functor f;
    unsigned num_int = 10;
    auto v = evaluate< double(&)(double), P>( function,0.,1., num_int);
    auto w(v);
    Operator<double,P> forward( DLT<P>::forward);
    //for( unsigned i=0; i<num_int; i++)
        //cout << v[i][0] << " "<<v[i][1]<< " "<<v[i][2]<<endl;

    for( unsigned i=0; i<num_int; i++)
        w[i] = forward*v[i];
    //for( unsigned i=0; i<num_int; i++)
        //cout << w[i][0] << " "<<w[i][1]<< " "<<w[i][2]<<endl;

    cout << "Square norm in x "<<square_norm<P>( v, XSPACE)<<endl;
    cout << "Square norm in l "<<square_norm<P>( w, LSPACE)<<endl;



    return 0;

}
