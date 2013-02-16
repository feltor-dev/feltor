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

    double norm_x=0, norm_l=0;
    for( unsigned i=0; i<num_int; i++)
        for( unsigned j=0; j<P; j++)
        {
            norm_x += DLT<P>::weight[j]*v[i][j]*v[i][j];
            norm_l += 2./(2.*j+1.)*w[i][j]*w[i][j];
        }

    cout << "Square norm in x "<<norm_x<<endl;
    cout << "Square norm in l "<<norm_l<<endl;



    return 0;

}
