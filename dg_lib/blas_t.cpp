#include <iostream>
#include <cmath>
#include "evaluation.h"
#include "blas.h"


using namespace std;

double function( double x)
{
    return sin(2*M_PI*x);
}
#define P 2
typedef std::vector<std::array<double,P>> ArrVec;
int main()
{
    unsigned N = 10;
    vector< array<double,P>> v = dg::evaluate< double(&)(double), P>( function,0.,1., N);
    vector< array<double,P>> w(v);
    cout << "v is a sine \n";
    for( unsigned i=0; i<N; i++)
    {
        for( unsigned j=0; j<P; j++)
            cout << v[i][j] << " ";
        cout << "\n";
    }
    dg::BLAS1<ArrVec>::daxpby( 10., w, 0., w);
    cout << "w is 10 sine \n";
    for( unsigned i=0; i<N; i++)
    {
        for( unsigned j=0; j<P; j++)
            cout << w[i][j] << " ";
        cout << "\n";
    }

}

