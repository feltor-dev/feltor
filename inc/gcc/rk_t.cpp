#include <iostream>
#include <array>
#include <cmath>

#include "rk.h"

using namespace std;

namespace dg{
template<>
struct BLAS1<array<double,2>> 
{
    typedef typename std::array<double,2> Vector;
    static void daxpby( double alpha, const Vector& x, double beta, Vector& y)
    {
        for( unsigned i=0; i<2; i++)
            y[i] = alpha*x[i]+beta*y[i];
    }
};
} //namespace dg

typedef array<double, 2> Vec;

struct Functor
{
    typedef std::array<double,2> Vector;
    void operator( )(const Vec& y0, Vec& y1){
        for( unsigned i=0; i<2; i++)
            y1[i] = y0[i]; 
    }
};

int main()
{
    const unsigned num_int = 100;
    Vec y0{ {1,1}};
    Vec y1 = y0;

    double h = 1./(double)num_int;
    dg::RK<2, Functor> rk(y0);
    Functor f;
    for( unsigned i=1; i<=num_int/2; i++)
    {
        rk( f, y0, y1, h);
        rk( f, y1, y0, h);
    }
    cout << "Solution is " << y0[0] << endl;
    cout << "Error is "<< (y0[0]-exp(1))/exp(1)<<endl;
    return 0;
}

