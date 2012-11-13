#include <iostream>
#include <cmath>
#include "karniadakis.h"
#include "matrix.h"

using namespace std;
using namespace toefl;


//try to integrate d/dt y = y from 0 to 1 with y a nonlinearity
// y(0) = 1;
int main()
{
    const size_t rows = 2, cols = 4;
    unsigned steps = 100;
    double dt = 1./(double)steps;
    double t = 0;
    Matrix<double, TL_NONE> m( rows, cols), n( rows, cols);
    for( size_t i = 0; i< rows; i++)
        for( size_t j = 0; j< cols; j++)
            m(i,j) = n(i,j) = 1;

    Karniadakis<TL_NONE> k_void( rows, cols, dt, false);
    try{ k_void.step<TL_EULER>( m,n);}
    catch( Message& m){ m.display();}
    Karniadakis<TL_NONE> k(rows, cols, dt);
    k.step<TL_EULER>( m, n);
    t += dt;
    for( size_t i = 0; i< rows; i++)
        for( size_t j = 0; j< cols; j++)
            n(i,j) = m(i,j) = m(i,j)/(Coefficients<TL_EULER>::gamma_0);
    k.step<TL_ORDER2>( m, n);
    t += dt;
    for( size_t i = 0; i< rows; i++)
        for( size_t j = 0; j< cols; j++)
            n(i,j) = m(i,j) = m(i,j)/(Coefficients<TL_ORDER2>::gamma_0);
    for( unsigned i = 2; i < steps; i++)
    {
        k.step<TL_ORDER3>( m, n);
        t += dt;
        for( size_t i = 0; i< rows; i++)
            for( size_t j = 0; j< cols; j++)
                n(i,j) = m(i,j) = m(i,j)/(Coefficients<TL_ORDER3>::gamma_0);
    }
    cout << "Exact solution is: "<< exp(1) << "\n";
    cout << "at time "<<t<<"\n";
    cout << "Approximate solution is: \n" <<m << endl;


    return 0;
}
