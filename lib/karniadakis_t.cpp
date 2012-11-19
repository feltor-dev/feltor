#include <iostream>
#include <cmath>
#include "karniadakis.h"
#include "matrix.h"
#include "quadmat.h"

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
    //initialize the Linearity which is zero
    Matrix< QuadMat<double,2> > coeff( rows, cols, Zero<2>());
    Matrix<double, TL_NONE> m( rows, cols, 1.), n( rows, cols, 1.);

    Karniadakis<2, TL_EULER, double, TL_NONE> k_euler( rows, cols, coeff, dt);
    Karniadakis<2, TL_ORDER2, double, TL_NONE> k_1( rows, cols, coeff, dt);
    Vector< Matrix<double>, 2> v, non;
    v[0].resize( rows, cols);
    v[1].resize( rows, cols);
    non[0].resize( rows, cols);
    non[1].resize( rows, cols);

    v[0] = v[1] = m;
    non[0] = non[1] = n;
    try{
    k_euler.prestep( v, non);
    k_euler.poststep( v);
    }catch(Message& m){m.display();}
    t += dt;

    swap_fields( k_euler, k_1);
    non = v;
    k_1.prestep( v, non);
    k_1.poststep( v);
    t += dt;
    swap_fields( k_2, k_1);
    non = v;
    for( unsigned i = 2; i < steps; i++)
    {
        k_2.prestep( v, non);
        k_2.poststep( v);
        t += dt;
        non = v;
    }
    cout << "Exact solution is: "<< exp(1) << "\n";
    cout << "at time "<<t<<"\n";
    cout << "Approximate solution is: \n" <<v << endl;


    return 0;
}
