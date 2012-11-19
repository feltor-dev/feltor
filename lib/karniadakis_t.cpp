#include <iostream>
#include <cmath>
#include "karniadakis.h"
#include "matrix.h"
#include "quadmat.h"

using namespace std;
using namespace toefl;


//try to integrate d/dt y = y + y from 0 to 1 with y a nonlinearity and a linearity
// y(0) = 1;
int main()
{
    const size_t rows = 2, cols = 4;
    unsigned steps = 100;
    double dt = 1./(double)steps;
    double t = 0;
    //initialize the linearity and the nonlinearity
    Matrix< QuadMat<double,2> > coeff( rows, cols, Eins<2>());
    Matrix<double, TL_NONE> m( rows, cols, 1.), n( rows, cols, 1.);
    Vector< Matrix<double>, 2> v, non;
    try{
        v[0].allocate( rows, cols);
        v[1].allocate( rows, cols);
        non[0].allocate( rows, cols);
        non[1].allocate( rows, cols);

        v[0] = v[1] = m;
        non[0] = non[1] = n;
        cout << v << non <<endl;
    }catch(Message& m){m.display();}

    Karniadakis<2,  TL_EULER, double, TL_NONE> k_euler( rows, cols, coeff, dt);
    Karniadakis<2, TL_ORDER2, double, TL_NONE> k_1( rows, cols, coeff, dt);
    Karniadakis<2, TL_ORDER3, double, TL_NONE> k_2( rows, cols, coeff, dt);
    k_euler.step_i( v, non);
    k_euler.step_ii( v);
    t += dt;
    swap_fields( k_euler, k_1);
    non = v;
    k_1.step_i( v, non);
    k_1.step_ii( v);
    t += dt;
    swap_fields( k_2, k_1);
    non = v;
    for( unsigned i = 2; i < steps; i++)
    {
        k_2.step_i( v, non);
        k_2.step_ii( v);
        t += dt;
        non = v;
    }
    cout << "Exact solution is: "<< exp(2) << "\n";
    cout << "at time "<<t<<"\n";
    cout << "Approximate solution is: \n" <<v << endl;
    cout << "with relative error: "<< (v[0](0,0)-exp(2))/exp(2) <<endl;


    return 0;
}
