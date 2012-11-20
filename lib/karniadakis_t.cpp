#include <iostream>
#include <cmath>
#include "karniadakis.h"
#include "timer.h"
#include "matrix.h"
#include "quadmat.h"

using namespace std;
using namespace toefl;


//try to integrate d/dt y = y + y from 0 to 1 with y a nonlinearity and a linearity
// y(0) = 1;
int main()
{
    const size_t rows = 2, cols = 4;
    unsigned steps = 1000;
    double dt = 1./(double)steps;
    double t = 0;
    //initialize the linearity and the nonlinearity
    Matrix< QuadMat<double,2> > coeff( rows, cols, Eins<2>());
    Matrix<double, TL_NONE> m( rows, cols, 1.), n( rows, cols, 1.);
    Vector< Matrix<double>, 2> v, non;
    Timer timer;
    try{
        v[0].allocate( rows, cols);
        v[1].allocate( rows, cols);
        non[0].allocate( rows, cols);
        non[1].allocate( rows, cols);

        v[0] = v[1] = m;
        non[0] = non[1] = n;
    }catch(Message& m){m.display();}

    //Test of Karniadakis scheme
    cout << "Construct Karniadakis object...\n";
    Karniadakis<2, double, TL_NONE> k( rows, cols, dt);
    cout << "make various steps...\n";
    k.init_coeff (coeff);

    timer.tic();
    k.step_i<TL_EULER>( v, non);
    k.step_ii<TL_EULER>( v, true);
    t += dt;
    non = v;
    k.step_i<TL_ORDER2>( v, non);
    k.step_ii<TL_ORDER2>( v, true);
    t += dt;
    non = v;
    k.step_i<TL_ORDER3>( v,non);
    k.step_ii<TL_ORDER3>( v, true);
    t += dt;
    non = v;
    cout << "Begin loop...\n";
    for( unsigned i = 3; i < steps; i++)
    {
        k.step_i<TL_ORDER3>( v, non);
        k.step_ii<TL_ORDER3>( v);
        t += dt;
        non = v;
    }
    timer.toc();
    cout << "Exact solution is: "<< exp(2) << "\n";
    cout << "at time "<<t<<"\n";
    cout << "Approximate solution is: \n" <<v;
    cout << "with relative error: "<< (v[0](0,0)-exp(2))/exp(2) <<endl;
    cout << "in "<<timer.diff()<<"s\n";


    return 0;
}
