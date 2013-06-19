#include <iostream>
#include <cmath>
#include "karniadakis.h"
#include "matrix.h"
#include "ghostmatrix.h"
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
    Matrix< QuadMat<double,2> > coeff( rows, cols, One<2>());
    Matrix<double, TL_NONE> m( rows, cols, 1.), n( rows, cols, 1.);
    std::array< Matrix<double>, 2> v{{m,m}}, non{{n,n}};

    //Test of Karniadakis scheme
    cout << "Construct Karniadakis object...\n";
    Karniadakis<2, double, TL_NONE> k( rows, cols, rows, cols, dt);
    k.init_coeff( coeff, 1. );
    cout << "Make various steps...\n";
    
    k.invert_coeff<TL_EULER> ( );
    k.step_i<TL_EULER>( v, non);
    k.step_ii( v);
    t += dt;
    non = v;
    k.step_i<TL_ORDER2>( v, non);
    k.invert_coeff<TL_ORDER2> ();
    k.step_ii( v);
    t += dt;
    non = v;
    k.invert_coeff<TL_ORDER3> ();
    for( unsigned i = 2; i < steps; i++)
    {
        k.step_i<TL_ORDER3>( v, non);
        k.step_ii( v);
        t += dt;
        non = v;
    }
    cout << "At time "<<t<<"\n"
         << "(with "<<steps<<" steps)\n"
         << "Exact solution is:       "<< exp(2) << "\n"
         << "Approximate solution is: "<<v[0](0,0) << endl
         << "Relative error:          "<< (v[0](0,0)-exp(2))/exp(2) <<endl;
    cout << "(Test passed when relative error is small!)\n";


    return 0;
}
