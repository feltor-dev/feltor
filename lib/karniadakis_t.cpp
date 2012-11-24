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
    //First test the multiply coefficients function
    Matrix< QuadMat<double, 2>> test_coeff( 2, 3);
    for( unsigned i=0; i<2; i++)
        for( unsigned j=0; j<3; j++)
            for( unsigned k=0; k<2; k++)
                for( unsigned q=0; q<2; q++)
                    test_coeff(i,j)(k,q) = i+j+k+q;

    cout << "Test coefficients are:\n "<<test_coeff<<endl;
    Matrix< double> test1( 2,3, 1.);
    Matrix< double> test2( 2,3, 2.);
    std::array< Matrix<double>, 2> test_v{{ test1, test2}};
    cout << "multiply coefficients are: " <<endl;
    multiply_coeff( test_coeff, test_v, test_v);
    cout <<test_v[0] <<endl << test_v[1] <<endl;



    //initialize the linearity and the nonlinearity
    Matrix< QuadMat<double,2> > coeff( rows, cols, Eins<2>());
    Matrix<double, TL_NONE> m( rows, cols, 1.), n( rows, cols, 1.);
    std::array< Matrix<double>, 2> v, non;
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
    k.init_coeff( coeff );
    cout << "make various steps...\n";
    
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
    cout << "Exact solution is: "<< exp(2) << "\n";
    cout << "at time "<<t<<"\n";
    cout << "Approximate solution is: \n" <<v[0] << endl;
    cout << "with relative error: "<< (v[0](0,0)-exp(2))/exp(2) <<endl;


    return 0;
}
