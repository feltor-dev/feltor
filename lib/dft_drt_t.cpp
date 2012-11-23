#include <iostream>
#include <iomanip>
#include <cmath>
#include <complex>
#include "dft_drt.h"

using namespace std;
using namespace toefl;


int main()
{
    Matrix<double, TL_DFT_1D> test( 5, 10);
    Matrix<complex<double> > test_( 5, 6);
    DFT_DRT dft_drt( 5, 10, FFTW_RODFT00);
    test.zero();
    cout << setprecision(2) << fixed;
    cout << "Testmatrix is\n" << test<<endl;

    dft_drt.r2c( test, test_);
    cout << "transformed matrix is\n" << test_<<endl;
    for( unsigned i=0; i<5; i++)
        test_( i, 5) = { sin(M_PI*(i+1)/6.), 0};
    cout << "Multiplied matrix is\n" << test_<<endl;

    dft_drt.c2r( test_, test);
    cout << "backtransformed is (should be input times 120)\n" << test << endl;


    return 0 ;
}
