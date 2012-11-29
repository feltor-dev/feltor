#include <iostream>
#include <iomanip>
#include <cmath>
#include <complex>
#include "dft_drt.h"

using namespace std;
using namespace toefl;


int main()
{
    Matrix<double, TL_DFT> test( 5, 10);
    Matrix<complex<double> > test_( 5, 6);
    DFT_DRT dft_drt( 5, 10, FFTW_RODFT10); //A DST2 is DFT10 NOT DFT01 !!
    test.zero();
    cout << setprecision(2) << fixed;
    for( unsigned i=0; i<5; i++)
        for( unsigned j=0; j<10; j++)
            test(i,j) = sin(M_PI*(i+1./2.)/5.);
    cout << "Testmatrix is\n" << test<<endl;

    dft_drt.r2c( test, test_);
    cout << "transformed matrix is\n" << test_<<endl;

    dft_drt.c2r( test_, test);
    cout << "backtransformed is (should be input times 100)\n" << test << endl;


    return 0 ;
}
