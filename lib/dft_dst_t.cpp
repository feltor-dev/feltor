#include <iostream>
#include <iomanip>
#include <cmath>
#include <complex>
#include "dft_dst.h"

using namespace std;
using namespace toefl;


int main()
{
    Matrix<double, TL_DFT_1D> test( 5, 10);
    Matrix<complex<double> > test_( 5, 6);
    DFT_DST dft_dst( 5, 10, FFTW_RODFT00);
    for( size_t i = 0; i < 5; i++)
        for( size_t j = 0; j < 10; j++)
            test(i,j) = sin( M_PI*(i+1)/(5.+1.))*cos( 2*M_PI*j/10.);
    cout << setprecision(2) << fixed;
    cout << "Testmatrix is\n" << test<<endl;

    dft_dst.r2c( test, test_);
    cout << "transformed matrix is\n" << test_<<endl;
    dft_dst.c2r( test_, test);
    cout << "backtransformed is (should be input times 120)\n" << test << endl;


    return 0 ;
}
