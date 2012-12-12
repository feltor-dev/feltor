#include <iostream>
#include <iomanip>
#include <cmath>
#include <complex>
#include "dft_drt.h"
#include "drt_drt.h"
#include "timer.h"

using namespace std;
using namespace toefl;

const size_t rows = 512, cols = 4*512;
fftw_r2r_kind kind = FFTW_RODFT10;
int main()
{
    Timer t;
    Matrix<double, TL_DFT> test( rows, cols);
    Matrix<complex<double> > test_( rows, cols/2 +1);
    DFT_DRT dft_drt( rows, cols, kind);
    for( size_t i = 0; i < rows; i++)
        for( size_t j = 0; j < cols; j++)
            test(i,j) = sin( M_PI*(i+1)/(rows+1.))*cos( 2*M_PI*j/cols);
    t.tic();
    dft_drt.r2c( test, test_);
    t.toc();
    cout << "Transformation took " << t.diff()<< "s\n";
    t.tic();
    dft_drt.c2r( test_, test);
    t.toc();
    cout << "Backtransformation took " << t.diff() <<"s\n";

    Matrix<double, TL_NONE> m0(rows, cols);
    DRT_DRT drt_drt( rows, cols, FFTW_R2HC, kind, FFTW_MEASURE);
    for( size_t i = 0; i < rows; i++)
        for( size_t j = 0; j < cols; j++)
            m0(i,j) = sin( M_PI*(i+1)/(rows+1.))*cos( 2*M_PI*j/cols);
    t.tic();
    drt_drt.forward( m0, m0);
    t.toc();
    cout << "DRT_DRT Transformation took " << t.diff()<< "s\n";
    t.tic();
    drt_drt.backward( m0, m0);
    t.toc();
    cout << "DRT_DRT Backtransformation took " << t.diff() <<"s\n";


    fftw_cleanup();
    return 0 ;
}
