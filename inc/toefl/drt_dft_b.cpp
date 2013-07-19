#include <iostream>
#include <iomanip>
#include <cmath>
#include <complex>
#include "drt_dft.h"
#include "drt_drt.h"
#include "timer.h"

using namespace std;
using namespace toefl;

const size_t rows = 512, cols = 4*512;
fftw_r2r_kind kind = FFTW_RODFT10;
int main()
{
    Timer t;
    Matrix<double, TL_DRT_DFT> test( rows, cols);
    Matrix<complex<double> > test_( cols, rows/2 +1 );
    DRT_DFT drt_dft( rows, cols, kind);
    for( size_t i = 0; i < rows; i++)
        for( size_t j = 0; j < cols; j++)
            test(i,j) = sin( M_PI*(i+1)/(rows+1.))*cos( 2*M_PI*j/cols);
    t.tic();
    drt_dft.r2c_T( test, test_);
    t.toc();
    cout << "Transformation took " << t.diff()<< "s\n";
    t.tic();
    drt_dft.c_T2r( test_, test);
    t.toc();
    cout << "Backtransformation took " << t.diff() <<"s\n";

    Matrix<double, TL_NONE> m0(rows, cols);
    DRT_DRT drt_drt( rows, cols, kind, FFTW_R2HC, FFTW_MEASURE);
    for( size_t i = 0; i < rows; i++)
        for( size_t j = 0; j < cols; j++)
            m0(i,j) = sin( M_PI*(i+1)/(rows+1.))*cos( 2*M_PI*j/cols);
    t.tic();
    drt_drt.forward( m0, m0);
    t.toc();
    cout << "DRT_DRT (R2HC) Transformation took " << t.diff()<< "s\n";
    t.tic();
    drt_drt.backward( m0, m0);
    t.toc();
    cout << "DRT_DRT (HC2R) Backtransformation took " << t.diff() <<"s\n";


    return 0 ;
}
