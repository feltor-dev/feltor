#include <iostream>
#include <cmath>
#include "timer.h"
#include "drt_drt.h"

using namespace std;
using namespace toefl;


const size_t rows = 512, cols = 4*512;
fftw_r2r_kind kind0 = FFTW_RODFT10;
fftw_r2r_kind kind1 = FFTW_R2HC;
fftw_r2r_kind kind0_inv = inverse_kind(kind0);
fftw_r2r_kind kind1_inv = inverse_kind(kind1);
int main()
{
    Timer t;
    Matrix<double, TL_NONE> test( rows, cols);

    DRT_DRT drt_drt( rows, cols, kind0, kind1);
    for( size_t i = 0; i < rows; i++)
        for( size_t j = 0; j < cols; j++)
            test(i,j) = i + j -1;

    t.tic();
    drt_drt.forward( test, test);
    t.toc();
    cout << "Transformation took "<<t.diff()<<"ms\n";
    t.tic();
    drt_drt.backward( test, test);
    t.toc();
    cout << "Backtransformation took "<<t.diff()<<"ms\n";

    ////////////////////////////////////////////////////////////////////////////////
    Matrix<double, TL_NONE> m0( rows, cols);
    Matrix<double, TL_NONE> m1( rows, cols);

    fftw_plan forward =  fftw_plan_r2r_2d( rows, cols, m0.getPtr(), m0.getPtr(), kind0, kind1, FFTW_MEASURE); 
    fftw_plan backward = fftw_plan_r2r_2d( rows, cols, m0.getPtr(), m0.getPtr(), kind0_inv, kind1_inv, FFTW_MEASURE); 
    for( size_t i = 0; i < rows; i++)
        for( size_t j = 0; j < cols; j++)
            m0(i,j) = i + j - 1;//sin( M_PI*(i+1)/(rows+1.))*sin( 2*M_PI*(j+1)/(cols+1.));
    t.tic();
    fftw_execute_r2r( forward, m0.getPtr(), m0.getPtr());
    t.toc();
    cout << "Direct FFTW Transformation took "<<t.diff()<<"ms\n";
    t.tic();
    fftw_execute_r2r( backward, m0.getPtr(), m0.getPtr());
    t.toc();
    cout << "Direct FFTW Backtransformation took "<<t.diff()<<"ms\n";

    return 0 ;
}
