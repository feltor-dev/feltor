#include <iostream>
#include <cmath>
#include "timer.h"
#include "dst_dst.h"

using namespace std;
using namespace toefl;


const size_t rows = 512, cols = 4*512;
fftw_r2r_kind kind = FFTW_RODFT10;
fftw_r2r_kind kind_inv = inverse_kind(kind);
int main()
{
    Timer t;
    Matrix<double, TL_NONE> test( rows, cols);
    Matrix<double, TL_NONE> test_T( cols, rows);

    DST_DST dst_dst( rows, cols, kind);
    for( size_t i = 0; i < rows; i++)
        for( size_t j = 0; j < cols; j++)
            test(i,j) = i + j -1;

    t.tic();
    dst_dst.r2r_T( test, test_T);
    t.toc();
    cout << "Transformation took "<<t.diff()<<"s\n";
    t.tic();
    dst_dst.r_T2r( test_T, test);
    t.toc();
    cout << "Backtransformation took "<<t.diff()<<"s\n";

    ////////////////////////////////////////////////////////////////////////////////
    Matrix<double, TL_NONE> m0( rows, cols);
    Matrix<double, TL_NONE> m1( rows, cols);

    fftw_plan forward =  fftw_plan_r2r_2d( rows, cols, m0.getPtr(), m0.getPtr(), kind, kind, FFTW_MEASURE); 
    fftw_plan backward = fftw_plan_r2r_2d( rows, cols, m0.getPtr(), m0.getPtr(), kind_inv, kind_inv, FFTW_MEASURE); 
    for( size_t i = 0; i < rows; i++)
        for( size_t j = 0; j < cols; j++)
            m0(i,j) = i + j - 1;//sin( M_PI*(i+1)/(rows+1.))*sin( 2*M_PI*(j+1)/(cols+1.));
    t.tic();
    fftw_execute_r2r( forward, m0.getPtr(), m0.getPtr());
    t.toc();
    cout << "Transformation took "<<t.diff()<<"s\n";
    t.tic();
    fftw_execute_r2r( backward, m0.getPtr(), m0.getPtr());
    t.toc();
    cout << "Backtransformation took "<<t.diff()<<"s\n";

    return 0 ;
}
