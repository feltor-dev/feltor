#include <iostream>
#include <complex>
#include "timer.h"
#include "matrix.h"
#include "fft.h"

using namespace std;
using namespace toefl;


unsigned rows = 500, cols = 4*500;

int main()
{
    Matrix<  double > a(rows, cols);
    Matrix<  double > b(cols, rows);
    fftw_plan transpose = plan_transpose( rows, cols/2, fftw_cast(a.getPtr()), fftw_cast(a.getPtr()), FFTW_MEASURE);
    Timer t;
    for( size_t i =0; i<rows; i++)
        for( size_t j=0; j<cols; j++)
            a(i,j) = 17 + i +j;
    t.tic();
    fftw_execute( transpose);
    t.toc();

    cout << "Transposition takes " << t.diff() << " seconds\n";

    Matrix<double, TL_DFT_1D> m2(rows,cols);
    Matrix<complex<double>, TL_NONE> m2_( rows, cols/2+1);
    fftw_plan forward_plan = plan_dft_1d_r2c(rows, cols, m2.getPtr(), reinterpret_cast<fftw_complex*>(m2.getPtr()), FFTW_MEASURE);

    double dx = 1./cols;
    for( size_t i = 0; i < m2.rows(); i++)
        for ( size_t j=0; j < m2.cols(); j++)
            m2(i, j) = cos( 2*(i+1)*M_PI*j*dx); 
    cout << "forward trafo\n";
    t.tic();
    fftw_execute_dft_r2c( forward_plan, m2.getPtr(), reinterpret_cast<fftw_complex*>(m2.getPtr()));
    swap_fields(m2,m2_);
    t.toc();
    cout << "Fourier trafo takes " << t.diff() << " seconds\n";



    return 0;
}
