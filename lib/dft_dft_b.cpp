#include <iostream>
#include <iomanip>
#include "dft_dft.h"
#include "timer.h"


using namespace std;
using namespace toefl;

unsigned rows = 2*512, cols = 2*512;
int main()
{
    Matrix<double, TL_DFT_DFT> m1(rows, cols);
    Matrix<complex<double> >   m1_( cols/2 + 1, rows);
    Timer t;
    DFT_DFT dft_dft( rows,cols);
    double dx = 1./(cols), dy = 1./rows;
    for( size_t i = 0; i < m1.rows(); i++)
        for ( size_t j=0; j < m1.cols(); j++)
            m1(i, j) = sin( 4.*M_PI*j*dx)*cos( 2.*M_PI*i*dy); //sin(kPix)*sin(qPiy)
    cout << "One mode in every line, One mode in every column\n"<< endl;

    t.tic();
    dft_dft.r2c_T( m1, m1_);
    t.toc();
    cout << "The transformed matrix "<< t.diff()<< "s\n";
    t.tic();
    dft_dft.c_T2r( m1_, m1);
    t.toc();
    cout << "The backtransformed matrix "<<t.diff()<<"s\n";


    // inplace is faster than out of place
    // r2c_2d is faster than r2c_1d, transpose, c2c_1d
    Matrix<double, TL_DFT_DFT> m0(rows, cols);
    Matrix<double, TL_DFT_DFT> m2(rows, cols);
    Matrix<complex<double> >   m0_(rows, cols/2 + 1);
    fftw_plan plan = fftw_plan_dft_r2c_2d( rows, cols, m0.getPtr(), fftw_cast(m0.getPtr()), FFTW_MEASURE);
    fftw_plan plan2 = fftw_plan_dft_c2r_2d( rows, cols, fftw_cast(m0.getPtr()), m0.getPtr(), FFTW_MEASURE);
    for( size_t i = 0; i < m0.rows(); i++)
        for ( size_t j=0; j < m0.cols(); j++)
            m0(i, j) = sin( 4.*M_PI*j*dx)*cos( 2.*M_PI*i*dy); //sin(kPix)*sin(qPiy)
    cout << "One mode in every line, One mode in every column\n"<< endl;

    t.tic();
    fftw_execute_dft_r2c(plan, m0.getPtr(), fftw_cast(m0.getPtr()));
    t.toc();
    cout << "The transformed matrix "<< t.diff()<< "s\n";
    t.tic();
    fftw_execute_dft_c2r( plan2,fftw_cast(m0.getPtr()), m0.getPtr());
    t.toc();
    cout << "The backtransformed matrix "<<t.diff()<<"s\n";


    return 0;
}
