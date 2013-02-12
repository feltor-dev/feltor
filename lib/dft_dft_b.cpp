#include <iostream>
#include <iomanip>
#include "dft_dft.h"
#include "timer.h"
#include "matrix_array.h"
#include "omp.h"


using namespace std;
using namespace toefl;

unsigned rows = 512, cols = 4*512;
int main()
{
    const unsigned kmax = 2;
    omp_set_num_threads(kmax);
    cout << "Test with "<<omp_get_max_threads()<< " threads\n";
    cout << "Size of one array element: " << sizeof( Matrix<double, TL_DFT>) <<"\n";
    auto m1 = MatrixArray<double, TL_DFT, kmax>::construct(rows, cols);
    auto m1_ = MatrixArray<complex<double>, TL_NONE, kmax >::construct( rows, cols/2 + 1);
    double a[100];
    Matrix<double, TL_DFT> mm( rows, cols);
    Matrix<complex<double>, TL_NONE> mm_( rows, cols/2+1);
    Timer t;
    DFT_DFT dft_dft( rows,cols, FFTW_MEASURE);
    DFT_DFT dft_dft2( rows,cols, FFTW_MEASURE);
    //double dx = 1./(cols), dy = 1./rows;
    for( unsigned k=0; k<kmax; k++)
    for( size_t i = 0; i < m1[0].rows(); i++)
        for ( size_t j=0; j < m1[0].cols(); j++)
            mm(i,j) = m1[k](i, j) = i * i + j - 17;//sin( 4.*M_PI*j*dx)*cos( 2.*M_PI*i*dy); //sin(kPix)*sin(qPiy)
#pragma omp parallel
    {
#pragma omp master
    t.tic();
#pragma omp sections 
    {
#pragma omp section
        dft_dft.r2c( m1[0], m1_[0]);
#pragma omp section
        dft_dft.r2c( mm, mm_);
    }
#pragma omp master
    t.toc();
    }
    cout << "Transformation of "<<kmax<<" matrices "<< t.diff()<< "s\n";
    t.tic();
    dft_dft.c2r( m1_[0], m1[0]);
    t.toc();
    cout << "Backtransformation of 1 matrix "<<t.diff()<<"s\n";


    // inplace is faster than out of place
    // r2c_2d is faster than r2c_1d, transpose, c2c_1d
    Matrix<double, TL_DFT> m0(rows, cols);
    Matrix<double, TL_DFT> m2(rows, cols);
    Matrix<complex<double> >   m0_(rows, cols/2 + 1);
    fftw_plan plan = fftw_plan_dft_r2c_2d( rows, cols, m0.getPtr(), fftw_cast(m0.getPtr()), FFTW_MEASURE);
    fftw_plan plan2 = fftw_plan_dft_c2r_2d( rows, cols, fftw_cast(m0.getPtr()), m0.getPtr(), FFTW_MEASURE);
    for( size_t i = 0; i < m0.rows(); i++)
        for ( size_t j=0; j < m0.cols(); j++)
            m0(i, j) = i * i + j - 17;//sin( 4.*M_PI*j*dx)*cos( 2.*M_PI*i*dy); //sin(kPix)*sin(qPiy)
    t.tic();
    fftw_execute_dft_r2c(plan, m0.getPtr(), fftw_cast(m0.getPtr()));
    t.toc();
    cout << "Transformation "<< t.diff()<< "s\n";
    t.tic();
    fftw_execute_dft_c2r( plan2,fftw_cast(m0.getPtr()), m0.getPtr());
    t.toc();
    cout << "Backtransformation "<<t.diff()<<"s\n";

    fftw_destroy_plan( plan);
    fftw_destroy_plan( plan2);

    fftw_cleanup();
    return 0;
}
