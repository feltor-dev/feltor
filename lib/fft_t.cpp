//deprecated
#include "fft.h"
#include "matrix.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <complex>

using namespace std;
using namespace toefl;

int main()
{
    cout << "Test \n";
    double dx;
    size_t rows = 1; 
    size_t cols = 10;
    Matrix<double, TL_DFT> m2(rows,cols);
    Matrix<complex<double>, TL_NONE> m2_( rows, cols/2+1, TL_VOID);
    fftw_plan forward_plan  = plan_dft_1d_r2c(rows, cols, m2.getPtr(), fftw_cast(m2.getPtr()), FFTW_MEASURE);
    fftw_plan backward_plan = plan_dft_1d_c2r(rows, cols, fftw_cast(m2.getPtr()), m2.getPtr(), FFTW_MEASURE);

    dx = 1./10.;
    for( size_t i = 0; i < m2.rows(); i++)
        for ( size_t j=0; j < m2.cols(); j++)
            m2(i, j) = cos( 2*(i+1)*M_PI*j*dx); 
    cout << "Trafo of:\n";
    cout << setprecision(2) << fixed;
    cout << m2 <<endl;
    cout << "forward trafo\n";
    fftw_execute_dft_r2c( forward_plan, m2.getPtr(), fftw_cast(m2.getPtr()));
    swap_fields( m2, m2_);
    cout << "result should be one cosine mode\n";
    cout << m2_ <<endl;
    swap_fields( m2, m2_);

    try{
        fftw_execute_dft_c2r( backward_plan, fftw_cast(m2.getPtr()), m2.getPtr());}
    catch(Message& m){m.display();}
    cout << "backtrafo\n";
    cout << m2 <<endl;


    return 0;
}



    






