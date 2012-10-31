//deprecated
#include "fft.h"
#include "fft_2d.h"
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
    Matrix<double, TL_DFT_1D> m2(1,10);
    Matrix<complex<double>, TL_NONE> m2_( 1, 6, TL_VOID);
    fftw_plan forward_plan = plan_dft_1d_r2c( m2);
    swap_fields( m2, m2_);
    fftw_plan backward_plan = plan_dft_1d_c2r( m2_, m2.cols()%2);
    swap_fields( m2, m2_);

    dx = 1./10.;
    for( size_t i = 0; i < m2.rows(); i++)
        for ( size_t j=0; j < m2.cols(); j++)
            m2(i, j) = cos( 2*(i+1)*M_PI*j*dx); 
    cout << "Trafo of:\n";
    cout << m2 <<endl;
    cout << "forward trafo\n";
    execute_dft_1d_r2c( forward_plan, m2, m2_);
    cout << "result should be one cosine mode\n";
    cout << m2_ <<endl;

    try{
    execute_dft_1d_c2r( backward_plan, m2_, m2);}
    catch(Message& m){m.display();}
    cout << "backtrafo\n";
    cout << m2 <<endl;


    return 0;
}



    






