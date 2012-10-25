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
    cout << "Test of the fft wrapper routines\n";
    cout << "Construct a real Matrix (1,9)\n";
    cout << "That means 9 inner points i.e. 10 intervalls and 11 grid points\n";
    Matrix<double> m1(1, 9);
    cout << "Make the plan befor initialization!\n";
    fftw_plan sine_plan = plan_dst_1d( m1);
    double dx = 1./(8.+2.);
    for( size_t i = 0; i < m1.rows(); i++)
        for ( size_t j=0; j < m1.cols(); j++)
            m1(i, j) = sin( (i+1)*M_PI*(j+1)*dx); //sin(kPix)
    cout <<fixed <<setprecision(2)<< m1 << endl;
    cout << "Every line has on sine Mode with k = (i+1)\n";
    cout << "The output should show one mode each \n";
    execute_dst_1d( sine_plan, m1);
    cout << m1<< endl;
    cout << "Backtrafo\n";
    cout << "Output should be input times 20\n";
    execute_dst_1d( sine_plan, m1);
    cout << m1 <<endl;

    cout << "Test 2\n";
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



    






