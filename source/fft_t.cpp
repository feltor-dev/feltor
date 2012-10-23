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
    cout << "Construct a real Matrix (2,9)\n";
    cout << "That means 9 inner points i.e. 10 intervalls and 11 grid points\n";
    Matrix<double> m1(2, 9);
    cout << "Make the plan befor initialization!\n";
    FFT_MANY_1D_SINE sine_trafo( m1);
    double dx = 1./(8.+2.);
    for( size_t i = 0; i < m1.rows(); i++)
        for ( size_t j=0; j < m1.cols(); j++)
            m1(i, j) = sin( (i+1)*M_PI*(j+1)*dx); //sin(kPix)
    cout <<fixed <<setprecision(2)<< m1 << endl;
    cout << "Every line has on sine Mode with k = (i+1)\n";
    cout << "The output should show one mode each \n";
    sine_trafo.r2r( m1);
    cout << m1<< endl;
    cout << "Backtrafo\n";
    cout << "Output should be input times 20\n";
    sine_trafo.r2r( m1);
    cout << m1 <<endl;

    cout << "Test 2\n";
    Matrix<double, TL_FFT_1D> m2(2,11);
    Matrix<complex<double>, TL_NONE> m2_( 2, 6, TL_VOID);
    FFT_MANY_1D<complex<double>> fourier_trafo( m2);

    dx = 1./11.;
    for( size_t i = 0; i < m2.rows(); i++)
        for ( size_t j=0; j < m2.cols(); j++)
            m2(i, j) = cos( 2*(i+1)*M_PI*j*dx); 
    cout << "Trafo of:\n";
    cout << m2 <<endl;
    cout << "forward trafo\n";
    fourier_trafo.r2c( m2);
    cout << m2 <<endl;
    swap_matrices( m2, m2_);
    cout << "result should be one cosine mode\n";
    cout << m2_ <<endl;

    try{
    fourier_trafo.c2r( m2_);}
    catch(Message& m){m.display();}
    swap_matrices( m2, m2_);
    cout << "backtrafo\n";
    cout << m2 <<endl;


    return 0;
}



    






