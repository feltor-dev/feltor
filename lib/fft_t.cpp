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
    cout << "Generating a reference case:\n";
    Matrix<double, TL_DFT_2D> m0(5, 10);
    Matrix<complex<double>>   m0_( 5, 10/2+1, TL_VOID);

    fftw_plan plan_f = fftw_plan_dft_r2c_2d( 5, 10, m0.getPtr(), reinterpret_cast<fftw_complex*>(m0.getPtr()), FFTW_ESTIMATE);
    fftw_plan plan_b = fftw_plan_dft_c2r_2d( 5, 10, reinterpret_cast<fftw_complex*>(m0.getPtr()),m0.getPtr(), FFTW_ESTIMATE);

    cout << "Test of the fft wrapper routines\n";
    cout << "Construct a real Matrix (5,10)\n";
    Matrix<double, TL_DFT_1D> m1(5, 10);
    Matrix<complex<double>>   m1_( 10/2+1, 5);
    DFT_2D dft_2d( 5,10);
    double dx = 1./(10.), dy = 1./5;
    for( size_t i = 0; i < m1.rows(); i++)
        for ( size_t j=0; j < m1.cols(); j++)
            m0(i,j) = m1(i, j) = sin( 4.*M_PI*j*dx)*cos( 2.*M_PI*i*dy); //sin(kPix)*sin(qPiy)
    cout <<fixed <<setprecision(2)<< m1 << endl;
    cout << "The output should show one mode \n";
    fftw_execute( plan_f);
    cout << "Reference output\n";
    swap_fields( m0, m0_);
    cout << m0_ <<endl;
    swap_fields( m0, m0_);
    try{dft_2d.execute_dft_2d_r2c( m1, m1_);}
    catch( Message& m){m.display();}
    cout << "New function output\n";
    cout << m1_<< endl;
    cout << "Backtrafo\n";
    cout << "Output should be input times 50\n";
    fftw_execute(plan_b);
    cout << "Reference output\n";
    cout << m0 <<endl;
    dft_2d.execute_dft_2d_c2r( m1_, m1);
    cout << "New function output\n";
    cout << m1 <<endl;

    /*
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
    */


    return 0;
}



    






