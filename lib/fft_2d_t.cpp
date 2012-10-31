#include <iostream>
#include <iomanip>
#include "fft_2d.h"



using namespace std;
using namespace toefl;

int main()
{
    Matrix<double, TL_DFT_1D> m1(5, 10);
    Matrix<complex<double>>   m1_( 10/2 + 1, 5);
    DFT_2D dft_2d( 5,10);
    double dx = 1./(10.), dy = 1./5;
    for( size_t i = 0; i < m1.rows(); i++)
        for ( size_t j=0; j < m1.cols(); j++)
            m1(i, j) = sin( 4.*M_PI*j*dx)*cos( 2.*M_PI*i*dy); //sin(kPix)*sin(qPiy)
    cout << setprecision(2) << fixed;
    cout << "One mode in every line, One mode in every column\n"<< m1<< endl;
    dft_2d.execute_dft_2d_r2c( m1, m1_);
    cout << "The transformed matrix\n"<<m1_<<endl;
    try{
        dft_2d.execute_dft_2d_c2r( m1_, m1);
    }catch( Message& m){m.display();}
    cout << "The backtransformed matrix (50 times input)\n"<<m1<<endl;

    return 0;
}
