#include <iostream>
#include <iomanip>
#include "dft_dft.h"



using namespace std;
using namespace toefl;

int main()
{
    Matrix<double, TL_DFT_DFT> m1(5, 10);
    Matrix<complex<double> >   m1_( 5, 10/2 + 1);
    DFT_DFT dft_dft( 5,10);
    double dx = 1./(10.), dy = 1./5;
    for( size_t i = 0; i < m1.rows(); i++)
        for ( size_t j=0; j < m1.cols(); j++)
            m1(i, j) = sin( 4.*M_PI*j*dx)*cos( 2.*M_PI*i*dy); //sin(kPix)*sin(qPiy)
    cout << setprecision(2) << fixed;
    cout << "One mode in every line, One mode in every column\n"<< m1<< endl;
    dft_dft.r2c( m1, m1_);
    cout << "The transformed matrix\n"<<m1_<<endl;
    try{
        dft_dft.c2r( m1_, m1);
    }catch( Message& m){m.display();}
    cout << "The backtransformed matrix (50 times input)\n"<<m1<<endl;

    Matrix<double, TL_DFT_DFT> m0( 5,10);
    Matrix<complex< double> > m0_(5,10/2 + 1);
    fftw_plan plan = fftw_plan_dft_r2c_2d( 5, 10, m0.getPtr(), fftw_cast(m0.getPtr()), FFTW_MEASURE);
    fftw_plan plan2 = fftw_plan_dft_c2r_2d( 5, 10, fftw_cast(m0.getPtr()), m0.getPtr(), FFTW_MEASURE);
    for( size_t i = 0; i < m0.rows(); i++)
        for ( size_t j=0; j < m0.cols(); j++)
            m0(i, j) = sin( 4.*M_PI*j*dx)*cos( 2.*M_PI*i*dy); //sin(kPix)*sin(qPiy)
    cout << m0 <<endl;
    

    fftw_execute( plan);
    fftw_execute( plan2);
    if( m1 != m0)
        cerr << "Transformation failed!\n m1: "<<m1<<"\n m2" <<m0<<endl;
    else
        cout << "Test passed\n";




    return 0;
}
