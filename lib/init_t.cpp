#include <iostream>
#include <iomanip>
#include <complex>

#include "init.h"
#include "matrix.h"
#include "dft_dft.h"

using namespace std;
using namespace toefl;

int main()
{
    unsigned rows = 10, cols = 5;
    Matrix<double, TL_DFT> m{ rows, cols, 0.};
    Matrix<complex<double>, TL_NONE> m_{ rows, cols/2+1, complex<double>(0)};
    init_gaussian( m, 0.5,0.5, 0.1,0.1, 1);
    DFT_DFT dft_dft( rows, cols);
    cout << setprecision(4)<<fixed;
    TurbulentBath bath( 10.0);
    double laplace;
    int ik;
    for( unsigned i=0; i<rows; i++)
        for( unsigned j=0; j<cols/2+1; j++)
        {
            ik = (i>=rows/2+1) ? (i-10):i;
            laplace = 4*M_PI*M_PI*(j*j+ik*ik);
            m_(i,j) = {bath( laplace), bath( laplace)};
        }
    cout << "Turbulent coefficients: \n"<<m_<<endl;

    try{
    cout << "Gaussian\n" <<m<<endl;
    dft_dft.c2r( m_, m);}
    catch( Message& m) {m.display();}
    cout << "Turbulence\n"<<m<<endl;
    return 0;
}




