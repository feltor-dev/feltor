#include <iostream>
#include <iomanip>
#include <complex>
#include "dft_dft.h"
#include "ghostmatrix.h"
#include "arakawa.h"
#include "matrix.h"

using namespace toefl;
using namespace std;

const unsigned rows = 3, cols = 10;
const double h = 1./cols;
int main()
{
    GhostMatrix<double, TL_DFT> lhs( rows, cols, TL_DST10, TL_PERIODIC), rhs( rows, cols, TL_DST10, TL_PERIODIC);
    Matrix<std::complex<double>> clhs( rows, cols/2+1), crhs( rows, cols);
    Matrix<std::complex<double>> cjac( rows, cols/2+1);
    Matrix<std::complex<double>> cjac_exact( rows, cols/2+1);
    Matrix<double, TL_DFT> jac( rows, cols);
    Arakawa arakawa(h);

    DFT_DFT dft_dft( rows, cols);

    clhs( 0, 1) = {0,-0.5};//one sine mode in x sin(2*Pi*x)
    cjac_exact( 0,1) = std::complex<double>(0,2*M_PI)*clhs(0,1); //the derivative

    dft_dft.c2r( clhs, lhs);
    
    //for( int i=-1; i<(int)rows+1; i++)
    //    for( int j=-1; j<(int)cols+1; j++)
    for( int i=0; i<(int)rows; i++)
        for( int j=-1; j<(int)cols+1; j++)
            rhs.at(i,j) = (double)(2*i+1)/6.;//(double)i*h; //rhs( y,x ) = y
    lhs.initGhostCells( );
    rhs.initGhostCells( ); //in the first case ghost cells are already initialized

    cout<< "Test whether d/dx(sin(2Pi x) ) is calculated correctly by arakawa scheme\n";

    //cout << lhs << endl << rhs <<endl;
    //lhs.display(cout);
    //cout << endl;
    //rhs.display(cout);
    //cout << endl;
    arakawa( lhs, rhs, jac);
    cout << jac <<endl;
    dft_dft.r2c( jac, cjac);
    //cout << cjac <<endl;
    cout << setprecision(6) <<scientific;
    double norm = (double)rows*cols;
    //cout << cjac(0,1)/norm<< endl;
    //cout << cjac_exact(0,1)<<endl;
    cout << "Difference with " << cols<< " cells: "<< (cjac(0,1)/norm - cjac_exact(0,1))<<endl;
    return 0;
}
