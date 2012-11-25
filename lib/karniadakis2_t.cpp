#include <iostream>
#include <iomanip>
#include <array>
#include <complex>
#include "karniadakis.h"
#include "dft_drt.h"
#include "matrix.h"


using namespace  std;
using namespace toefl;

typedef std::complex<double> Complex;
//initial mode
const double P = 10;
const double nx = 10;
const double nz = 12; //number of inner points
const double lz = 1.;
const double h = 1./(double)(nz+1);
const double lx = (double)nx*h;
const double dt = 1e-3;
//initialize fields with an Eigenmode of the Benard problem
//such that it lies on the neutral curve (growth rate 0)
/*(that means that the step_ii (EULER) function has to produce the exact
 same coefficients if these are correctly normalized*/
const unsigned ix = 1, iz = 1;
const double k = ix, q = iz +1.;
const double l_kq = -(k*k*4.*M_PI*M_PI/lx/lx + q*q*M_PI*M_PI/lz/lz);
const double dx_k = 2.*M_PI*k/lx;
const double R = -l_kq*l_kq*l_kq/dx_k/dx_k; 
const double omega_0 = 1.;
const double theta_0 = l_kq/dx_k*omega_0;

Karniadakis<2,Complex,TL_DFT_1D> karniadakis( nz,nx,dt);
DFT_DRT dft_drt( nz, nx, FFTW_RODFT00);
std::array< Matrix<Complex, TL_NONE>, 2> cfield = matrix_array<Complex>( nz, nx/2+1);
std::array< Matrix<double, TL_DFT_1D>, 2> field = matrix_array<double, TL_DFT_1D>( nz, nx);
Matrix< QuadMat< Complex, 2>> coefficients( nz, nx/2+1);

void rayleigh_equations( QuadMat< Complex,2>& coeff, const Complex dx, const Complex dy)
{
    double laplace = (dx*dx + dy*dy).real(); 
    coeff(0,0) = laplace, coeff(0,1) = -R*dx/laplace;
    coeff(1,0) = -P*dx  , coeff(1,1) =  P*laplace;
};

int main()
{
    //Construct coefficients and init karniadakis with correct normalisation
    const Complex kxmin { 0, 2.*M_PI/lx}, kzmin{ 0, M_PI/lz};
    for( unsigned i=0; i<nz; i++)
        for( unsigned j=0; j<nx/2+1; j++)
            rayleigh_equations( coefficients(i,j), (double)j*kxmin, (double)(i+1)*kzmin);
    karniadakis.init_coeff( coefficients, (double)(2*nx*(nz+1))); //swaps in coefficients
    //Initialize the complex fields ////////////////////////
    cfield[0].zero();
    cfield[1].zero();
    Complex ctheta_0 = { 0, -0.5*theta_0};
    Complex comega_0 = {0.5*omega_0, 0};
    cfield[0](iz, ix) = ctheta_0;
    cfield[1](iz, ix) = comega_0;
    //Transform and backtransform will multiply input
    dft_drt.c2r( cfield[0], field[0]), dft_drt.c2r( cfield[1], field[1]);
    dft_drt.r2c( field[0], cfield[0]), dft_drt.r2c( field[1], cfield[1]);
    //Hopefully the step_ii function will correctly normalize
    karniadakis.invert_coeff<TL_EULER>();
    karniadakis.step_ii( cfield);
    if( abs( cfield[0](iz,ix) - ctheta_0) > 1e-14 ||
        abs( cfield[1](iz,ix) - comega_0) > 1e-14 )
    {
        cout << "TEST FAILED: \n";
        cout << setprecision(6) << fixed;
        cout << "Rayleigh = "<< R <<endl;
        cout << "-0.5 theta_0 = " <<ctheta_0<< endl
             << " 0.5 omega_0 = "<< comega_0<<endl;
        cout << cfield[0](iz,ix) << endl<< cfield[1](iz,ix)<<endl;
        cout <<scientific<< "Difference: " <<abs( cfield[0](iz,ix) - ctheta_0 )<<endl;
        cout <<scientific<< "Difference: " <<abs( cfield[1](iz,ix) - comega_0 )<<endl;
    }
    else
        cout << "TEST PASSED!\n";
    return 0;
}
