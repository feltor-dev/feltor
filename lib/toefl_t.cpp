#include <iostream>
#include <iomanip>
#include <array>
#include <GL/glfw.h>
#include "quadmat.h"
#include "arakawa.h"
#include "karniadakis.h"
#include "matrix.h"
#include "ghostmatrix.h"
#include "dft_drt.h"

using namespace  std;
using namespace toefl;


typedef std::complex<double> Complex;

const double R = 500;
const double P = 10;
const double nx = 10;
const double nz = 10; //number of inner points
const double lz = 1.;
const double h = 1./(double)(nz+1);
const double lx = (double)nx*h;
const double dt = 1e-6;
//initial mode
const unsigned iz = 1;
const unsigned ix = 2;

Karniadakis<2,Complex,TL_DFT_1D> karniadakis( nz,nx,dt);
DFT_DRT dft_drt( nz, nx, FFTW_RODFT00, FFTW_MEASURE);
Arakawa arakawa(h);

void rayleigh_equations( QuadMat< Complex,2>& coeff, const Complex dx, const Complex dy)
{
    double laplace = (dx*dx + dy*dy).real(); 
    //std::cout << laplace<<std::endl;
    coeff(0,0) = laplace, coeff(0,1) = -R*dx/laplace;
    coeff(1,0) = -P*dx  , coeff(1,1) =  P*laplace;
};

inline void laplace_inverse( double& l_inv, const Complex dx, const Complex dy)
{
    l_inv = 1.0/(dx*dx+dy*dy).real();
}

//Felder 
std::array< Matrix<double, TL_DFT_1D>, 2>  field = matrix_array<double, TL_DFT_1D>( nz, nx);
std::array< Matrix<double, TL_DFT_1D>, 2>  nonlinear = matrix_array<double,TL_DFT_1D>( nz, nx);
      GhostMatrix<double, TL_DFT_1D>       ghostfield( nz, nx);
      GhostMatrix<double, TL_DFT_1D>       phi( nz, nx);
//Complex fields
std::array< Matrix<Complex, TL_NONE>, 2>    cfield = matrix_array<Complex>( nz, nx/2+1);
            Matrix<Complex, TL_NONE>        cphi( nz, nx/2+1);
//Coefficients
Matrix< QuadMat< Complex, 2>> coefficients( nz, nx/2+1);
Matrix< double, TL_NONE>   cphi_coefficients( nz, nx/2+1);

void multiply_coefficients();
template<enum stepper S>
void step();

int main()
{
    //////////////////////////////// glfw//////////////////////////////
    int running = GL_TRUE;
    if( !glfwInit()) { cerr << "ERROR: glfw couldn't initialize.\n";}
    if( !glfwOpenWindow( 300, 300,  0,0,0,  0,0,0, GLFW_WINDOW))
    { 
        cerr << "ERROR: glfw couldn't open window!\n";
    }
    //////////////////////////////////////////////////////////////////
    const Complex kxmin { 0, 2.*M_PI/lx},         kzmin{ 0, M_PI/lz};
    for( unsigned i=0; i<nz; i++)
        for( unsigned j=0; j<nx/2+1; j++)
        {
            rayleigh_equations( coefficients(i,j), (double)j*kxmin, (double)(i+1)*kzmin);
            laplace_inverse( cphi_coefficients(i,j), (double)j*kxmin, (double)(i+1)*kzmin);
        }
    //init solvers
    karniadakis.init_coeff( coefficients); //swaps in coefficients
    //init fields
    cfield[0].zero();
    cfield[1].zero();
    cphi.zero();
    cfield[0](iz, nx/2+1-ix) = cfield[0](iz, ix) = {0, -0.5};
    cfield[1](iz, nx/2+1-ix) = cfield[1](iz, ix) = {0,-0.5};
    multiply_coefficients();
    dft_drt.c2r( cfield[0], field[0]);
    dft_drt.c2r( cfield[1], field[1]);
    dft_drt.c2r( cphi, phi);
    field[1].zero();
    phi.zero();
    //first steps
    cout << setprecision(2)<<fixed;
    karniadakis.invert_coeff<TL_EULER>();
    cout << field[0] <<endl;
    cout << field[1] <<endl;
    step<TL_EULER>();
    cout << field[0] <<endl;
    cout << field[1] <<endl;
    step<TL_EULER>();
    cout << field[0] <<endl;
    cout << field[1] <<endl;
    //////////////////////////////////////////////////////////////////
    while( running)
    {
        glLoadIdentity();
        glBegin(GL_QUADS);
        int scale_z = 1.0;
        glClearColor(1.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        glLoadIdentity();
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0, -1.0*scale_z);
        glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0, -1.0*scale_z);
        glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0,  scale_z);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0,  scale_z);
        glEnd();
        glClear( GL_COLOR_BUFFER_BIT);
        glfwSwapBuffers();
        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    glfwTerminate();
    //////////////////////////////////////////////////////////////////
    return 0;
}


void multiply_coefficients()
{
    for( unsigned i=0; i<cphi.rows(); i++)
        for( unsigned j=0; j<cphi.cols(); j++)
            cphi(i,j) = cphi_coefficients(i,j)*cfield[1](i,j); //double - complex Mult.
}

template< enum stepper S>
void step()
{
    //std::cout << phi << std::endl<< field[0] << std::endl<< field[1];
    phi.initGhostCells( TL_DST00, TL_PERIODIC);
    for( unsigned i=0; i<2; i++)
    {
        swap_fields( field[i], ghostfield);// now field is void
        ghostfield.initGhostCells( TL_DST00, TL_PERIODIC);
        arakawa( phi, ghostfield, nonlinear[i]);
        swap_fields( field[i], ghostfield);// now ghostfield is void
    }
    karniadakis.step_i<S>( field, nonlinear);
    for( unsigned i=0; i<2; i++)
        dft_drt.r2c( field[i], cfield[i]);
    karniadakis.step_ii( cfield);
    swap_fields( cphi, phi); //now phi is void
    multiply_coefficients();
    for( unsigned i=0; i<2; i++)
        dft_drt.c2r( cfield[i], field[i]);
    dft_drt.c2r( cphi, phi); //field in ghosphi again
}


