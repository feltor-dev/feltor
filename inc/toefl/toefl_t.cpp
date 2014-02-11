#include <iostream>
#include <iomanip>
#include <array>
#include "toefl.h"
#include "draw/host_window.h"

using namespace  std;
using namespace toefl;


typedef std::complex<double> Complex;

//initial mode
const unsigned iz = 0;
const unsigned ix = 2;
//physical Parameter
const double R = 500000;
const double P = 10;
const double nu = 0.0001;
const double nx = 128;
const double nz = 64; //number of inner points
const double lz = 1.;
const double h = 1./(double)(nz);
const double lx = (double)nx*h;
const double dt = 5e-6;

const enum bc  bc_z = TL_DST10;
const double prefactor = (double)nx*fftw_normalisation( bc_z, nz);

Karniadakis<2,Complex,TL_DFT> karniadakis( nz,nx,nz,nx/2+1,dt);
DFT_DRT dft_drt( nz, nx, FFTW_RODFT10, FFTW_MEASURE);
Arakawa arakawa(h);

void rayleigh_equations( QuadMat< Complex,2>& coeff, const Complex dx, const Complex dy)
{
    double laplace = (dx*dx + dy*dy).real(); 
    //std::cout << laplace<<std::endl;
    coeff(0,0) = laplace - nu*laplace*laplace, coeff(0,1) = -R*dx/laplace;
    coeff(1,0) = -P*dx,                        coeff(1,1) =  P*laplace;
};

inline void laplace_inverse( double& l_inv, const Complex dx, const Complex dy)
{
    l_inv = 1.0/(dx*dx+dy*dy).real();
}

//Fields
auto field      = MatrixArray< double, TL_DFT, 2>::construct( nz, nx);
auto nonlinear  = MatrixArray< double, TL_DFT, 2>::construct( nz, nx);
auto cfield = MatrixArray<Complex, TL_NONE,2>::construct( nz, nx/2+1);
GhostMatrix<double, TL_DFT> ghostfield( nz, nx, bc_z, TL_PERIODIC);
GhostMatrix<double, TL_DFT> phi( nz, nx, bc_z, TL_PERIODIC);
Matrix<Complex, TL_NONE> cphi( nz, nx/2+1);
//Coefficients
Matrix< QuadMat< Complex, 2>> coefficients( nz, nx/2+1);
Matrix< double, TL_NONE>   cphi_coefficients( nz, nx/2+1);

void multiply_coefficients();
template<enum stepper S>
void step();

int main()
{
    ////////////////////////////////glfw//////////////////////////////
    GLFWwindow* w = draw::glfwInitAndCreateWindow( 600, 300, "Behold the convection!");
    draw::RenderHostData render(1,1);
    //////////////////////////////////////////////////////////////////
    const Complex kxmin { 0, 2.*M_PI/lx}, kzmin{ 0, M_PI/lz};
    for( unsigned i=0; i<nz; i++)
        for( unsigned j=0; j<nx/2+1; j++)
        {
            rayleigh_equations( coefficients(i,j), (double)j*kxmin, (double)(i+1)*kzmin);
            laplace_inverse( cphi_coefficients(i,j), (double)j*kxmin, (double)(i+1)*kzmin);
        }
    //init solvers
    karniadakis.init_coeff( coefficients, prefactor); //swaps in coefficients
    //init fields
    cfield[0].zero();
    cfield[1].zero();
    cphi.zero();
    cfield[0](iz, ix) = {0, -1000};
    cfield[1](iz, ix) = {0,-1000};
    multiply_coefficients();
    //cout << cphi<<endl;
    dft_drt.c2r( cfield[0], field[0]);
    dft_drt.c2r( cfield[1], field[1]);
    dft_drt.c2r( cphi, phi);
    //cout << phi<<endl;
    //first steps
    karniadakis.invert_coeff<TL_EULER>();
    step<TL_EULER>();
    karniadakis.invert_coeff<TL_ORDER2>();
    step<TL_ORDER2>();
    karniadakis.invert_coeff<TL_ORDER3>();
    step<TL_ORDER3>();
    //////////////////////////////////////////////////////////////////
    cout << "PRESS N TO SEE THE NEXT TIME STEP\n";
    draw::ColorMapRedBlueExt colors( R);
    std::vector<double> visual;
    while( !glfwWindowShouldClose( w))
    {
        visual = field[0].copy();
        render.renderQuad( visual, nz, nx, colors);
        glfwSwapBuffers( w ); 
        //Now wait until a key is pressed
        glfwWaitEvents(); //=glfwPollEvents() when an Event comes: reacts e.g. on mouse mvt or keyboard input
        if( glfwGetKey(w ,'N'))
        {
            step<TL_ORDER3>();
            cout << "Next Step\n";
        }
        static int i=0; 
        cout << "Event #"<<i++<<" occured!\n ";
    }
    glfwTerminate();
    //////////////////////////////////////////////////////////////////
    fftw_cleanup();
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
    phi.initGhostCells( );
    for( unsigned i=0; i<2; i++)
    {
        swap_fields( field[i], ghostfield);// now field is void
        ghostfield.initGhostCells( );
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
    dft_drt.c2r( cphi, phi); //field in phi again
}


