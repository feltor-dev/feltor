#ifndef _CONVECTION_SOLVER_
#define _CONVECTION_SOLVER_

#include <complex>

#include "toefl/toefl.h"

enum target{
    TEMPERATURE, VORTICITY, POTENTIAL
};

struct Parameter
{
    double R; 
    double P; 
    double nu;
    unsigned nx; 
    unsigned nz; 
    double lx; 
    double lz; 
    double h; 
    double dt;
    enum toefl::bc bc_z;
    template< class Ostream>
    void display( Ostream& os)
    {
        os << "R is: "<< R <<"\n";
        os << "P is: "<< P <<"\n";
        os << "nu is: "<<nu  <<"\n";
        os << "nx is: "<< nx <<"\n";
        os << "nz is: "<< nz <<"\n";
        os << "lx is: "<< lx <<"\n";
        os << "lz is: "<< lz <<"\n";
        os << "h is: "<< h <<"\n";
        os << "dt is: "<< dt <<"\n";
    }
};

typedef std::complex<double> Complex;

void rayleigh_equations( toefl::QuadMat< Complex,2>& coeff, const Complex dx, const Complex dy, const Parameter& p)
{
    double laplace = (dx*dx + dy*dy).real();
    coeff( 0,0) = laplace - p.nu*laplace*laplace,  coeff( 0,1) = -p.R*dx/laplace;
    coeff( 1,0) = -p.P*dx,    coeff( 1,1) = p.P*laplace - p.nu*laplace*laplace;
}

inline void laplace_inverse( double& l_inv, const Complex dx, const Complex dy)
{
    l_inv = 1.0/(dx*dx + dy*dy).real();
}
namespace toefl{
/*! @brief Solver for periodic boundary conditions of the toefl equations.
 * @ingroup solvers
 */
class Convection_Solver
{
  public:
    typedef Matrix<double, TL_DFT> Matrix_Type;
    /*! @brief Construct a solver for periodic boundary conditions
     *
     * The constructor allocates storage for the solver
     * and initializes all fourier coefficients as well as 
     * all low level solvers needed.  
     * @param blueprint Contains all the necessary parameters.
     * @throw Message If your parameters are inconsistent.
     */
    Convection_Solver( const Parameter& param);
    /*! @brief Prepare Solver for execution
     *
     * This function takes the fields and computes the missing 
     * one according to the target parameter passed. After that
     * it performs three initializing steps (one onestep-, 
     * one twostep-method and the threestep-method used in the step function)
     * in order to initialize the karniadakis scheme. The actual time is
     * thus T_0 + 3*dt after initialisation. 
     * @param v Container with three non void matrices
     * @param t which Matrix is missing?
     */
    void init( std::array< Matrix<double,TL_DFT>, 2>& v, enum target t);
    /*! @brief Perform a step by the 3 step Karniadakis scheme*/
    void step(){ step_<TL_ORDER3>();}
    /*! @brief Get the result
        
        You get the solution matrix of the current timestep.
        @param t The field you want
        @return A Read only reference to the field
        @attention The reference is only valid until the next call to 
            the step() function!
    */
    const Matrix<double, TL_DFT>& getField( enum target t) const;
    /*! @brief Get the result

        Use this function when you want to call step() without 
        destroying the solution. 
        @param m 
            In exchange for the solution matrix you have to provide
            storage for further calculations. The field is swapped in.
        @param t 
            The field you want. 
        @attention The fields you get are not the ones of the current
            timestep. You get the fields that are not needed any more. 
            This means the densities are 4 timesteps "old" whereas 
            the potential is the one of the last timestep.
    */
    void getField( Matrix<double, TL_DFT>& m, enum target t);
    /*! @brief Get the parameters of the solver.

        @return The parameters in use. 
        @note You cannot change parameters once constructed.
     */
    const Parameter& parameter() const { return param;}
  private:
    typedef std::complex<double> complex;
    //methods
    void init_coefficients( );
    void compute_cphi();//multiply cphi
    void first_steps(); 
    template< enum stepper S>
    void step_();
    //members
    const size_t rows, cols;
    const size_t crows, ccols;
    const Parameter param;
    /////////////////fields//////////////////////////////////
    //GhostMatrix<double, TL_DFT> ghostdens, ghostphi;
    std::array< Matrix_Type, 2> dens, nonlinear;
    GhostMatrix<double, TL_DFT> phi;
    /////////////////Complex (void) Matrices for fourier transforms///////////
    std::array< Matrix< complex>, 2> cdens;
    Matrix< complex> cphi;
    ///////////////////Solvers////////////////////////
    Arakawa arakawa;
    Karniadakis<2, complex, TL_DFT> karniadakis;
    DFT_DRT dft_drt;
    /////////////////////Coefficients//////////////////////
    Matrix< double > phi_coeff;
};

Convection_Solver::Convection_Solver( const Parameter& p):
    rows( p.nz ), cols( p.nx ),
    crows( rows), ccols( cols/2+1),
    param( p),
    //fields
    dens( MatrixArray<double, TL_DFT, 2>::construct( rows, cols)), nonlinear( dens),
    phi( rows, cols, param.bc_z, TL_PERIODIC),
    cdens( MatrixArray<complex, TL_NONE, 2>::construct( crows, ccols)), 
    cphi(crows, ccols), 
    //Solvers
    arakawa( p.h),
    karniadakis(rows, cols, crows, ccols, p.dt),
    dft_drt( rows, cols, fftw_convert( p.bc_z), FFTW_MEASURE),
    //Coefficients
    phi_coeff( crows, ccols)
{
    init_coefficients( );
}

void Convection_Solver::init_coefficients( )
{
    Matrix< QuadMat< complex, 2> > coeff( crows, ccols);
    const complex kxmin( 0, 2.*M_PI/param.lx), kzmin( 0, M_PI/param.lz);
    // dft_drt is not transposing so i is the y index by default
    for( unsigned i = 0; i<crows; i++)
        for( unsigned j = 0; j<ccols; j++)
        {
            rayleigh_equations( coeff( i,j), (double)j*kxmin, (double)(i+1)*kzmin, param);
            laplace_inverse( phi_coeff( i,j), (double)j*kxmin, (double)(i+1)*kzmin);
        }
    double norm = param.nx * fftw_normalisation( param.bc_z, param.nz);
    karniadakis.init_coeff( coeff, norm);
}
void Convection_Solver::init( std::array< Matrix<double, TL_DFT>,2>& v, enum target t)
{ 
    //fourier transform input into cdens
    for( unsigned k=0; k<2; k++)
    {
#ifdef TL_DEBUG
        if( v[k].isVoid())
            throw Message("You gave me a void Matrix!!", ping);
#endif
        dft_drt.r2c( v[k], cdens[k]);
    }
    //don't forget to normalize coefficients!!
    double norm = param.nx * fftw_normalisation( param.bc_z, param.nz);
    for( unsigned k=0; k<2; k++)
        for( unsigned i=0; i<crows; i++)
            for( unsigned j=0; j<ccols;j++)
                cdens[k](i,j) /= norm;
    switch( t) //which field must be computed?
    {
        case( TEMPERATURE): 
            throw Message( "Temperature independent", ping);
            break;
        case( VORTICITY):
            //bring cdens and cphi in the right order
            swap_fields( cphi, cdens[1]);
            //solve for cdens[1]
            for( unsigned i=0; i<crows; i++)
                for( unsigned j=0; j<ccols; j++)
                    cdens[1](i,j) = cphi(i,j) /phi_coeff(i,j);
            break;
        case( POTENTIAL):
            //solve for cphi
            for( unsigned i=0; i<crows; i++)
                for( unsigned j=0; j<ccols; j++)
                {
                    cphi(i,j) = cdens[1](i,j)*phi_coeff(i,j);
                }
            break;
    }
    //backtransform to x-space
    for( unsigned k=0; k<2; k++)
    {
        dft_drt.c2r( cdens[k], dens[k]);
    }
    dft_drt.c2r( cphi, phi);
    //now the density and the potential is given in x-space
    first_steps();
}

void Convection_Solver::getField( Matrix<double, TL_DFT>& m, enum target t)
{
#ifdef TL_DEBUG
    if(m.isVoid()) 
        throw Message( "You may not swap in a void Matrix!\n", ping);
#endif
    switch( t)
    {
        case( TEMPERATURE):    swap_fields( m, nonlinear[0]); break;
        case( VORTICITY):      swap_fields( m, nonlinear[1]); break;
        case( POTENTIAL):      swap_fields( m, cphi); break;
    }
}
const Matrix<double, TL_DFT>& Convection_Solver::getField( enum target t) const
{
    Matrix<double, TL_DFT> const * m = 0;
    switch( t)
    {
        case( TEMPERATURE):     m = &dens[0]; break;
        case( VORTICITY):       m = &dens[1]; break;
        case( POTENTIAL):       m = &phi; break;
    }
    return *m;
}

void Convection_Solver::first_steps()
{
    karniadakis.invert_coeff<TL_EULER>( );
    step_<TL_EULER>();
    karniadakis.invert_coeff<TL_ORDER2>();
    step_<TL_ORDER2>();
    karniadakis.invert_coeff<TL_ORDER3>();
    step_<TL_ORDER3>();
}

void Convection_Solver::compute_cphi()
{
#pragma omp for 
    for( size_t i = 0; i < crows; i++)
        for( size_t j = 0; j < ccols; j++)
            cphi(i,j) = phi_coeff(i,j)*cdens[1](i,j);
}

template< enum stepper S>
void Convection_Solver::step_()
{
    phi.initGhostCells(  );
#pragma omp parallel 
    {
    GhostMatrix<double, TL_DFT> ghostdens{ rows, cols, param.bc_z, TL_PERIODIC, TL_VOID};
    //1. Compute nonlinearity
#pragma omp for 
    for( unsigned k=0; k<2; k++)
    {
        swap_fields( dens[k], ghostdens); //now dens[k] is void
        ghostdens.initGhostCells( );
        arakawa( phi, ghostdens, nonlinear[k]);
        swap_fields( dens[k], ghostdens); //now ghostdens is void
    }
    //2. perform karniadakis step
    karniadakis.step_i<S>( dens, nonlinear);
    //3. solve linear equation
    //3.1. transform v_hut
#pragma omp for 
    for( unsigned k=0; k<2; k++)
        dft_drt.r2c( dens[k], cdens[k]);
    //3.2. perform karniadaksi step and multiply coefficients for phi
    karniadakis.step_ii( cdens);
    compute_cphi();
    //3.3. backtransform
#pragma omp for 
    for( unsigned k=0; k<2; k++)
        dft_drt.c2r( cdens[k], dens[k]);
    } //omp parallel
    dft_drt.c2r( cphi,  phi); //field in phi again
}

}//namespace toefl
#endif //_CONVECTION_SOLVER_
