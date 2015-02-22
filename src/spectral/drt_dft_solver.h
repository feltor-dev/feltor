#ifndef _DRT_DFT_SOLVER_
#define _DRT_DFT_SOLVER_

#include <complex>

#include "toefl/toefl.h"
#include "blueprint.h"
#include "equations.h"

namespace toefl
{

/*! @brief Solver for dirichlet type x-boundary conditions of the toefl equations.
 * @ingroup solvers
 */
template< size_t n>
class DRT_DFT_Solver
{
  public:
    typedef Matrix<double, TL_DRT_DFT> Matrix_Type;
    /*! @brief Construct a solver for dirichlet type boundary conditions
     *
     * The constructor allocates storage for the solver
     * and initializes all fourier coefficients as well as 
     * all low level solvers needed.  
     * @param blueprint Contains all the necessary parameters.
     * @throw Message If your parameters are inconsistent.
     */
    DRT_DFT_Solver( const Blueprint& blueprint);
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
    void init( std::array< Matrix<double,TL_DRT_DFT>, n>& v, enum target t);
    /**
     * @brief Perform first initializing step
     *
     */
    void first_step(); 
    /**
     * @brief Perform second initializing step
     *
     * After that the step function can be used
     */
    void second_step(); 
    /*! @brief Perform a step by the 3 step Karniadakis scheme
     *
     * @attention At least one call of first_step() and second_step() is necessary
     * */
    void step(){ step_<TL_ORDER3>();}
    /*! @brief Get the result
        
        You get the solution matrix of the current timestep.
        @param t The field you want
        @return A Read only reference to the field
        @attention The reference is only valid until the next call to 
            the step() function!
    */
    const Matrix<double, TL_DRT_DFT>& getField( enum target t) const;
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
    void getField( Matrix<double, TL_DRT_DFT>& m, enum target t);
    /*! @brief Get the parameters of the solver.

        @return The parameters in use. 
        @note You cannot change parameters once constructed.
     */
    const Blueprint& blueprint() const { return blue;}
  private:
    typedef std::complex<double> complex;
    //methods
    void init_coefficients( const Boundary& bound, const Physical& phys);
    void compute_cphi();//multiply cphi
    //void first_steps(); 
    template< enum stepper S>
    void step_();
    //members
    const size_t rows, cols;
    const size_t crows, ccols;
    const Blueprint blue;
    /////////////////fields//////////////////////////////////
    //GhostMatrix<double, TL_DRT_DFT> ghostdens, ghostphi;
    std::array< Matrix<double, TL_DRT_DFT>, n> dens, phi, nonlinear;
    /////////////////Complex (void) Matrices for fourier transforms///////////
    std::array< Matrix< complex>, n> cdens, cphi;
    ///////////////////Solvers////////////////////////
    Arakawa arakawa;
    Karniadakis<n, complex, TL_DRT_DFT> karniadakis;
    DRT_DFT drt_dft;
    /////////////////////Coefficients//////////////////////
    Matrix< std::array< double, n> > phi_coeff;
    std::array< Matrix< double>, n-1> gamma_coeff;
};

template< size_t n>
DRT_DFT_Solver<n>::DRT_DFT_Solver( const Blueprint& bp):
    rows( bp.algorithmic().ny ), cols( bp.algorithmic().nx ),
    crows( cols), ccols( rows/2+1),
    blue( bp),
    //fields
    dens( MatrixArray<double, TL_DRT_DFT,n>::construct( rows, cols)),
    phi( dens), nonlinear( dens),
    cdens( MatrixArray<complex, TL_NONE, n>::construct( crows, ccols)), 
    cphi(cdens), 
    //Solvers
    arakawa( bp.algorithmic().h),
    karniadakis(rows, cols, crows, ccols, bp.algorithmic().dt),
    drt_dft( rows, cols, fftw_convert( bp.boundary().bc_x), FFTW_MEASURE),
    //Coefficients
    phi_coeff( crows, ccols),
    gamma_coeff( MatrixArray< double, TL_NONE, n-1>::construct( crows, ccols))
{
    bp.consistencyCheck();
    Physical phys = bp.physical();
    if( bp.isEnabled( TL_GLOBAL))
    {
        std::cerr << "WARNING: GLOBAL solver not implemented yet! \n\
             Switch to local solver...\n";
    }
    init_coefficients( bp.boundary(), phys);
}

//aware of BC
template< size_t n>
void DRT_DFT_Solver<n>::init_coefficients( const Boundary& bound, const Physical& phys)
{
    Matrix< QuadMat< complex, n> > coeff( crows, ccols);
    double laplace;
    const complex dymin( 0, 2.*M_PI/bound.ly);
    const double kxmin2 = M_PI*M_PI/(double)(bound.lx*bound.lx),
                 kymin2 = 4.*M_PI*M_PI/(double)(bound.ly*bound.ly);
    double add;
    if( bound.bc_x == TL_DST00 || bound.bc_x == TL_DST10)
        add = 1.0;
    else
        add = 0.5;

    Equations e( phys, blue.isEnabled( TL_MHW));
    Poisson p( phys);
    // drt_dft is transposing so i is the x index 
    for( unsigned i = 0; i<crows; i++)
        for( unsigned j = 0; j<ccols; j++)
        {
            laplace = - kxmin2*(double)((i+add)*(i+add)) - kymin2*(double)(j*j);
            if( n == 2)
                gamma_coeff[0](i,j) = p.gamma1_i( laplace);
            else if( n == 3)
            {
                gamma_coeff[0](i,j) = p.gamma1_i( laplace);
                gamma_coeff[1](i,j) = p.gamma1_z( laplace);
            }
            e( coeff( i,j), laplace, (double)j*dymin);
            p( phi_coeff(i,j), laplace);  
        }
    double norm = fftw_normalisation( bound.bc_x, cols)*(double)rows;
    karniadakis.init_coeff( coeff, norm);
}
//unaware of BC except FFT 
template< size_t n>
void DRT_DFT_Solver<n>::init( std::array< Matrix<double, TL_DRT_DFT>,n>& v, enum target t)
{ 
    //fourier transform input into cdens
    for( unsigned k=0; k<n; k++)
    {
#ifdef TL_DEBUG
        if( v[k].isVoid())
            throw Message("You gave me a void Matrix!!", _ping_);
#endif
        drt_dft.r2c_T( v[k], cdens[k]);
    }
    //don't forget to normalize coefficients!!
    double norm = fftw_normalisation( blue.boundary().bc_x, cols)*(double)rows;
    for( unsigned k=0; k<n; k++)
        for( unsigned i=0; i<crows; i++)
            for( unsigned j=0; j<ccols;j++)
                cdens[k](i,j) /= norm;
    switch( t) //which field must be computed?
    {
        case( TL_ELECTRONS): 
            //bring cdens and cphi in the right order
            swap_fields( cphi[0], cdens[n-1]);
            for( unsigned k=n-1; k>0; k--)
                swap_fields( cdens[k], cdens[k-1]);
            //now solve for cdens[0]
            for( unsigned i=0; i<crows; i++)
                for( unsigned j=0; j<ccols; j++)
                {
                    cdens[0](i,j) = cphi[0](i,j)/phi_coeff(i,j)[0];
                    for( unsigned k=0; k<n && k!=0; k++)
                        cdens[0](i,j) -= cdens[k](i,j)*phi_coeff(i,j)[k]/phi_coeff(i,j)[0];
                }
            break;
        case( TL_IONS):
            //bring cdens and cphi in the right order
            swap_fields( cphi[0], cdens[n-1]);
            for( unsigned k=n-1; k>1; k--)
                swap_fields( cdens[k], cdens[k-1]);
            //solve for cdens[1]
            for( unsigned i=0; i<crows; i++)
                for( unsigned j=0; j<ccols; j++)
                {
                    cdens[1](i,j) = cphi[0](i,j) /phi_coeff(i,j)[1];
                    for( unsigned k=0; k<n && k!=1; k++) 
                        cdens[1](i,j) -= cdens[k](i,j)*phi_coeff(i,j)[k]/phi_coeff(i,j)[1];
                }
            break;
        case( TL_IMPURITIES):
            //bring cdens and cphi in the right order
            swap_fields( cphi[0], cdens[n-1]);
            for( unsigned k=n-1; k>2; k--) //i.e. never for n = 3
                swap_fields( cdens[k], cdens[k-1]);
            //solve for cdens[2]
            for( unsigned i=0; i<crows; i++)
                for( unsigned j=0; j<ccols; j++)
                {
                    cdens[2](i,j) = cphi[0](i,j) /phi_coeff(i,j)[2];
                    for( unsigned k=0; k<n && k!=2; k++) 
                        cdens[2](i,j) -= cdens[k](i,j)*phi_coeff(i,j)[k]/phi_coeff(i,j)[2];
                }
            break;
        case( TL_POTENTIAL):
            //solve for cphi
            for( unsigned i=0; i<crows; i++)
                for( unsigned j=0; j<ccols/2+1; j++)
                {
                    cphi[0](i,j) = 0;
                    for( unsigned k=0; k<n && k!=2; k++) 
                        cphi[0](i,j) += cdens[k](i,j)*phi_coeff(i,j)[k];
                }
            break;
        case( TL_ALL):
            throw Message( "TL_ALL not treated yet!", _ping_);
    }
    //compute the rest cphi[k]
    for( unsigned k=0; k<n-1; k++)
        for( size_t i = 0; i < crows; i++)
            for( size_t j = 0; j < ccols; j++)
                cphi[k+1](i,j) = gamma_coeff[k](i,j)*cphi[0](i,j);
    //backtransform to x-space
    for( unsigned k=0; k<n; k++)
    {
        drt_dft.c_T2r( cdens[k], dens[k]);
        drt_dft.c_T2r( cphi[k], phi[k]);
    }
    //now the density and the potential is given in x-space
    //first_steps();
}

template< size_t n>
void DRT_DFT_Solver<n>::getField( Matrix<double, TL_DRT_DFT>& m, enum target t)
{
#ifdef TL_DEBUG
    if(m.isVoid()) 
        throw Message( "You may not swap in a void Matrix!\n", _ping_);
#endif
    switch( t)
    {
        case( TL_ELECTRONS):    swap_fields( m, nonlinear[0]); break;
        case( TL_IONS):         swap_fields( m, nonlinear[1]); break;
        case( TL_IMPURITIES):   swap_fields( m, nonlinear[2]); break;
        case( TL_POTENTIAL):    swap_fields( m, cphi[0]); break;
        case( TL_ALL):          throw Message( "TL_ALL not allowed here", _ping_);
    }
}
template< size_t n>
const Matrix<double, TL_DRT_DFT>& DRT_DFT_Solver<n>::getField( enum target t) const
{
    Matrix<double, TL_DRT_DFT> const * m = 0;
    switch( t)
    {
        case( TL_ELECTRONS):    m = &dens[0]; break;
        case( TL_IONS):         m = &dens[1]; break;
        case( TL_IMPURITIES):   m = &dens[2]; break;
        case( TL_POTENTIAL):    m = &phi[0]; break;
        case( TL_ALL):          throw Message( "TL_ALL not allowed here", _ping_);
    }
    return *m;
}

template< size_t n>
void DRT_DFT_Solver<n>::first_step()
{
    karniadakis.template invert_coeff<TL_EULER>( );
    step_<TL_EULER>();
}

template< size_t n>
void DRT_DFT_Solver<n>::second_step()
{
    karniadakis.template invert_coeff<TL_ORDER2>();
    step_<TL_ORDER2>();
    karniadakis.template invert_coeff<TL_ORDER3>();
}

template< size_t n>
void DRT_DFT_Solver<n>::compute_cphi()
{
    if( n==2)
    {
#pragma omp parallel for
        for( size_t i = 0; i < crows; i++)
            for( size_t j = 0; j < ccols; j++)
                cphi[0](i,j) = phi_coeff(i,j)[0]*cdens[0](i,j) 
                             + phi_coeff(i,j)[1]*cdens[1](i,j);
#pragma omp parallel for
        for( size_t i = 0; i < crows; i++)
            for( size_t j = 0; j < ccols; j++)
                cphi[1](i,j) = gamma_coeff[0](i,j)*cphi[0](i,j);
    }
    else if( n==3)
    {
#pragma omp parallel for
        for( size_t i = 0; i < crows; i++)
            for( size_t j = 0; j < ccols; j++)
                cphi[0](i,j) = phi_coeff(i,j)[0]*cdens[0](i,j) 
                             + phi_coeff(i,j)[1]*cdens[1](i,j) 
                             + phi_coeff(i,j)[2]*cdens[2](i,j);
#pragma omp parallel for
        for( size_t i = 0; i < crows; i++)
            for( size_t j = 0; j < ccols; j++)
            {
                cphi[1](i,j) = gamma_coeff[0](i,j)*cphi[0](i,j);
                cphi[2](i,j) = gamma_coeff[1](i,j)*cphi[0](i,j);
            }
    }
}

//unaware of BC except FFT
template< size_t n>
template< enum stepper S>
void DRT_DFT_Solver<n>::step_()
{
    //1. Compute nonlinearity
#pragma omp parallel for 
    for( unsigned k=0; k<n; k++)
    {
        GhostMatrix<double, TL_DRT_DFT> ghostphi{ rows, cols, TL_PERIODIC, blue.boundary().bc_x};
        GhostMatrix<double, TL_DRT_DFT> ghostdens{ rows, cols, TL_PERIODIC, blue.boundary().bc_x};
        swap_fields( dens[k], ghostdens); //now dens[j] is void
        swap_fields( phi[k], ghostphi); //now phi[j] is void
        ghostdens.initGhostCells( );
        ghostphi.initGhostCells(  );
        arakawa( ghostdens, ghostphi, nonlinear[k]);
        swap_fields( dens[k], ghostdens); //now ghostdens is void
        swap_fields( phi[k], ghostphi); //now ghostphi is void
    }
    //2. perform karniadakis step
    karniadakis.template step_i<S>( dens, nonlinear);
    //3. solve linear equation
    //3.1. transform v_hut
#pragma omp parallel for
    for( unsigned k=0; k<n; k++)
        drt_dft.r2c_T( dens[k], cdens[k]);
    //3.2. perform karniadaksi step and multiply coefficients for phi
    karniadakis.step_ii( cdens);
    compute_cphi();
    //3.3. backtransform
#pragma omp parallel for
    for( unsigned k=0; k<n; k++)
    {
        drt_dft.c_T2r( cdens[k], dens[k]);
        drt_dft.c_T2r( cphi[k],  phi[k]);
    }
}
}//namespace toefl

#endif //_DRT_DFT_SOLVER_
