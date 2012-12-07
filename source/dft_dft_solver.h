#include <complex>
#include "toefl.h"
#include "blueprint.h"
#include "equations.h"

namespace toefl
{

/*! @brief Solver for periodic boundary conditions of the toefl equations.
 */
template< size_t n>
class DFT_DFT_Solver
{
  public:
    /*! @brief Construct a solver for periodic boundary conditions
     *
     * The constructor allocates storage for the solver
     * and initializes all fourier coefficients as well as 
     * all low level solver needed.  
     * @param blueprint Contains all the necessary parameters.
     * @throw Message If your parameters are inconsistent.
     */
    DFT_DFT_Solver( const Blueprint& blueprint);
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
    void init( std::array< Matrix<double,TL_DFT>, n>& v, enum target t);
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
    const Blueprint& blueprint() const { return blue;}
  private:
    typedef std::complex<double> complex;
    //methods
    void init_coefficients( const Boundary& bound, const Physical& phys);
    void compute_cphi();//multiply cphi
    void first_steps(); 
    template< enum stepper S>
    void step_();
    //members
    const size_t rows, cols;
    const Blueprint blue;
    /////////////////fields//////////////////////////////////
    GhostMatrix<double, TL_DFT> ghostdens, ghostphi;
    std::array< Matrix<double, TL_DFT>, n> dens, phi, nonlinear;
    /////////////////Complex (void) Matrices for fourier transforms///////////
    std::array< Matrix< complex>, n> cdens, cphi;
    ///////////////////Solvers////////////////////////
    Arakawa arakawa;
    Karniadakis<n, complex, TL_DFT> karniadakis;
    DFT_DFT dft_dft;
    /////////////////////Coefficients//////////////////////
    Matrix< std::array< double, n> > phi_coeff;
    std::array< Matrix< double>, n-1> gamma_coeff;
};

template< size_t n>
DFT_DFT_Solver<n>::DFT_DFT_Solver( const Blueprint& bp):
    rows( bp.getAlgorithmic().ny ), cols( bp.getAlgorithmic().nx ),
    blue( bp),
    //fields
    ghostdens{ rows, cols, TL_PERIODIC, TL_PERIODIC, TL_VOID},
    ghostphi{ ghostdens},
    dens{ MatrixArray<double, TL_DFT,n>::construct( rows, cols)},
    phi{ dens}, nonlinear{ dens},
    cdens{ MatrixArray<complex, TL_NONE, n>::construct( rows, cols/2+1)}, 
    cphi{cdens}, 
    //Solvers
    arakawa( bp.getAlgorithmic().h),
    karniadakis(rows, cols, rows, cols/2+1,bp.getAlgorithmic().dt),
    dft_dft( rows, cols, FFTW_MEASURE),
    //Coefficients
    phi_coeff{ rows, cols/2+1},
    gamma_coeff{ MatrixArray< double, TL_NONE, n-1>::construct( rows, cols/2+1)}
{
    bp.consistencyCheck();
    init_coefficients( bp.getBoundary(), bp.getPhysical());
}

template< size_t n>
void DFT_DFT_Solver<n>::init_coefficients( const Boundary& bound, const Physical& phys)
{
    Matrix< QuadMat< complex, n> > coeff( rows, cols/2+1);
    double laplace;
    int ik;
    const complex dymin( 0, 2.*M_PI/bound.ly);
    const double kxmin2 = 2.*2.*M_PI*M_PI/(double)(bound.lx*bound.lx),
                 kymin2 = 2.*2.*M_PI*M_PI/(double)(bound.ly*bound.ly);
    Equations e( phys);
    Poisson p( phys);
    // dft_dft is not transposing so i is the y index by default
    for( unsigned i = 0; i<rows; i++)
        for( unsigned j = 0; j<cols/2+1; j++)
        {
            ik = (i>rows/2) ? (i-rows) : i; //integer division rounded down
            laplace = - kxmin2*(double)(j*j) - kymin2*(double)(ik*ik);
            if( n == 2)
                gamma_coeff[0](i,j) = p.gamma1_i( laplace);
            else if( n == 3)
            {
                gamma_coeff[0](i,j) = p.gamma1_i( laplace);
                gamma_coeff[1](i,j) = p.gamma1_z( laplace);
            }
            if( rows%2 == 0 && i == rows/2) ik = 0;
            e( coeff( i,j), laplace, (double)ik*dymin);
            if( laplace == 0) continue;
            p( phi_coeff(i,j), laplace);  
        }
        //for periodic bc the constant is undefined
    for( unsigned k=0; k<n; k++)
        phi_coeff(0,0)[k] = 0;
    //coeff( 0,0).zero();
    //std::cout << coeff<<std::endl;
    karniadakis.init_coeff( coeff, (double)(rows*cols));
}
template< size_t n>
void DFT_DFT_Solver<n>::init( std::array< Matrix<double, TL_DFT>,n>& v, enum target t)
{ 
    //fourier transform input into cdens
    for( unsigned k=0; k<n; k++)
    {
#ifdef TL_DEBUG
        if( v[k].isVoid())
            throw Message("You gave me a void Matrix!!", ping);
#endif
        dft_dft.r2c( v[k], cdens[k]);
    }
    //don't forget to normalize coefficients!!
    for( unsigned k=0; k<n; k++)
        for( unsigned i=0; i<rows; i++)
            for( unsigned j=0; j<cols/2+1;j++)
                cdens[k](i,j) /= (double)(rows*cols);
    switch( t) //which field must be computed?
    {
        case( TL_ELECTRONS): 
            //bring cdens and cphi in the right order
            swap_fields( cphi[0], cdens[n-1]);
            for( unsigned k=n-1; k>0; k--)
                swap_fields( cdens[k], cdens[k-1]);
            //now solve for cdens[0]
            for( unsigned i=0; i<rows; i++)
                for( unsigned j=0; j<cols/2+1; j++)
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
            for( unsigned i=0; i<rows; i++)
                for( unsigned j=0; j<cols/2+1; j++)
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
            for( unsigned i=0; i<rows; i++)
                for( unsigned j=0; j<cols/2+1; j++)
                {
                    cdens[2](i,j) = cphi[0](i,j) /phi_coeff(i,j)[2];
                    for( unsigned k=0; k<n && k!=2; k++) 
                        cdens[2](i,j) -= cdens[k](i,j)*phi_coeff(i,j)[k]/phi_coeff(i,j)[2];
                }
            break;
        case( TL_POTENTIAL):
            //solve for cphi
            //std::cout << "ping\n";
            for( unsigned i=0; i<rows; i++)
                for( unsigned j=0; j<cols/2+1; j++)
                {
                    cphi[0](i,j) = 0;
                    for( unsigned k=0; k<n && k!=2; k++) 
                        cphi[0](i,j) += cdens[k](i,j)*phi_coeff(i,j)[k];
                }
            break;
    }
    //compute the rest cphi[k]
    for( unsigned k=0; k<n-1; k++)
        for( size_t i = 0; i < rows; i++)
            for( size_t j = 0; j < cols/2 + 1; j++)
                cphi[k+1](i,j) = gamma_coeff[k](i,j)*cphi[0](i,j);
    //backtransform to x-space
    for( unsigned k=0; k<n; k++)
    {
        dft_dft.c2r( cdens[k], dens[k]);
        dft_dft.c2r( cphi[k], phi[k]);
    }
    //now the density and the potential is given in x-space
    first_steps();
}

template< size_t n>
void DFT_DFT_Solver<n>::getField( Matrix<double, TL_DFT>& m, enum target t)
{
#ifdef TL_DEBUG
    if(m.isVoid()) 
        throw Message( "You may not swap in a void Matrix!\n", ping);
#endif
    switch( t)
    {
        case( TL_ELECTRONS):    swap_fields( m, nonlinear[0]); break;
        case( TL_IONS):         swap_fields( m, nonlinear[1]); break;
        case( TL_IMPURITIES):   swap_fields( m, nonlinear[2]); break;
        case( TL_POTENTIAL):    swap_fields( m, cphi[0]); break;
    }
}
template< size_t n>
const Matrix<double, TL_DFT>& DFT_DFT_Solver<n>::getField( enum target t) const
{
    Matrix<double, TL_DFT> const * m;
    switch( t)
    {
        case( TL_ELECTRONS):    m = &dens[0]; break;
        case( TL_IONS):         m = &dens[1]; break;
        case( TL_IMPURITIES):   m = &dens[2]; break;
        case( TL_POTENTIAL):    m = &phi[0]; break;
    }
    return *m;
}

template< size_t n>
void DFT_DFT_Solver<n>::first_steps()
{
    karniadakis.template invert_coeff<TL_EULER>( );
    step_<TL_EULER>();
    karniadakis.template invert_coeff<TL_ORDER2>();
    step_<TL_ORDER2>();
    karniadakis.template invert_coeff<TL_ORDER3>();
    step_<TL_ORDER3>();
}

template< size_t n>
void DFT_DFT_Solver<n>::compute_cphi()
{
    if( n==2)
    {
        for( size_t i = 0; i < rows; i++)
            for( size_t j = 0; j < cols/2 + 1; j++)
                cphi[0](i,j) = phi_coeff(i,j)[0]*cdens[0](i,j) 
                             + phi_coeff(i,j)[1]*cdens[1](i,j);
        for( size_t i = 0; i < rows; i++)
            for( size_t j = 0; j < cols/2 + 1; j++)
                cphi[1](i,j) = gamma_coeff[0](i,j)*cphi[0](i,j);
    }
    else if( n==3)
    {
        for( size_t i = 0; i < rows; i++)
            for( size_t j = 0; j < cols/2 + 1; j++)
                cphi[0](i,j) = phi_coeff(i,j)[0]*cdens[0](i,j) 
                             + phi_coeff(i,j)[1]*cdens[1](i,j) 
                             + phi_coeff(i,j)[2]*cdens[2](i,j);
        for( size_t i = 0; i < rows; i++)
            for( size_t j = 0; j < cols/2 + 1; j++)
            {
                cphi[1](i,j) = gamma_coeff[0](i,j)*cphi[0](i,j);
                cphi[2](i,j) = gamma_coeff[1](i,j)*cphi[0](i,j);
            }
    }
}

template< size_t n>
template< enum stepper S>
void DFT_DFT_Solver<n>::step_()
{
    //1. Compute nonlinearity
    for( unsigned j=0; j<n; j++)
    {
        swap_fields( dens[j], ghostdens); //now dens[j] is void
        swap_fields( phi[j], ghostphi); //now phi[j] is void
        ghostdens.initGhostCells( );
        ghostphi.initGhostCells(  );
        arakawa( ghostdens, ghostphi, nonlinear[j]);
        swap_fields( dens[j], ghostdens); //now ghostdens is void
        swap_fields( phi[j], ghostphi); //now ghostphi is void
    }
    //2. perform karniadakis step
    karniadakis.template step_i<S>( dens, nonlinear);
    //3. solve linear equation
    //3.1. transform v_hut
    for( unsigned j=0; j<n; j++)
        dft_dft.r2c( dens[j], cdens[j]);
    //3.2. perform karniadaksi step and multiply coefficients for phi
    karniadakis.step_ii( cdens);
    compute_cphi();
    //3.3. backtransform
    for( unsigned j=0; j<n; j++)
    {
        dft_dft.c2r( cdens[j], dens[j]);
        dft_dft.c2r( cphi[j],  phi[j]);
    }
}


} //namespace toefl
