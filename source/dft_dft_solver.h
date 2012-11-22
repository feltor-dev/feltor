#include <complex>
#include "tl_numerics.h"
#include "blueprint.h"
#include "equations.h"

namespace toefl
{
    template< size_t n>
    class DFT_DFT_Solver
    {
        typedef std::complex<double> complex;
        const size_t rows, cols;
        const bool imp;
        const double dt;
        Vector< GhostMatrix<double, TL_DFT_DFT>, n > dens, phi;
        Vector< Matrix<double, TL_DFT_DFT>, n > nonlinear;
        /////////////////Complex (void) Matrices for fourier transforms///////////
        Vector< Matrix< complex>, n> cdens, cphi;
        ///////////////////Solvers////////////////////////
        Arakawa arakawa;
        Karniadakis<n, complex, TL_DFT_DFT> k;
        DFT_DFT dft_dft;
        /////////////////////Coefficients//////////////////////
        Matrix< Vector< double, n> > coeff_phi;
        Vector< Matrix< double>, n-1> coeff_Gamma;
        void init_coefficients( const Boundary& bound, const Physical& phys);

        void multiply_coefficients();//multiply phi
        void first_steps(); 
        template< enum stepper S>
        void step_();
      public:
        /*! @brief Construct a solver for periodic boundary conditions
         *
         * @param blueprint Contains all the necessary parameters
         */
        DFT_DFT_Solver( const Blueprint& blueprint);
        /*! @brief Prepare Solver for execution
         *
         * This function initializes the Fourier Coefficients as well as 
         * all low level solver needed. It performs two 
         * initializing steps (by one onestep- and one twostep-method)
         * in order to get the karniadakis scheme ready. The actual time is
         * thus T_0 + 2*dt after initialisation. 
         * @param v Container with three non void matrices
         * @param t which Matrix is missing?
         */
        void init( Vector< Matrix<double,TL_DFT_DFT>, n>& v, enum target t);
        /*! @brief Perform a step by the 3 step Karniadakis scheme*/
        void step(){ step_<TL_ORDER3>();}
        /*! @brief Get the result*/
        void getField( enum target t, const Matrix<double, TL_DFT_DFT>& m);
    };

    template< size_t n>
    DFT_DFT_Solver<n>::DFT_DFT_Solver( const Blueprint& bp):
        rows( bp.getAlgorithmic().ny ), cols( bp.getAlgorithmic().nx ),
        imp( bp.isEnabled( TL_IMPURITY)),
        dt( bp.getAlgorithmic().dt),
        //fields
        //Solvers
        arakawa( bp.getAlgorithmic().h),
        k(rows, cols, dt),
        dft_dft( rows, cols, FFTW_MEASURE),
        coeff_phi( rows, cols/2+1)
    {
        //allocate vectors
        for( unsigned k=0; k<n; k++)
        {
            dens[k].allocate(rows, cols);
            phi[k].allocate( rows, cols);
            nonlinear[k].allocate( rows, cols);
            cdens[k].resize( rows, cols/2 +1);
            cphi[k].resize( rows, cols/2 +1);
        }
        for( unsigned k=0; k<n-1; k++)
        {
            coeff_Gamma[k].allocate( rows, cols/2 + 1);
        }
        bp.consistencyCheck();
        Physical phys = bp.getPhysical();
        if( !bp.isEnabled( TL_CURVATURE))
            phys.kappa_x = phys.kappa_y = 0; 
        init_coefficients(bp.getBoundary(), phys);
    }

    template< size_t n>
    void DFT_DFT_Solver<n>::init_coefficients( const Boundary& bound, const Physical& phys)
    {
        Matrix< QuadMat< complex, n> > coeff( rows, cols/2+1);
        double laplace;
        const complex kxmin ( 0, 2.*M_PI/bound.lx), kymin( 0, 2.*M_PI/bound.ly);
        const double kxmin2 = 2.*2.*M_PI*M_PI/bound.lx/bound.lx,
                     kymin2 = 2.*2.*M_PI*M_PI/bound.ly/bound.ly;
        Equations e( phys);
        Poisson p( phys);
        // ki = 2Pi*i/ly, 
        // dft_dft is not transposing so i is the y index by default
        for( unsigned i = 0; i<rows; i++)
            for( unsigned j = 0; j<cols/2+1; j++)
            {
                laplace = -kxmin2*(double)(j*j) - kymin2*(double)(i*i);
                p( coeff_phi(i,j), laplace);  
                e( coeff( i,j), (double)j*kxmin, (double)i*kymin);
            }
        if( n==2)
            for( unsigned i = 0; i<rows; i++)
                for( unsigned j = 0; j<cols/2+1; j++)
                    coeff_Gamma[0](i,j) = p.gamma1_i( laplace);
        if(n==3)
            for( unsigned i = 0; i<rows; i++)
                for( unsigned j = 0; j<cols/2+1; j++)
                {
                    coeff_Gamma[0](i,j) = p.gamma1_i( laplace);
                    coeff_Gamma[1](i,j) = p.gamma1_z( laplace);
                }
        k.init_coeff( coeff);
    }
    template< size_t n>
    void DFT_DFT_Solver<n>::multiply_coefficients()
    {
        if( n==2)
        {
            for( size_t i = 0; i < rows; i++)
                for( size_t j = 0; j < cols/2 + 1; j++)
                    cphi[0](i,j) = coeff_phi(i,j)[0]*cdens[0](i,j) 
                                 + coeff_phi(i,j)[1]*cdens[1](i,j);
            for( size_t i = 0; i < rows; i++)
                for( size_t j = 0; j < cols/2 + 1; j++)
                    cphi[1](i,j) = coeff_Gamma[0](i,j)*cphi[0](i,j);
        }
        else if( n==3)
        {
            for( size_t i = 0; i < rows; i++)
                for( size_t j = 0; j < cols/2 + 1; j++)
                    cphi[0](i,j) = coeff_phi(i,j)[0]*cdens[0](i,j) 
                                 + coeff_phi(i,j)[1]*cdens[1](i,j) 
                                 + coeff_phi(i,j)[2]*cdens[2](i,j);
            for( size_t i = 0; i < rows; i++)
                for( size_t j = 0; j < cols/2 + 1; j++)
                {
                    cphi[1](i,j) = coeff_Gamma[0](i,j)*cphi[0](i,j);
                    cphi[2](i,j) = coeff_Gamma[1](i,j)*cphi[0](i,j);
                }
        }
    }
    template< size_t n>
    void DFT_DFT_Solver<n>::init( const Vector< Matrix<double, TL_DFT_DFT>,n>& v, enum target t)
    {
        for( unsigned k=0; k<n; k++)
            dft_dft.r2c( v[k], cdens[k]);
        switch( t) //which field must be computed?
        {
            case( TL_ELECTRONS): //to be read again Gammas must be multiplied as well
                throw Message( "Electron feature not implemented yet!\n",ping);
                swap_fields( cphi[0], cdens[n-1]);
                for( unsigned k=n-1; k>0; k--)
                    swap_fields( cdens[k], cdens[k-1]);
                for( unsigned i=0; i<rows; i++)
                    for( unsigned j=0; j<cols/2+1; j++)
                    {
                        cdens[0](i,j) = cphi[0](i,j)/coeff_phi(i,j)[0];
                        for( unsigned k=1; k<n; k++)
                            cdens[0](i,j) -= cdens[k](i,j)*coeff_phi(i,j)[k]/coeff_phi(i,j)[0];
                    }
                break;
            case( TL_IONS):
                throw Message( "Ion feature not implemented yet!\n",ping);
                swap_fields( cphi[0], cdens[n-1]);
                for( unsigned k=n-1; k>1; k--)
                    swap_fields( cdens[k], cdens[k-1]);
                for( unsigned i=0; i<rows; i++)
                    for( unsigned j=0; j<cols/2+1; j++)
                    {
                        cdens[1](i,j) = (cphi[0](i,j) - coeff_phi(i,j)[2]*cdens[0](i,j)) /coeff_phi(i,j)[1];
                        for( unsigned k=2; k<n; k++) 
                            cdens[0](i,j) -= cdens[k](i,j)*coeff_phi(i,j)[k]/coeff_phi(i,j)[0];
                    }
                break;
            case( TL_IMPURITIES):
                throw Message( "Impurity feature not implemented yet!\n",ping);
                swap_fields( cphi[0], cdens[n-1]);
                for( unsigned k=n-1; k>2; k--) //i.e. never for n = 3
                    swap_fields( cdens[k], cdens[k-1]);
                break;
            case( TL_POTENTIAL):
                multiply_coefficients();
                break;
        }
        for( unsigned k=0; k<n; k++)
        {
            dft_dft.c2r( cdens[k], dens[k]);
            dft_dft.c2r( cphi[k], phi[k]);
        }
        //now the density and the potential is given
        first_steps();
    }

    template< size_t n>
    void DFT_DFT_Solver<n>::first_steps()
    {
        k.invert_coeff<TL_EULER>( );
        step_<TL_EULER>();
        k.invert_coeff<TL_ORDER2>( );
        step_<TL_ORDER2>();
        k.invert_coeff<TL_ORDER3>( );
    }

    template< size_t n>
    template< enum stepper S>
    void DFT_DFT_Solver<n>::step_()
    {
        //1. Compute nonlinearity
        for( unsigned j=0; j<n; j++)
        {
            dens[j].initGhostCells( TL_PERIODIC, TL_PERIODIC);
            phi[j].initGhostCells( TL_PERIODIC, TL_PERIODIC);
            arakawa( dens[j], phi[j], nonlinear[j]);
        }
        //2. perform karniadakis step
        k.step_i<S>( dens, nonlinear);
        //3. solve linear equation
        //3.1. transform v_hut
        for( unsigned j=0; j<n; j++)
            dft_dft.r2c( dens[j], cdens[j]);
        //3.2. perform karniadaksi step and multiply coefficients for phi
        k.step_ii( cdens);
        multiply_coefficients();
        //3.3. backtransform
        for( unsigned j=0; j<n; j++)
        {
            dft_dft.c2r( cdens[j], dens[j]);
            dft_dft.c2r( cphi[j], phi[j]);
        }
    }

}
