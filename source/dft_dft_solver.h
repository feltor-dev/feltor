#include <complex>
#include "tl_numerics.h"
#include "blueprint.h"
#include "equations.h"



namespace toefl
{
    class DFT_DFT_Solver
    {
        typedef std::complex<double> complex;
        const size_t rows, cols;
        const bool imp;
        const double dt;
        GhostMatrix<double, TL_DFT_DFT> ne, phi_e;
        GhostMatrix<double, TL_DFT_DFT> ni, phi_i;
        GhostMatrix<double, TL_DFT_DFT> nz, phi_z;
        Matrix<double, TL_DFT_DFT> nonlinear_e;
        Matrix<double, TL_DFT_DFT> nonlinear_i;
        Matrix<double, TL_DFT_DFT> nonlinear_z;
        /////////////////Complex (void) Matrices for fourier transforms///////////
        Matrix< complex> cne, cphi_e;
        Matrix< complex> cni, cphi_i;
        Matrix< complex> cnz, cphi_z;
        ///////////////////Solvers////////////////////////
        Arakawa arakawa;
        Karniadakis<TL_DFT_DFT> k_e, k_i, k_z;
        DFT_DFT dft_dft;
        /////////////////////Coefficients//////////////////////
        Matrix< QuadMat< complex, 2> > coeff_dim2;
        Matrix< QuadMat< complex, 3> > coeff_dim3;
        Matrix< Vector< double, 2> > coeff_phi_dim2;
        Matrix< Vector< double, 3> > coeff_phi_dim3;
        Matrix< double> coeff_Gamma_i, coeff_Gamma_z;
        void init_coefficients( const Boundary& bound, const Physical& phys);

        template< enum stepper S>
        void invert_coefficients();
        void multiply_coefficients();

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
         * all low level solver needed. After that it performs two 
         * initializing steps (by one onestep- and one twostep-method)
         * in order to get the karniadakis scheme ready. The actual time is
         * thus T_0 + 2*dt after initialisation. 
         */
        void init( enum target t);
        /*! @brief Perform a step by the 3 step Karniadakis scheme*/
        void step(){ step_<TL_ORDER3>();}
        /*! @brief Init field
         */
        void setField( enum target t, const Matrix<double, TL_DFT_DFT>& m);
        /*! @brief Get the result*/
        void getField( enum target t, const Matrix<double, TL_DFT_DFT>& m);
    };

    DFT_DFT_Solver::DFT_DFT_Solver( const Blueprint& bp):
        rows( bp.getAlgorithmic().ny ), cols( bp.getAlgorithmic().nx ),
        imp( bp.isEnabled( TL_IMPURITY)),
        dt( bp.getAlgorithmic().dt),
        //fields
        ne( rows, cols),      phi_e( ne),   
        ni( rows, cols),      phi_i( ni),   
        nz( rows, cols, imp), phi_z( nz),   
        nonlinear_e( rows, cols),
        nonlinear_i( rows, cols),                
        nonlinear_z( rows, cols, imp),
        //complex
        cne( rows, cols/2 +1, TL_VOID),  cphi_e( cne), 
        cni( rows, cols/2 +1, TL_VOID),  cphi_i( cne), 
        cnz( rows, cols/2 +1, TL_VOID),  cphi_z( cne), 
        //Solvers
        arakawa( bp.getAlgorithmic().h),
        k_e(rows, cols, dt), k_i( k_e), k_z(rows, cols, dt, imp),
        dft_dft( rows, cols, FFTW_MEASURE),
        coeff_dim2( rows, cols/2+1, TL_VOID),
        coeff_dim3( rows, cols/2+1, TL_VOID),
        coeff_phi_dim2( rows, cols/2+1, TL_VOID),
        coeff_phi_dim3( rows, cols/2+1, TL_VOID),
        coeff_Gamma_i( rows, cols/2+1),
        coeff_Gamma_z( rows, cols/2+1, imp)
    {
        bp.consistencyCheck();
        Physical phys = bp.getPhysical();
        if( !bp.isEnabled( TL_CURVATURE))
            phys.kappa_x = phys.kappa_y = 0; 
        init_coefficients(bp.getBoundary(), phys);
    }

    void DFT_DFT_Solver::init_coefficients( const Boundary& bound, const Physical& phys)
    {
        double laplace;
        const complex kxmin ( 0, 2.*M_PI/bound.lx);
        const complex kymin ( 0, 2.*M_PI/bound.ly);
        const double kxmin2 = 2.*2.*M_PI*M_PI/bound.lx/bound.lx;
        const double kymin2 = 2.*2.*M_PI*M_PI/bound.ly/bound.ly;
        Equations e( phys);
        Poisson p( phys);
        // ki = 2Pi*i/ly, 
        // dft_dft is not transposing so i is the y index by default
        // First the coefficients for the Poisson equation
        if(imp)
        {
            for( unsigned i = 0; i<rows; i++)
                for( unsigned j = 0; j<cols/2+1; j++)
                {
                    laplace = -kxmin2*(double)(j*j) - kymin2*(double)(i*i);
                    coeff_Gamma_i(i,j) = p.gamma1_i( laplace);
                    coeff_Gamma_z(i,j) = p.gamma1_z(laplace);
                    p( coeff_phi_dim3(i,j), laplace);  
                    e( coeff_dim3( i,j), (double)j*kxmin, (double)i*kymin);
                }
        }
        else
        {
            for( unsigned i = 0; i<rows; i++)
                for( unsigned j = 0; j<cols/2+1; j++)
                {
                    laplace = -kxmin2*(double)(j*j) - kymin2*(double)(i*i);
                    coeff_Gamma_i(i,j) = p.gamma1_i( laplace);
                    p( coeff_phi_dim2(i,j), laplace);  
                    e( coeff_dim2( i,j), (double)j*kxmin, (double)i*kymin);
                }
        }
    }
    template< enum stepper S>
    void DFT_DFT_Solver::invert_coefficients( )
    {
        if(imp)
        {
            for( unsigned i = 0; i<rows; i++)
                for( unsigned j = 0; j<cols/2+1; j++)
                {
                    for( unsigned k=0; k<3; k++)
                        coeff_dim3(i,j)(k,k) += Coefficients<S>::gamma_0 - dt*coeff_dim3(i,j)(k,k);
                    invert( coeff_dim3(i,j));
                }
        }
        else
        {
            for( unsigned i = 0; i<rows; i++)
                for( unsigned j = 0; j<cols/2+1; j++)
                {
                    for( unsigned k=0; k<2; k++)
                        coeff_dim2(i,j)(k,k) += Coefficients<S>::gamma_0 - dt*coeff_dim2(i,j)(k,k);
                    invert( coeff_dim2(i,j));
                }
        }
    }
    void DFT_DFT_Solver::multiply_coefficients()
    {
        if( !imp)
        {
            multiply_coeff( coeff_dim2, cne, cni);
            for( size_t i = 0; i < rows; i++)
                for( size_t j = 0; j < cols/2 + 1; j++)
                    cphi_e(i,j) = coeff_phi_dim2(i,j)[0]*cne(i,j) 
                                + coeff_phi_dim2(i,j)[1]*cni(i,j);
            for( size_t i = 0; i < rows; i++)
                for( size_t j = 0; j < cols/2 + 1; j++)
                    cphi_i(i,j) = coeff_Gamma_i(i,j)*cphi_e(i,j);
        }
        if(imp)
        {
            multiply_coeff( coeff_dim3, cne, cni, cnz);
            for( size_t i = 0; i < rows; i++)
                for( size_t j = 0; j < cols/2 + 1; j++)
                    cphi_e(i,j) = coeff_phi_dim3(i,j)[0]*cne(i,j) 
                                + coeff_phi_dim3(i,j)[1]*cni(i,j) 
                                + coeff_phi_dim3(i,j)[2]*cnz(i,j);
            for( size_t i = 0; i < rows; i++)
                for( size_t j = 0; j < cols/2 + 1; j++)
                {
                    cphi_i(i,j) = coeff_Gamma_i(i,j)*cphi_e(i,j);
                    cphi_z(i,j) = coeff_Gamma_z(i,j)*cphi_e(i,j);
                }
        }
    }
    void DFT_DFT_Solver::init( enum target t)
    {
        dft_dft.r2c( ne, cne);
        dft_dft.r2c( ni, cni);
        if( imp) 
            dft_dft.r2c( nz, cnz);
        switch( t) //which field must be computed?
        {
            case( TL_ELECTRONS):
                throw Message( "Electron feature not implemented yet!\n",ping);
                break;
            case( TL_IONS):
                throw Message( "Ion feature not implemented yet!\n",ping);
                break;
            case( TL_IMPURITIES):
                throw Message( "Impurity feature not implemented yet!\n",ping);
                break;
            case( TL_POTENTIAL):
                multiply_coefficients();
                break;
        }
        dft_dft.c2r( cne, ne);
        dft_dft.c2r( cni, ni);
        dft_dft.c2r( cphi_e, phi_e);
        dft_dft.c2r( cphi_i, phi_i);
        if( imp)
        {
            dft_dft.c2r( cnz, nz);
            dft_dft.c2r( cphi_z, phi_z);
        }
        first_steps();
    }

    void DFT_DFT_Solver::first_steps()
    {
        if(imp)
        {
            Matrix< QuadMat<complex, 3> > temp( coeff_dim3);
            invert_coefficients<TL_EULER>();
            step_<TL_EULER>();
            coeff_dim3 = temp;
            invert_coefficients<TL_ORDER2>();
            step_<TL_ORDER2>();
            coeff_dim3 = temp;
            invert_coefficients<TL_ORDER3>();
        }
        else
        {
            Matrix< QuadMat<complex, 2> > temp( coeff_dim2);
            invert_coefficients<TL_EULER>();
            step_<TL_EULER>();
            coeff_dim2 = temp;
            invert_coefficients<TL_ORDER2>();
            step_<TL_ORDER2>();
            coeff_dim2 = temp;
            invert_coefficients<TL_ORDER3>();
        }
    }

    template< enum stepper S>
    void DFT_DFT_Solver::step_()
    {
        //1. Compute nonlinearity
        arakawa( ne, phi_e, nonlinear_e);
        arakawa( ni, phi_i, nonlinear_i);
        if( imp)
            arakawa( nz, phi_z, nonlinear_z);
        //2. perform karniadakis step
        k_e.step<S>( ne, nonlinear_e);
        k_i.step<S>( ni, nonlinear_i);
        if( imp)
            k_z.step<S>( nz, nonlinear_z);
        //3. solve linear equation
        //3.1. transform v_hut
        dft_dft.r2c( ne, cne);
        dft_dft.r2c( ni, cni);
        if( imp) 
            dft_dft.r2c( nz, cnz);
        //3.2. multiply coefficients
        multiply_coefficients();
        //3.3. backtransform
        dft_dft.c2r( cne, ne);
        dft_dft.c2r( cni, ni);
        dft_dft.c2r( cphi_e, phi_e);
        dft_dft.c2r( cphi_i, phi_i);
        if( imp)
        {
            dft_dft.c2r( cnz, nz);
            dft_dft.c2r( cphi_z, phi_z);
        }
    }



}
