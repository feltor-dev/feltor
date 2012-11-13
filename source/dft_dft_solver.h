#include <complex>
#include "tl_numerics.h"
#include "blueprint.h"
#include "equations.h"



namespace toefl
{
    class DFT_DFT_Solver
    {
        typedef std::complex<double> complex;
        bool imp;
        KarniadakisField<TL_DFT_DFT> k;
        Matrix<double, P> phi_e, phi_i, phi_z;
        /////////////////Complex (void) Matrices for fourier transforms///////////
        Matrix< complex> cne, cni, cnz;
        Matrix< complex> cphi_e, cphi_i, cphi_z;
        Matrix< complex> cne_temp, cni_temp, cnz_temp;
        ///////////////////Solvers////////////////////////
        Arakawa arakawa;
        DFT_DFT dft_dft;
        /////////////////////Coefficients//////////////////////
        Matrix< QuadMat< complex, 2> > coeff_dim2;
        Matrix< QuadMat< complex, 3> > coeff_dim3;
        Matrix< Vector< double, 2> > coeff_phi_dim2;
        Matrix< Vector< double, 3> > coeff_phi_dim3;
        Matrix< double> coeff_Gamma_i, coeff_Gamma_z;
        void multiply_coefficients();
      public:
        DFT_DFT_Solver( const Blueprint& blueprint);
        void init();
        void execute();
        void setField( enum target t, const Matrix<double, TL_DFT_DFT>& m);
        void getField( enum target t, const Matrix<double, TL_DFT_DFT>& m);
    };

    DFT_DFT_Solver::DFT_DFT_Solver( const Blueprint& bp):
        rows( rows), cols( cols),
        imp( imp), k( rows, cols, imp),
        phi_e( rows, cols), phi_i(rows, cols), phi_z( rows, cols, imp),
        cne( rows, cols/2 +1, TL_VOID), cni( rows, cols/2 + 1, TL_VOID), cnz(rows, cols/2 +1, TL_VOID),
        cphi_e( rows, cols/2+1, TL_VOID), cphi_i( rows, cols/2+1, TL_VOID), cphi_z(rows, cols/2+1, TL_VOID)
        cne_temp( rows, cols/2 +1, TL_VOID), cni_temp( rows, cols/2 +1, TL_VOID), cnz_temp(rows, cols/2+1, TL_VOID),
        coeff_dim2( rows, cols/2+1, TL_VOID),
        coeff_dim3( rows, cols/2+1, TL_VOID),
        coeff_phi_dim2( rows, cols/2+1, TL_VOID),
        coeff_phi_dim3( rows, cols/2+1, TL_VOID),
        coeff_Gamma_i( rows, cols),
        coeff_Gamma_z( rows, cols, imp)
    {
        bp.consistencyCheck();
        Physical phys = bp.getPhysical();
        if( !bp.isEnabled( TL_CURVATURE))
            phys.kappa_x = phys.kappa_y = 0; 
        const Boundary bound = bp.getBoundary();
        const Algorithmic alg = bp.getAlgorithmic();
        Equations( phys);
        Poisson( phys);
        }





    void DFT_DFT_Solver::execute()
    {
        arakawa( k.ne0, k.phi_e, k.nonlinear_e0);
        arakawa( k.ni0, k.phi_i, k.nonlinear_i0);
        if( imp)
            arakawa( nz0, phi_z, nonlinear_z0);
        //2. perform karniadakis step
        k.step();
        //3. solve linear equation
        //3.1. transform v_temp
        dft_dft.r2c( ne_temp, cne_temp);
        dft_dft.r2c( ni_temp, cni_temp);
        if( imp) 
            dft_dft( nz_temp, cnz_temp);
        //3.2. multiply coefficients
        multiply_coefficients();
        //3.3. backtransform
        dft_dft.c2r( cne_temp, ne_temp);
        dft_dft.c2r( cni_temp, ni_temp);
        dft_dft.c2r( cphi_e, phi_e);
        dft_dft.c2r( cphi_i, phi_i);
        if( imp)
        {
            dft_dft.c2r( cnz_temp, nz_temp);
            dft_dft.c2r( cphi_z, phi_z);
        }
        //4. permute fields
        k.permute();
    }
    void DFT_DFT_Solver::multiply_coefficients()
    {
        if( !imp)
        {
            multiply_coeff( coeff_dim2, cne_temp, cni_temp);
            for( size_t i = 0; i < rows; i++)
                for( size_t j = 0; j < cols/2 + 1; j++)
                    cphi_e(i,j) = coeff_phi_dim2(i,j)[0]*cne(i,j) 
                                + coeff_phi_dim2(i,j)[1]*cni(i,j);
            for( size_t i = 0; i < rows; i++)
                for( size_t j = 0; j < cols/2 + 1; j++)
                    cphi_i(i,j) = coeff_Gamma_i(i,j)*cphi_e;
        }
        if(imp)
        {
            multiply_coeff( coeff_dim3, cne_temp, cni_temp, cnz_temp);
            for( size_t i = 0; i < rows; i++)
                for( size_t j = 0; j < cols/2 + 1; j++)
                    cphi_e(i,j) = coeff_phi_dim3(i,j)[0]*cne(i,j) 
                                + coeff_phi_dim3(i,j)[1]*cni(i,j) 
                                + coeff_phi_dim3(i,j)[2]*cnz(i,j);
            for( size_t i = 0; i < rows; i++)
                for( size_t j = 0; j < cols/2 + 1; j++)
                {
                    cphi_i(i,j) = coeff_Gamma_i(i,j)*cphi_e;
                    cphi_z(i,j) = coeff_Gamma_z(i,j)*cphi_e;
                }
        }
    }



}
