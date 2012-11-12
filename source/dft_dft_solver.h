#include <complex>
#include "matrix.h"
#include "ghostmatrix.h"
#include "arakawa.h"
#include "coeff.h"
#include "equations.h"



namespace toefl
{
    class DFT_DFT_Solver
    {
        typedef std::complex<double> complex;
        const size_t rows, cols;
        bool imp;
        GhostMatrix<double, TL_DFT_DFT> ne[3], ni[3], nz[3];
        GhostMatrix<double, TL_DFT_DFT> phi_e, phi_i, phi_z;
        Matrix<double, TL_DFT_DFT> nonlinear_e[3], nonlinear_i[3], nonlinear_z[3];
        Matrix<double, TL_DFT_DFT>& ne_temp; //reference to ne[2]
        Matrix<double, TL_DFT_DFT>& ni_temp, nz_temp;
        void karniadakis_permute();
        void karniadakis_step();
        /////////////////Complex (void) Matrices for fourier transforms///////////
        Matrix< complex> cne, cni, cnz;
        Matrix< complex> cphi_e, cphi_i, cphi_z;
        Matrix< complex> cne_temp, cni_temp, cnz_temp;
        ///////////////////Solvers////////////////////////
        Arakawa arakawa;
        double alpha[3], beta[3];
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

    DFT_DFT_Solver::DFT_DFT_Solver( const Blueprint& blueprint)

    void DFT_DFT_Solver::execute()
    {
        arakawa( ne[0], phi_e, nonlinear_e[0]);
        arakawa( ni[0], phi_i, nonlinear_i[0]);
        if( imp)
            arakawa( nz[0], phi_z, nonlinear_z[0]);
        //2. perform karniadakis step
        karniadakis_step();
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
        karniadakis_permute();
    }
    void DFT_DFT_Solver::karniadakis_step()
    {
        for( size_t i = 0; i < rows; i++)
            for( size_t j = 0; j < cols; j++)
            {
                ne_temp(i,j) =    alpha[0]*ne[0](i,j) 
                                + alpha[1]*ne[1](i,j) 
                                + alpha[2]*ne[2](i,j)
                                + beta[0]*nonlinear_e[0](i,j) 
                                + beta[1]*nonlinear_e[1](i,j) 
                                + beta[2]*nonlinear_e[2](i,j);
                ni_temp(i,j) =    alpha[0]*ni[0](i,j) 
                                + alpha[1]*ni[1](i,j) 
                                + alpha[2]*ni[2](i,j)
                                + beta[0]*nonlinear_i[0](i,j) 
                                + beta[1]*nonlinear_i[1](i,j) 
                                + beta[2]*nonlinear_i[2](i,j);
            }
        if( imp)
            for( size_t i = 0; i < rows; i++)
                for( size_t j = 0; j < cols; j++)
                {
                    nz_temp(i,j) =    alpha[0]*nz[0](i,j) 
                                    + alpha[1]*nz[1](i,j) 
                                    + alpha[2]*nz[2](i,j)
                                    + beta[0]*nonlinear_z[0](i,j) 
                                    + beta[1]*nonlinear_z[1](i,j) 
                                    + beta[2]*nonlinear_z[2](i,j);
                }
    }
    void DFT_DFT_Solver::karniadakis_permute()
    {
        permute_fields( ne[0], ne[1], ne[2]);
        permute_fields( ni[0], ni[1], ni[2]);
        permute_fields( nonlinear_e[0], nonlinear_e[1], nonlinear_e[2]);
        permute_fields( nonlinear_i[0], nonlinear_i[1], nonlinear_i[2]);
        if( imp)
        {
            permute_fields( nz[0], nz[1], nz[2]);
            permute_fields( nonlinear_z[0], nonlinear_z[1], nonlinear_z[2]);
        }
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
