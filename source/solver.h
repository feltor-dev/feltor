#include <complex>
#include "lib/matrix.h"
#include "lib/coeff.h"
#include "lib/arakawa.h"



namespace toefl
{

    class HW
    {
      private:
        typedef std::complex<double> complex;
        Blueprint p;
        Arakawa a;
        typedef struct{
            double alpha[3];
            double beta[3];
            double gamma_0;
        } karniadakis_coeff;
        const karniadakis_coeff k;
        const karniadakis_coeff k_euler;
        Matrix< double, TL_DFT_DFT> ne[3];
        Matrix< double, TL_DFT_DFT> ni[3];
        Matrix< double, TL_DFT_DFT> arakawa_e[3];
        Matrix< double, TL_DFT_DFT> arakawa_i[3];
        Matrix< double, TL_DFT_DFT> phi_e, phi_i;
        
        Matrix< double, TL_DFT_DFT>& ne, ni;

        Matrix< complex> ne__T, ni__T;
        Matrix< complex> phi_e__T, phi_i__T;
        Matrix< QuadMat<complex, 2> > coeff_n_T;
        Matrix< QuadMat<double, 2> > coeff_phi_T;

        DFT_DFT dft_dft;

        inline void karniadakis( const karniadakis_coeff& k);
      public:
        HW( const Blueprint& p);
        void Init();
        void execute();
    }

    void Init()
    {


    }

    void HW::execute()
    {
        a.per_per( ne0, phi_e, arakawa_e0);
        a.per_per( ni0, phi_i, arakawa_i0);
        karniadakis( );
        dft_dft.r2c_T( ne, ne__T);
        dft_dft.r2c_T( ni, ni__T);
        multiply_coeff( coeff_n_T, ne__T, ni__T);
        multiply_coeff( coeff_phi_T, phi_e__T, phi_i__T);
        dft_dft.c_T2r( ne__T, ne);
        dft_dft.c_T2r( ni__T, ni);
        dft_dft.c_T2r( phi_e__T, phi_e);
        dft_dft.c_T2r( phi_i__T, phi_i);
        permute_fields( ne0, ne1, ne2); 
        permute_fields( ni0, ni1, ni2); 
    }

    void HW::karniadakis()
    {
        for( size_t i = 0; i < p.ny; i++)
            for( size_t j = 0; j < p.nx; j++)
                ne(i,j) = k.alpha_0*ne0(i,j) + k.alpha_1*ne1(i,j) + k.alpha_2*ne2(i,j);
        for( size_t i = 0; i < p.ny; i++)
            for( size_t j = 0; j < p.nx; j++)
                ni(i,j) = k.alpha_0*ni0(i,j) + k.alpha_1*ni1(i,j) + k.alpha_2*ni2(i,j);
        for( size_t i = 0; i < p.ny; i++)
            for( size_t j = 0; j < p.nx; j++)
                ne(i,j) += k.beta_0*arakawa_e0(i,j) + k.beta_1*arakawa_e1(i,j) + k.beta_2*arakawa_e2(i,j);
        for( size_t i = 0; i < p.ny; i++)
            for( size_t j = 0; j < p.nx; j++)
                ni(i,j) += k.beta_0*arakawa_i0(i,j) + k.beta_1*arakawa_i1(i,j) + k.beta_2*arakawa_i2(i,j);
    }




}




        
        

        

