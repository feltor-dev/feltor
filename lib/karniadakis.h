#ifndef _TL_KARNIADAKIS_
#define _TL_KARNIADAKIS_
#include "matrix"

namespace toefl{
    enum stepper { TL_EULER, TL_ORDER2, TL_ORDER3};

    template< enum stepper>
    struct Coefficients
    {};

    template<>
    struct Coefficients< TL_EULER>
    {
        static const double gamma_0 = 1.;
        static const double alpha[3] = {1.,0.,0.}
        static const double beta[3] = {1.,0.,0.}
    }
    template<>
    struct Coefficients< TL_ORDER2>
    {
        static const double gamma_0 = 1.5;
        static const double alpha[3] = {2.,-0.5.,0.}
        static const double beta[3] = {2.,-1.,0.}
    }
    template<>
    struct Coefficients< TL_ORDER3>
    {
        static const double gamma_0 = 11./6.;
        static const double alpha[3] = {3.,-1.5.,1./3.}
        static const double beta[3] = {3.,-3.,1.}
    }

    template< enum Padding P>
    class Karniadakis
    {
        private:
        const size_t rows, cols;
        Matrix<double, P> v1, v2;
        Matrix<double, P> nonlinear1, nonlinear2;
        public:
        template<enum stepper S>
        void step( Matrix<double, P>& ne0, Matrix<double, P>& ni0, Matrix<double, P>& nz0,
                   Matrix<double, P>& nonlinear_e0, Matrix<double, P>& nonlinear_i0, Matrix<double, P>& nonlinear_z0,
                   const double dt)
        Karniadakis( const size_t rows, const size_t cols, const double dt, const bool imp):
            rows( rows), cols( cols),
            imp( imp),
            v1( rows, cols, allocate), v2( rows, cols, allocate),
            ne1( rows, cols), ne2( rows, cols), 
            ni1( rows, cols), ni2( rows, cols), 
            nz1( rows, cols, imp), nz2( rows, cols, imp), 
            nonlinear_e1( rows, cols), nonlinear_e2( rows, cols),
            nonlinear_i1( rows, cols), nonlinear_i2( rows, cols),
            nonlinear_z1( rows, cols, imp), nonlinear_z2( rows, cols, imp),
            dt(dt)
            {
                ne1.zero(), ne2.zero(), ni1.zero(); 
                ni2.zero(), nz1.zero(), nz2.zero();
                nonlinear_e1.zero(), nonlinear_e2.zero();
                nonlinear_i1.zero(), nonlinear_i2.zero();
                nonlinear_z1.zero(), nonlinear_z2.zero();
            }
    };
    template<enum Padding P>
    template<enum stepper S>
    void Karniadakis<P>::step( Matrix<double, P>& ne0, Matrix<double, P>& ni0, Matrix<double, P>& nz0,
                               Matrix<double, P>& nonlinear_e0, Matrix<double, P>& nonlinear_i0, Matrix<double, P>& nonlinear_z0,
                               const double dt)
    {
#ifdef TL_DEBUG
        if( ne0.isVoid()||ni0.isVoid()||(nz0.isVoid()&&imp)) 
            throw Message( "first swap in valid linear matrices\n", pint);
        if( nonlinear_e0.isVoid()||nonlinear_i0.isVoid()||(nonlinear_z0.isVoid&&imp))
            throw Message( "Swap in nonlinearities!\n", ping);
#endif
        for( size_t i = 0; i < rows; i++)
            for( size_t j = 0; j < cols; j++)
            {
                ne2(i,j) =    S::alpha[0]*ne0(i,j) 
                                + S::alpha[1]*ne1(i,j) 
                                + S::alpha[2]*ne2(i,j)
                                + S::beta[0]*nonlinear_e0(i,j) 
                                + S::beta[1]*nonlinear_e1(i,j) 
                                + S::beta[2]*nonlinear_e2(i,j);
                ni2(i,j) =    S::alpha[0]*ni0(i,j) 
                                + S::alpha[1]*ni1(i,j) 
                                + S::alpha[2]*ni2(i,j)
                                + S::beta[0]*nonlinear_i0(i,j) 
                                + S::beta[1]*nonlinear_i1(i,j) 
                                + S::beta[2]*nonlinear_i2(i,j);
            }
        if( imp)
            for( size_t i = 0; i < rows; i++)
                for( size_t j = 0; j < cols; j++)
                {
                    nz2(i,j) =    S::alpha[0]*nz0(i,j) 
                                    + S::alpha[1]*nz1(i,j) 
                                    + S::alpha[2]*nz2(i,j)
                                    + S::beta[0]*nonlinear_z0(i,j) 
                                    + S::beta[1]*nonlinear_z1(i,j) 
                                    + S::beta[2]*nonlinear_z2(i,j);
                }
        permute_fields( ne0, ne1, ne2);
        permute_fields( ni0, ni1, ni2);
        permute_fields( nonlinear_e0, nonlinear_e1, nonlinear_e2);
        permute_fields( nonlinear_i0, nonlinear_i1, nonlinear_i2);
        if( imp)
        {
            permute_fields( nz0, nz1, nz2);
            permute_fields( nonlinear_z0, nonlinear_z1, nonlinear_z2);
        }
    }
}
#endif //_TL_KARNIADAKIS_
