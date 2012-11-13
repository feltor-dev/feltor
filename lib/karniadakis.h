#ifndef _TL_KARNIADAKIS_
#define _TL_KARNIADAKIS_
#include "matrix.h"

namespace toefl{
    enum stepper { TL_EULER, TL_ORDER2, TL_ORDER3};

    template< enum stepper S>
    struct Coefficients
    {
        static double const gamma_0;
        static const double alpha[3];
        static const double beta[3];
    };
    template<>
    const double Coefficients<TL_EULER>::gamma_0 = 1;
    template<>
    const double Coefficients<TL_EULER>::alpha[3] = {1., 0,0};
    template<>
    const double Coefficients<TL_EULER>::beta[3] = {1., 0,0};

    template<>
    const double Coefficients<TL_ORDER2>::gamma_0 = 1.5;
    template<>
    const double Coefficients<TL_ORDER2>::alpha[3] = {2.,-0.5,0.};
    template<>
    const double Coefficients<TL_ORDER2>::beta[3] = {2.,-1.,0.};

    template<>
    const double Coefficients<TL_ORDER3>::gamma_0 = 11./6.;
    template<>
    const double Coefficients<TL_ORDER3>::alpha[3] = {3.,-1.5,1./3.};
    template<>
    const double Coefficients<TL_ORDER3>::beta[3] = {3.,-3.,1.};

    template< enum Padding P>
    class Karniadakis
    {
      private:
        const size_t rows, cols;
        Matrix<double, P> v1, v2;
        Matrix<double, P> n1, n2;
        const double dt;
      public:
        Karniadakis( const size_t rows, const size_t cols, const double dt, const bool allocate = true):
            rows( rows), cols( cols),
            v1( rows, cols, allocate), v2( rows, cols, allocate),
            n1( rows, cols, allocate), n2( rows, cols, allocate),
            dt(dt)
            {
                if( allocate)
                {
                    v1.zero(); 
                    v2.zero();
                    n1.zero();
                    n2.zero();
                }
            }
        template<enum stepper S>
        void step( Matrix<double, P>& v0, Matrix<double, P>& n0);
    };
    template<enum Padding P>
    template<enum stepper S>
    void Karniadakis<P>::step( Matrix<double, P>& v0, Matrix<double, P>& n0)
    {
#ifdef TL_DEBUG
        if( v0.isVoid()||n0.isVoid()) 
            throw Message( "ERROR: Cannot work on void matrices!\n", ping);
        if( v1.isVoid())
            throw Message( "ERROR: Karniadakis is void!\n", ping);
#endif
        for( size_t i = 0; i < rows; i++)
            for( size_t j = 0; j < cols; j++)
            {
                v2(i,j) =  Coefficients<S>::alpha[0]*v0(i,j) 
                         + Coefficients<S>::alpha[1]*v1(i,j) 
                         + Coefficients<S>::alpha[2]*v2(i,j)
                         + dt*( Coefficients<S>::beta[0]*n0(i,j) 
                              + Coefficients<S>::beta[1]*n1(i,j) 
                              + Coefficients<S>::beta[2]*n2(i,j));
            }
        permute_fields( n0, n1, n2);
        permute_fields( v0, v1, v2);
    }
}
#endif //_TL_KARNIADAKIS_
