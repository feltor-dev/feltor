#ifndef _TL_KARNIADAKIS_
#define _TL_KARNIADAKIS_
#include "matrix.h"
#include "vector.h"

namespace toefl{
    /*! @brief Kinds of Stepper coefficients for karniadakis scheme
     */
    enum stepper { TL_EULER, //!< Euler scheme (use for 1st step)
                   TL_ORDER2, //!< 2nd order scheme (use for 2nd step)
                   TL_ORDER3  //!< 3rd order scheme ( the "usual" karniadakis scheme)
                 };

    /*! @brief template traits class for various sets of coefficients in the karniadakis scheme
     */
    template< enum stepper S>
    struct Coefficients
    {
        static double const gamma_0;
        static const double alpha[3];
        static const double beta[3];
    };
    ///@cond
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
    ///@endcond

    /*! @brief Store the last two values in the karniadakis scheme
     *
     * Example code
     * \code
    Karniadakis<TL_NONE> k(rows, cols, dt);
    k.step<TL_EULER>( m, n);
    t += dt;
    for( size_t i = 0; i< rows; i++)
        for( size_t j = 0; j< cols; j++)
            n(i,j) = m(i,j) = m(i,j)/(Coefficients<TL_EULER>::gamma_0);
     * \endcode
     * @tparam P Padding of your matrices
     */
    template< const size_t n, typename coeff_T, enum Padding P>
    class Karniadakis
    {
      private:
        const size_t rows, cols;
        Vector< Matrix< double, P>, n> v;
        Vector< Matrix< double, P>, n> non;
        const Matrix< QuadMat< coeff_T,n>, TL_NONE> c;
        const double dt;
      public:
        /*! @brief Allocate storage for the last two fields in the karniadakis scheme
         *
         * @param rows rows of your matrices
         * @param cols columns of your matrices
         * @param dt the timestep
         */
        Karniadakis( const Matrix< QuadMat< coeff_T, n>, TL_NONE>& coeff, const double dt):
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
        /*! @brief Compute the first part of the Karniadakis scheme
         *
         * @tparam S Provide the set of coefficients to be used
         * @param v0 
         * The field at timestep n, that is stored by the class.
         * Contains v_{temp} on output.
         * @param n0
         * The nonlinearity at timestep n
         * Content undefined on output
         */
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
        if( v0.rows() != rows || v0.cols() != cols)
            throw Message( "ERROR: v0 has wrong size!\n", ping);
        if( n0.rows() != rows || n0.cols() != cols)
            throw Message( "ERROR: n0 has wrong size!\n", ping);
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
