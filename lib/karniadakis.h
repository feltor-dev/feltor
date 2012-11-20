#ifndef _TL_KARNIADAKIS_
#define _TL_KARNIADAKIS_
#include "matrix.h"
#include "vector.h"
#include "quadmat.h"

namespace toefl{
    /*! @brief Kinds of Stepper coefficients for karniadakis scheme
     */
    enum stepper { TL_EULER, //!< Euler scheme (use for 1st step)
                   TL_ORDER2, //!< 2nd order scheme (use for 2nd step)
                   TL_ORDER3  //!< 3rd order scheme ( the "usual" karniadakis scheme)
                 };

    /*! @brief template traits class for various sets of coefficients in the karniadakis scheme from the karniadakis paper
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

    /*! @brief pointwise multiply the n x n Matrix of coefficients by a n-vector of matrices  
     *
     * Compute the system m0 = c00*m0 + c01*m1 + c02*m2 + ..., m1 = ... where all
     * of the elements are matrices and matrix-matrix multiplications are done pointwise.
     * @tparam T1 type of the coefficients i.e. double or std::complex<double>
     * @tparam T type of the matrix elements, i.e. double or std::complex<double>
     * @param c the coefficient matrix 
     * @param in Input vector of matrices
     * @param out Output vector of matrices. Contains solution on output.
     *  Multiplication is done inplace if in and out reference the same object!
     */
    template< size_t n, typename T1, typename T>
    void multiply_coeff( const Matrix< QuadMat<T1,n>, TL_NONE>& c, 
                         const Vector< Matrix<T,TL_NONE>, n>& in,
                         Vector< Matrix<T,TL_NONE>, n>& out )
    {
        const size_t rows = c.rows(), cols = c.cols();
#ifdef TL_DEBUG
        if( c.isVoid())
            throw Message( "Cannot work with void Matrices!\n", ping);
        for( unsigned k=0; k<n; k++)
        {
            if( c.rows() != in[k].rows() || c.rows() != out[k].rows())
                if( c.cols() != in[k].cols() || c.cols() != out[k].cols())
                    throw Message( "Cannot multiply coefficients! Sizes not equal!", ping);
            if( in[k].isVoid() || out[k].isVoid() )
                throw Message( "Cannot work with void Matrices!\n", ping);
        }
#endif
        QuadMat<T, n> temp;
        for( size_t i = 0; i<rows; i++)
            for( size_t j=0; j<cols; j++)
            {
                //Matrix-Vector multiplication
                for( unsigned k=0; k<n; k++)
                    for( unsigned q=0; q<n; q++)
                        temp(k,q) = c(i,j)(k,q)*in[q](i,j);
                for( unsigned k=0; k<n; k++)
                {
                    out[k](i,j) = 0;
                    for( unsigned q=0; q<n; q++)
                        out[k](i,j) += temp(k,q);
                }
            }
    }

    /*! @brief Multistep timestepper object 
     *
     * Construction is a bit clumsy but usage is easy. This object is a solution to the
     * problem of computing the two steps in the karniadakis scheme. 
     * One is in x-space the other in fourier space. The goal was to implement a solver
     * which is oblivious to the type of boundary conditions used and can be used for two or
     * three equations. 
     * \todo n equations are available only when an implementation of an LU decomposition is available. (LAPACK?)
     * @tparam n size of the equations (2 or 3)
     * @tparam T_k the type of fourier Coefficients used (double or std::complex<double>)
     * @tparam P_x Padding of your (real) matrices
     */
    template< size_t n, typename T_k, enum Padding P_x>
    class Karniadakis
    {
      private:

        const size_t rows, cols;
        Vector< Matrix< double, P_x>, n> v1, v2;
        Vector< Matrix< double, P_x>, n> n1, n2;
        Matrix< QuadMat< T_k, n>, TL_NONE> c;
#ifdef TL_DEBUG
        Matrix< QuadMat< T_k, n>, TL_NONE> c_temp; //contains the coeff of first call
#endif

        const double dt;
      public:
        /*! @brief Allocate storage for the last two fields in the karniadakis scheme.
         *
         * @param rows_x rows of your x-space matrices
         * @param cols_x columns of your x-space matrices
         * @param dt the timestep
         */
        Karniadakis(const size_t rows_x, const size_t cols_x, const double dt);

        /*! @brief Init the Coefficients for step_ii
         *
         * This copies the coefficients for later use in the step_ii function.
         * @param coeff_k The linear part of your equations in fourier space
         */
        void init_coeff( const Matrix< QuadMat< T_k, n>, TL_NONE>& coeff_k);

        /*! @brief Compute the first part of the Karniadakis scheme
         *
         * @param v0 
         * The field at timestep n, that is stored by the class.
         * Contains v_{temp} on output.
         * @param n0
         * The nonlinearity at timestep n.
         * Content undefined on output.
         * @tparam S The set of Karniadakis-Coefficients you want to use.
         */
        template< enum stepper S>
        void step_i( Vector< Matrix<double, P_x>, n>& v0, Vector< Matrix<double, P_x>, n> & n0);
        /*! @brief Compute the second part of the Karniadakis scheme
         *
         * Inverts the fourier Coefficients on first call.
         * @param v 
         * The fourier transposed result of step_i on input.
         * Contains the multiplied coefficients on output
         * @tparam Fourier_T The value type of the fourier transposed matrices
         * @tparam S The set of Karniadakis-Coefficients you want to use.
         */
        template< enum stepper S, class Fourier_T>
        inline void step_ii( Vector< Matrix< Fourier_T, TL_NONE>, n>& v);
    };

    template< size_t n, typename T, enum Padding P>
    Karniadakis<n,T,P>::Karniadakis( const size_t rows, 
                 const size_t cols, 
                 const double dt):
            rows( rows), cols( cols),
            dt( dt)
            {
                //allocate vectors
                for(unsigned k=0; k<n; k++)
                {
                    v1[k].allocate(rows, cols, 0.);
                    v2[k].allocate(rows, cols, 0.);
                    n1[k].allocate(rows, cols, 0.);
                    n2[k].allocate(rows, cols, 0.);
                }
            }
    template< size_t n, typename T, enum Padding P>
    void Karniadakis< n,T,P>::init_coeff( const Matrix< QuadMat< T, n>, TL_NONE>& coeff)
    {
        c.allocate( coeff.rows(), coeff.cols());
        c = coeff;
    }

    template< size_t n, typename T, enum Padding P>
    template< enum stepper S>
    void Karniadakis<n,T,P>::step_i( Vector< Matrix<double, P>, n>& v0, Vector< Matrix<double, P>, n> & n0)
    {
        for( unsigned k=0; k<n; k++)
        {
#ifdef TL_DEBUG
            if( v0[k].isVoid()||n0[k].isVoid()) 
                throw Message( "ERROR: Cannot work on void matrices!\n", ping);
            if( v0[k].rows() != rows || v0[k].cols() != cols)
                throw Message( "ERROR: One of the v0 has wrong size!\n", ping);
            if( n0[k].rows() != rows || n0[k].cols() != cols)
                throw Message( "ERROR: One of the n0 has wrong size!\n", ping);
#endif
            for( size_t i = 0; i < rows; i++)
                for( size_t j = 0; j < cols; j++)
                {
                    v2[k](i,j) =  Coefficients<S>::alpha[0]*v0[k](i,j) 
                             + Coefficients<S>::alpha[1]*v1[k](i,j) 
                             + Coefficients<S>::alpha[2]*v2[k](i,j)
                             + dt*( Coefficients<S>::beta[0]*n0[k](i,j) 
                                  + Coefficients<S>::beta[1]*n1[k](i,j) 
                                  + Coefficients<S>::beta[2]*n2[k](i,j));
                }
            permute_fields( n0[k], n1[k], n2[k]);
            permute_fields( v0[k], v1[k], v2[k]);
        }
    }
    template< size_t n, typename T, enum Padding P>
    template< enum stepper S, class Fourier_T>
    void Karniadakis<n,T,P>::step_ii( Vector< Matrix< Fourier_T, TL_NONE>, n>& v)
    {
        static Matrix< QuadMat< T,n>, TL_NONE> coeff;
        if( coeff.isVoid()) //invert coefficients first
        {
#ifdef TL_DEBUG
        if( c.isVoid()) 
            throw Message( "Init coefficients first!",ping);
#endif
            coeff.allocate( coeff.rows(), coeff.cols());
            //invert coefficients
            for(unsigned i=0; i<c.rows(); i++)
                for( unsigned j=0; j<c.cols(); j++)
                {
                    for( unsigned k=0; k<n; k++)
                        coeff(i,j)(k,k) = Coefficients<S>::gamma_0 - dt*coeff(i,j)(k,k);
                    invert( coeff(i,j), coeff(i,j));
                }
        }
        multiply_coeff< n,T_k,Fourier_T>( coeff,v,v);
    }


}
#endif //_TL_KARNIADAKIS_
