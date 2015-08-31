#ifndef _TL_KARNIADAKIS_
#define _TL_KARNIADAKIS_
#include <array>
#include "matrix.h"
#include "matrix_array.h"
#include "quadmat.h"

namespace toefl{
/*! @brief Kinds of Stepper coefficients for karniadakis scheme
 * @ingroup algorithms
 */
enum stepper { TL_EULER, //!< Euler scheme (use for 1st step)
               TL_ORDER2, //!< 2nd order scheme (use for 2nd step)
               TL_ORDER3  //!< 3rd order scheme ( the "usual" karniadakis scheme)
             };

/*! @brief template traits class for various sets of coefficients in the karniadakis scheme from the karniadakis paper
 * @ingroup algorithms
 */
template< enum stepper S>
struct Coefficients
{
    static double const gamma_0; //!< Coefficient in the dft part
    static const double alpha[3]; //!< Coefficients for the timestep
    static const double beta[3]; //!< Coefficients fot the nonlinear part in the timestep.
};
///@cond
template<>
const double Coefficients<TL_EULER>::gamma_0 = 1.;
template<>
const double Coefficients<TL_EULER>::alpha[3] = {1., 0.,0.};
template<>
const double Coefficients<TL_EULER>::beta[3] = {1., 0.,0.};

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
 * @ingroup algorithms
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
                     const std::array< Matrix<T,TL_NONE>, n>& in,
                     std::array< Matrix<T,TL_NONE>, n>& out )
{
    const size_t rows = c.rows(), cols = c.cols();
#ifdef TL_DEBUG
    if( c.isVoid())
        throw Message( "Cannot work with void Matrices!\n", _ping_);
    for( unsigned k=0; k<n; k++)
    {
        if( c.rows() != in[k].rows() || c.rows() != out[k].rows())
            if( c.cols() != in[k].cols() || c.cols() != out[k].cols())
                throw Message( "Cannot multiply coefficients! Sizes not equal!", _ping_);
        if( in[k].isVoid() || out[k].isVoid() )
            throw Message( "Cannot work with void Matrices!\n", _ping_);
    }
#endif
#pragma omp parallel for 
    for( size_t i = 0; i<rows; i++)
    {
        QuadMat<T, n> temp;
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
}

/*! @brief Multistep timestepper object 
 *
 * @ingroup algorithms
 * Construction is a bit clumsy but usage is easy. This object is a solution to the
 * problem of computing the two steps in the karniadakis scheme. 
 * One is in x-space the other in fourier space. The goal was to implement a solver
 * which is oblivious to the type of boundary conditions used and can be used for two or
 * three equations. 
 * \todo n equations are available only when an implementation of an LU decomposition is available. (LAPACK?)
 * @tparam n size of the equations (2 or 3)
 * @tparam T_k the type of fourier coefficients used (double or std::complex<double>)
 * @tparam P_x Padding of your (real) matrices
 */
template< size_t n, typename T_k, enum Padding P_x>
class Karniadakis
{
  public:
    /*! @brief Allocate storage for the last two fields in the karniadakis scheme.
     *
     * @param rows_x # of rows of your x-space matrices
     * @param cols_x # of columns of your x-space matrices
     * @param rows_k # of rows of your k-space coefficients
     * @param cols_k # of columns of your k-space coefficients
     * @param dt the timestep
     */
    Karniadakis(const size_t rows_x, const size_t cols_x, const size_t rows_k, const size_t cols_k, const double dt);

    /*! @brief Swap in the fourier coefficients.
     *
     * Swaps the coefficients into the object and allocates internal storage for the
     * inverse matrix.
     * @param coeff_origin Set of fourier coefficients, void on output.
     * @param normalisation 
        A numerical discrete fourier transformation followed by its inverse usually
        yields the input times a constant factor. State this factor here to normalize the 
        output of the step_ii function. 
     */
    void init_coeff( Matrix<QuadMat<T_k, n> > & coeff_origin, const double normalisation);

    /*! @brief Init the coefficients for step_ii
     *
     * Inverts your fourier coefficients with the correct gamma_0.
     * @tparam S The set of Karniadakis-Coefficients you want to use
     * @attention This function has to be called BEFORE a call of step_ii AND/OR
     *   AFTER you switched steppers.
     */
    template< enum stepper S>
    void invert_coeff( );

    /*! @brief Compute the first part of the Karniadakis scheme
     *
     * @param v0 
     * The field at timestep n, that is stored by the class.
     * Contains v_{temp} on output.
     * @param n0
     * The nonlinearity at timestep n.
     * Contains the old v2 on output.
     * @tparam S The set of Karniadakis-Coefficients you want to use
     */
    template< enum stepper S>
    void step_i( std::array< Matrix<double, P_x>, n>& v0, std::array< Matrix<double, P_x>, n> & n0);
    /*! @brief Compute the second part of the Karniadakis scheme
     *
     * The result is normalized with the inverse of the normalisation factor 
     * you specified in the init_coeff function. 
     * @param v 
     * The fourier transposed result of step_i on input.
     * Contains the multiplied coefficients on output
     * @tparam Fourier_T The value type of the fourier transposed matrices
     * @attention Call invert_coeff BEFORE the first call to step_ii with 
     *   a new stepper.
     */
    template< class Fourier_T>
    inline void step_ii( std::array< Matrix< Fourier_T, TL_NONE>, n>& v)
    {
#ifdef TL_DEBUG
        if( c_origin.isVoid())
            throw Message( "Init coefficients first!", _ping_);
#endif
        multiply_coeff< n,T_k,Fourier_T>( c_inv,v,v);
    }

    /*! @brief Display the original and the inverted coefficients
     *
     * @param os The outstream, the coefficients are streamed to.
     */
    void display( std::ostream& os = std::cout) const
    {
        os << "The current coefficients are \n"<< c_origin
            <<"The current inverse is\n" << c_inv<<std::endl;
    }
  private:
    const size_t rows, cols;
    std::array< Matrix< double, P_x>, n> v1, v2;
    std::array< Matrix< double, P_x>, n> n1, n2;
    Matrix< QuadMat< T_k, n>, TL_NONE> c_inv;
    Matrix< QuadMat< T_k, n>, TL_NONE> c_origin; //contains the coeff of first call
    double prefactor;
    const double dt;

};

template< size_t n, typename T, enum Padding P>
Karniadakis<n,T,P>::Karniadakis( 
             const size_t rows, 
             const size_t cols, 
             const size_t crows, 
             const size_t ccols, 
             const double dt):
        rows( rows), cols( cols),
        v1( MatrixArray<double,P,n>::construct( rows, cols)), v2(v1), n1(v1), n2(n1),
        c_inv( crows, ccols, TL_VOID), c_origin(c_inv),
        prefactor(0.),
        dt( dt)
{ }
template< size_t n, typename T_k, enum Padding P>
void Karniadakis<n,T_k,P>::init_coeff( Matrix<QuadMat<T_k, n> > & coeff_origin, const double normalisation)
{
#ifdef TL_DEBUG
    if( normalisation < 1.)
        throw Message( "Yield the prefactor, not its inverse!", _ping_);
    if( coeff_origin.isVoid())
        throw Message("Your coefficients are void!", _ping_);
    if( coeff_origin.rows() != c_origin.rows() || coeff_origin.cols() != c_origin.cols())
        throw Message("Your coefficients have wrong size!\n", _ping_);
#endif
    prefactor = normalisation; 
    if( c_origin.isVoid())
    {
        swap_fields( c_origin, coeff_origin);
        c_inv.allocate( );
    }
    else
        throw Message("You've already initialized coefficients", _ping_);
}
template< size_t n, typename T, enum Padding P>
template< enum stepper S>
void Karniadakis< n,T,P>::invert_coeff( )
{
#ifdef TL_DEBUG
    if( c_origin.isVoid())
        throw Message( "Init your coefficients first!", _ping_);
#endif
    //invert coefficients
    for(unsigned i=0; i<c_inv.rows(); i++)
        for( unsigned j=0; j<c_inv.cols(); j++)
        {
            for( unsigned k=0; k<n; k++)
            {
                for( unsigned q=0; q<n; q++)
                    c_inv(i,j)(k,q) = -prefactor*dt*c_origin(i,j)(k,q);
                c_inv(i,j)(k,k) += prefactor*Coefficients<S>::gamma_0;
                
            }
            invert( c_inv(i,j), c_inv(i,j));
        }
    //std::cout << "C_inv "<<c_inv<<std::endl;

}

template< size_t n, typename T, enum Padding P>
template< enum stepper S>
void Karniadakis<n,T,P>::step_i( std::array< Matrix<double, P>, n>& v0, std::array< Matrix<double, P>, n> & n0)
{
    for( unsigned k=0; k<n; k++)
    {
#ifdef TL_DEBUG
        if( v0[k].isVoid()||n0[k].isVoid()) 
            throw Message( "ERROR: Cannot work on void matrices!\n", _ping_);
        if( v0[k].rows() != rows || v0[k].cols() != cols)
            throw Message( "ERROR: One of the v0 has wrong size!\n", _ping_);
        if( n0[k].rows() != rows || n0[k].cols() != cols)
            throw Message( "ERROR: One of the n0 has wrong size!\n", _ping_);
#endif
#pragma omp parallel for 
        for( size_t i = 0; i < rows; i++)
        {
            for( size_t j = 0; j < cols; j++)
            {
                n2[k](i,j) =  Coefficients<S>::alpha[0]*v0[k](i,j) 
                         + Coefficients<S>::alpha[1]*v1[k](i,j) 
                         + Coefficients<S>::alpha[2]*v2[k](i,j)
                         + dt*( Coefficients<S>::beta[0]*n0[k](i,j) 
                              + Coefficients<S>::beta[1]*n1[k](i,j) 
                              + Coefficients<S>::beta[2]*n2[k](i,j));
            }
        }
        swap_fields( n2[k], v2[k]); //we want to keep v2 not n2

        permute_fields( n0[k], n1[k], n2[k]);
        permute_fields( v0[k], v1[k], v2[k]);
    }
}


} //namespace toefl
#endif //_TL_KARNIADAKIS_
