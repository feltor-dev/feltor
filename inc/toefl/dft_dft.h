#ifndef _TL_DFT_DFT_
#define _TL_DFT_DFT_

#include <complex>
#include "matrix.h"
#include "fftw3.h"
#include "fft.h"

namespace toefl{

/*! @brief Class for 2d discrete fourier transformations of Matrix using FFTW
 *
 * @ingroup fftw
 * Actually just a wrapper for the dft_c2r_2d/r2c_2d plans and execute
 * functions. Note that the dft_c2r_2d execution of fftw is faster than 
 * the procedure 
 * linewise transformation - transposition - linewise transformation
 * @attention When computing derivatives using 2d r2c/c2r transform pay attention on 
 *  the correct column coefficients in fourier space. 
 *  The first dimension (the column transform) is effectively a c2c transform
 *  and you thus have to be careful about coefficients >N/2+1 in order to keep the 
 *  minimal oscillating trigonometric interpolation property. (probably you
 *  have to worry about (N-k) factors to get the derivative. )
 *  The last dimension (i.e. the row transform) is a r2c transform and thus only 
 *  stores N/2+1 coefficients. Here you don't have the problem since
 *  the c2r backtransform automatically takes care of the coefficients >N/2+1.
 */
class DFT_DFT
{
  private:
    typedef std::complex<double> complex;
    const size_t rows, cols;
    fftw_plan forward;
    fftw_plan backward;
  public:
    /*! @brief Prepare a 2d discrete fourier transformation of given size
     *
     * @param real_rows # of rows in the real matrix
     * @param real_cols # of colums in the real matrix
     * @param flags flags for plan creation 
     */
    DFT_DFT( const size_t real_rows, const size_t real_cols, const unsigned flags = FFTW_MEASURE);
    /*! @brief Execute a r2c transformation on given Matrix
     *
     * @param inout non void matrix of size specified in the constructor.
     * i.e. (real_rows, real_cols)
     * Content on output is the one of swap on input.
     * @param swap Can be void. Size has to be (real_rows, real_cols/2 + 1).
     * Contains the solution on output.
     */
    inline void r2c( Matrix<double, TL_DFT>& inout, Matrix<complex, TL_NONE>& swap);

    /*! @brief Execute a c2r transformation of the given Matrix
     *
     * @param inout
     * Non void matrix of size ( real_rows, real_cols/2 + 1)
     * Content on output is the one of swap on input.
     * @param swap 
     * Can be void. Size has to be (real_rows, real_cols).
     * Contains the solution on output.
     * @attention Are you sure you normalized your coefficients with 
     * (real_rows*real_cols) before backtrafo?
     */
    inline void c2r( Matrix<complex, TL_NONE>& inout, Matrix<double, TL_DFT>& swap);

    /**
     * @brief Compute the scalar product in fourier space
     *
     * @param m1 left hand side
     * @param m2 right hand side
     * @note the normalisation missing is 1./rows/rows/cols/cols
     *
     * @return the scalar product 
     */
    double dot( const Matrix<complex, TL_NONE>& m1, const Matrix<complex, TL_NONE>& m2)
    {
#ifdef TL_DEBUG
    if( m1.rows() != m2.rows() || m1.rows() != rows)
        throw Message( "Matrix rows don't match!", _ping_);
    if( m1.cols() != m2.cols() || m1.cols() != cols/2+1 )
        throw Message( "Matrix columns don't match!", _ping_);
#endif
        complex sum=0;
        for( unsigned i=0; i<rows; i++)
        {
            sum += m1(i,0)*conj( m2(i,0));
            for( unsigned j=1; j<cols/2; j++)
                sum += 2.*m1(i,j)*conj( m2(i,j));
            if( cols%2)
                sum += m1(i,cols/2)*conj(m2(i,cols/2));
            else
                sum += 2.*m1(i,cols/2)*conj( m2(i,cols/2));
        }
        return real( sum);
    }

    /*! @brief This class shall not be copied 
     *
     * Mainly because fftw_plans are not copyable
     */
    DFT_DFT( DFT_DFT& ) = delete;
    /*! @brief This class shall not be copy assigned
     *
     * Mainly because fftw_plans are not copyable
     */
    DFT_DFT& operator=( DFT_DFT&) = delete;

    /*! @brief Free the fftw plans
     */
    ~DFT_DFT();
};

DFT_DFT::DFT_DFT( const size_t r, const size_t c, const unsigned flags):rows(r), cols(c)
{
    Matrix<double, TL_DFT> temp(rows, cols);
    forward = fftw_plan_dft_r2c_2d( rows, cols, temp.getPtr(), fftw_cast(temp.getPtr()), flags);
    backward = fftw_plan_dft_c2r_2d( rows, cols, fftw_cast(temp.getPtr()), temp.getPtr(), flags);
    if(forward == 0 )
        throw Message( "Forward Planner routine failed!", _ping_);
    if(backward == 0 )
        throw Message( "Backward Planner routine failed!", _ping_);
}
DFT_DFT::~DFT_DFT()
{
    fftw_destroy_plan( forward);
    fftw_destroy_plan( backward);
}


void DFT_DFT::r2c( Matrix<double, TL_DFT>& inout, Matrix<complex, TL_NONE>& swap)
{
#ifdef TL_DEBUG
    if( inout.rows() != rows|| inout.cols() != cols )
        throw Message( "Matrix for transformation doesn't have the right size!", _ping_);
    if( swap.rows() != rows || swap.cols() != cols/2+1 ) 
        throw Message( "Swap Matrix in r2c doesn't have the right size!", _ping_);
#endif
    fftw_execute_dft_r2c(forward, inout.getPtr(), fftw_cast(inout.getPtr()));
    swap_fields( inout, swap);

}
void DFT_DFT::c2r( Matrix<complex, TL_NONE>& inout, Matrix<double, TL_DFT>& swap)
{
#ifdef TL_DEBUG
    if( inout.rows() != rows || inout.cols() != cols/2+1) 
        throw Message( "Matrix for transformation doesn't have the right size!", _ping_);
    if( swap.rows()  != rows || swap.cols()  != cols)
        throw Message( "Swap Matrix in 2d_c2r doesn't have the right size!", _ping_);
#endif
    swap_fields( inout, swap);
    fftw_execute_dft_c2r( backward, fftw_cast(swap.getPtr()), swap.getPtr());
}

} //namespace toefl
#endif // _TL_DFT_DFT_

