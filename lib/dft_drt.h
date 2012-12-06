#ifndef _TL_DFT_DRT_
#define _TL_DFT_DRT_

#include <complex>
#include "fftw3.h"
#include "matrix.h"
#include "fft.h"

namespace toefl
{
/*! @brief Expansion class of DRT_DRT for periodic BC in the horizontal direction
 *
 * First transforms linewise r2c, then transposes (r2r) and transforms linewise r2r.
 * The result is transposed back (r2r). 
 * The Backward transform goes the same way in the other direction.
 * @note Because of the extra transposes the transformation is not as fast
 * as it could be. The fastest would be a 2d r2r transform of r2hc type. 
 * But then the halfcomplex format is so ugly that we don't want to do this.
 */
class DFT_DRT
{
  private:
    typedef std::complex<double> complex;
    const size_t rows, cols;
    fftw_plan forward;
    fftw_plan backward; 
    fftw_plan transpose_forward;
    fftw_plan transpose_backward; 
    fftw_plan r2r_forward;
    fftw_plan r2r_backward;
  public:
    DFT_DRT( const size_t rows, const size_t cols, const fftw_r2r_kind kind, const unsigned = FFTW_MEASURE);
    ~DFT_DRT();
    void r2c( Matrix<double, TL_DFT>& inout, Matrix<complex, TL_NONE>& swap_T);
    void c2r( Matrix<complex, TL_NONE>& inout_T, Matrix<double, TL_DFT>& swap);

    /*! @brief This class shall not be copied 
     *
     * Mainly because fftw_plans are not copyable
     */
    DFT_DRT( DFT_DRT& ) = delete;
    /*! @brief This class shall not be copy assigned
     *
     * Mainly because fftw_plans are not copyable
     */
    DFT_DRT& operator=( DFT_DRT&) = delete;
};

   /*! @brief prepare the fftw_plans
    * 
    * @param rows # of rows in the real matrix
    @param cols # of cols in the real matrix 
    @param kind one of the fftw_r2r_kind
    \param flags one of the fftw flags
    */
DFT_DRT::DFT_DRT( const size_t rows, const size_t cols, const fftw_r2r_kind kind, const unsigned flags):rows(rows), cols(cols)
{
    Matrix< double, TL_DFT> temp(rows, cols);
    forward = plan_dft_1d_r2c( rows, cols, temp.getPtr(), reinterpret_cast<fftw_complex*>(temp.getPtr()), flags);
    backward = plan_dft_1d_c2r( rows, cols, reinterpret_cast<fftw_complex*>(temp.getPtr()), temp.getPtr(), flags);
    transpose_forward = plan_transpose( rows, cols + 2 -cols%2, temp.getPtr(), temp.getPtr(), flags);
    transpose_backward = plan_transpose( cols + 2 -cols%2, rows, temp.getPtr(), temp.getPtr(), flags);

    fftw_r2r_kind kind_fw = kind;
    fftw_r2r_kind kind_bw = inverse_kind( kind);

    r2r_forward  = plan_drt_1d( cols + 2 - cols%2, rows, temp.getPtr(), temp.getPtr(), kind_fw, flags);
    r2r_backward = plan_drt_1d( cols + 2 - cols%2, rows, temp.getPtr(), temp.getPtr(), kind_bw, flags);
#ifdef TL_DEBUG
    if( forward == NULL || backward == NULL)
        throw Message( "r2c plan creation failed!", ping);
    if( transpose_forward == NULL || transpose_backward == NULL)
        throw Message( "transpose plan creation failed!", ping);
    if( r2r_forward == NULL || r2r_backward == NULL)
        throw Message( "r2r plan creation failed!", ping);
#endif

}
/*! @brief frees the fftw plans*/
DFT_DRT::~DFT_DRT()
{
    fftw_free( forward);
    fftw_free( backward);
    fftw_free( transpose_forward);
    fftw_free( transpose_backward);
    fftw_free( r2r_forward);
    fftw_free( r2r_backward);
}

/*! @brief Perform a r2c transformation

Transformations are always done inplace, if you want to preserve input copy it first. 
\param m contains values to be transformed, contains memory of swap on output //A DST2 is DFT10 NOT DFT01 !!
\param swap contains transformed values on output (maybe void)
*/
void DFT_DRT::r2c( Matrix<double, TL_DFT>& m, Matrix<complex, TL_NONE>& swap)
{
#ifdef TL_DEBUG
    if( m.rows() != rows || m.cols() != cols)
        throw Message( "Matrix hasn't got the right size!", ping);
    if( m.isVoid())  
        throw Message( "Cannot transform a void matrix!", ping);
    if( swap.rows() != rows || swap.cols()!= cols/2 +1)
        throw Message( "Swap Matrix has wrong size!", ping);
#endif
    fftw_execute_dft_r2c( forward, m.getPtr(), reinterpret_cast<fftw_complex*>(m.getPtr()));
    fftw_execute_r2r( transpose_forward, m.getPtr(), m.getPtr());
    fftw_execute_r2r( r2r_forward, m.getPtr(), m.getPtr());
    fftw_execute_r2r( transpose_backward, m.getPtr(), m.getPtr());
    swap_fields( m, swap);
}
/*! @brief Perform a c2r transformation

 * Transformations are always done inplace, if you want to preserve input copy it first. 
 * \param m contains values to be transformed, contains memory of swap on output
 * \param swap contains transformed values on output (maybe void)
 * @attention Are you sure you normalized your coefficients before backtrafo?
*/
void DFT_DRT::c2r( Matrix<complex, TL_NONE>& m, Matrix<double, TL_DFT>& swap)
{
#ifdef TL_DEBUG
    if( swap.rows() != rows || swap.cols() != cols)
        throw Message( "Matrix hasn't got the right size!", ping);
    if( m.isVoid())  
        throw Message( "Cannot transform a void matrix!", ping);
    if( m.rows() != rows || m.cols()!= cols/2 +1)
        throw Message( "Swap Matrix has wrong size!", ping);
#endif
    swap_fields( m, swap);
    fftw_execute_r2r( transpose_forward, swap.getPtr(), swap.getPtr());
    fftw_execute_r2r( r2r_backward, swap.getPtr(), swap.getPtr());
    fftw_execute_r2r( transpose_backward, swap.getPtr(), swap.getPtr());
    fftw_execute_dft_c2r( backward, reinterpret_cast<fftw_complex*>(swap.getPtr()), swap.getPtr());
}
} //namespace toefl

#endif //_TL_DFT_DRT_
