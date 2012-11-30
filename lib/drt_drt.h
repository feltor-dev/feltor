#ifndef _TL_DRT_DRT_
#define _TL_DRT_DRT_

#include "fftw3.h"
#include "matrix.h"
#include "fft.h"

namespace toefl
{
/*! @brief Class for 2d r2r transformation of Matrix using FFTW
 * 
 * Actually just a wrapper for the r2r_2d plans and execute functions
 * Note that RODFT00 IS significantly slower ( up to a factor of 10)
 * than RODFT01. 
 * In or out-of place takes approx same time
 */
class DRT_DRT
{
  private:
    const size_t rows, cols;
    fftw_plan forward_;
    fftw_plan backward_;
  public:
    /*! @brief Prepare a 2d discrete fourier transformation of given size
     *
     * @param rows # of rows in the real matrix
     * @param cols # of colums in the real matrix
     * @param horizontal_kind hoizontal kind of transformation
     * @param vertical_kind vertical kind of transformation 
     * @param flags flags for plan creation 
     */
    DRT_DRT( const size_t rows, const size_t cols, const fftw_r2r_kind horizontal_kind , const fftw_r2r_kind vertical_kind, const unsigned = FFTW_MEASURE);
    ~DRT_DRT();
    /*! @brief Forward 2d r2r transform
     *
     * @param inout
     * Non void real matrix of size given on construction.
     * Content on output is the one of swap on input.
     * @param swap
     * Can be void. Size has to be equal to inout.
     * Contains the solution on output.
     */
    void forward( Matrix<double, TL_NONE>& inout, Matrix<double, TL_NONE>& swap );
    /*! @brief Backward 2d r2r transform
     *
     * The kind of the backward transformations given by the inverse_kind() function
     * in fft.h.
     * @param inout
     * Non void real matrix of size given on construction.
     * Content on output is the one of swap on input.
     * @param swap
     * Can be void. Size has to be equal to inout.
     * Contains the solution on output.
     */
    void backward( Matrix<double, TL_NONE>& inout, Matrix<double, TL_NONE>& swap );
    /*! @brief This class shall not be copied 
     *
     * Mainly because fftw_plans are not copyable
     */
    DRT_DRT( DRT_DRT& ) = delete;
    /*! @brief This class shall not be copy assigned
     *
     * Mainly because fftw_plans are not copyable
     */
    DRT_DRT& operator=( DRT_DRT&) = delete;
};


DRT_DRT::DRT_DRT( const size_t rows, const size_t cols, const fftw_r2r_kind kind0, const fftw_r2r_kind kind1, const unsigned flags):rows(rows), cols(cols)
{
    Matrix<double> m0(rows, cols);
    fftw_r2r_kind kind_inv0 = inverse_kind( kind0);
    fftw_r2r_kind kind_inv1 = inverse_kind( kind1);
    forward_  = fftw_plan_r2r_2d( rows, cols, m0.getPtr(), m0.getPtr(), kind1, kind0, flags); 
    backward_ = fftw_plan_r2r_2d( rows, cols, m0.getPtr(), m0.getPtr(), kind_inv1, kind_inv0, flags); 
    if( forward_ == 0)
        throw Message( "Forward_ plan creation failed for DRT_DRT!", ping);
    if( backward_ == 0 )
        throw Message( "Backward_ plan creation failed for DRT_DRT!",ping);
}
/*! @brief free all created fftw plans
 */
DRT_DRT::~DRT_DRT()
{
    fftw_free( forward_);
    fftw_free( backward_);
}

void DRT_DRT::forward( Matrix<double, TL_NONE>& m, Matrix<double, TL_NONE>& swap)
{
#ifdef TL_DEBUG 
    if( m.isVoid()) 
        throw Message ("Cannot r2r transpose a void Matrix", ping);
    if( m.rows() != rows || m.cols() != cols)
        throw Message( "Matrix for r2r transposition doesn't have the right size!", ping);
    if( swap.rows() != rows || swap.cols() != cols)
        throw Message( "Swap Matrix doesn't have the right size!", ping);
#endif
    fftw_execute_r2r( forward_, m.getPtr(), m.getPtr());
    swap_fields( m, swap);
}

void DRT_DRT::backward( Matrix<double, TL_NONE>& m, Matrix<double, TL_NONE>& swap)
{
#ifdef TL_DEBUG 
    if( m.isVoid()) 
        throw Message ("Cannot r2r transpose a void Matrix", ping);
    if( m.rows() != rows || m.cols() != cols)
        throw Message( "Matrix for r2r transposition doesn't have the right size!", ping);
    if( swap.rows() != rows || swap.cols() != cols)
        throw Message( "Swap Matrix doesn't have the right size!", ping);
#endif
    fftw_execute_r2r( backward_, m.getPtr(), m.getPtr());
    swap_fields( m, swap);
}


}

#endif // _TL_DRT_DRT_
