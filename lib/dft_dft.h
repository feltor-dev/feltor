#ifndef _TL_DFT_DFT_
#define _TL_DFT_DFT_

#include <complex>
#include "matrix.h"
#include "fftw3.h"
#include "fft.h"

namespace toefl{

/*! @brief Class for 2d discrete fourier transformations of Matrix using FFTW
 *
 * Actually just a wrapper for the dft_c2r_2d/r2c_2d plans and execute
 * functions.
 * \note Do not copy or assign any Objects of this class!!
 * \note The dft_c2r_2d execution of fftw is faster than the procedure linewise transformation - 
    transposition - linewise transformation
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
     */
    inline void c2r( Matrix<complex, TL_NONE>& inout, Matrix<double, TL_DFT>& swap);

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
        throw Message( "Forward Planner routine failed!", ping);
    if(backward == 0 )
        throw Message( "Backward Planner routine failed!", ping);
}
DFT_DFT::~DFT_DFT()
{
    fftw_free( forward);
    fftw_free( backward);
}


void DFT_DFT::r2c( Matrix<double, TL_DFT>& inout, Matrix<complex, TL_NONE>& swap)
{
#ifdef TL_DEBUG
    if( inout.rows() != rows|| inout.cols() != cols )
        throw Message( "Matrix for transformation doesn't have the right size!", ping);
    if( swap.rows() != rows || swap.cols() != cols/2+1 ) 
        throw Message( "Swap Matrix in r2c doesn't have the right size!", ping);
#endif
    fftw_execute_dft_r2c(forward, inout.getPtr(), fftw_cast(inout.getPtr()));
    swap_fields( inout, swap);

}
void DFT_DFT::c2r( Matrix<complex, TL_NONE>& inout, Matrix<double, TL_DFT>& swap)
{
#ifdef TL_DEBUG
    if( inout.rows() != rows || inout.cols() != cols/2+1) 
        throw Message( "Matrix for transformation doesn't have the right size!", ping);
    if( swap.rows()  != rows || swap.cols()  != cols)
        throw Message( "Swap Matrix in 2d_c2r doesn't have the right size!", ping);
#endif
    swap_fields( inout, swap);
    fftw_execute_dft_c2r( backward, fftw_cast(swap.getPtr()), swap.getPtr());
}

} //namespace toefl
#endif // _TL_DFT_DFT_

