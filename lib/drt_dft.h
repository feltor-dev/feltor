#ifndef _TL_DRT_DFT_
#define _TL_DRT_DFT_

#include <complex>
#include "matrix.h"
#include "fftw3.h"
#include "fft.h"

namespace toefl{

    /*! @brief class for twodimensional fourier transformation of Matrix
     *
     * A r2r transformation is performed horizontally, i.e. along the lines 
     * of the matrices, and a discrete fourier transformation is performed vertically, i.e. along the columns. 
     * \note Do not copy or assign any Objects of this class!!
     */
    class DRT_DFT
    {
      private:
        typedef std::complex<double> complex;
        const size_t rows, cols;
        fftw_plan real_forward;
        fftw_plan real_backward;
        fftw_plan forward;
        fftw_plan backward;
      public:
        /*! @brief prepare transformations of given size
         *
         * @param real_rows # of rows in the real matrix
         * @param real_cols # of colums in the real matrix
         * @param kind Kind of the r2r transformation (the backtransform kind is automatically inferred from this)
         * @param flags one of the fftw performance flags
         */
        DRT_DFT( const size_t real_rows, const size_t real_cols, const fftw_r2r_kind kind, const unsigned flags = FFTW_MEASURE);
        /*! @brief execute a r2c transposing transformation
         *
         * First perform a linewise discrete r2r transform followed
         * by a transposition and a linewise discrete fourier transform.
         * @param inout non void matrix of size specified in the constructor.
         * i.e. (real_rows, real_cols)
         * Content on output is the one of swap on input.
         * @param swap_T Can be void. Size has to be (real_cols, real_rows/2 + 1).
         * Contains the solution on output.
         */
        void r2c_T( Matrix<double, TL_DRT_DFT>& inout, Matrix<complex, TL_NONE>& swap_T);
        /*! @brief execute a c2r transposing transformation
         *
         * First perform a linewise discrete fourier transform followed
         * by a transposition and a linewise discrete real transform.
         * @param inout_T
         * Non void complex matrix of size (real_cols, real_rows/2 + 1)
         * Content on output is the one of swap on input.
         * @param swap
         * Can be void. Size has to be (real_rows, real_cols).
         * Contains the solution on output.
         */
        void c_T2r( Matrix<complex, TL_NONE>& inout_T, Matrix<double, TL_DRT_DFT>& swap);
        //make copy construction impossible because fftw_plan cannot be copied
        /*! @brief Free all fftw plans
         */
        ~DRT_DFT();
    };

    DRT_DFT::DRT_DFT( const size_t rows, const size_t cols, const fftw_r2r_kind kind, const unsigned flags): rows(rows), cols(cols)
    {
        Matrix<double, TL_DRT_DFT> temp( rows, cols);
        fftw_r2r_kind kind_fw = kind;
        fftw_r2r_kind kind_bw = inverse_kind(kind);
        real_forward = plan_drt_1d( rows, cols, temp.getPtr(), temp.getPtr(),kind_fw, flags);
        real_backward = plan_drt_1d( rows, cols, temp.getPtr(), temp.getPtr(),kind_bw, flags);
        forward = plan_dft_1d_r_T2c( rows, cols, temp.getPtr(), reinterpret_cast<fftw_complex*>(temp.getPtr()), FFTW_MEASURE);
        backward = plan_dft_1d_c2r_T( rows, cols, reinterpret_cast<fftw_complex*>(temp.getPtr()), temp.getPtr(), FFTW_MEASURE);
#ifdef TL_DEBUG
        if(forward == 0|| backward == 0)
            throw Message( "r2c Planner routine failed!", ping);
        if( real_forward == 0 || real_backward == 0)
            throw Message( "Sine trafo Planner routine failed!", ping);
#endif
    }
    DRT_DFT::~DRT_DFT()
    {
        fftw_free( forward);
        fftw_free( backward);
        fftw_free( real_forward);
        fftw_free( real_backward);
    }

    void DRT_DFT::r2c_T( Matrix<double, TL_DRT_DFT>& inout, Matrix<complex, TL_NONE>& swap)
    {
#ifdef TL_DEBUG
        if( inout.rows() != rows|| inout.cols() != cols)
            throw Message( "Matrix for transformation doesn't have the right size!", ping);
        if( swap.rows() != cols|| swap.cols() != rows/2 + 1) 
            throw Message( "Swap Matrix in 2d_r2c doesn't have the right size!", ping);
#endif
        fftw_execute_r2r( real_forward, inout.getPtr(), inout.getPtr());
        fftw_execute_dft_r2c( forward, inout.getPtr(), reinterpret_cast<fftw_complex*>(inout.getPtr()));
        swap_fields( inout, swap);
    }

    void DRT_DFT::c_T2r( Matrix<complex, TL_NONE>& inout, Matrix<double, TL_DRT_DFT>& swap)
    {
#ifdef TL_DEBUG
        if( inout.rows() != cols || inout.cols() != rows/2 + 1)
            throw Message( "Matrix for transformation doesn't have the right size!", ping);
        if( swap.rows() != rows || swap.cols() != cols) 
            throw Message( "Swap Matrix in 2d_r2c doesn't have the right size!", ping);
#endif
        swap_fields( inout, swap);
        fftw_execute_dft_c2r( backward, reinterpret_cast<fftw_complex*>(swap.getPtr()),swap.getPtr());
        fftw_execute_r2r( real_backward, swap.getPtr(), swap.getPtr());
    }


}
#endif //_TL_DRT_DFT_
