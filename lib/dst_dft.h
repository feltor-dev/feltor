#ifndef _TL_DST_DFT_
#define _TL_DST_DFT_

#include <complex>
#include "matrix.h"
#include "fftw3.h"
#include "fft.h"

namespace toefl{

    /*! @brief class for twodimensional fourier transformation of Matrix
     *
     * A sine transformation is performed horizontally, i.e. along the lines 
     * of the matrices, and a discrete fourier transformation is performed vertically, i.e. along the columns. 
     * \note Do not copy or assign any Objects of this class!!
     */
    class DST_DFT
    {
      private:
        typedef std::complex<double> complex;
        const size_t rows, cols;
        fftw_plan sine_forward;
        fftw_plan sine_backward;
        fftw_plan forward;
        fftw_plan backward;
      public:
        /*! @brief prepare transformations of given size
         *
         * Uses fftw. 
         * @param real_rows # of rows in the real matrix
         * @param real_cols # of colums in the real matrix
         * @param kind Kind of the sine transformation (the backtransform kind is automatically inferred from this)
         */
        DST_DFT( const size_t real_rows, const size_t real_cols, const fftw_r2r_kind kind);
        /*! @brief execute a r2c transposing transformation
         *
         * First performs a linewise discrete sine transform followed
         * by a transposition and a linewise discrete fourier transform.
         * @param inout non void matrix of size specified in the constructor.
         * i.e. (real_rows, real_cols)
         * Content on output is the one of swap on input.
         * @param swap_T Can be void. Size has to be (real_cols, real_rows/2 + 1).
         * Contains the solution on output.
         */
        void r2c_T( Matrix<double, TL_DST_DFT>& inout, Matrix<complex, TL_NONE>& swap_T);
        /*! @brief execute a c2r transposing transformation
         *
         * First performs a linewise discrete fourier transform followed
         * by a transposition and a linewise discrete sine transform.
         * @param inout_T
         * Non void complex matrix of size (real_cols, real_rows/2 + 1)
         * Content on output is the one of swap on input.
         * @param swap
         * Can be void. Size has to be (real_rows, real_cols).
         * Contains the solution on output.
         */
        void c_T2r( Matrix<complex, TL_NONE>& inout_T, Matrix<double, TL_DST_DFT>& swap);
        //make copy construction impossible because fftw_plan cannot be copied
        /*! @brief frees all fftw plans
         */
        ~DST_DFT();
    };

    DST_DFT::DST_DFT( const size_t rows, const size_t cols, const fftw_r2r_kind kind): rows(rows), cols(cols)
    {
        const unsigned flags = FFTW_MEASURE;
        Matrix<double, TL_DST_DFT> temp( rows, cols);
        fftw_r2r_kind kind_fw = kind;
        fftw_r2r_kind kind_bw;
        switch( kind)
        {
            case( FFTW_RODFT00): kind_bw = FFTW_RODFT00;
            break;
            case( FFTW_RODFT01): kind_bw = FFTW_RODFT10;
            break;
            case( FFTW_RODFT10): kind_bw = FFTW_RODFT01;
            break;
            case( FFTW_RODFT11): kind_bw = FFTW_RODFT11;
            break;
            default: throw Message( "Kind doesn't match!", ping);
        }
        sine_forward = plan_dst_1d( rows, cols, temp.getPtr(), temp.getPtr(),kind_fw, flags);
        sine_backward = plan_dst_1d( rows, cols, temp.getPtr(), temp.getPtr(),kind_bw, flags);
        forward = plan_dft_1d_r_T2c( rows, cols, temp.getPtr(), reinterpret_cast<fftw_complex*>(temp.getPtr()), FFTW_MEASURE);
        backward = plan_dft_1d_c2r_T( rows, cols, reinterpret_cast<fftw_complex*>(temp.getPtr()), temp.getPtr(), FFTW_MEASURE);
#ifdef TL_DEBUG
        if(forward == 0|| backward == 0)
            throw Message( "r2c Planner routine failed!", ping);
        if( sine_forward == 0 || sine_backward == 0)
            throw Message( "Sine trafo Planner routine failed!", ping);
#endif
    }
    DST_DFT::~DST_DFT()
    {
        fftw_free( forward);
        fftw_free( backward);
        fftw_free( sine_forward);
        fftw_free( sine_backward);
    }

    void DST_DFT::r2c_T( Matrix<double, TL_DST_DFT>& inout, Matrix<complex, TL_NONE>& swap)
    {
#ifdef TL_DEBUG
        if( inout.rows() != rows|| inout.cols() != cols)
            throw Message( "Matrix for transformation doesn't have the right size!", ping);
        if( swap.rows() != cols|| swap.cols() != rows/2 + 1) 
            throw Message( "Swap Matrix in 2d_r2c doesn't have the right size!", ping);
#endif
        fftw_execute_r2r( sine_forward, inout.getPtr(), inout.getPtr());
        fftw_execute_dft_r2c( forward, inout.getPtr(), reinterpret_cast<fftw_complex*>(inout.getPtr()));
        swap_fields( inout, swap);
    }

    void DST_DFT::c_T2r( Matrix<complex, TL_NONE>& inout, Matrix<double, TL_DST_DFT>& swap)
    {
#ifdef TL_DEBUG
        if( inout.rows() != cols || inout.cols() != rows/2 + 1)
            throw Message( "Matrix for transformation doesn't have the right size!", ping);
        if( swap.rows() != rows || swap.cols() != cols) 
            throw Message( "Swap Matrix in 2d_r2c doesn't have the right size!", ping);
#endif
        swap_fields( inout, swap);
        fftw_execute_dft_c2r( backward, reinterpret_cast<fftw_complex*>(swap.getPtr()),swap.getPtr());
        fftw_execute_r2r( sine_backward, swap.getPtr(), swap.getPtr());
    }


}
#endif //_TL_DST_DFT_
