#ifndef _TL_DST_DST_
#define _TL_DST_DST_

#include "fftw3.h"
#include "matrix.h"
#include "fft.h"

namespace toefl
{
    /*! @brief class to make a r2r Transformation using FFTW
     * 
     * Note that RODFT00 IS significantly slower ( approx 1 oder of magnitude)
     * than RODFT01  
     */
    class DST_DST
    {
      private:
        const size_t rows, cols;
        fftw_plan sine_fw, sine_bw;
        fftw_plan sine_fw_T, sine_bw_T;
        fftw_plan transpose_fw;
        fftw_plan transpose_bw;
      public:
        DST_DST( const size_t rows, const size_t cols, const fftw_r2r_kind kind, const unsigned flags = FFTW_MEASURE);
        void r2r_T( Matrix<double, TL_NONE>& inout, Matrix<double, TL_NONE>& swap_T );
        void r_T2r( Matrix<double, TL_NONE>& inout_T, Matrix<double, TL_NONE>& swap );
    };


    DST_DST::DST_DST( const size_t rows, const size_t cols, const fftw_r2r_kind kind, const unsigned flags):rows(rows), cols(cols)
    {
        Matrix<double> temp(rows, cols);
        fftw_r2r_kind kind_fw = kind;
        fftw_r2r_kind kind_bw = inverse_kind( kind);
        sine_fw = plan_dst_1d( rows, cols, temp.getPtr(), temp.getPtr(), kind_fw, flags);
        sine_bw = plan_dst_1d( rows, cols, temp.getPtr(), temp.getPtr(), kind_bw, flags);
        transpose_fw = plan_transpose( rows, cols, temp.getPtr(), temp.getPtr(), flags);
        transpose_bw = plan_transpose( cols, rows, temp.getPtr(), temp.getPtr(), flags);
        sine_fw_T = plan_dst_1d( cols, rows, temp.getPtr(), temp.getPtr(), kind_fw, flags);
        sine_bw_T = plan_dst_1d( cols, rows, temp.getPtr(), temp.getPtr(), kind_bw, flags);
#ifdef TL_DEBUG
        if( sine_fw == 0 || sine_bw == 0 || transpose_fw == 0 || transpose_bw == 0)
            throw Message( "Plan creation failed for DST_DST!", ping);
        if( sine_fw_T == 0 || sine_bw_T == 0)
            throw Message( "Plan creation failed for transposed sine plans!",ping);
#endif
    }

    void DST_DST::r2r_T( Matrix<double, TL_NONE>& m, Matrix<double, TL_NONE>& swap)
    {
#ifdef TL_DEBUG 
        if( m.isVoid()) 
            throw Message ("Cannot r2r transpose a void Matrix", ping);
        if( m.rows() != rows || m.cols() != cols)
            throw Message( "Matrix for r2r transposition doesn't have the right size!", ping);
        if( swap.rows() != cols || swap.cols() != rows)
            throw Message( "Swap Matrix doesn't have the right size!", ping);
#endif
        fftw_execute_r2r( sine_fw, m.getPtr(), m.getPtr());
        fftw_execute_r2r( transpose_fw, m.getPtr(), m.getPtr());
        fftw_execute_r2r( sine_fw_T, m.getPtr(), m.getPtr());
        swap_fields( m, swap);
    }

    void DST_DST::r_T2r( Matrix<double, TL_NONE>& m, Matrix<double, TL_NONE>& swap)
    {
#ifdef TL_DEBUG 
        if( m.isVoid()) 
            throw Message ("Cannot r2r transpose a void Matrix", ping);
        if( m.rows() != cols || m.cols() != rows)
            throw Message( "Matrix for r2r transposition doesn't have the right size!", ping);
        if( swap.rows() != rows || swap.cols() != cols)
            throw Message( "Swap Matrix doesn't have the right size!", ping);
#endif
        fftw_execute_r2r( sine_bw_T, m.getPtr(), m.getPtr());
        fftw_execute_r2r( transpose_bw, m.getPtr(), m.getPtr());
        fftw_execute_r2r( sine_bw, m.getPtr(), m.getPtr());
        swap_fields( m, swap);
    }
   

}

#endif // _TL_DST_DST_
