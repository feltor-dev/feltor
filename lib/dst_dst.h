#ifndef _TL_DST_DST_
#define _TL_DST_DST_

#include "fftw3.h"
#include "matrix.h"
#include "fft.h"

namespace toefl
{
    class DST_DST
    {
      private:
        const size_t rows, cols;
        fftw_plan sine_fw, sine_bw;
        fftw_plan sine_fw_T, sine_bw_T;
        fftw_plan transpose_fw;
        fftw_plan transpose_bw;
      public:
        DST_DST( const size_t rows, const size_t cols, const fftw_r2r_kind kind);
        void r2r_T( Matrix<double, TL_NONE>& inout, Matrix<double, TL_NONE>& swap_T );
        void r_T2r( Matrix<double, TL_NONE>& inout_T, Matrix<double, TL_NONE>& swap );
    };


    DST_DST::DST_DST( const size_t rows, const size_t cols, const fftw_r2r_kind kind):rows(rows), cols(cols)
    {
        Matrix<double> temp(rows, cols);
        const unsigned flags = FFTW_MEASURE; /* other flags are possible */
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
