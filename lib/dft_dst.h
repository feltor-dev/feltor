#ifndef _TL_DFT_1D_
#define _TL_DFT_1D_

#include <complex>
#include "fftw3.h"
#include "matrix.h"
#include "fft.h"

namespace toefl
{
    class DFT_DST
    {
      private:
        typedef std::complex<double> complex;
        const size_t rows, cols;
        fftw_plan forward;
        fftw_plan backward; 
        fftw_plan transpose_forward;
        fftw_plan transpose_backward; 
        fftw_plan sine_forward;
        fftw_plan sine_backward;
      public:
        DFT_DST( const size_t rows, const size_t cols, const fftw_r2r_kind kind);
        ~DFT_DST();
        void r2c( Matrix<double, TL_DFT_1D>& inout, Matrix<complex, TL_NONE>& swap_T);
        void c2r( Matrix<complex, TL_NONE>& inout_T, Matrix<double, TL_DFT_1D>& swap);
    };

    DFT_DST::DFT_DST( const size_t rows, const size_t cols, const fftw_r2r_kind kind):rows(rows), cols(cols)
    {
        Matrix< double, TL_DFT_1D> temp(rows, cols);
        const unsigned flags = FFTW_MEASURE;
        forward = plan_dft_1d_r2c( rows, cols, temp.getPtr(), reinterpret_cast<fftw_complex*>(temp.getPtr()), flags);
        backward = plan_dft_1d_c2r( rows, cols, reinterpret_cast<fftw_complex*>(temp.getPtr()), temp.getPtr(), flags);
        transpose_forward = plan_transpose( rows, cols + 2 -cols%2, temp.getPtr(), temp.getPtr(), flags);
        transpose_backward = plan_transpose( cols + 2 -cols%2, rows, temp.getPtr(), temp.getPtr(), flags);

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

        sine_forward  = plan_dst_1d( cols + 2 - cols%2, rows, temp.getPtr(), temp.getPtr(), kind_fw, flags);
        sine_backward = plan_dst_1d( cols + 2 - cols%2, rows, temp.getPtr(), temp.getPtr(), kind_bw, flags);
#ifdef TL_DEBUG
        if( forward == NULL || backward == NULL)
            throw Message( "r2c plan creation failed!", ping);
        if( transpose_forward == NULL || transpose_backward == NULL)
            throw Message( "transpose plan creation failed!", ping);
        if( sine_forward == NULL || sine_backward == NULL)
            throw Message( "sine plan creation failed!", ping);
#endif

    }
    DFT_DST::~DFT_DST()
    {
        fftw_free( forward);
        fftw_free( backward);
        fftw_free( transpose_forward);
        fftw_free( transpose_backward);
        fftw_free( sine_forward);
        fftw_free( sine_backward);
    }

    void DFT_DST::r2c( Matrix<double, TL_DFT_1D>& m, Matrix<complex, TL_NONE>& swap)
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
        fftw_execute_r2r( sine_forward, m.getPtr(), m.getPtr());
        fftw_execute_r2r( transpose_backward, m.getPtr(), m.getPtr());
        swap_fields( m, swap);
    }
    void DFT_DST::c2r( Matrix<complex, TL_NONE>& m, Matrix<double, TL_DFT_1D>& swap)
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
        fftw_execute_r2r( sine_backward, swap.getPtr(), swap.getPtr());
        fftw_execute_r2r( transpose_backward, swap.getPtr(), swap.getPtr());
        fftw_execute_dft_c2r( backward, reinterpret_cast<fftw_complex*>(swap.getPtr()), swap.getPtr());
    }
}

#endif //_TL_DFT_1D_
