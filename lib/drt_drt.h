#ifndef _TL_DRT_DRT_
#define _TL_DRT_DRT_

#include "fftw3.h"
#include "matrix.h"
#include "fft.h"

namespace toefl
{
    /*! @brief class to make a r2r transformation using FFTW
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
        DRT_DRT( const size_t, const size_t , const fftw_r2r_kind , const fftw_r2r_kind, const unsigned = FFTW_MEASURE);
        ~DRT_DRT();
        void forward( Matrix<double, TL_NONE>& inout, Matrix<double, TL_NONE>& swap );
        void backward( Matrix<double, TL_NONE>& inout, Matrix<double, TL_NONE>& swap );
    };


    /*! @brief Prepare a 2d discrete fourier transformation of given size
     *
     * @param rows # of rows in the real matrix
     * @param cols # of colums in the real matrix
     * @param kind0 hoizontal kind of transformation
     * @param kind1 vertical kind of transformation 
     * @param flags flags for plan creation 
     */
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

    /*! @brief 
     */
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
