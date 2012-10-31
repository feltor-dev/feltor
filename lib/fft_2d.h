#ifndef _TL_FFT_2D_
#define _TL_FFT_2D_

#include <complex>
#include "matrix.h"
#include "fftw3.h"

namespace toefl{



    class DFT_2D
    {
      private:
        typedef std::complex<double> complex;
        const size_t rows, cols;
        fftw_plan forward_a;
        fftw_plan forward_b;
        fftw_plan backward_a;
        fftw_plan backward_b;
        void plan_forward_a( Matrix<double, TL_DFT_1D>&);
        void plan_forward_b( Matrix<double, TL_DFT_1D>&);
        void plan_backward_a( Matrix<double, TL_DFT_1D>&);
        void plan_backward_b( Matrix<double, TL_DFT_1D>&);
      public:
        DFT_2D( const size_t real_rows, const size_t real_cols);
        void execute_dft_2d_r2c( Matrix<double, TL_DFT_1D>& inout, Matrix<complex, TL_NONE>& swap);
        void execute_dft_2d_c2r( Matrix<complex, TL_NONE>& inout, Matrix<double, TL_DFT_1D>& swap);
        //make copy construction impossible because fftw_plan cannot be copied
        ~DFT_2D();
    };

    DFT_2D::DFT_2D( const size_t r, const size_t c):rows(r), cols(c)
    {
        Matrix<double, TL_DFT_1D> temp(rows, cols);
        plan_forward_a(temp);
        plan_forward_b(temp);
        plan_backward_a(temp);
        plan_backward_b(temp);
#ifdef TL_DEBUG
        if(forward_a == 0 || forward_b == 0)
            throw Message( "Forward Planner routine failed!", ping);
        if(backward_a == 0 || backward_b == 0)
            throw Message( "Backward Planner routine failed!", ping);
#endif
    }
    //plan a r2c inplace routine of lines without transpositions
    void DFT_2D::plan_forward_a( Matrix<double, TL_DFT_1D>& temp)
    {
        int rank = 1;
        fftw_iodim dims[rank];
        dims[0].n  = cols;
        dims[0].is = 1; //(double)
        dims[0].os = 1; //(complex)
        int howmany_rank = 1;
        fftw_iodim howmany_dims[howmany_rank];
        howmany_dims[0].n  = rows;
        howmany_dims[0].is = cols + 2 - cols%2; //(double)
        howmany_dims[0].os = cols/2 + 1;//(complex)
        forward_a = fftw_plan_guru_dft_r2c( rank, dims, howmany_rank, howmany_dims, temp.getPtr(), reinterpret_cast<fftw_complex*>(temp.getPtr()), FFTW_MEASURE);
    }

    //plan a c2c transposing transformation
    void DFT_2D::plan_forward_b( Matrix<double, TL_DFT_1D>& temp)
    {
        int rank = 1;
        fftw_iodim dims[rank];
        dims[0].n  = rows;
        dims[0].is = cols/2 + 1;
        dims[0].os = 1; //transpose the output
        int howmany_rank = 1;
        fftw_iodim howmany_dims[howmany_rank];
        howmany_dims[0].n  = cols/2 + 1;
        howmany_dims[0].is = 1;
        howmany_dims[0].os = rows; //transpose the output
        forward_b = fftw_plan_guru_dft( rank, dims, howmany_rank, howmany_dims, reinterpret_cast<fftw_complex*>(temp.getPtr()), reinterpret_cast<fftw_complex*>(temp.getPtr()), FFTW_FORWARD, FFTW_MEASURE);
    }

    //transposing c2c transformation
    void DFT_2D::plan_backward_a( Matrix<double, TL_DFT_1D>& temp)
    {
        //backward_a
        int rank = 1;
        fftw_iodim dims[rank];
        dims[0].n  = rows;
        dims[0].is = 1; //(complex)
        dims[0].os = cols/2 + 1; // transpose (complex)
        int howmany_rank = 1;
        fftw_iodim howmany_dims[howmany_rank];
        howmany_dims[0].n  = cols/2+1;
        howmany_dims[0].is = rows;
        howmany_dims[0].os = 1; //transpose
        backward_a = fftw_plan_guru_dft( rank, dims, howmany_rank, howmany_dims, reinterpret_cast<fftw_complex*>(temp.getPtr()), reinterpret_cast<fftw_complex*>(temp.getPtr()), FFTW_BACKWARD, FFTW_MEASURE);
    }

    //linewise c2r transformation
    void DFT_2D::plan_backward_b( Matrix<double, TL_DFT_1D>& temp)
    {
        int rank = 1;
        fftw_iodim dims[rank];
        dims[0].n  = cols;
        dims[0].is = 1; //(complex)
        dims[0].os = 1; //(double)
        int howmany_rank = 1;
        fftw_iodim howmany_dims[howmany_rank];
        howmany_dims[0].n  = rows;
        howmany_dims[0].is = cols/2 + 1; //(complex)
        howmany_dims[0].os = cols + 2 - cols%2; //(double)
        backward_b = fftw_plan_guru_dft_c2r( rank, dims, howmany_rank, howmany_dims, reinterpret_cast<fftw_complex*>(temp.getPtr()), temp.getPtr(), FFTW_MEASURE);
    }
    DFT_2D::~DFT_2D()
    {
        fftw_free( forward_a);
        fftw_free( backward_a);
        fftw_free( forward_b);
        fftw_free( backward_b);
    }
    void DFT_2D::execute_dft_2d_r2c( Matrix<double, TL_DFT_1D>& inout, Matrix<complex, TL_NONE>& swap)
    {
#ifdef TL_DEBUG
        if( inout.rows() != rows|| inout.cols() != cols)
            throw Message( "Matrix for transformation doesn't have the right size!", ping);
        if( swap.rows() != cols/2+1|| swap.cols() != rows) 
            throw Message( "Swap Matrix in 2d_r2c doesn't have the right size!", ping);
#endif
        fftw_execute_dft_r2c( forward_a, inout.getPtr(), reinterpret_cast<fftw_complex*>(inout.getPtr()));
        fftw_execute_dft    ( forward_b, reinterpret_cast<fftw_complex*>(inout.getPtr()), reinterpret_cast<fftw_complex*>(inout.getPtr()));
        swap_fields( inout, swap);

    }
    void DFT_2D::execute_dft_2d_c2r( Matrix<complex, TL_NONE>& inout, Matrix<double, TL_DFT_1D>& swap)
    {
#ifdef TL_DEBUG
        if( inout.rows() != cols/2+1|| inout.cols() != rows) 
            throw Message( "Matrix for transformation doesn't have the right size!", ping);
        if( swap.rows() != rows || swap.cols() != cols)
            throw Message( "Swap Matrix in 2d_c2r doesn't have the right size!", ping);
#endif
        fftw_execute_dft( backward_a, reinterpret_cast<fftw_complex*>(inout.getPtr()), reinterpret_cast<fftw_complex*>(inout.getPtr()));
        fftw_execute_dft_c2r( backward_b, reinterpret_cast<fftw_complex*>(inout.getPtr()), reinterpret_cast<double*>(inout.getPtr()));
        swap_fields( inout, swap);
    }

}
#endif // _TL_FFT_2D_

