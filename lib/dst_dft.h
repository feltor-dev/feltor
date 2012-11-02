#ifndef _TL_DST_DFT_
#define _TL_DST_DFT_

#include <complex>
#include "matrix.h"
#include "fftw3.h"

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
        fftw_plan forward_a;
        fftw_plan forward_b;
        fftw_plan backward_a;
        fftw_plan backward_b;
        void plan_forward_a ( Matrix<double, TL_DST_DFT>&);
        void plan_forward_b ( Matrix<double, TL_DST_DFT>&);
        void plan_backward_a( Matrix<double, TL_DST_DFT>&);
        void plan_backward_b( Matrix<double, TL_DST_DFT>&);
      public:
        /*! @brief prepare transformations of given size
         *
         * Uses fftw. 
         * @param real_rows # of rows in the real matrix
         * @param real_cols # of colums in the real matrix
         */
        DST_DFT( const size_t real_rows, const size_t real_cols);
        /*! @brief execute a r2c transposing transformation
         *
         * First performs a linewise discrete sine transform followed
         * by a transposition and a linewise discrete fourier transform.
         * @param inout non void matrix of size specified in the constructor.
         * i.e. (real_rows, real_cols)
         * Content on output is the one of swap on input.
         * @param swap Can be void. Size has to be (real_cols, real_rows/2 + 1).
         * Contains the solution on output.
         */
        void r2c( Matrix<double, TL_DST_DFT>& inout, Matrix<complex, TL_NONE>& swap_T);
        /*! @brief execute a c2r transposing transformation
         *
         * First performs a linewise discrete fourier transform followed
         * by a transposition and a linewise discrete sine transform.
         * @param inout
         * Non void complex matrix of size (real_cols, real_rows/2 + 1)
         * Content on output is the one of swap on input.
         * @param swap 
         * Can be void. Size has to be (real_rows, real_cols).
         * Contains the solution on output.
         */
        void c2r( Matrix<complex, TL_NONE>& inout_T, Matrix<double, TL_DST_DFT>& swap);
        //make copy construction impossible because fftw_plan cannot be copied
        /*! @brief frees all fftw plans
         */
        ~DST_DFT();
    };

    DST_DFT::DST_DFT( const size_t rows, const size_t cols): rows(rows), cols(cols)
    {
        Matrix<double, TL_DST_DFT> temp( rows, cols);
        plan_forward_a(temp);
        plan_forward_b(temp);
        plan_backward_a(temp);
        plan_backward_b(temp);
#ifdef TL_DEBUG
        if(forward_a == 0|| forward_b == 0)
            throw Message( "Forward Planner routine failed!", ping);
        if(backward_a == 0 || backward_b == 0)
            throw Message( "Backward Planner routine failed!", ping);
#endif
    }
    DST_DFT::~DST_DFT()
    {
        fftw_free( forward_a);
        fftw_free( backward_a);
        fftw_free( forward_b);
        fftw_free( backward_b);
    }
    //transposing sine trafo into padded array
    //DFT_1D Padding not possible for result i.e. padding is in the end
    //transposition not possible
    void DST_DFT::plan_forward_a ( Matrix<double, TL_DST_DFT>& temp)
    {
        int rank = 1;
        fftw_r2r_kind kind[] = {FFTW_RODFT00};
        fftw_iodim dims[rank];
        dims[0].n  = cols;
        dims[0].is = 1;
        dims[0].os = 1;
        int howmany_rank = 1;
        fftw_iodim howmany_dims[howmany_rank];
        howmany_dims[0].n  = rows;
        howmany_dims[0].is = cols;
        howmany_dims[0].os = cols;
        forward_a = fftw_plan_guru_r2r( rank, dims, howmany_rank, howmany_dims, temp.getPtr(), temp.getPtr(), kind, FFTW_MEASURE);
    }

    //transposing linewise r2c trafo
    void DST_DFT::plan_forward_b ( Matrix<double, TL_DST_DFT>& temp)
    {
        int rank = 1;
        fftw_iodim dims[rank];
        dims[0].n  = rows;
        dims[0].is = cols; //(double)
        dims[0].os = 1; //(complex)
        int howmany_rank = 1;
        fftw_iodim howmany_dims[howmany_rank];
        howmany_dims[0].n  = cols;
        howmany_dims[0].is = 1;
        howmany_dims[0].os = rows/2 + 1;
        forward_b = fftw_plan_guru_dft_r2c( rank, dims, howmany_rank, howmany_dims, temp.getPtr(), reinterpret_cast<fftw_complex*>(temp.getPtr()), FFTW_MEASURE);

    }

    // transposing c2r
    // pad result in the end (cf forward trafo)
    void DST_DFT::plan_backward_a( Matrix<double, TL_DST_DFT>& temp)
    {
        int rank = 1;
        fftw_iodim dims[rank];
        dims[0].n  = rows;
        dims[0].is = 1;
        dims[0].os = cols;
        int howmany_rank = 1;
        fftw_iodim howmany_dims[howmany_rank];
        howmany_dims[0].n  = cols;
        howmany_dims[0].is = rows/2 + 1;
        howmany_dims[0].os = 1;
        backward_a = fftw_plan_guru_dft_c2r( rank, dims, howmany_rank, howmany_dims, reinterpret_cast<fftw_complex*>(temp.getPtr()), temp.getPtr(),  FFTW_MEASURE);
    }
    //transposing r2r transform of columns
    void DST_DFT::plan_backward_b( Matrix<double, TL_DST_DFT>& temp)
    {
        int rank = 1;
        fftw_r2r_kind kind[] = {FFTW_RODFT00};
        fftw_iodim dims[rank];
        dims[0].n  = cols;
        dims[0].is = 1;
        dims[0].os = 1;
        int howmany_rank = 1;
        fftw_iodim howmany_dims[howmany_rank];
        howmany_dims[0].n  = rows;
        howmany_dims[0].is = cols;
        howmany_dims[0].os = cols;
        backward_b = fftw_plan_guru_r2r( rank, dims, howmany_rank, howmany_dims, temp.getPtr(), temp.getPtr(), kind, FFTW_MEASURE);
    }

    void DST_DFT::r2c( Matrix<double, TL_DST_DFT>& inout, Matrix<complex, TL_NONE>& swap)
    {
#ifdef TL_DEBUG
        if( inout.rows() != rows|| inout.cols() != cols)
            throw Message( "Matrix for transformation doesn't have the right size!", ping);
        if( swap.rows() != cols|| swap.cols() != rows/2 + 1) 
            throw Message( "Swap Matrix in 2d_r2c doesn't have the right size!", ping);
#endif
        fftw_execute_r2r( forward_a, inout.getPtr(), inout.getPtr());
        fftw_execute_dft_r2c( forward_b, inout.getPtr(), reinterpret_cast<fftw_complex*>(inout.getPtr()));
        swap_fields( inout, swap);
    }

    void DST_DFT::c2r( Matrix<complex, TL_NONE>& inout, Matrix<double, TL_DST_DFT>& swap)
    {
#ifdef TL_DEBUG
        if( inout.rows() != cols || inout.cols() != rows/2 + 1)
            throw Message( "Matrix for transformation doesn't have the right size!", ping);
        if( swap.rows() != rows || swap.cols() != cols) 
            throw Message( "Swap Matrix in 2d_r2c doesn't have the right size!", ping);
#endif
        swap_fields( inout, swap);
        fftw_execute_dft_c2r( backward_a, reinterpret_cast<fftw_complex*>(swap.getPtr()),swap.getPtr());
        fftw_execute_r2r( backward_b, swap.getPtr(), swap.getPtr());
    }


}
#endif //_TL_DST_DFT_
