#ifndef _TL_DFT_DFT_
#define _TL_DFT_DFT_

#include <complex>
#include "matrix.h"
#include "fftw3.h"
#include "fft.h"

namespace toefl{

/* fftw guru interface
 *
 * fftw_iodim{int n,  // Größe der Dimension des Index 
 *            int is, //in stride
 *            int os} //out strid
 * rank, fftw_iodim dims[rank] //describe how you come to the next point inside a trafo for every index i.e. dims[0] describes the first index of the matrix m[i0][i1]...[i_rank-1]
 * howmany_rank, fftw_iodim howmany_dims[howmany_rank] //describe how you come to the first point of the next trafo
 */


    /*! @brief Class for 2d discrete fourier transformations of Matrix
     *
     * \note Do not copy or assign any Objects of this class!!
     */
    class DFT_DFT
    {
      private:
        typedef std::complex<double> complex;
        const size_t rows, cols;
        fftw_plan forward_a;
        fftw_plan forward_b;
        fftw_plan backward_b;
        fftw_plan backward_a;
        void plan_forward_b( Matrix<double, TL_DFT_DFT>&);
        void plan_backward_b( Matrix<double, TL_DFT_DFT>&);
      public:
        /*! @brief Prepare a 2d discrete fourier transformation of given size
         *
         * Uses fftw. 
         * @param real_rows # of rows in the real matrix
         * @param real_cols # of colums in the real matrix
         */
        DFT_DFT( const size_t real_rows, const size_t real_cols);
        /*! @brief Execute a r2c transposing transformation
         *
         * First performs a linewise discrete fourier transform followed
         * by a transposition and another linewise discrete fourier transform.
         * @param inout non void matrix of size specified in the constructor.
         * i.e. (real_rows, real_cols)
         * Content on output is the one of swap on input.
         * @param swap_T Can be void. Size has to be (real_cols/2 + 1, real_rows).
         * Contains the solution on output.
         */
        void r2c_T( Matrix<double, TL_DFT_DFT>& inout, Matrix<complex, TL_NONE>& swap_T);

        /*! @brief Execute a c2r transposing transformation
         *
         * First perform a linewise discrete fourier transform followed
         * by a transposition and another linewise discrete fourier transform.
         * @param inout_T
         * Non void matrix of size (real_cols/2 + 1, real_rows)
         * Content on output is the one of swap on input.
         * @param swap 
         * Can be void. Size has to be (real_rows, real_cols).
         * Contains the solution on output.
         */
        void c_T2r( Matrix<complex, TL_NONE>& inout_T, Matrix<double, TL_DFT_DFT>& swap);
        //make copy construction impossible because fftw_plan cannot be copied
        /*! @brief Free the fftw plans
         */
        ~DFT_DFT();
    };

    DFT_DFT::DFT_DFT( const size_t r, const size_t c):rows(r), cols(c)
    {
        Matrix<double, TL_DFT_DFT> temp(rows, cols);
        forward_a = plan_dft_1d_r2c( rows, cols, temp.getPtr(), reinterpret_cast<fftw_complex*>(temp.getPtr()), FFTW_MEASURE);
        plan_forward_b(temp);
        plan_backward_b(temp);
        backward_a = plan_dft_1d_c2r( rows, cols, reinterpre_cast<fftw_complex*>(temp.getPtr()), FFTW_MEASURE) ;
#ifdef TL_DEBUG
        if(forward_a == 0 || forward_b == 0)
            throw Message( "Forward Planner routine failed!", ping);
        if(backward_b == 0 || backward_a == 0)
            throw Message( "Backward Planner routine failed!", ping);
#endif
    }
    DFT_DFT::~DFT_DFT()
    {
        fftw_free( forward_a);
        fftw_free( backward_b);
        fftw_free( forward_b);
        fftw_free( backward_a);
    }
    //plan a r2c inplace routine of lines without transpositions
    void DFT_DFT::plan_forward_a( Matrix<double, TL_DFT_DFT>& temp)
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

    //plan a c_T2c transposing transformation (i.e. read transposed)
    void DFT_DFT::plan_forward_b( Matrix<double, TL_DFT_DFT>& temp)
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

    //transposing c2c_T transformation (i.e. write transposed)
    void DFT_DFT::plan_backward_b( Matrix<double, TL_DFT_DFT>& temp)
    {
        //backward_b
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
        backward_b = fftw_plan_guru_dft( rank, dims, howmany_rank, howmany_dims, reinterpret_cast<fftw_complex*>(temp.getPtr()), reinterpret_cast<fftw_complex*>(temp.getPtr()), FFTW_BACKWARD, FFTW_MEASURE);
    }

    //linewise c2r transformation
    void DFT_DFT::plan_backward_a( Matrix<double, TL_DFT_DFT>& temp)
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
        backward_a = fftw_plan_guru_dft_c2r( rank, dims, howmany_rank, howmany_dims, reinterpret_cast<fftw_complex*>(temp.getPtr()), temp.getPtr(), FFTW_MEASURE);
    }
    void DFT_DFT::r2c_T( Matrix<double, TL_DFT_DFT>& inout, Matrix<complex, TL_NONE>& swap)
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
    void DFT_DFT::c_T2r( Matrix<complex, TL_NONE>& inout, Matrix<double, TL_DFT_DFT>& swap)
    {
#ifdef TL_DEBUG
        if( inout.rows() != cols/2+1|| inout.cols() != rows) 
            throw Message( "Matrix for transformation doesn't have the right size!", ping);
        if( swap.rows() != rows || swap.cols() != cols)
            throw Message( "Swap Matrix in 2d_c2r doesn't have the right size!", ping);
#endif
        fftw_execute_dft( backward_b, reinterpret_cast<fftw_complex*>(inout.getPtr()), reinterpret_cast<fftw_complex*>(inout.getPtr()));
        fftw_execute_dft_c2r( backward_a, reinterpret_cast<fftw_complex*>(inout.getPtr()), reinterpret_cast<double*>(inout.getPtr()));
        swap_fields( inout, swap);
    }

}
#endif // _TL_DFT_DFT_

