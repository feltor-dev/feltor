/*! \file
 * @brief wrapper functions for easy fftw plan creation 
 * @author Matthias Wiesenberger
 * @email Matthias.Wiesenberger@uibk.ac.at
 */
#ifndef _TL_FFT_
#define _TL_FFT_

#include <complex>
#include "matrix.h"
#include "fftw3.h"

//NOTE: The 2d r2c fftw padding is not like described in the documentation! 
//It's like for a 1d r2c trafo
namespace toefl{

    inline fftw_complex* fftw_cast( double * const ptr){ return reinterpret_cast<fftw_complex*> (ptr);}
    
    //plan sine transformation of many rows
    fftw_plan plan_dst_1d( const size_t rows, const size_t cols, double *in, double *out, const fftw_r2r_kind kind, const unsigned flags)
    {
        int rank = 1, howmany_rank = 1;
        fftw_iodim dims[rank], howmany_dims[howmany_rank];
        fftw_r2r_kind kind_[] = {kind};
        dims[0].n  = cols;
        dims[0].is = 1;
        dims[0].os = 1;
        howmany_dims[0].n  = rows;
        howmany_dims[0].is = cols;
        howmany_dims[0].os = cols;
        return fftw_plan_guru_r2r( rank, dims, howmany_rank, howmany_dims, in, out, kind_, flags);
    }
    fftw_plan plan_dst_1d_c2c( const size_t rows, const size_t cols, double* in, double* out, const int offset, const unsigned flags);
    //from fftws website
    /* You can use the FFTW guru interface to create a rank-0 transform of vector rank 2 where the vector strides are transposed. 
     * (A rank-0 transform is equivalent to a 1D transform of size 1, which. just copies the input into the output.) 
     * Specifying the same location for the input and output makes the transpose in-place.

     * For double-valued data stored in row-major format, plan creation looks like this: *
     * (Takes less time than a copy of matrices...)
     */
    fftw_plan plan_transpose( const size_t rows, const size_t cols, double *in, double *out, const unsigned flags)
    {
        fftw_iodim howmany_dims[2];
        howmany_dims[0].n  = rows;
        howmany_dims[0].is = cols;
        howmany_dims[0].os = 1;
        howmany_dims[1].n  = cols;
        howmany_dims[1].is = 1;
        howmany_dims[1].os = rows;

        return fftw_plan_guru_r2r(/*rank=*/ 0, /*dims=*/ NULL,
                                  /*howmany_rank=*/ 2, howmany_dims,
                                  in, out, /*kind=*/ NULL, flags);
    }


    fftw_plan plan_transpose( const size_t rows, const size_t cols, fftw_complex *in, fftw_complex *out, const unsigned flags)
    {
        fftw_iodim howmany_dims[2];
        howmany_dims[0].n  = rows;
        howmany_dims[0].is = cols;
        howmany_dims[0].os = 1;
        howmany_dims[1].n  = cols;
        howmany_dims[1].is = 1;
        howmany_dims[1].os = rows;

        return fftw_plan_guru_dft(/*rank=*/ 0, /*dims=*/ NULL,
                                  /*howmany_rank=*/ 2, howmany_dims,
                                  in, out,FFTW_FORWARD, flags);
    }
    fftw_plan plan_dft_1d_r2c_T( const size_t real_rows, const size_t real_cols, double* in, fftw_complex* out, const unsigned flags);
    fftw_plan plan_dft_1d_c_T2r( const size_t real_rows, const size_t real_cols, fftw_complex* in, double* out, const unsigned flags);

    fftw_plan plan_dft_1d_r_T2c( const size_t real_rows, const size_t real_cols, double* in, fftw_complex* out, const unsigned flags);
    fftw_plan plan_dft_1d_c2r_T( const size_t real_rows, const size_t real_cols, fftw_complex* in, double* out, const unsigned flags);
    fftw_plan plan_dft_1d_r2c( const size_t real_rows, const size_t real_cols, double* in, fftw_complex* out, const unsigned flags);
    fftw_plan plan_dft_1d_c2r( const size_t real_rows, const size_t real_cols, fftw_complex* in, double* out, const unsigned flags);


/////////////////////Definitions/////////////////////////////////////////////////////////
    //offset is 0 or 1
    fftw_plan plan_dst_1d_c2c( const size_t rows, const size_t cols, double* in, double* out, fftw_r2r_kind kind, const unsigned flags, const int offset)
    {
        int rank = 1, howmany_rank = 1;
        fftw_iodim dims[rank], howmany_dims[howmany_rank];
        fftw_r2r_kind kind_[] = {kind};
        dims[0].n  = cols;
        dims[0].is = 2; 
        dims[0].os = 2; 
        howmany_dims[0].n  = rows;
        howmany_dims[0].is = cols + 2 - cols%2; //(double)
        howmany_dims[0].os = cols + 2 - cols%2; //(double)
        return fftw_plan_guru_r2r( rank, dims, howmany_rank, howmany_dims, in + offset, out, kind_, flags);
    }

    fftw_plan plan_dft_1d_r2c( const size_t rows, const size_t cols, double* in, fftw_complex* out, const unsigned flags)
    {
        int rank = 1, howmany_rank = 1;
        fftw_iodim dims[rank], howmany_dims[howmany_rank];
        dims[0].n  = cols;
        dims[0].is = 1; //(double)
        dims[0].os = 1; //(complex)
        howmany_dims[0].n  = rows;
        howmany_dims[0].is = cols + 2 - cols%2;
        howmany_dims[0].os = cols/2 + 1;
        return fftw_plan_guru_dft_r2c( rank, dims, howmany_rank, howmany_dims, in, out, flags);
    }
    fftw_plan plan_dft_1d_r2c_T( const size_t rows, const size_t cols, double* in, fftw_complex* out, const unsigned flags)
    {
        int rank = 1, howmany_rank = 1;
        fftw_iodim dims[rank], howmany_dims[howmany_rank];
        dims[0].n  = cols;
        dims[0].is = 1; //(double)
        dims[0].os = rows; //(complex)
        howmany_dims[0].n  = rows;
        howmany_dims[0].is = cols + 2 - cols%2;
        howmany_dims[0].os = 1;
        return fftw_plan_guru_dft_r2c( rank, dims, howmany_rank, howmany_dims, in, out, flags);
    }

    //die letzten Zeilen sind padding
    fftw_plan plan_dft_1d_r_T2c( const size_t rows, const size_t cols, double* in, fftw_complex* out, const unsigned flags)
    {
        int rank = 1, howmany_rank = 1;
        fftw_iodim dims[rank], howmany_dims[howmany_rank];
        dims[0].n  = rows;
        dims[0].is = cols; //(double)
        dims[0].os = 1; //(complex)
        howmany_dims[0].n  = cols;
        howmany_dims[0].is = 1;
        howmany_dims[0].os = rows/2 + 1;
        return fftw_plan_guru_dft_r2c( rank, dims, howmany_rank, howmany_dims, in, out, flags);
    }

    fftw_plan plan_dft_1d_c2r_T( const size_t rows, const size_t cols, fftw_complex* in, double* out, const unsigned flags)
    {
        int rank = 1, howmany_rank = 1;
        fftw_iodim dims[rank], howmany_dims[howmany_rank];
        dims[0].n  = rows;
        dims[0].is = 1;
        dims[0].os = cols;
        howmany_dims[0].n  = cols;
        howmany_dims[0].is = rows/2 + 1;
        howmany_dims[0].os = 1;
        return fftw_plan_guru_dft_c2r( rank, dims, howmany_rank, howmany_dims, in, out, flags);
    }
    fftw_plan plan_dft_1d_c_T2r( const size_t rows, const size_t cols, fftw_complex* in, double* out, const unsigned flags)
    {
        int rank = 1, howmany_rank = 1;
        fftw_iodim dims[rank], howmany_dims[howmany_rank];
        dims[0].n  = cols;
        dims[0].is = rows;
        dims[0].os = 1;
        howmany_dims[0].n  = rows;
        howmany_dims[0].is = 1;
        howmany_dims[0].os = cols + 2 - cols%2;
        return fftw_plan_guru_dft_c2r( rank, dims, howmany_rank, howmany_dims, in, out, flags);
    }
    //padding am Zeilenende
    fftw_plan plan_dft_1d_c2r( const size_t rows, const size_t cols, fftw_complex* in, double* out, const unsigned flags)
    {
        int rank = 1, howmany_rank = 1;
        fftw_iodim dims[rank], howmany_dims[howmany_rank];
        dims[0].n  = cols;
        dims[0].is = 1;
        dims[0].os = 1;
        howmany_dims[0].n  = rows;
        howmany_dims[0].is = cols/2 + 1;
        howmany_dims[0].os = cols + 2 - cols%2;
        return fftw_plan_guru_dft_c2r( rank, dims, howmany_rank, howmany_dims, in, out, flags);
    }



    fftw_plan plan_dft_1d_c2c( const size_t rows, const size_t cols, fftw_complex* in, fftw_complex* out, const int sign, const unsigned flags)
    {
        int rank = 1, howmany_rank = 1;
        fftw_iodim dims[rank], howmany_dims[howmany_rank];
        dims[0].n  = cols;
        dims[0].is = 1; 
        dims[0].os = 1; 
        howmany_dims[0].n  = rows;
        howmany_dims[0].is = cols; 
        howmany_dims[0].os = cols; 
        return fftw_plan_guru_dft( rank, dims, howmany_rank, howmany_dims, in, out, sign, flags);
    }


}
#endif //_TL_FFT_
