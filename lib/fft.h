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

// was nicht geprüft wird ist, ob der Plan in der execute Funktion den richtigen "Typ" hat und (evtl macht das die fftw selbst)
// ob der Plan für die Größe der Matrix passt (das macht die fftw aber auch nicht)
namespace toefl{

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

     * For double-valued data stored in row-major format, plan creation looks like this: */
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


    fftw_plan plan_transpose( const size_t rows, const size_t cols, fftw_complex *in, fftw_complex *out, const unsigned flags);
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

    /*
    //1d dft_r2c plans and execute functions
    template< typename Complex>
    fftw_plan plan_dft_1d_r2c( Matrix<double, TL_NONE>& in, Matrix<Complex, TL_NONE>& out);

    template< typename Complex>
    void execute_dft_1d_r2c( const fftw_plan plan, Matrix< double, TL_DFT_1D>& inout, Matrix< Complex, TL_NONE>& swap);//implemented
    template< typename Complex>
    void execute_dft_1d_r2c( const fftw_plan plan, Matrix< double, TL_NONE>& in, Matrix< Complex, TL_NONE>& out);

    //1d dft_c2r plans and execute functions
    template< typename Complex>
    fftw_plan plan_dft_1d_c2r( Matrix<Complex, TL_NONE>& inout,  bool odd); //init with n%2//implemented
    template< typename Complex>
    fftw_plan plan_dft_1d_c2r( Matrix<Complex, TL_NONE>& in,     Matrix<double, TL_NONE>& out); 

    template< typename Complex>
    void execute_dft_1d_c2r( const fftw_plan plan, Matrix< Complex, TL_NONE>& inout, Matrix< double, TL_DFT_1D>& swap);//implemented
    template< typename Complex>
    void execute_dft_1d_c2r( const fftw_plan plan, Matrix< Complex, TL_NONE>& in,    Matrix< double, TL_NONE>& out);
    
    //2d dft_r2c plans and execute functions
    fftw_plan plan_dft_2d_r2c( Matrix<double, TL_DFT_2D>& inout);
    template< typename Complex>
    fftw_plan plan_dft_2d_r2c( Matrix<double, TL_NONE>& in, Matrix<Complex, TL_NONE>& out);

    template< typename Complex>
    void execute_dft_2d_r2c( const fftw_plan plan, Matrix< double, TL_DFT_2D>& inout, Matrix< Complex, TL_NONE>& swap );
    template< typename Complex>
    void execute_dft_2d_r2c( const fftw_plan plan, Matrix< double, TL_NONE>& in, Matrix< Complex, TL_NONE>& out);

    //2d dft_c2r plans and execute functions
    template< typename Complex>
    fftw_plan plan_dft_2d_c2r( Matrix<Complex, TL_NONE>& inout, bool odd);
    template< typename Complex>
    fftw_plan plan_dft_2d_c2r( Matrix<Complex, TL_NONE>& in, Matrix<double, TL_NONE>& out);
    //non transposing r2c of lines
    fftw_plan plan_dft_1d_r2c( const size_t rows, const size_t cols, double* in, fftw_complex* out, const unsigned flags)
    {
        int n[] = { (int)m.cols()}; //length of each transform
        fftw_plan plan = fftw_plan_many_dft_r2c(  1,  //dimension 1D
                                    n,  //size of each dimension
                                    m.rows(), //number of transforms
                                    &m(0,0), //input
                                    NULL, //embed
                                    1, //stride in units of double
                                    TotalNumberOf<TL_DFT_1D>::cols( m.cols()), //distance between trafos
                                    reinterpret_cast<fftw_complex*>(&m(0,0)),
                                    NULL,
                                    1, //stride in units of fftw_complex
                                    m.cols()/2 + 1, //distance between trafos
                                    FFTW_MEASURE);
    }

    fftw_plan plan_dft_1d_c2r( const size_t rows, const size_t cols, fftw_complex* in, double* out, const unsigned flags) 
    {
        int n[] ={2*(int)m.cols() - ((odd==true)?1:2) }; //{ (int)m.cols()};  //length of each transform (double)
        fftw_plan plan = fftw_plan_many_dft_c2r(  1,  //dimension 1D
                                    n,  //size of each dimension (in complex)
                                    m.rows(), //number of transforms
                                    reinterpret_cast<fftw_complex*>(&m(0,0)), //input
                                    NULL, //embed
                                    1, //stride in units of complex
                                    m.cols(), //distance between trafos (in complex)
                                    reinterpret_cast<double*>(&m(0,0)),
                                    NULL,
                                    1, //stride in units of double
                                    2*(int)m.cols(), //distance between trafos (in double)
                                    FFTW_MEASURE);
        return plan;
    }
    */


}
#endif //_TL_FFT_
