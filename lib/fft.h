/*! \file
 * @brief wrapper functions for easy fftw plan creation 
 * @author Matthias Wiesenberger
 *  Matthias.Wiesenberger@uibk.ac.at
 */
#ifndef _TL_FFT_
#define _TL_FFT_

#include <complex>
#include "matrix.h"
#include "ghostmatrix.h"
#include "fftw3.h"

//NOTE: The 2d r2c fftw padding is not like described in the documentation! 
//It's like for a 1d r2c trafo
/* fftw guru interface
 *
 * fftw_iodim{int n,  // Größe der Dimension des Index 
 *            int is, //in stride
 *            int os} //out strid
 * rank, fftw_iodim dims[rank] //describe how you come to the next point inside a trafo for every index i.e. dims[0] describes the first index of the matrix m[i0][i1]...[i_rank-1]
 * howmany_rank, fftw_iodim howmany_dims[howmany_rank] //describe how you come to the first point of the next trafo
 */
namespace toefl{

/*! @brief reinterpret_cast a double to a fftw_complex pointer
*
* @param ptr a pointer to double
* @return pointer to fftw_complex
*/
inline fftw_complex* fftw_cast( double * const ptr){ return reinterpret_cast<fftw_complex*> (ptr);}
/*! @brief reinterpret_cast a std::complex<double> to a fftw_complex pointer
*
* @param ptr a pointer to std::complex<double> 
* @return pointer to fftw_complex
*/
inline fftw_complex* fftw_cast( std::complex<double> * const ptr){ return reinterpret_cast<fftw_complex*> (ptr);}
/*! @brief return the inverse kind of a r2r transformation
 * 
 * @param kind Kind of the r2r transformation
 * @return its inverse kind according to fftw documentation
 */
fftw_r2r_kind inverse_kind( fftw_r2r_kind kind);
/*! @brief Convert toefl enum in fftw kind
 *
 * @param bc Boundary condition 
 * @return The corresponding fftw kind
 */
fftw_r2r_kind fftw_convert( enum bc bc);
/*! @brief Compute normalisation factor for given boundary type
 * 
 * Computes the normalisation according to fftw documentation.
 * @param bc Boundary condition
 * @param n Number of elements you transform
 */
double fftw_normalisation( enum bc bc, unsigned n);
/*! @brief plan many linewise real transformations

 * @param rows # of rows of the Matrix
 * @param cols # of columns of the Matrix
 * @param in the input for the plan creation
 * @param out the output for the plan creation
 * @param kind on of RODFT00, REDFT00, RODFT01, ...
 * @param flags fftw flags
 * @return the plan
 */
fftw_plan plan_drt_1d( const size_t rows, const size_t cols, double *in, double *out, const fftw_r2r_kind kind, const unsigned flags);
/*! @brief plan for real Matrix transposition
 *
 * (From the FFTW FAQ section)
 * You can use the FFTW guru interface to create a rank-0 transform of vector rank 2 where the vector strides are transposed. 
 * (A rank-0 transform is equivalent to a 1D transform of size 1, which. just copies the input into the output.) 
 * Specifying the same location for the input and output makes the transpose in-place.
 *
 * The storage is row major
 * (Takes less time than a copy of matrices, i.e. <0.01s for 1e6 elements)
 * @param rows # of rows of the Matrix
 * @param cols # of columns of the Matrix
 * @param in the input for the plan creation
 * @param out the output for the plan creation
 * @param flags fftw flags
 * @return the transposition plan
 * \note execute with fftw_execute_r2r( plan, ptr, ptr);
 * \see http://www.fftw.org/faq/section3.html#transpose
 */
fftw_plan plan_transpose( const size_t rows, const size_t cols, double *in, double *out, const unsigned flags);
/*! @brief Plan a complex matrix transposition
 *
 * Same as the real version just with complex numbers.
 * @param rows # of rows of the Matrix
 * @param cols # of columns of the Matrix
 * @param in the input for the plan creation
 * @param out the output for the plan creation
 * @param flags fftw flags
 * @return the transposition plan
 * \note execute with fftw_execute( plan, ptr, ptr);
 */
fftw_plan plan_transpose( const size_t rows, const size_t cols, fftw_complex *in, fftw_complex *out, const unsigned flags);
/*! @brief plan many linewise r2c transformations

 * @param real_rows # of rows of the real Matrix
 * @param real_cols # of columns of the real Matrix
 * @param in the input for the plan creation
 * @param out the output for the plan creation( has to padded for inplace transformation)
 * @param flags fftw flags
 * @return the plan
 */
fftw_plan plan_dft_1d_r2c( const size_t real_rows, const size_t real_cols, double* in, fftw_complex* out, const unsigned flags);
/*! @brief plan many linewise c2r transformations

 * @param real_rows # of rows of the real Matrix
 * @param real_cols # of columns of the real Matrix
 * @param in the input for the plan creation
 * @param out the output for the plan creation( has to padded for inplace transformation)
 * @param flags fftw flags
 * @return the plan
 */
fftw_plan plan_dft_1d_c2r( const size_t real_rows, const size_t real_cols, fftw_complex* in, double* out, const unsigned flags);

///@cond
fftw_plan plan_dft_1d_r2c_T( const size_t real_rows, const size_t real_cols, double* in, fftw_complex* out, const unsigned flags);

fftw_plan plan_dft_1d_c_T2r( const size_t real_rows, const size_t real_cols, fftw_complex* in, double* out, const unsigned flags);

fftw_plan plan_dft_1d_r_T2c( const size_t real_rows, const size_t real_cols, double* in, fftw_complex* out, const unsigned flags);

fftw_plan plan_dft_1d_c2r_T( const size_t real_rows, const size_t real_cols, fftw_complex* in, double* out, const unsigned flags);

fftw_plan plan_dft_1d_c2c( const size_t rows, const size_t cols, fftw_complex* in, fftw_complex* out, const int sign, const unsigned flags);


/////////////////////Definitions/////////////////////////////////////////////////////
//from fftws website
fftw_plan plan_transpose( const size_t rows, const size_t cols, double *in, double *out, const unsigned flags = FFTW_MEASURE)
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
                              in, out, FFTW_FORWARD, flags);
}

fftw_r2r_kind fftw_convert( enum bc bc)
{
    fftw_r2r_kind kind = FFTW_RODFT00;
    switch( bc)
    {
        case( TL_PERIODIC): 
            throw Message( "Cannot convert TL_PERIODIC to fftw_r2r_kind!", ping);
            break;
        case( TL_DST00) : kind = FFTW_RODFT00; break;
        case( TL_DST10) : kind = FFTW_RODFT10; break;
        case( TL_DST01) : kind = FFTW_RODFT01; break;
        case( TL_DST11) : kind = FFTW_RODFT11; break;
    }
    return kind;
}

double fftw_normalisation( enum bc bc, unsigned n)
{
    double norm = 0;
    switch( bc)
    {
        case( TL_PERIODIC): norm = (double)n;           break;
        case( TL_DST00):    norm = (double)(2*(n+1));   break;
        case( TL_DST10):    norm = (double)(2*n);       break;
        case( TL_DST01):    norm = (double)(2*n);       break;
        case( TL_DST11):    norm = (double)(2*n);       break;
    }
    return norm;
}

fftw_plan plan_drt_1d( const size_t rows, const size_t cols, double *in, double *out, const fftw_r2r_kind kind, const unsigned flags = FFTW_MEASURE)
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

fftw_r2r_kind inverse_kind( fftw_r2r_kind kind)
{
    switch( kind)
    {
        case( FFTW_RODFT00): return FFTW_RODFT00;
        case( FFTW_RODFT01): return FFTW_RODFT10;
        case( FFTW_RODFT10): return FFTW_RODFT01;
        case( FFTW_RODFT11): return FFTW_RODFT11;
        case( FFTW_REDFT00): return FFTW_REDFT00;
        case( FFTW_REDFT01): return FFTW_REDFT10;
        case( FFTW_REDFT10): return FFTW_REDFT01;
        case( FFTW_REDFT11): return FFTW_REDFT11;
        case( FFTW_R2HC): return FFTW_HC2R;
        case( FFTW_HC2R): return FFTW_R2HC;
        default: throw Message( "fftw r2r kind unknown!", ping);
        return kind;
    }
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



///@endcond


}
#endif //_TL_FFT_
