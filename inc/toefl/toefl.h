/*! @file 
 * @brief This file includes all header files of the toefl library
 */
#ifndef _TL_TOEFL_
#define _TL_TOEFL_

//Message handling
#include "message.h"
#include "exceptions.h"
//input reading and visualization
//#include "read_input.h" //moved to file lib
//#include "texture.h" //moved to draw lib
//benchmarking
#include "timer.h"
//Matrices 
#include "quadmat.h"
#include "padding.h"
#include "matrix.h"
#include "matrix_array.h"
#include "ghostmatrix.h"
//Arkawa and karniadakis scheme
#include "arakawa.h"
#include "karniadakis.h"
//Fourier transforms
#include "fft.h"
#include "dft_dft.h"
#include "dft_drt.h"
#include "drt_dft.h"
#include "drt_drt.h"
namespace toefl{
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
//inits

fftw_r2r_kind fftw_convert( enum bc bc)
{
    fftw_r2r_kind kind = FFTW_R2HC; //least likely used
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
}//namespace toefl
#include "init.h"

#endif //_TL_TOEFL_
