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
#include "texture.h"
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
//inits
#include "init.h"


#endif //_TL_TOEFL_
