#error Documentation only
/*! @mainpage
 *
 * This version of exblas is an adaption of the original exblas library for FELTOR.
 * The algorithm is described in the paper \n
 <a href = "https://hal.archives-ouvertes.fr/hal-00949355v3">Sylvain Collange, David Defour, Stef Graillat, Roman Iakymchuk. "Numerical Reproducibility for the Parallel Reduction on Multi- and Many-Core Architectures", 2015. </a>
 */
namespace dg{
/*!
 * @brief This is the namespace for all functions and
 *     classes defined and used in the exblas library
 *
 * In principle you can use this as a standalone library but it is much easier
 * to just use the \c dg::blas1::dot and \c dg::blas2::dot functions for general
 * purpose usage
 */
namespace exblas{
/*!
 * @brief gpu (CUDA) versions of primitive functions
 */
namespace  gpu{}

/*!
 * @brief cpu versions of the primitive functions
 */
namespace cpu{}
}
}//namespace dg
