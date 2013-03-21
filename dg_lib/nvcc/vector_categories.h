#ifndef _DG_VECTOR_CATEGORIES_
#define _DG_VECTOR_CATEGORIES_

namespace dg{

/**
 * @brief The Thrust Vector concept
 *
 * A Thrust vector must have the two methods begin() and end() which
 * return thrust compatible iterators and the value_type typedef
 */
struct ThrustVectorTag {};


}//namespace dg

#endif //_DG_VECTOR_CATEGORIES_
