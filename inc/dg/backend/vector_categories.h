#ifndef _DG_VECTOR_CATEGORIES_
#define _DG_VECTOR_CATEGORIES_

#if defined(__INTEL_COMPILER)
  // On Intel compiler, you need to pass the -restrict compiler flag in addition to your own compiler flags.
# define RESTRICT restrict
#elif defined(__GNUG__)
# define RESTRICT __restrict__
#else
# warning Missing restrict keyword for this compiler
# define RESTRICT
#endif

namespace dg{

struct AnyVectorTag{};

struct StdVectorTag:public AnyVectorTag {};

/**
 * @brief The Thrust Vector concept
 *
 * A Thrust vector must have the two methods begin() and end() which
 * return thrust compatible iterators and the value_type typedef
 */
struct ThrustVectorTag: public AnyVectorTag {};

struct CuspVectorTag: public ThrustVectorTag {};

struct MPIVectorTag: public AnyVectorTag{};


}//namespace dg

#endif //_DG_VECTOR_CATEGORIES_
