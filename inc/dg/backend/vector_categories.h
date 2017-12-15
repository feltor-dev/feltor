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

#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
#if defined(__INTEL_COMPILER)

#if __INTEL_COMPILER < 1500
#warning "icc version >= 15.0 recommended to activate OpenMP 4 support"
#define SIMD
#else//>1500
#define SIMD simd
#endif//__INTEL_COMPILER

#elif defined(__GNUG__)

#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#if GCC_VERSION < 40900
#warning "gcc version >= 4.9 recommended to activate OpenMP 4 support"
#define SIMD
#else
#define SIMD simd
#endif //GCC_VERSION

#endif //__GNUG__
#endif //THRUST_DEVICE_SYSTEM

namespace dg{

//here we introduce the concept of data access

struct AnyVectorTag{};

struct SharedVectorTag  : public AnyVectorTag {};   //shared vectors
struct MPIVectorTag     : public AnyVectorTag {};   //MPI vectors, contains a shared vector

struct VectorVectorTag  : public AnyVectorTag {};   //container of containers (either Shared or MPI Vectors
struct ArrayVectorTag   : public VectorVectorTag{}; //std::array of containers

struct ThrustVectorTag  : public SharedVectorTag {};
struct CuspVectorTag    : public ThrustVectorTag {};
struct StdArrayTag      : public ThrustVectorTag {};



}//namespace dg

#endif //_DG_VECTOR_CATEGORIES_
