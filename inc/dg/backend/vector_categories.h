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

struct AnyVectorTag{};

struct StdVectorTag:public AnyVectorTag {};

struct ThrustVectorTag: public AnyVectorTag {};

struct ThrustSerialTag : public ThrustVectorTag{};
struct ThrustCudaTag : public ThrustVectorTag{};
struct ThrustOmpTag : public ThrustVectorTag{};

struct CuspVectorTag: public ThrustVectorTag {};

struct MPIVectorTag: public AnyVectorTag{};


}//namespace dg

#endif //_DG_VECTOR_CATEGORIES_
