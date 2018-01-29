#ifndef _DG_VECTOR_CATEGORIES_
#define _DG_VECTOR_CATEGORIES_

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
