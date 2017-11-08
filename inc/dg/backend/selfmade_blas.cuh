#ifndef _DG_BLAS_SELFMADE_
#define _DG_BLAS_SELFMADE_
//
///@cond
namespace dg{
namespace blas2{
namespace detail{


template<class Matrix1, class Matrix2>
inline void doTransfer( const Matrix1& x, Matrix2& y, SelfMadeMatrixTag, SelfMadeMatrixTag)
{
    y = (Matrix2)x; //try to invoke the explicit conversion construction
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv(  
              Matrix& m, 
              Vector1& x,
              Vector2& y, 
              SelfMadeMatrixTag,
              AnyVectorTag,
              AnyVectorTag)
{
    m.symv( x,y);
}

template< class Matrix, class Vector1, class Vector2>
inline void doGemv(  
              Matrix& m, 
              Vector1& x,
              Vector2& y, 
              SelfMadeMatrixTag,
              AnyVectorTag,
              AnyVectorTag)
{
    m.symv( x,y);
}

template< class Matrix, class Vector>
inline void doSymv(  
              typename VectorTraits<Vector>::value_type alpha, 
              const Matrix& m,
              const Vector& x, 
              typename VectorTraits<Vector>::value_type beta, 
              Vector& y, 
              SelfMadeMatrixTag,
              AnyVectorTag)
{
    m.symv( alpha, x, beta, y);
}

template< class Matrix, class Vector>
inline void doGemv(  
              typename VectorTraits<Vector>::value_type alpha, 
              const Matrix& m,
              const Vector& x, 
              typename VectorTraits<Vector>::value_type beta, 
              Vector& y, 
              SelfMadeMatrixTag,
              AnyVectorTag)
{
    m.symv( alpha, x, beta, y);
}

} //namespace detail
} //namespace blas2
} //namespace dg
///@endcond
//
#endif//_DG_BLAS_SELFMADE_
