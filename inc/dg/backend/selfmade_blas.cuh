#ifndef _DG_BLAS_SELFMADE_
#define _DG_BLAS_SELFMADE_

namespace dg{
namespace blas2{
namespace detail{

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

} //namespace detail
} //namespace blas2
} //namespace dg
#endif//_DG_BLAS_SELFMADE_
