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
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
#endif //DG_DEBUG
    m.symv( x,y);
}
//template< class Matrix, class Vector>
//inline void doSymv(  
//              Matrix& m, 
//              const std::vector<Vector>& x,
//              std::vector<Vector>& y, 
//              SelfMadeMatrixTag,
//              StdVectorTag)
//{
//#ifdef DG_DEBUG
//    assert( x.size() == y.size() );
//#endif //DG_DEBUG
//    m.symv( x,y);
//}

} //namespace detail
} //namespace blas2
} //namespace dg
#endif//_DG_BLAS_SELFMADE_
