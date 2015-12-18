#pragma once

#include "../blas1.h"

namespace dg
{
//categories
//
struct CurvilinearTag{}; 
struct CurvilinearCylindricalTag: public CurvilinearTag{}; //perpVol, vol(), g_xx, g_xy, g_yy
struct OrthonormalCylindricalTag:public CurvilinearCylindricalTag{}; //vol()
struct OrthonormalTag: public OrthonormalCylindricalTag{};


///@cond
template <class Geometry>
struct GeometryTraits{
    typedef typename Geometry::metric_category metric_category;
    typedef typename Geometry::memory_category memory_category;
};

template<class MemoryTag>
struct HostVec {
};
template<>
struct HostVec< SharedTag>
{
    typedef thrust::host_vector<double> host_vector;
};

#ifdef MPI_VERSION
template<>
struct HostVec< MPITag>
{
    typedef MPI_Vector<thrust::host_vector<double> > host_vector;
};
#endif //MPI_VERSION

namespace geo{
namespace detail{

template <class container, class Geometry>
void doMultiplyVolume( container& inout, const Geometry& g, OrthonormalTag)
{
};

template <class container, class Geometry>
void doMultiplyVolume( container& inout, const Geometry& g, CurvilinearTag)
{
    dg::blas1::pointwiseDot( inout, g.vol(), inout);
};
template <class container, class Geometry>
void doDivideVolume( container& inout, const Geometry& g, OrthonormalTag)
{
};

template <class container, class Geometry>
void doDivideVolume( container& inout, const Geometry& g, CurvilinearTag)
{
    dg::blas1::pointwiseDivide( inout, g.vol(), inout);
};

template <class container, class Geometry>
void doMultiplyPerpVolume( container& inout, const Geometry& g, OrthonormalCylindricalTag)
{
};

template <class container, class Geometry>
void doMultiplyPerpVolume( container& inout, const Geometry& g, CurvilinearCylindricalTag)
{
    dg::blas1::pointwiseDot( inout, g.perpVol(), inout);
};

template <class container, class Geometry>
void doDividePerpVolume( container& inout, const Geometry& g, OrthonormalCylindricalTag)
{
};

template <class container, class Geometry>
void doDividePerpVolume( container& inout, const Geometry& g, CurvilinearCylindricalTag)
{
    dg::blas1::pointwiseDivide( inout, g.perpVol(), inout);
};

template <class container, class Geometry>
void doRaisePerpIndex( container& in1, container& in2, container& out1, container& out2, const Geometry& g, OrthonormalCylindricalTag)
{
    in1.swap( out1);
    in2.swap( out2);
};

template <class container, class Geometry>
void doRaisePerpIndex( container& in1, container& in2, container& out1, container& out2, const Geometry& g, CurvilinearCylindricalTag)
{
    dg::blas1::pointwiseDot( g.g_xx(), in1, out1); //gxx*v_x
    dg::blas1::pointwiseDot( g.g_xy(), in1, out2); //gyx*v_x
    dg::blas1::pointwiseDot( 1., g.g_xy(), in2, 1., out1);//gxy*v_y
    dg::blas1::pointwiseDot( 1., g.g_yy(), in2, 1., out2); //gyy*v_y
};

template<class TernaryOp1, class TernaryOp2, class Geometry> 
void doPushforwardPerp( TernaryOp1 f1, TernaryOp2 f2, 
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& out1, 
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& out2,
        const Geometry& g, OrthonormalCylindricalTag)
{
    out1 = evaluate( f1, g);
    out2 = evaluate( f2, g);
}

template<class TernaryOp1, class TernaryOp2, class Geometry> 
void doPushforwardPerp( TernaryOp1 f1, TernaryOp2 f2, 
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& out1, 
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& out2,
        const Geometry& g, CurvilinearCylindricalTag)
{
    typedef typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector container;
    out1 = pullback( f1, g);
    out2 = pullback( f2, g);
    container temp1( out1), temp2( out2);
    dg::blas1::pointwiseDot( g.xR(), temp1, out1);
    dg::blas1::pointwiseDot( 1., g.xZ(), temp2, 1., out1);
    dg::blas1::pointwiseDot( g.yR(), temp1, out2);
    dg::blas1::pointwiseDot( 1., g.yZ(), temp2, 1., out2);
}


}//namespace detail 
}//namespace geo

namespace create{
namespace detail{

template< class Geometry>
typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector doCreateVolume( const Geometry& g, OrthonormalTag)
{
    return dg::create::weights( g);
}

template< class Geometry>
typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector doCreateInvVolume( const Geometry& g, OrthonormalTag)
{
    return dg::create::inv_weights( g);
}
template< class Geometry>
typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector doCreateVolume( const Geometry& g, CurvilinearTag)
{
    typedef typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector host_vector;
    host_vector temp = dg::create::weights( g);
    host_vector vol = g.vol();
    dg::blas1::pointwiseDot( vol, temp, temp);
    return temp;
}

template< class Geometry>
typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector doCreateInvVolume( const Geometry& g, CurvilinearTag)
{
    typedef typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector host_vector;
    host_vector temp = dg::create::inv_weights( g);
    host_vector vol = g.vol();
    dg::blas1::pointwiseDivide( temp, vol, temp);
    return temp;
}

}//namespace detail
}//namespace create
///@endcond

} //namespace dg
