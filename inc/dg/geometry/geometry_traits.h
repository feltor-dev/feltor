#pragma once

namespace dg
{
//categories
//
struct CurvilinearTag{}; 
struct CurvilinearCylindricalTag: public CurvilinearTag{}; //perpVol, vol(), g_xx, g_xy, g_yy
struct OrthonormalCylindricalTag:public CurvilinearCylindricalTag{}; //vol()
struct OrthonormalTag: public OrhonormalCylindricalTag{};


///@cond
template <class Geometry>
struct GeometryTraits{
    typedef typename Geometry::metric_category metric_category
    typedef typename Geometry::memory_category memory_category
};

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
    dg::blas1::pointwiseDot( g.g_xy(), in1, in1); //gyx*v_x
    dg::blas1::pointwiseDot( g.g_xy(), in2, out2);//gxy*v_y
    dg::blas1::pointwiseDot( g.g_yy(), in2, in2); //gyy*v_y
    dg::blas1::axpby( 1., in1, 1., in2, out2); //gyx*v_x + gyy*v_y
    dg::blas1::axpby( 1., out1, 1., out2, out1);//gxx*v_x + gxy*v_y
};

template<class TernaryOp1, class TernaryOp2, class Geometry> 
void doPushforwardPerp( TernaryOp1 f1, TenaryOp2& f2, 
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& out1, 
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& out2,
        const Geometry& g, OrthonormalCylindricalTag)
{
    out1 = evaluate( f1, g);
    out2 = evaluate( f2, g);
}

template<class TernaryOp1, class TernaryOp2, class Geometry> 
void doPushforwardPerp( TernaryOp1 f1, TenaryOp2& f2, 
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& out1, 
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& out2,
        const Geometry& g, CurvilinearCylindricalTag)
{
    typedef typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector container;
    out1 = pullback( f1, g);
    out2 = pullback( f2, g);
    container temp1( out1), temp2( out2);
    dg::blas1::pointwiseDot( g.xR(), out1, temp1);
    dg::blas1::pointwiseDot( g.xZ(), out2, temp2);
    dg::blas1::pointwiseDot( g.yR(), out1, out1);
    dg::blas1::pointwiseDot( g.yZ(), out2, out2);
    dg::blas1::axpby( 1., out1, 1., out2, out2);
    dg::blas1::axpby( 1., temp1, 1., temp2, out1);
}


}//namespace detail 
}//namespace geo
///@endcond

} //namespace dg
