#pragma once

#include "../blas1.h"

namespace dg
{
//categories
//
struct CurvilinearTag{}; 
struct CurvilinearCylindricalTag: public CurvilinearTag{}; //perpVol, vol(), g_xx, g_xy, g_yy
struct OrthogonalCylindricalTag:public CurvilinearCylindricalTag{}; //perpVol, vol(), g_xx, g_yy
struct ConformalCylindricalTag:public OrthogonalCylindricalTag{}; //perpVol, vol(), g_xx, g_yy
struct OrthonormalCylindricalTag:public ConformalCylindricalTag{}; //vol(), cylindrical grid
struct OrthonormalTag: public OrthonormalCylindricalTag{}; //cartesian grids

//memory_category and dimensionality Tags are already defined in grid.h

///@cond
template <class Geometry>
struct GeometryTraits{
    typedef typename Geometry::metric_category metric_category;
    typedef typename Geometry::memory_category memory_category; //either shared or distributed
    typedef typename Geometry::dimensionality dimensionality; //either shared or distributed
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
void doRaisePerpIndex( container& in1, container& in2, container& out1, container& out2, const Geometry& g, OrthogonalCylindricalTag)
{
    dg::blas1::pointwiseDot( g.g_xx(), in1, out1); //gxx*v_x
    dg::blas1::pointwiseDot( g.g_yy(), in2, out2); //gyy*v_y
};
template <class container, class Geometry>
void doRaisePerpIndex( container& in1, container& in2, container& out1, container& out2, const Geometry& g, CurvilinearCylindricalTag)
{
    dg::blas1::pointwiseDot( g.g_xx(), in1, out1); //gxx*v_x
    dg::blas1::pointwiseDot( g.g_xy(), in1, out2); //gyx*v_x
    dg::blas1::pointwiseDot( 1., g.g_xy(), in2, 1., out1);//gxy*v_y
    dg::blas1::pointwiseDot( 1., g.g_yy(), in2, 1., out2); //gyy*v_y
};

template <class container, class Geometry>
void doVolRaisePerpIndex( container& in1, container& in2, container& out1, container& out2, const Geometry& g, ConformalCylindricalTag)
{
    in1.swap( out1);
    in2.swap( out2);
};
template <class container, class Geometry>
void doVolRaisePerpIndex( container& in1, container& in2, container& out1, container& out2, const Geometry& g, OrthogonalCylindricalTag)
{
    dg::blas1::pointwiseDot( g.g_xx(), in1, out1); //gxx*v_x
    dg::blas1::pointwiseDot( g.g_yy(), in2, out2); //gyy*v_y
    dg::blas1::pointwiseDot( out1, g.perpVol(), out1);
    dg::blas1::pointwiseDot( out2, g.perpVol(), out2);
};
template <class container, class Geometry>
void doVolRaisePerpIndex( container& in1, container& in2, container& out1, container& out2, const Geometry& g, CurvilinearCylindricalTag)
{
    dg::blas1::pointwiseDot( g.g_xx(), in1, out1); //gxx*v_x
    dg::blas1::pointwiseDot( g.g_xy(), in1, out2); //gyx*v_x
    dg::blas1::pointwiseDot( 1., g.g_xy(), in2, 1., out1);//gxy*v_y
    dg::blas1::pointwiseDot( 1., g.g_yy(), in2, 1., out2); //gyy*v_y
    dg::blas1::pointwiseDot( out1, g.perpVol(), out1);
    dg::blas1::pointwiseDot( out2, g.perpVol(), out2);
};


template<class TernaryOp1, class TernaryOp2, class Geometry> 
void doPushForwardPerp( TernaryOp1 f1, TernaryOp2 f2, 
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& out1, 
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& out2,
        const Geometry& g, OrthonormalCylindricalTag)
{
    out1 = evaluate( f1, g);
    out2 = evaluate( f2, g);
}

template<class TernaryOp1, class TernaryOp2, class Geometry> 
void doPushForwardPerp( TernaryOp1 f1, TernaryOp2 f2, 
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& out1, 
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& out2,
        const Geometry& g, CurvilinearCylindricalTag)
{
    typedef typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector container;
    out1 = pullback( f1, g);
    out2 = pullback( f2, g);
    container temp1( out1), temp2( out2);
    dg::blas1::pointwiseDot( g.xr(), temp1, out1);
    dg::blas1::pointwiseDot( 1., g.xz(), temp2, 1., out1);
    dg::blas1::pointwiseDot( g.yr(), temp1, out2);
    dg::blas1::pointwiseDot( 1., g.yz(), temp2, 1., out2);
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
    host_vector temp, vol;
    dg::blas1::transfer( dg::create::weights( g), temp);
    dg::blas1::transfer( g.vol(), vol); //g.vol might be on device
    dg::blas1::pointwiseDot( vol, temp, temp);
    return temp;
}

template< class Geometry>
typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector doCreateInvVolume( const Geometry& g, CurvilinearTag)
{
    typedef typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector host_vector;
    host_vector temp, vol;
    dg::blas1::transfer( dg::create::inv_weights( g), temp);
    dg::blas1::transfer( g.vol(), vol); //g.vol might be on device
    dg::blas1::pointwiseDivide( temp, vol, temp);
    return temp;
}

}//namespace detail
}//namespace create

namespace detail
{
////pullbacks
//template< class Geometry>
//thrust::host_vector<double> doPullback( double(f)(double,double), const Geometry& g, CurvilinearTag, TwoDimensionalTag, SharedTag)
//{
//    return doPullback<double(double,double), Geometry>( f, g);
//}
//template< class Geometry>
//thrust::host_vector<double> pullback( double(f)(double,double,double), const Geometry& g, CurvilinearTag, ThreeDimensionalTag, SharedTag)
//{
//    return doPullback<double(double,double,double), Geometry>( f, g);
//}

template< class BinaryOp, class Geometry>
thrust::host_vector<double> doPullback( BinaryOp f, const Geometry& g, CurvilinearTag, TwoDimensionalTag, SharedTag)
{
    thrust::host_vector<double> vec( g.size());
    for( unsigned i=0; i<g.size(); i++)
        vec[i] = f( g.r()[i], g.z()[i]);
    return vec;
}

template< class TernaryOp, class Geometry>
thrust::host_vector<double> doPullback( TernaryOp f, const Geometry& g, CurvilinearTag, ThreeDimensionalTag, SharedTag)
{
    thrust::host_vector<double> vec( g.size());
    unsigned size2d = g.n()*g.n()*g.Nx()*g.Ny();
    Grid1d<double> gz( g.z0(), g.z1(), 1, g.Nz());
    thrust::host_vector<double> absz = create::abscissas( gz);
    for( unsigned k=0; k<g.Nz(); k++)
        for( unsigned i=0; i<size2d; i++)
            vec[k*size2d+i] = f( g.r()[k*size2d+i], g.z()[k*size2d+i], absz[k]);
    return vec;
}
template< class BinaryOp, class Geometry>
thrust::host_vector<double> doPullback( BinaryOp f, const Geometry& g, OrthonormalCylindricalTag, TwoDimensionalTag, SharedTag)
{
    return evaluate( f, g);
}
template< class TernaryOp, class Geometry>
thrust::host_vector<double> doPullback( TernaryOp f, const Geometry& g, OrthonormalCylindricalTag, ThreeDimensionalTag, SharedTag)
{
    return evaluate( f,g);
}

}//namespace detail
///@endcond

} //namespace dg
