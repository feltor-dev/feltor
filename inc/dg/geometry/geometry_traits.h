#pragma once

#include "../blas1.h"

namespace dg
{
//categories
//
struct CurvilinearTag{};  //3d curvilinear
struct CurvilinearCylindricalTag: public CurvilinearTag{}; //perpVol, vol(), g_xx, g_xy, g_yy
struct OrthogonalTag:public CurvilinearCylindricalTag{}; //perpVol, vol(), g_xx, g_yy
struct ConformalCylindricalTag:public OrthogonalTag{}; //perpVol, vol(), g_xx, g_yy
struct ConformalTag:public ConformalCylindricalTag{}; //A 2d conformal 
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
void doRaisePerpIndex( container& in1, container& in2, container& out1, container& out2, const Geometry& g, OrthogonalTag)
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
void doVolRaisePerpIndex( container& in1, container& in2, container& out1, container& out2, const Geometry& g, OrthogonalTag)
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

template<class FunctorRR, class FunctorRZ, class FunctorZZ, class Geometry> 
void doPushForwardPerp( FunctorRR chiRR, FunctorRZ chiRZ, FunctorZZ chiZZ,
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& chixx, 
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& chixy,
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& chiyy, 
        const Geometry& g, OrthonormalCylindricalTag)
{
    chixx = evaluate( chiRR, g);
    chixy = evaluate( chiRZ, g);
    chiyy = evaluate( chiZZ, g);
}

template<class FunctorRR, class FunctorRZ, class FunctorZZ, class Geometry> 
void doPushForwardPerp( FunctorRR chiRR_, FunctorRZ chiRZ_, FunctorZZ chiZZ_,
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& chixx, 
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& chixy,
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& chiyy, 
        const Geometry& g, CurvilinearCylindricalTag)
{
    //compute the rhs
    typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector chiRR, chiRZ, chiZZ;
    chixx = chiRR = evaluate( chiRR_, g);
    chixy = chiRZ = evaluate( chiRZ_, g);
    chiyy = chiZZ = evaluate( chiZZ_, g);
    //compute the transformation matrix
    typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector t00(chixx), t01(t00), t02(t00), t10(t00), t11(t00), t12(t00), t20(t00), t21(t00, t22(t00);
    dg::blas1::pointwiseDot( g.xr(), g.xr(), t00);
    dg::blas1::pointwiseDot( g.xr(), g.xz(), t01);
    dg::blas1::scal( t01, 2.);
    dg::blas1::pointwiseDot( g.xz(), g.xz(), t02);

    dg::blas1::pointwiseDot( g.xr(), g.yr(), t10);
    dg::blas1::pointwiseDot( g.xr(), g.yz(), t11);
    dg::blas1::pointwiseDot( 1., g.yr(), g.xz(), 1., t11);
    dg::blas1::pointwiseDot( g.xz(), g.yz(), t12);

    dg::blas1::pointwiseDot( g.yr(), g.yr(), t20);
    dg::blas1::pointwiseDot( g.yr(), g.yz(), t21);
    dg::blas1::scal( t21, 2.);
    dg::blas1::pointwiseDot( g.yz(), g.yz(), t22);
    //now multiply
    dg::blas1::pointwiseDot( t00, chiRR, chixx);
    dg::blas1::pointwiseDot( 1., t01, chiRZ, 1., chixx);
    dg::blas1::pointwiseDot( 1., t02, chiZZ, 1., chixx);
    dg::blas1::pointwiseDot( t10, chiRR, chixy);
    dg::blas1::pointwiseDot( 1., t11, chiRZ, 1., chixy);
    dg::blas1::pointwiseDot( 1., t12, chiZZ, 1., chixy);
    dg::blas1::pointwiseDot( t20, chiRR, chiyy);
    dg::blas1::pointwiseDot( 1., t21, chiRZ, 1., chiyy);
    dg::blas1::pointwiseDot( 1., t22, chiZZ, 1., chiyy);

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
