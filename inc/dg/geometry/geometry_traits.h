#pragma once

#include "../blas1.h"

namespace dg
{
//categories
//
struct CurvilinearTag{};  //! 3d curvilinear
struct CurvilinearCylindricalTag: public CurvilinearTag{}; //! perpVol, vol(), g_xx, g_xy, g_yy
struct OrthonormalCylindricalTag:public CurvilinearCylindricalTag{}; //! vol(), cylindrical grid
struct OrthonormalTag: public OrthonormalCylindricalTag{}; //! Cartesian grids

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
void doRaisePerpIndex( container& in1, container& in2, container& out1, container& out2, const Geometry& g, CurvilinearCylindricalTag)
{
    if( g.isOrthogonal())
    {
        dg::blas1::pointwiseDot( g.g_xx(), in1, out1); //gxx*v_x
        dg::blas1::pointwiseDot( g.g_yy(), in2, out2); //gyy*v_y
        return;
    }
    dg::blas1::pointwiseDot( g.g_xx(), in1, out1); //gxx*v_x
    dg::blas1::pointwiseDot( g.g_xy(), in1, out2); //gyx*v_x
    dg::blas1::pointwiseDot( 1., g.g_xy(), in2, 1., out1);//gxy*v_y
    dg::blas1::pointwiseDot( 1., g.g_yy(), in2, 1., out2); //gyy*v_y
};

template <class container, class Geometry>
void doVolRaisePerpIndex( container& in1, container& in2, container& out1, container& out2, const Geometry& g, OrthonormalCylindricalTag)
{
    in1.swap( out1);
    in2.swap( out2);
};
template <class container, class Geometry>
void doVolRaisePerpIndex( container& in1, container& in2, container& out1, container& out2, const Geometry& g, CurvilinearCylindricalTag)
{
    if( g.isConformal())
    {
        in1.swap( out1);
        in2.swap( out2);
        return;
    }
    if( g.isOrthogonal())
    {
        dg::blas1::pointwiseDot( g.g_xx(), in1, out1); //gxx*v_x
        dg::blas1::pointwiseDot( g.g_yy(), in2, out2); //gyy*v_y
        dg::blas1::pointwiseDot( out1, g.perpVol(), out1);
        dg::blas1::pointwiseDot( out2, g.perpVol(), out2);
        return;
    }
    dg::blas1::pointwiseDot( g.g_xx(), in1, out1); //gxx*v_x
    dg::blas1::pointwiseDot( g.g_xy(), in1, out2); //gyx*v_x
    dg::blas1::pointwiseDot( 1., g.g_xy(), in2, 1., out1);//gxy*v_y
    dg::blas1::pointwiseDot( 1., g.g_yy(), in2, 1., out2); //gyy*v_y
    dg::blas1::pointwiseDot( out1, g.perpVol(), out1);
    dg::blas1::pointwiseDot( out2, g.perpVol(), out2);
};


template<class TernaryOp1, class TernaryOp2, class container, class Geometry> 
void doPushForwardPerp( TernaryOp1 f1, TernaryOp2 f2, 
        container& vx, container& vy,
        const Geometry& g, OrthonormalCylindricalTag)
{
    typedef typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector host_vec;
    host_vec out1 = evaluate( f1, g);
    host_vec out2 = evaluate( f2, g);
    dg::blas1::transfer( out1, vx);
    dg::blas1::transfer( out2, vy);
}

template<class TernaryOp1, class TernaryOp2, class container, class Geometry> 
void doPushForwardPerp( TernaryOp1 f1, TernaryOp2 f2, 
        container& vx, container& vy,
        const Geometry& g, CurvilinearCylindricalTag)
{
    typedef typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector host_vec;
    host_vec out1 = pullback( f1, g), temp1(out1);
    host_vec out2 = pullback( f2, g), temp2(out2);
    dg::blas1::pointwiseDot( g.xr(), temp1, out1);
    dg::blas1::pointwiseDot( 1., g.xz(), temp2, 1., out1);
    dg::blas1::pointwiseDot( g.yr(), temp1, out2);
    dg::blas1::pointwiseDot( 1., g.yz(), temp2, 1., out2);
    dg::blas1::transfer( out1, vx);
    dg::blas1::transfer( out2, vy);
}

template<class FunctorRR, class FunctorRZ, class FunctorZZ, class container, class Geometry> 
void doPushForwardPerp( FunctorRR chiRR, FunctorRZ chiRZ, FunctorZZ chiZZ,
        container& chixx, container& chixy, container& chiyy,
        const Geometry& g, OrthonormalCylindricalTag)
{
    typedef typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector host_vec;
    host_vec chixx_ = evaluate( chiRR, g);
    host_vec chixy_ = evaluate( chiRZ, g);
    host_vec chiyy_ = evaluate( chiZZ, g);
    dg::blas1::transfer( chixx_, chixx);
    dg::blas1::transfer( chixy_, chixy);
    dg::blas1::transfer( chiyy_, chiyy);
}

template<class FunctorRR, class FunctorRZ, class FunctorZZ, class container, class Geometry> 
void doPushForwardPerp( FunctorRR chiRR_, FunctorRZ chiRZ_, FunctorZZ chiZZ_,
        container& chixx, container& chixy, container& chiyy,
        const Geometry& g, CurvilinearCylindricalTag)
{
    //compute the rhs
    typedef typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector host_vec;
    host_vec chixx_ = pullback( chiRR_, g), chiRR(chixx_);
    host_vec chixy_ = pullback( chiRZ_, g), chiRZ(chixy_);
    host_vec chiyy_ = pullback( chiZZ_, g), chiZZ(chiyy_);
    //compute the transformation matrix
    host_vec t00(chixx), t01(t00), t02(t00), t10(t00), t11(t00), t12(t00), t20(t00), t21(t00), t22(t00);
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
    dg::blas1::pointwiseDot(     t00, chiRR, chixx_);
    dg::blas1::pointwiseDot( 1., t01, chiRZ, 1., chixx_);
    dg::blas1::pointwiseDot( 1., t02, chiZZ, 1., chixx_);
    dg::blas1::pointwiseDot(     t10, chiRR, chixy_);
    dg::blas1::pointwiseDot( 1., t11, chiRZ, 1., chixy_);
    dg::blas1::pointwiseDot( 1., t12, chiZZ, 1., chixy_);
    dg::blas1::pointwiseDot(     t20, chiRR, chiyy_);
    dg::blas1::pointwiseDot( 1., t21, chiRZ, 1., chiyy_);
    dg::blas1::pointwiseDot( 1., t22, chiZZ, 1., chiyy_);
    dg::blas1::transfer( chixx_, chixx);
    dg::blas1::transfer( chixy_, chixy);
    dg::blas1::transfer( chiyy_, chiyy);

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
    host_vector temp = dg::create::weights( g), vol(temp);
    dg::blas1::transfer( g.vol(), vol); //g.vol might be on device
    dg::blas1::pointwiseDot( vol, temp, temp);
    return temp;
}

template< class Geometry>
typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector doCreateInvVolume( const Geometry& g, CurvilinearTag)
{
    typedef typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector host_vector;
    host_vector temp = dg::create::inv_weights(g), vol(temp);
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
    Grid1d gz( g.z0(), g.z1(), 1, g.Nz());
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
