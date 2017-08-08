#pragma once


namespace dg
{
///@cond
template<class Geometry>
class GeometryTraits
{
    typedef typename MemoryTraits< typename TopologyTraits<Geometry>::memory_category>::host_vector host_vector;

};
template<class MemoryTag>
struct MemoryTraits {
};
template<>
struct MemoryTraits< SharedTag>
{
    typedef thrust::host_vector<double> host_vector;
};

///@endcond
/**
 * @brief This function pulls back a function defined in some basic coordinates to the curvilinear coordinate system
 *
 * e.g. F(x,y) = f(R(x,y), Z(x,y)) in 2d 
 * @tparam Functor The binary or ternary function object 
 * @param f The function defined in cartesian coordinates
 * @param g a two- or three dimensional Geometry
 * @note Template deduction for the Functor will fail if you overload functions with different 
 dimensionality (e.g. double sine( double x) and double sine(double x, double y) )
 * You will want to rename those uniquely
 *
 * @return A set of points representing F
 * @ingroup pullback
 */
template< class Functor>
thrust::host_vector<double> pullback( Functor f, const aGeometry2d& g)
{
    const SharedContainers<thrust::host_vector<double> >& map = g.map();
    if( !map.isSet(0) && !map.isSet(1)) //implicit
        return evaluate(f,g);
    thrust::host_vector<double> vec( g.size());
    for( unsigned i=0; i<g.size(); i++)
        vec[i] = f( map.getValue(0)[i], map.getValue(1)[i]);
    return vec;
}

///@copydoc pullback(Functor,const aGeometry2d&)
///@ingroup pullback
template< class Functor>
thrust::host_vector<double> pullback( Functor f, const aGeometry3d& g)
{
    const SharedContainers<thrust::host_vector<double> >& map = g.map();
    if( !map.isSet(0) && !map.isSet(1) && !map.isSet(2)) //implicit
        return evaluate(f,g);

    thrust::host_vector<double> vec( g.size());
    for( unsigned i=0; i<g.size(); i++)
        vec[i] = f( map.getValue(0)[i], map.getValue(1)[i], map.getValue(2)[i]);
    return vec;
}

#ifdef MPI_VERSION

///@cond
template<>
struct MemoryTraits< MPITag>
{
    typedef MPI_Vector<thrust::host_vector<double> > host_vector;
};
///@endcond

///@copydoc pullback(Functor,const aGeometry2d&)
///@ingroup pullback
template< class Functor>
MPI_Vector<thrust::host_vector<double> > pullback( Functor f, const aMPIGeometry2d& g)
{
    const SharedContainers<MPI_Vector<thrust::host_vector<double> >& map = g.map();
    if( !map.isSet(0) && !map.isSet(1)) //implicit
        return evaluate(f,g);
    thrust::host_vector<double> vec( g.size());
    for( unsigned i=0; i<g.size(); i++)
        vec[i] = f( map.getValue(0).data()[i], map.getValue(1).data()[i]);
    return vec;
}

///@copydoc pullback(Functor,const aGeometry2d&)
///@ingroup pullback
template< class Functor>
MPI_Vector<thrust::host_vector<double> > pullback( Functor f, const aMPIGeometry3d& g)
{
    const SharedContainers<MPI_Vector<thrust::host_vector<double> >& map = g.map();
    if( !map.isSet(0) && !map.isSet(1) && !map.isSet(2)) //implicit
        return evaluate(f,g);

    thrust::host_vector<double> vec( g.size());
    for( unsigned i=0; i<g.size(); i++)
        vec[i] = f( map.getValue(0).data()[i], map.getValue(1).data()[i], map.getValue(2).data()[i]);
    return vec;
}

#endif //MPI_VERSION

/**
 * @brief Push forward a vector from cylindrical or Cartesian to a new coordinate system
 *
 * Computes \f[ v^x(x,y) = x_R (x,y) v^R(R(x,y), Z(x,y)) + x_Z v^Z(R(x,y), Z(x,y)) \\
                v^y(x,y) = y_R (x,y) v^R(R(x,y), Z(x,y)) + y_Z v^Z(R(x,y), Z(x,y)) \f]
   where \f$ x_R = \frac{\partial x}{\partial R}\f$, ... 
 * @tparam Functor1 Binary or Ternary functor
 * @tparam Geometry The Geometry class
 * @param vR input R-component in cylindrical coordinates
 * @param vZ input Z-component in cylindrical coordinates
 * @param vx x-component of vector (gets properly resized)
 * @param vy y-component of vector (gets properly resized)
 * @param g The geometry object
 * @ingroup pullback
 */
template<class Functor1, class Functor2, class container, class Geometry> 
void pushForwardPerp( Functor1 vR, Functor2 vZ, 
        container& vx, container& vy,
        const Geometry& g)
{
    typedef typename GeometryTraits< Geometry>::host_vector host_vec;
    host_vec out1 = pullback( vR, g), temp1(out1);
    host_vec out2 = pullback( vZ, g), temp2(out2);
    dg::tensor::multiply(g.map(), out1, out2, temp1, temp2);
    dg::blas1::transfer( out1, vx);
    dg::blas1::transfer( out2, vy);
}

/**
 * @brief Push forward a vector from cylindrical or Cartesian to a new coordinate system
 *
 * Computes \f[ v^x(x,y) = x_R (x,y) v^R(R(x,y), Z(x,y)) + x_Z v^Z(R(x,y), Z(x,y)) \\
                v^y(x,y) = y_R (x,y) v^R(R(x,y), Z(x,y)) + y_Z v^Z(R(x,y), Z(x,y)) \f]
   where \f$ x_R = \frac{\partial x}{\partial R}\f$, ... 
 * @tparam Functor1 Binary or Ternary functor
 * @tparam Geometry The Geometry class
 * @param vR input R-component in cartesian or cylindrical coordinates
 * @param vZ input Z-component in cartesian or cylindrical coordinates
 * @param vPhi input Z-component in cartesian or cylindrical coordinates
 * @param vx x-component of vector (gets properly resized)
 * @param vy y-component of vector (gets properly resized)
 * @param vz z-component of vector (gets properly resized)
 * @param g The geometry object
 * @ingroup pullback
 */
template<class Functor1, class Functor2, class Functor3 class container, class Geometry> 
void pushForward( Functor1 vR, Functor2 vZ, Functor3 vPhi,
        container& vx, container& vy, container& vz,
        const Geometry& g)
{
    typedef typename GeometryTraits< Geometry>::host_vector host_vec;
    host_vec out1 = pullback( vR, g), temp1(out1);
    host_vec out2 = pullback( vZ, g), temp2(out2);
    host_vec out3 = pullback( vPhi, g), temp3(out3);
    dg::tensor::multiply(g.map(), out1, out2, out3, temp1, temp2, temp3);
    dg::blas1::transfer( out1, vx);
    dg::blas1::transfer( out2, vy);
    dg::blas1::transfer( out3, vz);
}

/**
 * @brief Push forward a symmetric 2d tensor from cylindrical or Cartesian to a new coordinate system
 *
 * Computes \f[ 
 \chi^{xx}(x,y) = x_R x_R \chi^{RR} + 2x_Rx_Z \chi^{RZ} + x_Zx_Z\chi^{ZZ} \\
 \chi^{xy}(x,y) = x_R x_R \chi^{RR} + (x_Ry_Z+y_Rx_Z) \chi^{RZ} + x_Zx_Z\chi^{ZZ} \\
 \chi^{yy}(x,y) = y_R y_R \chi^{RR} + 2y_Ry_Z \chi^{RZ} + y_Zy_Z\chi^{ZZ} \\
               \f]
   where \f$ x_R = \frac{\partial x}{\partial R}\f$, ... 
 * @tparam Geometry The Geometry class
 * @param chiRR input RR-component in cylindrical coordinates
 * @param chiRZ input RZ-component in cylindrical coordinates
 * @param chiZZ input ZZ-component in cylindrical coordinates
 * @param chixx xx-component of tensor (gets properly resized)
 * @param chixy xy-component of tensor (gets properly resized)
 * @param chiyy yy-component of tensor (gets properly resized)
 * @param g The geometry object
 * @ingroup pullback
 */
template<class FunctorRR, class FunctorRZ, class FunctorZZ, class container, class Geometry> 
void pushForwardPerp( FunctorRR chiRR, FunctorRZ chiRZ, FunctorZZ chiZZ,
        container& chixx, container& chixy, container& chiyy,
        const Geometry& g)
{
    typedef typename GeometryTraits< Geometry>::host_vector host_vec;
    host_vec chiRR_ = pullback( chiRR, g);
    host_vec chiRZ_ = pullback( chiRZ, g);
    host_vec chiZZ_ = pullback( chiZZ, g);
    //transfer to device
    if(g.map().isEmpty())
    {
        chiRR_.swap(chixx);
        chiRZ_.swap(chixy);
        chiZZ_.swap(chiyy);
        return;
    }
    const dg::SparseTensor<container> jac = g.map();
    std::vector<container> values( 3); 
    values[0] = chiRR_, values[1] = chiRZ_, values[2] = chiZZ_;
    SparseTensor<container> chi(values);
    chi(0,0)=0, chi(0,1)=chi(1,0)=1, chi(1,1)=2;

    SparseTensor<container> d = jac.dense(); //now we have a dense tensor
    container tmp00(d.getValue(0,0)), tmp01(tmp00), tmp10(tmp00), tmp11(tmp00);
    // multiply Chi*t -> tmp
    dg::tensor::const_multiply( chi, d.getValue(0,0), d.getValue(1,0), tmp00, tmp10);
    dg::tensor::const_multiply( chi, d.getValue(0,1), d.getValue(1,1), tmp01, tmp11);
    // multiply tT * tmp -> Chi
    SparseTensor<container> transpose = jac.transpose();
    dg::tensor::multiply( transpose, tmp00, tmp01, chixx, chixy);
    dg::tensor::multiply( transpose, tmp10, tmp11, chixy, chiyy);
}

} //namespace dg
