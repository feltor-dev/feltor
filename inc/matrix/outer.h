#pragma once


#include <vector>
#include "dg/algorithm.h"

namespace dg{

template<class ContainerType0, class ContainerType1, class ContainerType2>
void outer_product( const ContainerType0& vx, const ContainerType1& vy, ContainerType2& y)
{
    using value_type = get_value_type<ContainerType0>;
    unsigned size = y.size();
    unsigned Nx = vx.size(), Ny = vy.size();
    unsigned product_size = Nx*Ny;
    if( size != product_size )
        throw dg::Error( Message( _ping_) << "Size " << size << " incompatible with outer produt size "<<product_size<<"!");
    dg::blas2::parallel_for([Nx ] DG_DEVICE(
        unsigned i, value_type* y, const value_type* vx, const value_type* vy)
    {
        y[i] = vx[i%Nx]*vy[i/Nx];
    }, size, y, vx, vy);
}

template<class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3>
void outer_product( const ContainerType0& vx, const ContainerType1& vy, ContainerType2& vz, ContainerType3& y)
{
    using value_type = get_value_type<ContainerType0>;
    unsigned size = y.size();
    unsigned Nx = vx.size(), Ny = vy.size(), Nz = vz.size();
    unsigned product_size = Nx*Ny*Nz;
    if( size != product_size )
        throw dg::Error( Message( _ping_) << "Size " << size << " incompatible with outer produt size "<<product_size<<"!");
    dg::blas2::parallel_for( [Nx, Ny] DG_DEVICE(
        unsigned i, value_type* y, const value_type* vx, const value_type* vy, const value_type* vz)
    {
        y[i] = vx[i%Nx]*vy[(i/Nx)%Ny]*vz[i/(Nx*Ny)];
    }, size, y, vx, vy, vz);
}

#ifdef MPI_VERSION
// Not tested yet
template<class ContainerType0, class ContainerType1, class ContainerType2>
void outer_product( const MPI_Vector<ContainerType0>& vx, const MPI_Vector<ContainerType1>& vy, MPI_Vector<ContainerType2>& y)
{
    outer_product( vx.data(), vy.data(), y.data());
}
template<class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3>
void outer_product( const MPI_Vector<ContainerType0>& vx, const MPI_Vector<ContainerType1>& vy, MPI_Vector<ContainerType2>& vz, MPI_Vector<ContainerType3>& y)
{
    outer_product( vx.data(), vy.data(), vz.data(), y.data());
}
#endif // MPI_VERSION



} // namespace dg
