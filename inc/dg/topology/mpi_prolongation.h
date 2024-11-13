#pragma once

#include "mpi_projection.h"
#include "prolongation.h"

/*! @file
  @brief Useful MPI typedefs and overloads of interpolation and projection
  */

namespace dg
{
namespace create
{
template<class MPITopology, size_t Md, typename = std::enable_if_t<is_mpi_grid<MPITopology>::value >>
dg::MIHMatrix_t<typename MPITopology::value_type> prolongation(
    const MPITopology& g_new, std::array<unsigned,Md> axes)
{
    using real_type = typename MPITopology::value_type;
    dg::IHMatrix_t<real_type> mat = dg::create::prolongation(
        g_new.local(), axes); // local rows, local cols
    dg::GeneralComm< dg::iHVec, thrust::host_vector<real_type>> comm; // no communication
    return dg::MIHMatrix_t<real_type>( mat, comm, dg::row_dist);
}

template<class MPITopology, size_t Md, typename = std::enable_if_t<is_mpi_grid<MPITopology>::value >>
dg::MIHMatrix_t<typename MPITopology::value_type> reduction(
    std::array<unsigned,Md> axes, const MPITopology& g_old)
{
    using real_type = typename MPITopology::value_type;
    // We need to create a grid where are all reduction directions are global
    // and the remaining ones are local
    //std::array<RealGrid<real_type,1>,MPITopology::ndim()> gs;
    //auto bools = detail::axes2bools( g_old, axes);
    //for( unsigned u=0; u<MPITopology::ndim(); u++)
    //{
    //    if( bools[u])
    //        gs[u] = g_old.grid(u).global();
    //    else
    //        gs[u] = g_old.grid(u).local();
    //}
    //RealGrid<real_type,MPITopology::ndim()> g(gs);
    //dg::IHMatrix_t<real_type> mat = dg::create::reduction(
    //    axes, g); // local rows, global cols
    //
    std::vector<MPI_Comm> comms(Md);
    for( unsigned u=0; u<Md; u++)
        comms[u] = g_old.comm(axes[u]);

    dg::IHMatrix_t<real_type> mat = dg::create::reduction(
        axes, g_old.local()); // local rows, global cols
    return {  mat, dg::mpi_cart_kron( comms) };
}
template<class MPITopology, size_t Md, typename = std::enable_if_t<is_mpi_grid<MPITopology>::value >>
dg::MIHMatrix_t<typename MPITopology::value_type> projection(
    std::array<unsigned,Md> axes, const MPITopology& g_old)
{
    using real_type = typename MPITopology::value_type;
    // We need to create a grid where are all reduction directions are global
    // and the remaining ones are local
    //std::array<RealGrid<real_type,1>,MPITopology::ndim()> gs;
    //auto bools = detail::axes2bools( g_old, axes);
    //for( unsigned u=0; u<MPITopology::ndim(); u++)
    //{
    //    if( bools[u])
    //        gs[u] = g_old.grid(u).global();
    //    else
    //        gs[u] = g_old.grid(u).local();
    //}
    //RealGrid<real_type,MPITopology::ndim()> g(gs);
    //dg::IHMatrix_t<real_type> mat = dg::create::projection(
    //    axes, g); // local rows, global cols
    ////std::cout << "Mat "<<mat.num_rows<<" "<<mat.num_cols<<" "<<mat.num_entries<<"\n";
    //return convert(  mat, g_old);
    std::vector<MPI_Comm> comms(Md);
    for( unsigned u=0; u<Md; u++)
        comms[u] = g_old.comm(axes[u]);
    MPI_Comm new_comm = dg::mpi_cart_kron( comms);
    dg::IHMatrix_t<real_type> mat = dg::create::projection(
        axes, g_old.local()); // local rows, global cols
    return {  mat, new_comm };
}
}//namespace create
}//namespace dg
