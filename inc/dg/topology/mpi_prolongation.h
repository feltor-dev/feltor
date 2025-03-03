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
///@copydoc dg::create::prolongation(const aRealTopology<real_type,Nd>&,std::array<unsigned,Md>)
///@note In MPI there is no communication involved
template<class MPITopology, size_t Md, typename = std::enable_if_t< dg::is_vector_v<
    typename MPITopology::host_vector, MPIVectorTag>>>
dg::MIHMatrix_t<typename MPITopology::value_type> prolongation(
    const MPITopology& g_new, std::array<unsigned,Md> axes)
{
    using real_type = typename MPITopology::value_type;
    dg::IHMatrix_t<real_type> mat = dg::create::prolongation(
        g_new.local(), axes); // local rows, local cols
    return dg::MIHMatrix_t<real_type>( mat);
}

///@copydoc dg::create::reduction(std::array<unsigned,Md>,const aRealTopology<real_type,Nd>&)
///@note In MPI this represents an "Allreduce" i.e. all ranks along the reduction direction get the result
template<class MPITopology, size_t Md, typename = std::enable_if_t< dg::is_vector_v<
    typename MPITopology::host_vector, MPIVectorTag>>>
dg::MIHMatrix_t<typename MPITopology::value_type> reduction(
    std::array<unsigned,Md> axes, const MPITopology& g_old)
{
    using real_type = typename MPITopology::value_type;
    std::vector<MPI_Comm> comms(Md);
    for( unsigned u=0; u<Md; u++)
        comms[u] = g_old.comm(axes[u]);

    dg::IHMatrix_t<real_type> mat = dg::create::reduction(
        axes, g_old.local()); // local rows, global cols
    return {  mat, dg::mpi_cart_kron( comms) };
}

///@copydoc dg::create::projection(std::array<unsigned,Md>,const aRealTopology<real_type,Nd>&)
///@note In MPI this represents an "Allreduce" i.e. all ranks along the reduction direction get the result
template<class MPITopology, size_t Md, typename = std::enable_if_t< dg::is_vector_v<
    typename MPITopology::host_vector, MPIVectorTag>>>
dg::MIHMatrix_t<typename MPITopology::value_type> projection(
    std::array<unsigned,Md> axes, const MPITopology& g_old)
{
    using real_type = typename MPITopology::value_type;
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
