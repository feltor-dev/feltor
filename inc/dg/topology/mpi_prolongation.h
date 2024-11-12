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
template<class MPITopology, typename = std::enable_if_t<is_mpi_grid<MPITopology>::value >>
dg::MIHMatrix_t<typename MPITopology::value_type> prolongation(
    const MPITopology& g_new, std::array<bool,MPITopology::ndim()> remains)
{
    using real_type = typename MPITopology::value_type;
    dg::IHMatrix_t<real_type> mat = dg::create::prolongation(
        g_new.local(), remains); // local rows, local cols
    dg::GeneralComm< dg::iHVec, thrust::host_vector<real_type>> comm; // no communication
    return dg::MIHMatrix_t<real_type>( mat, comm, dg::row_dist);
}
template<class MPITopology, typename = std::enable_if_t<is_mpi_grid<MPITopology>::value >>
dg::MIHMatrix_t<typename MPITopology::value_type> average(
    std::array<bool,MPITopology::ndim()> remains, const MPITopology& g_old)
{
    using real_type = typename MPITopology::value_type;
    dg::IHMatrix_t<real_type> mat = dg::create::average(
        remains, g_old.local()); // local rows, local cols
    convertLocal2GlobalCols( mat, g_old);
    return convert(  mat, g_old);
}
}//namespace create
}//namespace dg
