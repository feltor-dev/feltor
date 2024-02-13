#pragma once

#include "weights.h"
#include "mpi_grid.h"



/**@file
* @brief MPI weights
*/
namespace dg
{
namespace create
{

///@addtogroup highlevel
///@{

///@copydoc hide_weights_doc
///@copydoc hide_code_mpi_evaluate1d
///@copydoc hide_code_mpi_evaluate2d
///@copydoc hide_code_mpi_evaluate3d
template<class MPITopology, typename = std::enable_if_t<dg::is_mpi_grid<MPITopology>::value >>
typename MPITopology::host_vector weights( const MPITopology& g)
{
    return { dg::create::weights(g.local()), g.communicator()};
}
///@copydoc hide_inv_weights_doc
template<class MPITopology, typename = std::enable_if_t<dg::is_mpi_grid<MPITopology>::value>>
typename MPITopology::host_vector inv_weights( const MPITopology& g)
{
    return { dg::create::inv_weights(g.local()), g.communicator()};
}
///@copydoc hide_weights_coo_doc
///@tparam Coord either \c dg::coo2d or \c dg::coo3d
template<class MPITopology, class Coord, typename = std::enable_if_t<dg::is_mpi_grid<MPITopology>::value >>
typename MPITopology::host_vector weights( const MPITopology& g, Coord coo)
{
    return { dg::create::weights(g.local(), coo), g.communicator()};
}

///@}
}//namespace create

}//namespace dg
