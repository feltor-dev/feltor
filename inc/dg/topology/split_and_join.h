#pragma once
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "dg/backend/blas1_dispatch_shared.h"
#include "dg/backend/view.h"
#include "grid.h"
#ifdef MPI_VERSION
#include "dg/backend/mpi_vector.h"
#include "mpi_grid.h"
#endif //MPI_VERSION

namespace dg
{

///@ingroup scatter
///@{

/** @brief  Split a vector into planes along the last dimension
*
* @param in contiguous 3d vector (must be of size \c grid.size())
* @param out contains \c grid.Nz() 2d vector views of 2d size on output (gets resized if necessary)
* @param grid provide dimensions in 3rd and first two dimensions
* @tparam SharedContainer \c TensorTraits exists for this class and the \c tensor_category derives from \c SharedVectorTag
*/
template<class SharedContainer>
void split( SharedContainer& in, std::vector<View<SharedContainer>>& out, const aTopology3d& grid)
{
    Grid3d l( grid);
    unsigned size2d=l.n()*l.n()*l.Nx()*l.Ny();
    out.resize( l.Nz());
    for(unsigned i=0; i<l.Nz(); i++)
        out[i].construct( thrust::raw_pointer_cast(in.data()) + i*size2d, size2d);
}

#ifdef MPI_VERSION
///@brief MPI Version of split
///@copydetails dg::split()
///@tparam MPIContainer An MPI_Vector of a \c SharedContainer
///@note every plane in \c out holds a 2d Cartesian MPI_Communicator
///@note two seperately split vectors have congruent (not identical) MPI_Communicators (Note here the MPI concept of congruent vs. identical communicators)
template<class MPIContainer, class SharedContainer>
void split( MPIContainer& in, std::vector<MPI_Vector<View<SharedContainer>> >& out, const aMPITopology3d& grid)
{
    static_assert( std::is_same< typename MPIContainer::container_type, typename std::remove_cv<SharedContainer>::type >::value, "Both types in dg::split must be compatible!");
    static_assert( (std::is_const<MPIContainer>::value && std::is_const<SharedContainer>::value) || (!std::is_const<MPIContainer>::value && !std::is_const<SharedContainer>::value), "Both types in dg::split must be either const or non-const!");
    int result;
    MPI_Comm_compare( in.communicator(), grid.communicator(), &result);
    assert( result == MPI_CONGRUENT || result == MPI_IDENT);
    MPI_Comm planeComm = grid.get_perp_comm(), comm_mod, comm_mod_reduce;
    exblas::mpi_reduce_communicator( planeComm, &comm_mod, &comm_mod_reduce);
    //local size2d
    Grid3d l = grid.local();
    unsigned size2d=l.n()*l.n()*l.Nx()*l.Ny();
    out.resize( l.Nz());
    for(unsigned i=0; i<l.Nz(); i++)
    {
        out[i].data().construct( thrust::raw_pointer_cast(in.data().data()) + i*size2d, size2d);
        out[i].set_communicator( planeComm, comm_mod, comm_mod_reduce);
    }
}
#endif //MPI_VERSION

///@}
}//namespace dg
