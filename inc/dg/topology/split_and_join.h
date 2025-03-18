#pragma once
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "dg/backend/blas1_dispatch_shared.h"
#include "dg/backend/view.h"
#include "dg/blas1.h"
#include "grid.h"
#ifdef MPI_VERSION
#include "dg/backend/mpi_vector.h"
#include "mpi_grid.h"
#include "mpi_evaluation.h"
#endif //MPI_VERSION

namespace dg
{
    // TODO The only place this is tested implicitly is in fieldaligned and mpi_fieldaligned ?!
    // And in elliptic_mpib

///@addtogroup scatter
///@{

/** @brief Split a vector into planes along the last dimension (fast version)
*
* @param in contiguous 3d vector (must be of size \c grid.size())
* @param out contains <tt>grid.nz()*grid.Nz()</tt> 2d vector views of 2d size on output
* @param grid provide dimensions in 3rd and first two dimensions
* @attention out will NOT be resized
* @tparam SharedContainer \c TensorTraits exists for this class and the
*   \c tensor_category derives from \c SharedVectorTag
*/
template<class SharedContainer, class real_type>
void split( SharedContainer& in, std::vector<View<SharedContainer>>& out, const aRealTopology3d<real_type>& grid)
{
    assert( out.size() == grid.shape(2));
    unsigned size2d=grid.shape(0)*grid.shape(1);
    for(unsigned i=0; i<grid.shape(2); i++)
        out[i].construct( thrust::raw_pointer_cast(in.data()) + i*size2d, size2d);
}

/** @brief Split a vector into planes along the last dimension (construct version)
*
* @param in contiguous 3d vector (must be of size \c grid.size())
* @param grid provide dimensions in 3rd and first two dimensions
* @return \c out contains <tt>grid.nz()*grid.Nz()</tt> 2d vector views of 2d size on output
* @tparam SharedContainer \c TensorTraits exists for this class and the
*   \c tensor_category derives from \c SharedVectorTag
*/
template<class SharedContainer, class real_type>
std::vector<View<SharedContainer>> split( SharedContainer& in, const aRealTopology3d<real_type>& grid)
{
    std::vector<View<SharedContainer>> out;
    unsigned size2d=grid.shape(0)*grid.shape(1);
    out.resize( grid.shape(2));
    for(unsigned i=0; i<grid.shape(2); i++)
        out[i].construct( thrust::raw_pointer_cast(in.data()) + i*size2d, size2d);
    return out;
}

/**
 * @brief Construct a 3d vector given a 2d host vector
 *
 * A shortcut for
 * @code{.cpp}
    out = dg::construct<Container>(dg::kronecker ( dg::cooX2d, in2d, grid.abscissas(2)));
 * @endcode
 * @param in2d the 2d input
 * @param out output (memory will be allocated)
 * @param grid provide dimensions in 3rd and first two dimensions
 */
template<class Container, class host_vector, class Topology>
void assign3dfrom2d( const host_vector& in2d, Container&
        out, const Topology& grid)
{
    out = dg::construct<Container>(dg::kronecker ( dg::cooX2d, in2d, grid.abscissas(2)));
}


#ifdef MPI_VERSION

template<class MPIContainer>
using get_mpi_view_type =
    std::conditional_t< std::is_const<MPIContainer>::value,
    MPI_Vector<View<const typename MPIContainer::container_type>>,
    MPI_Vector<View<typename MPIContainer::container_type>> >;

/** @brief MPI Version of split (fast version)
 *
 * @note every plane in \c out must hold a 2d Cartesian MPI_Communicator
 * congruent (same process group) or ident (same process group, same context)
 * with the communicator in \c grid
 * @attention This version will NOT adapt the communicators in \c out
 * @attention out will NOT be resized
 * @param in contiguous 3d vector (must be of size \c grid.size())
 * @param out contains <tt>grid.nz()*grid.Nz()</tt> 2d vector views of 2d size on output
 * @param grid provide dimensions in 3rd and first two dimensions
 * @tparam MPIContainer An MPI_Vector of a \c SharedContainer
*/
template<class MPIContainer, class real_type>
void split( MPIContainer& in, std::vector<get_mpi_view_type<MPIContainer> >&
    out, const aRealMPITopology3d<real_type>& grid)
{
    //local size2d
    RealGrid3d<real_type> l = grid.local();
    unsigned size2d=l.shape(0)*l.shape(1);
    for(unsigned i=0; i<l.shape(2); i++)
    {
        out[i].data().construct( thrust::raw_pointer_cast(in.data().data()) +
            i*size2d, size2d);
    }
}
/** @brief MPI Version of split (construct version)
*
* may take longer due to the many calls to MPI group creation functions
* @param in contiguous 3d vector (must be of size \c grid.size())
* @param grid provide dimensions in 3rd and first two dimensions
* @return \c out contains <tt>grid.nz()*grid.Nz()</tt> 2d vector views of 2d size on output
* @note two seperately split vectors have congruent (not identical) MPI_Communicators Note here the MPI concept of congruent (same process group, different contexts) vs. identical (same process group, same context) communicators.
* @tparam MPIContainer An MPI_Vector of a \c SharedContainer
*/
template< class MPIContainer, class real_type>
std::vector<get_mpi_view_type<MPIContainer> > split(
    MPIContainer& in, const aRealMPITopology3d<real_type>& grid)
{
    std::vector<get_mpi_view_type<MPIContainer>> out;
    int result;
    MPI_Comm_compare( in.communicator(), grid.communicator(), &result);
    assert( result == MPI_CONGRUENT || result == MPI_IDENT);
    MPI_Comm planeComm = grid.get_perp_comm();
    //local size2d
    RealGrid3d<real_type> l = grid.local();
    unsigned size2d=l.shape(0)*l.shape(1);
    out.resize( l.shape(2));
    for(unsigned i=0; i<l.shape(2); i++)
    {
        out[i].data().construct( thrust::raw_pointer_cast(in.data().data())
            + i*size2d, size2d);
        out[i].set_communicator( planeComm);
    }
    return out;
}

#endif //MPI_VERSION

///@}
}//namespace dg
