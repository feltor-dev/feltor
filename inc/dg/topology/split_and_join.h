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

///@addtogroup scatter
///@{

/** @brief Split a vector into planes along the last dimension (fast version)
*
* @param in contiguous 3d vector (must be of size \c grid.size())
* @param out contains \c grid.nz()*grid.Nz() 2d vector views of 2d size on output
* @param grid provide dimensions in 3rd and first two dimensions
* @attention out will NOT be resized
* @tparam SharedContainer \c TensorTraits exists for this class and the
*   \c tensor_category derives from \c SharedVectorTag
*/
template<class SharedContainer, class real_type>
void split( SharedContainer& in, std::vector<View<SharedContainer>>& out, const aRealTopology3d<real_type>& grid)
{
    assert( out.size() == grid.Nz());
    unsigned size2d=grid.nx()*grid.ny()*grid.Nx()*grid.Ny();
    for(unsigned i=0; i<grid.nz()*grid.Nz(); i++)
        out[i].construct( thrust::raw_pointer_cast(in.data()) + i*size2d, size2d);
}

/** @brief Split a vector into planes along the last dimension (construct version)
*
* @param in contiguous 3d vector (must be of size \c grid.size())
* @param grid provide dimensions in 3rd and first two dimensions
* @return \c out contains \c grid.nz()*grid.Nz() 2d vector views of 2d size on output
* @tparam SharedContainer \c TensorTraits exists for this class and the
*   \c tensor_category derives from \c SharedVectorTag
*/
template<class SharedContainer, class real_type>
std::vector<View<SharedContainer>> split( SharedContainer& in, const aRealTopology3d<real_type>& grid)
{
    std::vector<View<SharedContainer>> out;
    RealGrid3d<real_type> l( grid);
    unsigned size2d=l.nx()*l.ny()*l.Nx()*l.Ny();
    out.resize( l.nz()*l.Nz());
    for(unsigned i=0; i<l.nz()*l.Nz(); i++)
        out[i].construct( thrust::raw_pointer_cast(in.data()) + i*size2d, size2d);
    return out;
}

/**
 * @brief Construct a 3d vector given a 2d host vector
 *
 * Conceptually the same as a split of the out vector followed by assigning
 * the input to each plane
 * @param in2d the 2d input
 * @param out output (memory will be allocated)
 * @param grid provide dimensions in 3rd and first two dimensions
 */
template<class Container, class real_type>
void assign3dfrom2d( const thrust::host_vector<real_type>& in2d, Container&
        out, const aRealTopology3d<real_type>& grid)
{
    thrust::host_vector<real_type> vector( grid.size());
    std::vector<dg::View< thrust::host_vector<real_type>>> view =
        dg::split( vector, grid); //3d vector
    for( unsigned i=0; i<grid.nz()*grid.Nz(); i++)
        dg::blas1::copy( in2d, view[i]);
    dg::assign( vector, out);
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
 * @param out contains \c grid.nz()*grid.Nz() 2d vector views of 2d size on output
 * @param grid provide dimensions in 3rd and first two dimensions
 * @tparam MPIContainer An MPI_Vector of a \c SharedContainer
*/
template<class MPIContainer, class real_type>
void split( MPIContainer& in, std::vector<get_mpi_view_type<MPIContainer> >&
    out, const aRealMPITopology3d<real_type>& grid)
{
    //local size2d
    RealGrid3d<real_type> l = grid.local();
    unsigned size2d=l.nx()*l.ny()*l.Nx()*l.Ny();
    for(unsigned i=0; i<l.nz()*l.Nz(); i++)
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
* @return \c out contains \c grid.Nz() 2d vector views of 2d size on output
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
    MPI_Comm planeComm = grid.get_perp_comm(), comm_mod, comm_mod_reduce;
    exblas::mpi_reduce_communicator( planeComm, &comm_mod, &comm_mod_reduce);
    //local size2d
    RealGrid3d<real_type> l = grid.local();
    unsigned size2d=l.nx()*l.ny()*l.Nx()*l.Ny();
    out.resize( l.nz()*l.Nz());
    for(unsigned i=0; i<l.nz()*l.Nz(); i++)
    {
        out[i].data().construct( thrust::raw_pointer_cast(in.data().data())
            + i*size2d, size2d);
        out[i].set_communicator( planeComm, comm_mod, comm_mod_reduce);
    }
    return out;
}

/**
 * @brief MPI Version of assign3dfrom2d
 *
 * Conceptually the same as a split of the out vector followed by assigning
 * the input to each plane
 * @param in2d the 2d input (communicator is ignored)
 * @param out output (memory will be allocated)
 * @param grid provide dimensions in 3rd and first two dimensions
 */
template<class LocalContainer, class real_type>
void assign3dfrom2d( const MPI_Vector<thrust::host_vector<real_type>>& in2d,
        MPI_Vector<LocalContainer>& out,
        const aRealMPITopology3d<real_type>& grid)
{
    MPI_Vector<thrust::host_vector<real_type>> vector = dg::evaluate( dg::zero, grid);
    std::vector<MPI_Vector<dg::View<thrust::host_vector<real_type>>> > view =
        dg::split( vector, grid); //3d vector
    for( unsigned i=0; i<grid.nz()*grid.local().Nz(); i++)
        dg::blas1::copy( in2d, view[i]);
    dg::assign( vector, out);
}
#endif //MPI_VERSION

///@}
}//namespace dg
