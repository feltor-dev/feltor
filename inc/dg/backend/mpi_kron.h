#pragma once
#include <vector>
#include <map>
#include "exceptions.h"


namespace dg
{
///@cond
namespace detail{
struct MPICartInfo
{
    MPI_Comm root;
    std::vector<int> remains;
    // std::vector<int> periods; // no need to track because the root and all the children have the same periods
};
//we keep track of communicators that were created in the past
static std::map<MPI_Comm, MPICartInfo> mpi_cart_info_map;
}
///@endcond
/*! @brief register a call to \c MPI_Cart_create with the dg library
 *
 * The \c comm_cart parameter is the \c MPI_Comm that will be registered
 * @note The function does not check if \c comm_cart is already registered
 * (it will simply overwrite an existing entry)
 * @param comm_old parameter used in \c MPI_Cart_create
 * @param ndims parameter used in \c MPI_Cart_create
 * @param dims parameter used in \c MPI_Cart_create
 * @param periods parameter used in \c MPI_Cart_create
 * @param reorder parameter used in \c MPI_Cart_create
 * @param comm_cart parameter used in \c MPI_Cart_create
 * @ingroup mpi_structures
 */
static void register_mpi_cart_create( MPI_Comm comm_old, int ndims, const int dims[],
                    const int periods[], int reorder, MPI_Comm comm_cart)
{
    std::vector<int> remains( ndims, 1);
    detail::MPICartInfo info{ comm_cart, remains};
    detail::mpi_cart_info_map[comm_cart] = info;
}
/*! @brief Call and register a call to \c MPI_Cart_create with the dg library
 *
 * If MPI_Cart_create is successful this function is equivalent to
 * @code{.cpp}
    MPI_Cart_create( comm_old, ndims, dims, periods, reorder, comm_cart);
    dg::register_mpi_cart_create( comm_old, ndims, dims, periods, reorder, *comm_cart);
    return MPI_SUCCESS;
 * @endcode
 * @ingroup mpi_structures
 */
static int mpi_cart_create( MPI_Comm comm_old, int ndims, const int dims[],
                    const int periods[], int reorder, MPI_Comm * comm_cart)
{
    int err = MPI_Cart_create( comm_old, ndims, dims, periods, reorder, comm_cart);
    if( err != MPI_SUCCESS)
        return err;
    register_mpi_cart_create( comm_old, ndims, dims, periods, reorder, *comm_cart);
    return err;
}

/*! @brief register a call to \c MPI_Cart_sub with the dg library
 *
 * The \c newcomm parameter is the \c MPI_Comm that will be registered
 * @note \c comm needs to be already registered
 * @param comm parameter used in \c MPI_Cart_sub
 * @param remain_dims parameter used in \c MPI_Cart_sub
 * @param newcomm parameter used in \c MPI_Cart_sub
 * @ingroup mpi_structures
 */
static void register_mpi_cart_sub( MPI_Comm comm, const int remain_dims[], MPI_Comm newcomm)
{
    detail::MPICartInfo info = detail::mpi_cart_info_map.at(comm);
    for( unsigned u=0; u<info.remains.size(); u++)
        info.remains[u]  = remain_dims[u];
    detail::mpi_cart_info_map[newcomm] = info;
}

/*! @brief Call and register a call to \c MPI_Cart_sub with the dg library
 *
 * If \c MPI_Cart_sub is successful and an equivalent sub communicator does not eixst already,
 * this function is equivalent to
 * @code{.cpp}
    MPI_Cart_sub( comm, remain_dims, newcomm);
    dg::register_mpi_cart_sub( comm, remain_dims, *newcomm);
    return MPI_SUCCESS;
 * @endcode
 * @note \c comm needs to be already registered
 * @param comm parameter of \c MPI_Cart_sub
 * @param remain_dims parameter of \c MPI_Cart_sub
 * @param newcomm parameter of \c MPI_Cart_sub
 * @param duplicate Determines what happens in case \c MPI_Cart_sub was already reigstered with the
 * same input parameters \c comm and \c remain_dims. True: call \c MPI_Cart_sub and register
 * the novel communicator even if a duplicate exists. False: first check if a communicator
 * that was subbed from \c comm with \c remain_dims was previously registered. In case one is found
 * set *newcomm = existing_comm. Else, call and register \c MPI_Cart_sub.
 * @ingroup mpi_structures
 */
static int mpi_cart_sub( MPI_Comm comm, const int remain_dims[], MPI_Comm *newcomm, bool duplicate = false)
{
    detail::MPICartInfo info = detail::mpi_cart_info_map.at(comm);
    for( unsigned u=0; u<info.remains.size(); u++)
        info.remains[u]  = remain_dims[u];
    if( ! duplicate)
    {
    for (auto it = detail::mpi_cart_info_map.begin(); it != detail::mpi_cart_info_map.end(); ++it)
        if( it->second.root == info.root && it->second.remains == info.remains)
        {
            *newcomm = it->first;
            return MPI_SUCCESS; // already registered
        }
    }
    int err = MPI_Cart_sub( comm, remain_dims, newcomm);
    if( err != MPI_SUCCESS)
        return err;
    register_mpi_cart_sub( comm, remain_dims, *newcomm);
    return err;
}

/*! @brief Form a Kronecker product among cartesian communicators
 *
 * All input comms must be registered in the dg library as Cartesian communicators
 * that derive from the same root Cartesian communicator.
 * Furthermore the comms must be mutually orthogonal i.e. any \c true
 * entry in \c remain_dims can exist in only exactly one comm.
 * The resulting \c remain_dims of the output is then the union of all \c remain_dims
 * of the inputs.
 *
 * The returned communicator is then the one that hypothetically generated all input comms
 * through <tt> MPI_Cart_sub( return_comm, remain_dims[u], comms[u]); </tt>
 * for all <tt>u < comms.size()</tt>;
 * @note The order of communicators matters. The function will not transpose communicators
 * @param comms input communicators (their order is irrelevant, the result is the same if reordered)
 * @return Kronecker product of communicators (is automatically registered)
 * @ingroup mpi_structures
 */
static MPI_Comm mpi_cart_kron( std::vector<MPI_Comm> comms)
{
    if ( comms.empty())
        return MPI_COMM_NULL;
    std::vector<detail::MPICartInfo> infos(comms.size());

    for( unsigned u=0; u<comms.size(); u++)
        infos [u] = detail::mpi_cart_info_map.at(comms[u]);
    MPI_Comm root = infos[0].root;
    for( unsigned u=0; u<comms.size(); u++)
        if( infos[u].root != root)
            throw Error(Message(_ping_)<<
                    "In mpi_cart_kron all comms must have same root comm "
                    <<root<<" Offending comm number "<<u<<" with root "
                    <<infos[u].root);
    auto root_info = detail::mpi_cart_info_map.at(root);
    size_t ndims = root_info.remains.size();
    std::vector<int> remains( ndims, 0) ;
    unsigned current_free_k=0;
    for( unsigned u=0; u<comms.size(); u++)
    {
        for( unsigned k=0; k<ndims; k++)
        if( infos[u].remains[k])
        {
            if( remains[k])
                throw Error(Message(_ping_)<<
                    "Cannot form kronecker product with given communicators as remaining dims must be orthogonal ");
            if( k < current_free_k)
                throw Error(Message(_ping_)<<
                    "Cannot form kronecker product with communicators in reverse order compared to original ");
            remains[k] = infos[u].remains[k];
            current_free_k = k+1;
        }
    }
    MPI_Comm newcomm;
    int err = dg::mpi_cart_sub( root_info.root, &remains[0], &newcomm, false);
    if( err != MPI_SUCCESS)
        throw Error(Message(_ping_)<<
                "Cannot form kronecker product with given communicators");
    return newcomm;
}

/*! @brief Split a Cartesian communicator along each dimensions
 *
 * using repeated calls to \c dg::mpi_cart_sub
 * @tparam Nd number of dimensions
 * @param comm input Cartesian communicator must be of dimension Nd
 */
template<size_t Nd>
std::array<MPI_Comm, Nd> mpi_cart_split( MPI_Comm comm)
{
    // Should there be a std::vector version?
    // TODO assert dimensionality of comm
    std::array<MPI_Comm, Nd> comms;
    int remain_dims[Nd];
    for( unsigned u=0; u<Nd; u++)
    {
        for( unsigned k=0; k<Nd; k++)
            remain_dims[k]=0;
        remain_dims[u] = 1;
        mpi_cart_sub( comm, remain_dims, &comms[u], false);
    }

    return comms;
}


// Need to think about those again
// /*! @brief unregister a communicator
// * For example if a communicator was freed
// * @param comm delete associated registry entry
// * @ingroup mpi_structures
// */
//void unregister_mpi_comm( MPI_Comm comm)
//{
//    detail::mpi_cart_info_map.erase( comm);
//}
//
// /*! @brief call \c MPI_Comm_free(comm) followed by \c dg::unregister_mpi_comm(comm)
// * @param comm free communicator and delete associated registry entry
// * @ingroup mpi_structures
// */
//void mpi_comm_free( MPI_Comm * comm)
//{
//    MPI_Comm_free(comm);
//    unregister_mpi_comm( *comm);
//}


}
