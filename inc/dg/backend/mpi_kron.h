#pragma once
#include <cassert>
#include <vector>
#include <algorithm> // std::copy_n
#include <map>
#include "mpi_cart.h"
#include "exceptions.h"


namespace dg
{
/*! @class hide_mpi_cart_rationale
 * @note The reason the dg library provides \c dg::mpi_cart_sub and \c
 * dg::mpi_cart_kron is that unfortunately the MPI standard does not provide a
 * way to form the Kronecker product of Cartesian communicators without
 * manually tracking their parent Cartesian communicators.  However, this is
 * needed in the \c dg::blas1::kronecker and \c dg::kronecker functions.
 */
///@cond
namespace detail{
// I think the mpi registry should not be exposed to the User
// It may be an idea to hint at it in the grid documentation
// inline declared functions generate no warning about "defined but not used"...
// and can be defined in multiple translation units
struct MPICartInfo
{
    MPI_Comm root;
    std::vector<int> remain_dims;
    // std::vector<int> periods; // no need to track because the root and all the children have the same periods
};
//we keep track of communicators that were created in the past
// inline variables have the same address in multiple translation units
// and have external linkage by default
inline std::map<MPI_Comm, MPICartInfo> mpi_cart_registry;

inline void mpi_cart_registry_display( std::ostream& out = std::cout)
{
    out << "MPI Cart registry\n";
    for ( auto pair : detail::mpi_cart_registry)
    {
        auto& re = pair.second.remain_dims;
        out << "Comm "<<pair.first<<" Root "<<pair.second.root<<" Remains size "<<re.size();
        out << " r:";
        for( unsigned i=0; i<re.size(); i++)
            out << " "<<re[i];
        out << std::endl;
    }
}

inline void mpi_cart_registry_clear( )
{
    mpi_cart_registry.clear();
}

inline void mpi_cart_register_cart( MPI_Comm comm)
{
    // Register Cartesian comm if not already there
    auto it = detail::mpi_cart_registry.find( comm);
    if( it == detail::mpi_cart_registry.end())
    {
        int ndims;
        dg::mpi_cartdim_get( comm, &ndims);
        std::vector<int> remain_dims( ndims, true);
        detail::MPICartInfo info{ comm, remain_dims};
        detail::mpi_cart_registry[comm] = info;
    }
}
}

/*! @brief Manually register a call to \c MPI_Cart_sub with the dg library
 *
 * @param comm communicator with Cartesian structure (handle) (parameter used
 * in \c MPI_Cart_sub)
 * @param remain_dims The actual parameter used in \c MPI_Cart_sub
 * corresponding to \c comm
 * @param newcomm communicator containing the subgrid that includes the calling
 * process (handle) (parameter used in \c MPI_Cart_sub)
 * @copydoc hide_mpi_cart_rationale
 */
inline void register_mpi_cart_sub( MPI_Comm comm, const int remain_dims[], MPI_Comm newcomm)
{
    // How should we handle row-major vs column-major issue?
    detail::mpi_cart_register_cart( comm);
    detail::MPICartInfo info = detail::mpi_cart_registry.at(comm);
    unsigned ndims = info.remain_dims.size();
    for( unsigned u=0; u<ndims; u++)
        info.remain_dims[u]  = remain_dims[u];
    detail::mpi_cart_registry[newcomm] = info;
}
// Can't decide whether to make register_mpi_cart_sub public ...
///@endcond
///@addtogroup mpi_utility
///@{

/*! @brief Call and register a call to \c MPI_Cart_sub with the dg library
 *
 * @param comm communicator with Cartesian structure (handle) (parameter used
 * in \c MPI_Cart_sub)
 * @param remain_dims the i-th entry of \c remain_dims specifies whether the
 * (N-i)-th dimension is kept in the subgrid (true) or is dropped (false), must
 * have \c ndims entries. (parameter used in \c MPI_Cart_sub)
 * @note Notice the transposition in \c remain_dims ! In the dg-library we
 * always transpose Cartesian communicators to effectively make the process
 * ranking column-major instead of row-major.
 * @param duplicate Determines what happens in case \c MPI_Cart_sub was already
 * registered with \c comm and the same \c
 * remain_dims. True: call \c MPI_Cart_sub and generate a novel communicator
 * even if a duplicate exists. False: first check if a communicator that was
 * subbed from \c comm with \c remain_dims was previously registered. In case
 * one is found <tt>return existing_comm;</tt>.  Else, call and register \c
 * MPI_Cart_sub.
 * @return communicator containing the subgrid that includes the calling
 * process (handle) (parameter used in \c MPI_Cart_sub)
 * @copydoc hide_mpi_cart_rationale
 */
inline MPI_Comm mpi_cart_sub( MPI_Comm comm, std::vector<int> remain_dims, bool
    duplicate = false)
{
    int ndims;
    dg::mpi_cartdim_get( comm, &ndims);
    assert( (unsigned) ndims == remain_dims.size());

    detail::mpi_cart_register_cart( comm);
    detail::MPICartInfo info = detail::mpi_cart_registry.at(comm);
    // info.remain_dims may be larger than remain_dims because comm may have a parent
    // but exactly remain_dims.size() entries are true
    int counter =0;
    for( unsigned u=0; u<info.remain_dims.size(); u++)
        if( info.remain_dims[u])
        {
            info.remain_dims[u]  = remain_dims[counter];
            counter ++;
        }
    assert( counter == (int)remain_dims.size());
    if( not duplicate)
    {
        for (auto it = detail::mpi_cart_registry.begin(); it !=
            detail::mpi_cart_registry.end(); ++it)
        {
//            int comp_root;
//            MPI_Comm_compare( it->second.root, info.root, &comp_root);
//            if( (comp_root == MPI_IDENT or comp_root == MPI_CONGRUENT) and
//                it->second.remain_dims == info.remain_dims)
            if( it->second.root == info.root && it->second.remain_dims ==
                info.remain_dims)
            {
                return it->first;
            }
        }
    }
    MPI_Comm newcomm;
    int err = dg::mpi_cart_sub( comm, &remain_dims[0], &newcomm);
    if( err != MPI_SUCCESS)
        throw Error(Message(_ping_)<<
                "Cannot create Cartesian sub comm from given communicator");
    register_mpi_cart_sub( comm, &remain_dims[0], newcomm);
    return newcomm;
}

/*! @brief Form a (transposed) Kronecker product among Cartesian communicators
 *
 * Find communicator as the one that hypothetically generated all
 * input comms through <tt>dg::mpi_cart_sub( return_comm, remain_dims[u],
 * comms[u]);</tt> for all <tt>u < comms.size();</tt>
 *
 * Unless \c comms is empty or contains only 1 communicator all input comms
 * must be registered in the dg library as Cartesian communicators that derive
 * from the same root Cartesian communicator.
 * Furthermore the comms must be mutually orthogonal i.e. any \c true entry in
 * \c remain_dims can exist in only exactly one comm.  The resulting \c
 * remain_dims of the output is then the union of all \c remain_dims of the
 * inputs.
 *
 * For example
 * @snippet{trimleft} mpi_kron_mpit.cpp split and join
 *
 * @attention The order of communicators matters. The function will always
 * transpose communicators, such that the first dimension in \c cmoms is the
 * last dimension in the return Cartesian communicator
 * @param comms input communicators
 * @return Kronecker product of communicators
 * @copydoc hide_mpi_cart_rationale
 */
inline MPI_Comm mpi_cart_kron( std::vector<MPI_Comm> comms)
{
    // This non-template interface must exist so compiler can deduce call
    // mpi_cart_kron( {comm0, comm1});
    if ( comms.empty())
        return MPI_COMM_NULL;
    if( comms.size() == 1)
        return comms[0];

    std::vector<detail::MPICartInfo> infos(comms.size());

    for( unsigned u=0; u<comms.size(); u++)
    {
    try{
        infos [u] = detail::mpi_cart_registry.at(comms[u]);
    }catch( std::exception& e)
    {
        std::cerr << "Did not find "<<comms[u]<<" in Registry!\n";
        detail::mpi_cart_registry_display();
        throw e;

    }
    }
    MPI_Comm root = infos[0].root;
    for( unsigned u=0; u<comms.size(); u++)
        if( infos[u].root != root)
            throw Error(Message(_ping_)<<
                    "In mpi_cart_kron all comms must have same root comm "
                    <<root<<" Offending comm number "<<u<<" with root "
                    <<infos[u].root);
    auto root_info = detail::mpi_cart_registry.at(root);
    size_t ndims = root_info.remain_dims.size();
    std::vector<int> remain_dims( ndims, false) ;
    unsigned current_free_k=0;
    for( unsigned u=0; u<comms.size(); u++)
    {
        for( unsigned k=0; k<ndims; k++)
        if( infos[u].remain_dims[k])
        {
            if( remain_dims[k])
                throw Error(Message(_ping_)<<
                    "Cannot form kronecker product with given communicators as remaining dims must be orthogonal ");
            if( k < current_free_k)
                throw Error(Message(_ping_)<<
                    "Cannot form kronecker product with communicators in reverse order compared to original ");
            remain_dims[k] = infos[u].remain_dims[k];
            current_free_k = k+1;
        }
    }
    return dg::mpi_cart_sub( root_info.root, remain_dims, false);
}

/*!
 * @brief Convenience shortcut for <tt> return mpi_cart_kron(
 * std::vector<MPI_Comm>(comms.begin(), comms.end()));</tt>
 */
template<class Vector>
MPI_Comm mpi_cart_kron( const Vector& comms)
{
    return mpi_cart_kron( std::vector<MPI_Comm>(comms.begin(), comms.end()));
}


/*! @brief Split a Cartesian communicator along each dimensions
 *
 * using repeated calls to \c dg::mpi_cart_sub
 * @param comm input Cartesian communicator
 * @return Array of 1-dimensional Cartesian communicators. The i-th
 * communicator corresponds to the (N-i)-th axis in \c comm
 * @attention Notice the tranpose. The first dimension in \c comm is the last
 * dimension in the return vector.
 */
inline std::vector<MPI_Comm> mpi_cart_split( MPI_Comm comm)
{
    // Check that there is a Comm that was already split
    int ndims;
    dg::mpi_cartdim_get( comm, &ndims);

    std::vector<MPI_Comm> comms(ndims);
    for( int u=0; u<ndims; u++)
    {
        std::vector<int> remain_dims(ndims, 0);
        remain_dims[u] = 1;
        comms[u] = mpi_cart_sub( comm, remain_dims, false);
    }
    return comms;
}
/*!
 * @brief Same as \c mpi_cart_split but different return type
 *
 * @snippet{trimleft} mpi_kron_mpit.cpp split and join
 * @tparam Nd Number of dimensions to copy from \c mpi_cart_split
 * @param comm input Cartesian communicator ( <tt>Nd <= comm.ndims</tt>)
 * @return Array of 1-dimensional Cartesian communicators
 * @attention Notice the tranpose. The first dimension in \c comm is the last
 * dimension in the return vector.
 */
template<size_t Nd>
std::array<MPI_Comm, Nd> mpi_cart_split_as( MPI_Comm comm)
{
    auto split = mpi_cart_split( comm);
    std::array<MPI_Comm, Nd> arr;
    std::copy_n( split.begin(), Nd, arr.begin());
    return arr;
}



// Need to think about those again
// /*! @brief unregister a communicator
// * For example if a communicator was freed
// * @param comm delete associated registry entry
// */
//void unregister_mpi_comm( MPI_Comm comm)
//{
//    detail::mpi_cart_registry.erase( comm);
//}
//
// /*! @brief call \c MPI_Comm_free(comm) followed by \c dg::unregister_mpi_comm(comm)
// * @param comm free communicator and delete associated registry entry
// */
//void mpi_comm_free( MPI_Comm * comm)
//{
//    MPI_Comm_free(comm);
//    unregister_mpi_comm( *comm);
//}

///@}


}
