#pragma once

#include <mpi.h>

namespace dg
{
///@cond
template<class value_type>
static inline MPI_Datatype getMPIDataType(){ assert( false && "Type not supported!\n" ); return MPI_DOUBLE; }
template<>
inline MPI_Datatype getMPIDataType<double>(){ return MPI_DOUBLE;}
template<>
inline MPI_Datatype getMPIDataType<float>(){ return MPI_FLOAT;}
template<>
inline MPI_Datatype getMPIDataType<int>(){ return MPI_INT;}
template<>
inline MPI_Datatype getMPIDataType<bool>(){ return MPI_C_BOOL;}
template<>
inline MPI_Datatype getMPIDataType<unsigned>(){ return MPI_UNSIGNED;}
///@endcond

/**
 * @brief Struct that performs collective scatter and gather operations across processes
 * on distributed vectors using MPI
 *
 * In order to understand what this class does you should first really(!) understand what
 gather and scatter operations are, so grab pen and paper:

 First, we note that gather and scatter are most often used in the context
 of memory buffers. The buffer needs to be filled wih values (gather) or these
 values need to be written back into the original place (scatter).

 @b Gather: imagine a buffer vector w and a map (the "gather map") that gives to every element in this vector w
 an index into a source vector v where the value of this element should be taken from
 i.e. \f$ w[i] = v[\text{idx}[i]] \f$
 Note that an index into the source vector v can appear several times or not at all.
 This is why the source vector v can have any size and even be smaller than w.

 @b Scatter: imagine a buffer vector w and a map that gives to every element in the buffer w an
 index into a target vector v where this element should go to.
 Note again that an index into v can appear several times or never at all.
 If the index appears more than once, we perform a reduction operation  (we sum up all elements)
 on these indices initializing the sum with 0.
 Note that \f$ v[\text{idx}[i]] = w[i] \f$ is an INCORRECT definition of this, because
 it does not show the reduction.

It is more accurate to represent the gather and scatter operation
by a matrix. The gather matrix \f$ G\f$ is just a
(permutation) matrix of 1's and 0's with exactly one "1" in each line.
In a "coo" formatted sparse matrix format the values array would consist only of "1"s,
the row array is just the index and the column array is the gather map.
\f[ \vec w = G \vec v \f]
where \f$ G \in \mathbb{R}^{N_w \times N_v}\f$  and \f$N_w\f$ and \f$ N_v\f$ can have
any sizes. We have \f$ G_{ij} = \delta_{\text{idx}_ij}\f$
The above definition uniquely identifies the <b> scatter matrix as the transpose of the gather matrix
of the same index map</b>
\f[ S\equiv G^\mathrm{T}\f]
We have \f$ S_{ij} = \delta_{\text{idx}_j i}\f$ and \f$ \vec v = S \vec w\f$.
A simple consistency test is given by \f$ (Gv)\cdot (Gv) = S(Gv)\cdot v\f$.

The scatter matrix can thus have zero, one or more "1"s in each line.

@note There is no "scatter map" or "gather map", there is just an integer index map idx\f$(i)\f$
that maps each index \f$ i \f$ in \f$ w\f$ to an index \f$ \text{idx}(i)\f$ in \f$ v\f$.
Think of it more in terms of a "connector".
In code this is represented by the vector "idx" with elements idx[i] = idx\f$(i)\f$.

We distinguish between

@b bijective: If the gather map idx[i] is bijective, each element of the source vector v maps
to exactly one location in the buffer vector w. In this case the scatter matrix S
is the inverse of G and v and w have the same size.

@b surjective: If the gather map idx[i] is surjective, each element of the source vector v
maps to at least one location in the buffer vector w. This means that the scatter matrix S
can have more than one 1's in each line and w has at least the size of v.

@b general: In general the gather map idx[i] might or might not map an element of
the source vector v. This means that the scatter matrix S can have one or more
empty lines and w may be smaller than v.

@note If v is filled with its indices, i.e. \f$ v[i] = i \f$, then
the gather operation will reproduce the index map in the buffer w or
\f[ \vec{\text{idx}} = G \vec i\f]
In case the index map is bijective the index map can be inverted and we have
\f[ \vec{\text{idx}^{-1}} = S \vec i\f]

This class performs these operations for the case that v and w are distributed across processes.
We always assume that the source vector v is distributed equally among processes, i.e.
each process holds a chunk of v of equal size. On the other hand the local size
of w may vary among processes depending on the gather/scatter map.

 * @tparam LocalContainer a container on a shared memory system (must be default constructible)
 * @ingroup mpi_structures
 */
template< class LocalContainer>
struct aCommunicator
{
    using value_type = get_value_type<LocalContainer>;//!< reveal value type

    using container_type = LocalContainer; //!< reveal local container type

    /**
     * @brief Allocate a buffer object of size
     * <tt> buffer_size() </tt>
     * @return a buffer object on the stack
     * @note if <tt> buffer_size()==0 </tt> the default constructor of \c LocalContainer is called
     */
    LocalContainer allocate_buffer( )const{
        if( do_size() == 0 ) return LocalContainer();
        return do_make_buffer();
    }

    /**
     * @brief \f$ w = G v\f$. Globally (across processes) gather data into a buffer
     *
     * The transpose operation is <tt> global_scatter_reduce() </tt>
     * @param values source vector v; data is collected from this vector
     * @param buffer on output holds the gathered data ( must be of size <tt> buffer_size() </tt>)
     * @note if <tt> buffer_size()==0 </tt>nothing happens
     */
    void global_gather( const value_type* values, LocalContainer& buffer)const
    {
        if( do_size() == 0 ) return;
        do_global_gather( values, buffer);
    }

    /**
     * @brief \f$ w = G v\f$. Globally (across processes) gather data into a buffer (memory allocating version)
     *
     * The transpose operation is <tt> global_scatter_reduce() </tt>
     * @param values source vector v; data is collected from this vector (must have <tt> local_size() </tt> elements)
     * @return object of size <tt> buffer_size() </tt> that holds the gathered data
     * @note if <tt> buffer_size()==0 </tt> the default constructor of \c LocalContainer is called
     */
    LocalContainer global_gather( const value_type* values) const
    {
        LocalContainer tmp = do_make_buffer();
        do_global_gather( values, tmp);
        return tmp;
    }

    /**
     * @brief \f$ v = G^\mathrm{T} w\f$. Globally (across processes) scatter data accross processes and reduce on multiple indices
     *
     * This is the transpose operation of <tt> global_gather() </tt>
     * @param toScatter buffer vector w; (has to be of size given by <tt> buffer_size() </tt>)
     * @param values target vector v; on output contains values from other processes sent back to the origin (must have <tt> local_size() </tt> elements)
     * @note if <tt> buffer_size()==0 </tt> nothing happens
     */
    void global_scatter_reduce( const LocalContainer& toScatter, value_type* values) const{
        if( do_size() == 0 ) return;
        do_global_scatter_reduce(toScatter, values);
    }

    /**
    * @brief The local size of the buffer vector w = local map size
    *
    * Consider that both the source vector v and the buffer w are distributed across processes.
    * In Feltor the vector v is distributed equally among processes and the local size
    * of v is the same for all processes. However, the buffer size might be different for each process.
    * @return buffer size (may be different for each process)
    * @note may return 0
    * @attention it is NOT enough to check for zero buffer size if you want to find out whether a given process
    * needs to send MPI messages or not. See <tt> isCommunicating() </tt> for an explanation
    * @sa local_size() isCommunicating()
    */
    unsigned buffer_size() const{return do_size();}
    /**
    * @brief The local size of the source vector v = local size of the
    * <tt> dg::MPI_Vector </tt>
    *
    * Consider that both the source vector v and the buffer w are distributed across processes.
    * In Feltor the vector v is distributed equally among processes and the local size
    * of v is the same for all processes.
    * @return local size of v (same for all processes)
    * @note Important only for general scatter operations where some elements of v might have to be set to zero
    * @sa buffer_size()
    */
    unsigned local_size() const{return m_source_size;}
    /**
     * @brief True if the gather/scatter operation involves actual MPI communication
     *
     * This is more than just a test for zero message size.
     * This is because even if a process has zero message size indicating that it
     * technically does not need to send any data at all it might still need to participate in an MPI communication (sending an empty message to
     * indicate that a certain point in execution has been reached). Only if NONE of the processes in the process group has anything to send will
     * this function return false.
     * This test can be used to avoid the gather operation alltogether in e.g. the construction of a MPI distributed matrix.
     * @note this check involves MPI communication itself, because a process needs to check if itself or any other process in its
     * group is communicating.
     *
     * @return False, if the global gather can be done without MPI communication (i.e. the indices are all local to each calling process), or if the communicator is \c MPI_COMM_NULL. True else.
     * @sa buffer_size()
     */
    bool isCommunicating() const{
        return do_isCommunicating();
    }
    /**
    * @brief The internal MPI communicator used
    *
    * can be e.g. used to assert that communicators of matrix and vector are the same
    * @return MPI Communicator
    */
    MPI_Comm communicator() const{return do_communicator();}
    ///@brief Generic copy method
    ///@return pointer to allocated object
    virtual aCommunicator* clone() const =0;
    ///@brief vritual destructor
    virtual ~aCommunicator(){}
    protected:
    ///@brief only derived classes can construct
    ///@param local_size the local size of the source vector v
    aCommunicator(unsigned local_size=0):m_source_size(local_size){}
    ///@brief only derived classes can copy
    ///@param src source
    aCommunicator(const aCommunicator& src):m_source_size( src.m_source_size){ }
    ///@brief only derived classes can assign
    ///@param src source
    ///@return *this
    aCommunicator& operator=(const aCommunicator& src){
        m_source_size = src.m_source_size;
        return *this;
    }
    ///@brief Set the local size of the source vector v
    ///@param new_size the new local size for the source vector v
    void set_local_size( unsigned new_size){
        m_source_size = new_size;
    }
    private:
    unsigned m_source_size;
    virtual MPI_Comm do_communicator() const=0;
    virtual unsigned do_size() const=0;
    virtual LocalContainer do_make_buffer( )const=0;
    virtual void do_global_gather( const value_type* values, LocalContainer& gathered)const=0;
    virtual void do_global_scatter_reduce( const LocalContainer& toScatter, value_type* values) const=0;
    virtual bool do_isCommunicating( ) const {
        return true;
    }
};





}//namespace dg
