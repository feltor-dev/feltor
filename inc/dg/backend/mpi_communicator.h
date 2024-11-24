#pragma once

#include <mpi.h>
#include <complex.h>
#include <thrust/complex.h>

namespace dg
{
///@cond
template<class value_type>
static inline MPI_Datatype getMPIDataType(){ assert( false && "Type not supported!\n" ); return MPI_DATATYPE_NULL; }
template<> inline MPI_Datatype getMPIDataType<char>(){ return MPI_CHAR;}
template<> inline MPI_Datatype getMPIDataType<signed char>(){ return MPI_SIGNED_CHAR;}
template<> inline MPI_Datatype getMPIDataType<unsigned char>(){ return MPI_UNSIGNED_CHAR;}
template<> inline MPI_Datatype getMPIDataType<wchar_t>(){ return MPI_WCHAR;}
template<> inline MPI_Datatype getMPIDataType<signed short>(){ return MPI_SHORT;}
template<> inline MPI_Datatype getMPIDataType<unsigned short>(){ return MPI_UNSIGNED_SHORT;}
template<> inline MPI_Datatype getMPIDataType<int>(){ return MPI_INT;}
template<> inline MPI_Datatype getMPIDataType<unsigned int>(){ return MPI_UNSIGNED;}
template<> inline MPI_Datatype getMPIDataType<signed long int>(){ return MPI_LONG;}
template<> inline MPI_Datatype getMPIDataType<unsigned long int>(){ return MPI_UNSIGNED_LONG;}
template<> inline MPI_Datatype getMPIDataType<signed long long int>(){ return MPI_LONG_LONG;}
template<> inline MPI_Datatype getMPIDataType<unsigned long long int>(){ return MPI_UNSIGNED_LONG_LONG;}
template<> inline MPI_Datatype getMPIDataType<float>(){ return MPI_FLOAT;}
template<> inline MPI_Datatype getMPIDataType<double>(){ return MPI_DOUBLE;}
template<> inline MPI_Datatype getMPIDataType<long double>(){ return MPI_LONG_DOUBLE;}
//template<> inline MPI_Datatype getMPIDataType<std::int8_t>(){ return MPI_INT8_T;}
//template<> inline MPI_Datatype getMPIDataType<std::int16_t>(){ return MPI_INT16_T;}
//template<> inline MPI_Datatype getMPIDataType<std::int32_t>(){ return MPI_INT32_T;}
//template<> inline MPI_Datatype getMPIDataType<std::int64_t>(){ return MPI_INT64_T;}
//template<> inline MPI_Datatype getMPIDataType<std::uint8_t>(){ return MPI_UINT8_T;}
//template<> inline MPI_Datatype getMPIDataType<std::uint16_t>(){ return MPI_UINT16_T;}
//template<> inline MPI_Datatype getMPIDataType<std::uint32_t>(){ return MPI_UINT32_T;}
//template<> inline MPI_Datatype getMPIDataType<std::uint64_t>(){ return MPI_UINT64_T;}
template<> inline MPI_Datatype getMPIDataType<bool>(){ return MPI_C_BOOL;}
template<> inline MPI_Datatype getMPIDataType<std::complex<float>>(){ return MPI_C_COMPLEX;}
template<> inline MPI_Datatype getMPIDataType<std::complex<double>>(){ return MPI_C_DOUBLE_COMPLEX;}
template<> inline MPI_Datatype getMPIDataType<std::complex<long double>>(){ return MPI_C_LONG_DOUBLE_COMPLEX;}
template<> inline MPI_Datatype getMPIDataType<thrust::complex<float>>(){ return MPI_C_COMPLEX;}
template<> inline MPI_Datatype getMPIDataType<thrust::complex<double>>(){ return MPI_C_DOUBLE_COMPLEX;}
template<> inline MPI_Datatype getMPIDataType<thrust::complex<long double>>(){ return MPI_C_LONG_DOUBLE_COMPLEX;}
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

 Imagine a buffer vector w and an index map \f$ \text{g}[i]\f$
 that gives to every index \f$ i\f$ in this vector w
 an index \f$ \text{g}[i]\f$ into a source vector v.

We can now define:
 @b Gather values from v and put them into w according to
  \f$ w[i] = v[\text{g}[i]] \f$

 Loosely we think of @b Scatter as the reverse operation, i.e. take the values
 in w and write them back into v. However, simply writing
 \f$ v[\text{g}[j]] = w[j] \f$ is a very **bad** definition.
 What should happen if \f$ g[j] = g[k]\f$
 for some j and k? What if some indices \f$ v_i\f$ are not mapped at all?

It is more accurate to represent the gather and scatter operation
by a matrix.

@b Gather matrix: A matrix \f$ G\f$ of size \f$ m \times N\f$ is a gather
 matrix if it consists of only 1's and 0's and has exactly one "1" in each row.
 \f$ m\f$ is the buffer size, \f$ N \f$ is the vector size and \f$ N\f$ may be smaller,
 same or larger than \f$m\f$.
 If \f$ \text{g}[i]\f$ is the index map then \f[ G_{ij} := \delta_{\text{g}[i] j}\f]
 We have \f$ w = G v\f$

@b Scatter matrix: A matrix \f$ S \f$ is a
 scatter matrix if its transpose is a gather matrix.

 This means that \f$ S\f$ has size \f$ N \times m \f$
 consists of only 1's and 0's and has exactly one "1" in each column.
 If \f$ \text{g}[j]\f$ is the index map then \f[ S_{ij} := \delta_{i \text{g}[j]}\f]
 We have \f$ v = S w\f$

All of the following statements are true

- The transpose of a gather matrix is a scatter matrix \f$ S  = G^\mathrm{T}\f$.
    The associated index map of \f$ S\f$ is identical to the index map of \f$ G\f$.
- The transpose of a scatter matrix is a gather matrix \f$ G  = S^\mathrm{T}\f$.
    The associated index map of \f$ G\f$ is identical to the index map of \f$ S\f$.
- From a given index map we can construct two matrices (\f$ G \f$ and \f$ S\f$)
- A simple consistency test is given by \f$ (Gv)\cdot (Gv) = S(Gv)\cdot v\f$.
- A scatter matrix can have zero, one or more "1"s in each row.
- A gather matrix can have zero, one or more "1"s in each column.
- If v is filled with its indices i.e. \f$ v_i = i\f$ then \f$ m = Gv\f$ i.e. the gather operation
    reproduces the index map
- If the entries of w are \f$ w_j = j\f$ then \f$ m \neq Sw\f$ does **not**
    reproduce the index map
- In a "coo" formatted sparse matrix format the gather matrix is obtained:
    \f$ m \f$ rows, \f$ N\f$ columns and \f$ m\f$ non-zeroes,
    the values array would consist only of "1"s,
    the row array is just the index \f$i\f$
    and the column array is the map \f$ g[i]\f$.
- In a "coo" formatted sparse matrix format the scatter matrix is obtained:
    \f$ N \f$ rows, \f$ m\f$ columns and \f$ m\f$ non-zeroes,
    the values array would consist only of "1"s,
    the row array is the map \f$g[j]\f$
    and the column array is the index \f$ j\f$.
- \f$ G' = G_1 G_2 \f$, i.e. the multiplication of two gather matrices is again a gather
- \f$ S' = S_1 S_2 \f$, i.e. the multiplication of two scatter matrices is again a scatter

Of the scatter and gather matrices permutations are especially interesting
A matrix is a **permutation** if and only if it is both a scatter and a gather matrix.
    In such a case it is square \f$ m \times m\f$ and \f[ P^{-1} = P^T\f].
    The buffer \f$ w\f$ and vector \f$ v\f$ have the same size \f$m\f$.


The following statements are all true
- The index map of a permutation is bijective i.e. invertible i.e. each element
    of the source vector v maps to exactly one location in the buffer vector w.
- The scatter matrix \f$ S = G^T \equiv G'\neq G\f$ is a gather matrix (in
    general unequal \f$ G\f$) with the associate index map \f$ m^{-1}\f$.
    Since the index map is recovered by applying the gather operation to the vector
    containing its index as values, we have
    \f[ m^{-1} = G' \vec i = S \vec i\f]
- \f$ S' = P_1 S P_2 \f$, i.e. multiplication of a scatter matrix by a permutation is again a scatter matrix
- \f$ G' = P_1 G P_2 \f$, i.e. multiplication of a gather matrix by a permutation is again a gather matrix
- A Permutation is **symmetric** if and only if it has identical scatter and gather maps
- Symmetric permutations can be implemented "in-place" i.e. the source and buffer can be identical



This class performs these operations for the case that v and w are distributed
across processes.  Accordingly, the index map \f$ g\f$  is also distributed
across processes (in the same way w is).  The elements of \f$ g\f$ are
**global** indices into v that have to be transformed to pairs (local index
        into v, rank in communicator) according to a user provided function. Or
the user can directly provide the index map as vector of mentioned pairs.

Imagine now that we want to perform a globally distributed gather operation.
Then, the following steps are performed
 - From the given index array a MPI communication matrix (of size
 \f$ s \times s\f$ where \f$ s\f$ is the number of processes in the MPI
 communicator) can be inferred. Each row shows how many elements a
 given rank ( the row index) receives from each of the other ranks in the
 communicator (the column indices). Each column of this map describe the
 sending pattern, i.e. how many elements a given rank (the column index) has to
 send each of the other ranks in the communicator.  If the MPI communication
 matrix is symmetric we can  perform MPI communications **in-place**
 - The information from the communication matrix can be used to allocate
 appropriately sized MPI send and receive buffers. Furthermore, it is possible
 to define a **permutation** across different processes. It is important to
 note that the index map associated to that permutation is immplementation
 defined i.e.  the implementation analyses the communication matrix and chooses
 an optimal call of MPI Sends and Recvs. The implementation then provides two
 index maps. The first one must be used to gather values from v into the
 MPI send buffer and the second one can be used to gather values from the
 receive buffer into the target buffer. Notice that these two operations are
 **local** and require no MPI communication.

 In total we thus describe the global gather as
 \f[ w = G v = G_1 P_{G,MPI} G_2 v\f]

 The global scatter operation is then simply
 \f[ v = S w = G_2^T P^T_{G,MPI} G^T_1 w = S_2 P_{S,MPI} S_1 w \f]
 (The scatter operation is constructed the same way as the gather operation, it is just the execution that is different)

 @note If the scatter/gather operations are part of a matrix-vector multiplication
 then \f$ G_1\f$ or \f$ S_1\f$ can be absorbed into the matrix

 \f[ M v = R G v  = R G_1 P_{G,MPI} G_2 v = R' P_{G,MPI} G_2 v\f]. If R was a
 coo matrix the simple way to obtain R' is replacing the column indices with
 the map \f$ g_1\f$.
 @note To give the involved vectors unique names we call v the "vector", \f$ s = G_2 v\f$ is the "store" and, \f$ b = P s\f$ is the "buffer".

 For \f[ M v = S C v = S_2 P_{S,MPI} S_1 C v = S_2 P_{S,MPI} C' v\f]. Again, if
 C was a coo matrix the simple way to obtain C' is replacing the row indices
 with the map \f$ g_1\f$.

 Simplifications can be achieved if \f$ G_2 = S_2 = I\f$ is the identity
 or if \f$ P_{G,MPI} = P_{S,MPI} = P_{MPI}\f$ is symmetric, which means that
 in-place communication can be used.

 @note Locally, a gather operation is trivially parallel but a scatter operation
 is not in general (because of the possible reduction operation).
 @sa LocalGatherMatrix


 * @tparam value_type The type of data that is being sent i.e. the value type of the vector \f$ v\f$
 * @ingroup mpi_structures
 * @code
 int i = myrank;
 double values[8] = {i,i,i,i, 9,9,9,9};
 thrust::host_vector<double> hvalues( values, values+8);
 int pids[8] =      {0,1,2,3, 0,1,2,3};
 thrust::host_vector<int> hpids( pids, pids+8);
 BijectiveComm coll( hpids, MPI_COMM_WORLD);
 thrust::host_vector<double> hrecv = coll.global_gather( hvalues); //for e.g. process 0 hrecv is now {0,9,1,9,2,9,3,9}
 thrust::host_vector<double> hrecv2( hvalues.size());
 coll.global_scatter_reduce( hrecv, hrecv2); //hrecv2 now equals hvalues independent of process rank
 @endcode
 */
template< template<class > class Device, class value_type>
struct MPIGatherScatter
{
    using container_type = Device<value_type>; //!< reveal local container type

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
     * @note If <tt> !isCommunicating() </tt> then this call will not involve
     * MPI communication but will still gather values according to the given
     * index map
     */
    void global_gather( const value_type* values, LocalContainer& buffer)const
    {
        do_global_gather( values, buffer);
    }

    /**
     * @brief \f$ w = G v\f$. Globally (across processes) gather data into a buffer (memory allocating version)
     *
     * The transpose operation is <tt> global_scatter_reduce() </tt>
     * @param values source vector v; data is collected from this vector (must
     * have <tt> local_size() </tt> elements)
     * @return object of size <tt> buffer_size() </tt> that holds the gathered data
     * @note If <tt> !isCommunicating() </tt> then this call will not involve
     * MPI communication but will still gather values according to the given
     * index map
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
     * @param values target vector v; on output contains values from other
     * processes sent back to the origin (must have <tt> local_size() </tt>
     * elements)
     * @note If <tt> !isCommunicating() </tt> then this call will not involve
     * MPI communication but will still scatter and reduce values according to
     * the given index map
     */
    void global_scatter_reduce( const LocalContainer& toScatter, value_type* values) const{
        do_global_scatter_reduce(toScatter, values);
    }

    /**
    * @brief The local size of the buffer vector w = local map size
    *
    * Consider that both the source vector v and the buffer w are distributed across processes.
    * In Feltor the vector v is (usually) distributed equally among processes and the local size
    * of v is the same for all processes. However, the buffer size might be different for each process.
    * @return buffer size (may be different for each process)
    * @note may return 0
    * @attention it is NOT a good idea to check for zero buffer size if you
    * want to find out whether a given process needs to send MPI messages or
    * not. The first reason is that even if no communication is happening the
    * buffer_size is not zero as there may still be local gather/scatter
    * operations. The right way to do it is to call <tt> isCommunicating() </tt>
    * @sa local_size() isCommunicating()
    */
    unsigned buffer_size() const{return do_size();}
    /**
    * @brief The local size of the source vector v = local size of the
    * <tt> dg::MPI_Vector </tt>
    *
    * Consider that both the source vector v and the buffer w are distributed across processes.
    * In Feltor the vector v is (usually) distributed equally among processes and the local size
    * of v is the same for all processes.
    * @return local size of v (same for all processes)
    * @note Important only for general scatter operations where some elements of v might have to be set to zero
    * @sa buffer_size()
    */
    unsigned local_size() const{return m_source_size;}
    /**
     * @brief True if the gather/scatter operation involves actual MPI communication
     *
     * This is more than just a test for zero message size.  This is because
     * even if a process has zero message size indicating that it technically
     * does not need to send any data at all it might still need to participate
     * in an MPI communication (sending an empty message to indicate that a
     * certain point in execution has been reached). Only if NONE of the
     * processes in the process group has anything to send will this function
     * return false.  This test can be used to avoid the gather operation
     * alltogether in e.g. the construction of a MPI distributed matrix.
     * @note this check may involve MPI communication itself, because a process
     * needs to check if itself or any other process in its group is
     * communicating.
     *
     * @return False, if the global gather can be done without MPI
     * communication (i.e. the indices are all local to each calling process),
     * or if the communicator is \c MPI_COMM_NULL. True else.
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
