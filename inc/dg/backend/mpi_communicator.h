#pragma once

#include <mpi.h>

namespace dg
{
///@cond
template<class value_type>
MPI_Datatype getMPIDataType(){ assert( false && "Type not supported!\n" ); return; }
template<>
MPI_Datatype getMPIDataType<double>(){ return MPI_DOUBLE;}
template<>
MPI_Datatype getMPIDataType<float>(){ return MPI_FLOAT;}
template<>
MPI_Datatype getMPIDataType<int>(){ return MPI_INT;} 
template<>
MPI_Datatype getMPIDataType<unsigned>(){ return MPI_UNSIGNED;} 
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

 @b Gather: imagine a buffer vector w and a map that gives to every element in this vector w
 an index into a source vector v where the value of this element should be taken from
 i.e. \f$ w[i] = v[\text{idx}[i]] \f$ 
 Note that an index into the source vector v can appear several times or not at all. 
 This is why the source vector v can have any size and even be smaller than w. 

 @b Scatter: imagine a buffer vector w and a map that gives to every element in the buffer w an
 index into a target vector v where this element should go to. 
 Note again that an index into v can appear several times or never at all. 
 If the index appears more than once, we perform a reduction operation  (we sum up all elements) 
 on these indices initializing the sum with 0.
 Note that \f$ v[\text{idx}[i]] = w[i] \f$ is NOT the correct definition of this, because
 it does not show the reduction.

It is more accurate to represent the gather and scatter operation 
by a matrix. The gather matrix is just a 
(permutation) matrix of 1's and 0's with exactly one "1" in each line.
In a "coo" formatted sparse matrix format the values array would consist only of "1"s, 
row array is just the index and column array is the gather map.
We uniquely define the corresponding <b> scatter matrix </b> as the <b> transpose of the gather matrix</b>. 
The scatter matrix can have zero, one or more "1"s in each line.
\f[ w_1 = G v_1 \\
    v_2 = S w_2 = G^\mathrm{T} w_2 \f]
where \f$ v_1\f$ and \f$ v_2\f$ are data vectors and \f$ w_1\f$ and \f$ w_2\f$ are buffer vectors.

This class performs these operations for the case that v and w are distributed across processes.
We always assume that the source vector v is distributed equally among processes, i.e. 
each process holds a chunk of v of equal size. On the other hand the local size 
of w may vary among processes depending on the gather/scatter map. 

@note The scatter matrix S is the actual inverse of G if and only if the gather map is bijective.
In this case the buffer and the vector can swap their roles. 

@note Finally note that when v is filled with its indices, i.e. \f$ v[i] = i \f$, then
the gather operation will reproduce the index map in the buffer w \f$ w[i] = \text{idx}[i]\f$ .

 * @tparam LocalContainer a container on a shared memory system (must be default constructible)
 * @ingroup mpi_structures
 */
template< class LocalContainer>
struct aCommunicator
{
    typedef LocalContainer container_type; //!< reveal local container type

    /**
     * @brief Allocate a buffer object of size size()
     * @return a buffer object on the stack
     * @note if size()==0 the default constructor of LocalContainer is called
     */
    LocalContainer allocate_buffer( )const{
        if( do_size() == 0 ) return LocalContainer();
        return do_make_buffer();
    }

    /**
     * @brief Globally (across processes) gather data into a buffer 
     *
     * This is the transpose operation of global_scatter_reduce()
     * @param values data; other processes collect data from this vector
     * @param buffer on output holds the gathered data ( must be of size size())
     * @note if size()==0 nothing happens
     */
    void global_gather( const LocalContainer& values, LocalContainer& buffer)const
    {
        if( do_size() == 0 ) return;
        do_global_gather( values, buffer);
    }

    /**
     * @brief Globally (across processes) gather data into a buffer (memory allocating version)
     *
     * This is the transpose operation of global_scatter_reduce()
     * @param values data; other processes collect data from this vector
     * @return object that holds the gathered data
     * @note if size()==0 the default constructor of LocalContainer is called
     */
    LocalContainer global_gather( const LocalContainer& values) const
    {
        LocalContainer tmp = do_make_buffer();
        do_global_gather( values, tmp);
        return tmp;
    }

    /**
     * @brief Globally (across processes) scatter data accross processes and reduce on multiple indices
     *
     * This is the transpose operation of global_gather()
     * @param toScatter buffer vector; (has to be of size given by size())
     * @param values on output contains values from other processes sent back to the origin 
     * @note if size()==0 nothing happens
     */
    void global_scatter_reduce( const LocalContainer& toScatter, LocalContainer& values) const{
        if( do_size() == 0 ) return;
        do_global_scatter_reduce(toScatter, values);
    }

    /**
    * @brief The size of the local buffer vector w = local map size
    *
    * Consider that both the source vector v and the buffer w are distributed across processes.
    * In Feltor the vector v is distributed equally among processes and the local size
    * of v is the same for all processes. However the buffer size might be different for each process. 
    * @return buffer size
    * @note may return 0 to indicate that no MPI communication is needed 
    * @note we assume that, contrary to size(), the vector size is always the local size of a dg::MPI_Vector
    */
    unsigned size() const{return do_size();}
    /**
    * @brief The internal MPI communicator used 
    *
    * e.g. used to assert that communicators of matrix and vector are the same
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
    aCommunicator(){}
    ///@brief only derived classes can copy
    ///@param src source 
    aCommunicator(const aCommunicator& src){ }
    ///@brief only derived classes can assign
    ///@param src source 
    ///@return *this
    aCommunicator& operator=(const aCommunicator& src){ return *this; }
    private:
    virtual MPI_Comm do_communicator() const=0;
    virtual unsigned do_size() const=0;
    virtual LocalContainer do_make_buffer( )const=0;
    virtual void do_global_gather( const LocalContainer& values, LocalContainer& gathered)const=0;
    virtual void do_global_scatter_reduce( const LocalContainer& toScatter, LocalContainer& values) const=0;
};





}//namespace dg
