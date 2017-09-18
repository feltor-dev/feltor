#pragma once

namespace dg
{

/**
 * @brief Struct that performs collective scatter and gather operations across processes
 * on distributed vectors using MPI
 *
 * In order to understand the issue you must first really(!) understand what 
 gather and scatter operations are, so grab pen and paper: 

 First we note that gather and scatter are most often used in the context
 of memory buffers. The buffer needs to be filled wih values (gather) or these
 values need to be written back into the original place (scatter).

 Gather: imagine a buffer vector w and a map that gives to every element in this vector w
 an index into a source vector v where the value of this element should be taken from
 i.e. \f$ w[i] = v[\text{idx}[i]] \f$ 
 Note that an index in the source vector can appear several times or not at all. 
 This is why the source vector w can have any size and even be smaller than w. 

 Scatter: imagine a buffer vector w and a map that gives to every element in the buffer w an
 index in a target vector v where this element should go to, 
 i.e. \f$ v[\text{idx}[i]] = w[i] \f$. This is ill-defined.
 Note again that an index in v can appear several times or never at all. 
 Then in our case we perform a reduction operation (we sum up all elements) beginning
 with 0 which remedies the defintion. 

Since it is a vector operation the gather and scatter operation can 
also be represented/defined by a matrix. The gather matrix is just a 
(permutation) matrix of 1's and 0's with exactly one "1" in each line.
In a "coo" formatted sparse matrix format the values array would consist only of "1"s, 
row array is just the index and column array is the gather map.
We uniquely define the corresponding scatter matrix as the transpose of the gather matrix. 
The scatter matrix can have zero, one or more "1"s in each line.
\f[ w = G v \\
    v = S w \f]

The scatter matrix S is the actual inverse of G if and only if the gather map is bijective.
In this case the buffer and the vector can swap their roles. 

Finally note that when v is filled with its indices, i.e. \f$ v[i] = i \f$, then
the gather operation will reproduce the index map in the buffer w \f$ w[i] = \text{idx}[i]\f$ .

 * @tparam LocalContainer a container on a shared memory system
 * @ingroup mpi_structures
 */
template< class LocalContainer>
struct aCommunicator
{

    /**
     * @brief Allocate a LocalContainer object of size size()
     * @return an object on the stack
     */
    LocalContainer make_buffer( )const{
        return do_make_buffer();
    }


    /**
     * @brief Gather data across processes
     * @param values data to gather from
     * @param gather object to hold the gathered data ( must be of size size())
     */
    void global_gather( const LocalContainer& values, LocalContainer& gather)const
    {
        do_global_gather( values, gather);
    }

    /**
     * @brief Gather data across processes (memory allocating version)
     * @param values data to gather from
     * @return object to hold the gathered data
     */
    LocalContainer global_gather( const LocalContainer& values) const
    {
        LocalContainer tmp = do_make_buffer();
        do_global_gather( values, tmp);
        return tmp;
    }

    /**
     * @brief Scatters data accross processes and reduces on double indices
     *
     * @param toScatter buffer vector (has to be of size given by size())
     * @param values contains values from other processes sent back to the origin 
     */
    void global_scatter_reduce( const LocalContainer& toScatter, LocalContainer& values) const{
        do_global_scatter_reduce(toScatter, values);
    }

    /**
    * @brief The size of the local buffer = local map size
    *
    * Consider that both the vector v and the buffer w are distributed across processes.
    * In Feltor the vector v is distributed equally among processes and the local size
    * of v is the same for all processes. However the buffer size might be different for each process. 
    * @note may return 0 to indicate identity between v and w and that no MPI communication is needed 
    * @note we assume that the vector size is always the local size of a dg::MPI_Vector
    * @return buffer size
    */
    unsigned size() const;
    /**
    * @brief The internal MPI communicator used 
    *
    * e.g. used to assert that communicators of matrix and vector are the same
    * @return MPI Communicator
    */
    MPI_Comm communicator() const{return m_comm;}
    virtual aCommunicator* clone() const =0;
    virtual ~aCommunicator(){}
    protected:
    /**
     * @brief default constructor
     * @param comm defaults to MPI_COMM_WORLD
     */
    aCommunicator(MPI_Comm comm = MPI_COMM_WORLD){m_comm = comm;}
    aCommunicator(const aCommunicator& src){ m_comm = src.m_comm; }
    aCommunicator& operator=(const aCommunicator& src){
        m_comm = src.m_comm;
        return *this 
    }
    private:
    MPI_Comm m_comm;
    virtual LocalContainer do_make_buffer( )const=0;
    virtual void do_global_gather( const LocalContainer& values, LocalContainer& gathered)const=0;
    virtual void do_global_scatter_reduce( const LocalContainer& toScatter, LocalContainer& values) const=0;
};

}//namespace dg
