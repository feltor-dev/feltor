#ifndef _DG_VECTOR_
#define _DG_VECTOR_

#include "vector_categories.h"
#include "thrust/host_vector.h"

namespace dg
{

//TODO some safety measurments
/**
 * @brief DG View on a Vector
 *
 * @ingroup utilities
 * Data is not owned by this class.
 * @tparam T The value type 
 * @tparam n The number of polynomial coefficients per cell
 * @tparam container The underlying container data type
 */
template< class container = thrust::host_vector<double> >
class ArrVec1d_View
{
  public:
        /**
         * @brief Data type of the underlying container
         */
    typedef container Vector;
    /**
     * @brief Data type of the elements in the container
     */
    typedef typename container::value_type value_type;

    typedef ThrustVectorTag vector_category;
    /**
     * @brief Initialize a reference to a container
     *
     * @param v This reference is stored by the object.
     */
    ArrVec1d_View(unsigned n, container& v ):n(n),hv(v){ }
    /**
     * @brief Access to a value
     *
     * @param i Line 
     * @param k Coefficient
     *
     * @return Reference to value.
     */
    value_type& operator()( unsigned i, unsigned k) {return hv[ i*n+k];}
    /**
     * @brief Const Access to a vlue
     *
     * @param i Line
     * @param k Coefficient
     *
     * @return Reference to value
     */
    const value_type& operator()( unsigned i, unsigned k) const
    { 
        return hv[i*n+k];
    }
    /**
     * @brief Access the underlying container object
     *
     * @return The stored reference
     */
    container& data(){ return hv;}
    /**
     * @brief Const Access the underlying container object
     *
     * @return The stored reference
     */
    const container& data() const {return hv;}
    const unsigned& n() const {return n;}

    /**
     * @brief Stream the underlying object
     *
     * @tparam Ostream e.g. std::cout 
     * @param os Object of type Ostream
     * @param v Oject to stream
     *
     * @return Reference to Ostream
     */
    template< class Ostream>
    friend Ostream& operator<<( Ostream& os, const ArrVec1d_View& v)
    {
        unsigned N = v.hv.size()/n;
        for( unsigned i=0; i<N; i++)
        {
            for( unsigned j=0; j<n; j++)
                os << v(i,j) << " ";
            os << "\n";
        }
        return os;
    }
  private:
    unsigned n;
    container& hv;
};

//an Array is a View but owns the data it views
/**
 * @brief DG View on a Vector it owns
 *
 * @ingroup utilities
 * Data is owned by this class.
 * @tparam T The value type 
 * @tparam n The number of polynomial coefficients per cell
 * @tparam container The underlying container data type
 */
template< class container = thrust::host_vector<double> >
class ArrVec1d : public ArrVec1d_View<container>
{
    public:
        /**
         * @brief The View type, i.e. parent class
         */
    typedef ArrVec1d_View<container> View;
    /**
     * @brief Construct an empty vector
     *
     */
    ArrVec1d() : View(hv){}
    /**
     * @brief Copy a containter object 
     *
     * @param c A container must be copyconstructible from c. 
     */
    ArrVec1d( unsigned n, const container& c): View(n, hv), hv(c) {}

    /**
      * @brief Construct a container by size and value
      * 
      * @param size Number of lines ( actual container is n*size long)
      * @param value Elements are initialized to this value
      */
    ArrVec1d( unsigned size, double value=0) : View(hv), hv( n*size, value){}

    //we need explicit copy constructors because of the reference to hv
    ArrVec1d( const ArrVec1d& src): View( hv), hv( src.hv){}

    template< class OtherContainer >
    ArrVec1d( const ArrVec1d< OtherContainer >& src): View( hv), hv( src.data()) {}

    ArrVec1d& operator=( const ArrVec1d& src)
    {
        hv = src.hv;
        return *this;
    }
    template< class OtherContainer >
    ArrVec1d& operator=(const ArrVec1d< OtherContainer>& src) 
    {
        hv = src.data(); //this might trigger warnings from thrust 
        return *this;
    }


    private:
    container hv;
};

}//namespace dg


#endif //_DG_VECTOR_
