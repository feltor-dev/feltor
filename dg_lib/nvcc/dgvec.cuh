#ifndef _DG_VECTOR_
#define _DG_VECTOR_

namespace dg
{

//TODO some safety measurments
/**
 * @brief DG View on a Vector
 *
 * @ingroup containers
 * Data is not owned by this class.
 * @tparam T The value type 
 * @tparam n The number of polynomial coefficients per cell
 * @tparam container The underlying container data type
 */
template< typename T, size_t n, class container = thrust::host_vector<T> >
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
    typedef T value_type;
    /**
     * @brief Initialize a reference to a container
     *
     * @param v This reference is stored by the object.
     */
    ArrVec1d_View( container& v ):hv(v){ }
    /**
     * @brief Access to a value
     *
     * @param i Line 
     * @param k Coefficient
     *
     * @return Reference to value.
     */
    T& operator()( unsigned i, unsigned k) {return hv[ i*n+k];}
    /**
     * @brief Const Access to a vlue
     *
     * @param i Line
     * @param k Coefficient
     *
     * @return Reference to value
     */
    const T& operator()( unsigned i, unsigned k) const
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
    container& hv;
};

//an Array is a View but owns the data it views
/**
 * @brief DG View on a Vector it owns
 *
 * @ingroup containers
 * Data is owned by this class.
 * @tparam T The value type 
 * @tparam n The number of polynomial coefficients per cell
 * @tparam container The underlying container data type
 */
template< typename T, size_t n, class container = thrust::host_vector<T> >
class ArrVec1d : public ArrVec1d_View<T, n, container>
{
    public:
        /**
         * @brief The View type, i.e. parent class
         */
    typedef ArrVec1d_View<T, n, container> View;
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
    ArrVec1d( const container& c): View(hv), hv(c) {}

    /**
      * @brief Construct a container by size and value
      * 
      * @param size Number of lines ( actual container is n*size long)
      * @param value Elements are initialized to this value
      */
    ArrVec1d( unsigned size, double value=0) : View(hv), hv( n*size, value){}
    private:
    container hv;
};

}//namespace dg

#endif //_DG_VECTOR_
