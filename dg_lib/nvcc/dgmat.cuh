#ifndef _DG_MATRIX_
#define _DG_MATRIX_

namespace dg
{

//TODO some safety measurments
/**
 * @brief DG View on a Vector
 *
 * @ingroup containers
 * Data is not owned by this class, nor does the class check for consistent 
 * container sizes, rows and columns.
 * @tparam T The value type 
 * @tparam n The number of polynomial coefficients per cell
 * @tparam container The underlying container data type.
 * 
 * Valid expression:
 * \li c.size() must return the number of elements in the container c
 */
template< typename T, size_t n, class container = thrust::host_vector<T> >
class ArrVec2d_View
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
     * The referenced object is interpreted as a DG Matrix
     * @param v This reference is stored by the object.
     * @param columns Number of lines of the matrix
     */
    ArrVec2d_View( container& v, unsigned columns ) : hv(v), cols( columns) { }
    /**
     * @brief Access a value
     *
     * @param i Line 
     * @param k Coefficient
     *
     * @return Reference to value.
     */
    T& operator()( unsigned i, unsigned j, unsigned k, unsigned l)
    { 
        //assert( k, l <n ) ??
        return hv[ i*n*n*cols + j*n*n + k*n + l];
    }
    /**
     * @brief Const Access a value
     *
     * @param i Line
     * @param k Coefficient
     *
     * @return Reference to value
     */
    const T& operator()( unsigned i, unsigned j, unsigned k, unsigned l) const
    { 
        return hv[i*n*n*cols + j*n*n + k*n + l];
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
    unsigned cols;
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
class ArrVec2d : public ArrVec2d_View<T, n, container>
{
  public:
        /**
         * @brief The View type, i.e. parent class
         */
    typedef ArrVec2d_View<T, n, container> View;
    /**
     * @brief Construct an empty vector
     *
     */
    ArrVec2d() : View(hv, 0){}
    /**
     * @brief Copy a containter object 
     *
     * Copy given object and interpret as a DG Matrix.
     * @param c A container must be copyconstructible from c. 
     */
    ArrVec2d( const container& c, unsigned cols): View(hv, cols), hv(c) {
    }

    /**
      * @brief Construct a container by size and value
      * 
      * @param rows Number of lines 
      * @param cols Number of columns ( actual container is n*n*rows*cols long)
      * @param value Elements are initialized to this value
      */
    ArrVec2d( unsigned rows, unsigned cols, double value=0) : View(hv, cols), hv( n*n*rows*cols, value){}
  private:
    container hv;
};

}//namespace dg

#endif //_DG_MATRIX_
