#ifndef _DG_MATRIX_
#define _DG_MATRIX_

namespace dg
{

//TODO some safety measurments
/**
 * @brief DG View on a Vector
 *
 * @ingroup utilities
 * Data is not owned by this class, nor does the class check for consistent 
 * container sizes, rows and columns.
 * @tparam T The value type 
 * @tparam n The number of polynomial coefficients per cell
 * @tparam container The underlying container data type.
 * 
 * Valid expression:
 * \li c.size() must return the number of elements in the container c
 */
template< class container = thrust::host_vector<double> >
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
    typedef typename container::value_type value_type;
    /**
     * @brief Initialize a reference to a container
     *
     * The referenced object is interpreted as a DG Matrix
     * @param v This reference is stored by the object.
     * @param columns Number of lines of the matrix
     */
    ArrVec2d_View( container& v, unsigned n, unsigned columns ) : hv(v), n_(n), cols_( columns) { }
    /**
     * @brief Access a value
     *
     * @param i Line 
     * @param k Coefficient
     *
     * @return Reference to value.
     */
    value_type& operator()( unsigned i, unsigned j, unsigned k, unsigned l)
    { 
        //assert( k, l <n ) ??
        return hv[ i*n_*n_*cols_ + j*n_*n_ + k*n_ + l];
    }
    /**
     * @brief Const Access a value
     *
     * @param i Line
     * @param k Coefficient
     *
     * @return Reference to value
     */
    const value_type& operator()( unsigned i, unsigned j, unsigned k, unsigned l) const
    { 
        return hv[i*n_*n_*cols_ + j*n_*n_ + k*n_ + l];
    }
    unsigned& cols() {return cols_;}
    const unsigned& n() const {return n_;}
    const unsigned& cols() const {return cols_;}

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
     * @brief Stream the underlying object in 2D view
     *
     * @tparam Ostream e.g. std::cout 
     * @param os Object of type Ostream
     * @param v Oject to stream
     *
     * @return Reference to Ostream
     * @note For best readability use <<setprecision(2) and <<fixed for cout
     */
    template< class Ostream>
    friend Ostream& operator<<( Ostream& os, const ArrVec2d_View& v)
    {
        unsigned N = v.hv.size()/v.n_/v.n_;
        unsigned n_ = v.n_;
        unsigned rows = N/v.cols_;
        for( unsigned i=0; i<rows; i++)
        {
            for( unsigned k=0; k<n_; k++)
            {
                for( unsigned j=0; j<v.cols_; j++)
                {
                    for( unsigned l=0; l<n_; l++)
                        os << v.hv[i*n_*n_*v.cols_ + j*n_*n_+ k*n_ + l] << " ";
                    os << "\t";
                }
                os << "\n";
            }
            os << "\n";
        }
        return os;
    }
  private:
    container& hv;
    unsigned n_, cols_;
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
class ArrVec2d : public ArrVec2d_View< container>
{
  public:
    /**
     * @brief The View type, i.e. parent class
     */
    typedef ArrVec2d_View< container> View;
    /**
     * @brief Construct an empty vector
     *
     */
    ArrVec2d() : View(hv, 0, 0){}
    /**
     * @brief Copy a containter object 
     *
     * Copy given object and interpret as a DG Matrix.
     * @param c A container must be copyconstructible from c. 
     */
    ArrVec2d( const container& c, unsigned n, unsigned cols): View(hv, n, cols), hv(c) {
    }

    /**
      * @brief Construct a container by size and value
      * 
      * @param rows Number of lines 
      * @param cols Number of columns ( actual container is n*n*rows*cols long)
      * @param value Elements are initialized to this value
      */
    ArrVec2d( unsigned n, unsigned rows, unsigned cols, double value=0) : View(hv, cols), hv( n*n*rows*cols, value){}
    ArrVec2d( const ArrVec2d& src): View( hv, src.n(), src.cols()), hv( src.hv){}

    template< class OtherContainer >
    ArrVec2d( const ArrVec2d< OtherContainer >& src): View( hv, src.n(), src.cols() ), hv( src.data()) {}

    ArrVec2d& operator=( const ArrVec2d& src)
    {
        this->cols() = src.cols();
        hv = src.hv;
        return *this;
    }
    template< class OtherContainer >
    ArrVec2d& operator=(const ArrVec2d< OtherContainer>& src) 
    {
        this->cols() = src.cols();
        hv = src.data(); //this might trigger warnings from thrust 
        return *this;
    }
  private:
    container hv;
};

}//namespace dg

#endif //_DG_MATRIX_
