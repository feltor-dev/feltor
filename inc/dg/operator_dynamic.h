#ifndef _DG_OPERATORS_DYN_
#define _DG_OPERATORS_DYN_

#include <vector>
#include <iterator>
#include <stdexcept>
#ifdef DG_DEBUG
#include <cassert>
#endif

namespace dg{

/**
* @brief Helper class mainly for the assembly of Matrices
*
* @ingroup lowlevel
* In principle it's an enhanced quadratic dynamic matrix
* but it's not meant for performance critical code. 
* @tparam T value type
*/
template< class T>
class Operator
{
  public:
    typedef T value_type; //!< typically double or float 
    /**
    * @brief Construct empty Operator
    */
    Operator(){}
    Operator( const unsigned n): n_(n), data_(n_*n_){}
    /**
    * @brief Initialize elements.
    *
    * @param n matrix is of size n x n
    * @param value Every element is initialized to.
    */
    Operator( const unsigned n, const T& value): n_(n), data_(n_*n_, value) {}
    template< class InputIterator>
    Operator( InputIterator first, InputIterator last): data_(first, last)
    {
        unsigned n = std::distance( first, last);
        n_ = (unsigned)sqrt( (double)n);
#ifdef DG_DEBUG
        assert( n_*n_ == n);
#endif
    }
    Operator( const std::vector<T>& src): data_(src)
    {
        unsigned n = src.size();
        n_ = (unsigned)sqrt( (double)n);
#ifdef DG_DEBUG
        assert( n_*n_ == n);
#endif
    }

    /*! @brief access operator
     *
     * A range check is performed if DG_DEBUG is defined
     * @param i row index
     * @param j column index
     * @return reference to value at that location
     */
    T& operator()(const size_t i, const size_t j){
#ifdef DG_DEBUG
        assert( i<n_ && j < n_);
#endif
        return data_[ i*n_+j];
    }
    /*! @brief const access operator
     *
     * @param i row index
     * @param j column index
     * @return const value at that location
     */
    const T& operator()(const size_t i, const size_t j) const {
#ifdef DG_DEBUG
        assert( i<n_ && j < n_);
#endif
        return data_[ i*n_+j];
    }

    unsigned size() const { return n_;}
    void resize( unsigned m) { data_.resize( m*m);}

    /**
    * @brief Transposition
    *
    * @return  A newly generated Operator containing the transpose.
    */
    Operator transpose() const 
    {
        T temp;
        Operator o(*this);
        for( unsigned i=0; i<n_; i++)
            for( unsigned j=0; j<i; j++)
            {
                temp = o.data_[i*n_+j];
                o.data_[i*n_+j] = o.data_[j*n_+i];
                o.data_[j*n_+i] = temp;
            }
        return o;
    }

    Operator operator-() const
    {
        Operator temp(n_, 0.);
        for( unsigned i=0; i<n_*n_; i++)
            temp.data_[i] = -data_[i];
        return temp;
    } 
    Operator& operator+=( const Operator& op)
    {
#ifdef DG_DEBUG
        assert( op.size() == this->size());
#endif//DG_DEBUG
        for( unsigned i=0; i<n_*n_; i++)
            data_[i] += op.data_[i];
        return *this;
    }
    Operator& operator-=( const Operator& op)
    {
#ifdef DG_DEBUG
        assert( op.size() == this->size());
#endif//DG_DEBUG
        for( unsigned i=0; i<n_*n_; i++)
            data_[i] -= op.data_[i];
        return *this;
    }
    Operator& operator*=( const T& value )
    {
        for( unsigned i=0; i<n_*n_; i++)
            data_[i] *= value;
        return *this;
    }
    friend Operator operator+( const Operator& lhs, const Operator& rhs) 
    {
        Operator temp(lhs); 
        temp+=rhs;
        return temp;
    }
    friend Operator operator-( const Operator& lhs, const Operator& rhs)
    {
        Operator temp(lhs); 
        temp-=rhs;
        return temp;
    }
    friend Operator operator*( const T& value, const Operator& rhs )
    {
        Operator temp(rhs); 
        temp*=value;
        return temp;
    }
    friend Operator operator*( const Operator& lhs, const T& value)
    {
        return  value*lhs;
    }
    friend Operator operator*( const Operator& lhs, const Operator& rhs)
    {
        unsigned n_ = lhs.n_;
#ifdef DG_DEBUG
        assert( lhs.size() == rhs.size());
#endif//DG_DEBUG
        Operator temp(n_, 0.);
        for( unsigned i=0; i< n_; i++)
            for( unsigned j=0; j<n_; j++)
            {
                temp.data_[i*n_+j] = (T)0;
                for( unsigned k=0; k<n_; k++)
                    temp.data_[i*n_+j] += lhs.data_[i*n_+k]*rhs.data_[k*n_+j];
            }
        return temp;
    }
    /*! @brief puts a matrix linewise in output stream
     *
     * @tparam Ostream The stream e.g. std::cout
     * @param os the outstream
     * @param mat the matrix to output
     * @return the outstream
     */
    template< class Ostream>
    friend Ostream& operator<<(Ostream& os, const Operator& mat)
    {
        unsigned n_ = mat.n_;
        for( size_t i=0; i < n_ ; i++)
        {
            for( size_t j = 0;j < n_; j++)
                os << mat(i,j) << " ";
            os << "\n";
        }
        return os;
    }
    /*! @brief Read values into a Matrix from given istream
     *
     * The values are filled linewise into the matrix. Values are seperated by 
     * whitespace characters. (i.e. newline, blank, etc)
     * @tparam Istream The stream e.g. std::cin
     * @param is The istream
     * @param mat The Matrix into which the values are written
     * @return The istream
     */
    template< class Istream>
    friend Istream& operator>> ( Istream& is, Operator<T>& mat){
        unsigned n_ = mat.n_;
        for( size_t i=0; i<n_; i++)
            for( size_t j=0; j<n_; j++)
                is >> mat(i, j);
        return is;
    }

  private:
    unsigned n_;
    std::vector<T> data_;
};

namespace create
{
///@addtogroup lowlevel
///@{
/**
 * @brief Create the unit matrix
 *
 * @param n # of polynomial coefficients
 *
 * @return Operator
 */
Operator<double> delta( unsigned n)
{
    Operator<double> op(n, 0);
    for( unsigned i=0; i<n; i++)
        op( i,i) = 1.;
    return op;
}
/**
 * @brief Create the S-matrix
 *
 * @param n # of polynomial coefficients
 *
 * @return Operator
 */
Operator<double> pipj( unsigned n)
{
    Operator<double> op(n, 0);
    for( unsigned i=0; i<n; i++)
        op( i,i) = 2./(double)(2*i+1);
    return op;
}
/**
 * @brief Create the T-matrix
 *
 * @param n # of polynomial coefficients
 *
 * @return Operator
 */
Operator<double> pipj_inv( unsigned n)
{
    Operator<double> op(n, 0);
    for( unsigned i=0; i<n; i++)
        op( i,i) = (double)(2*i+1)/2.;
    return op;
}
/**
 * @brief Create the D-matrix
 *
 * @param n # of polynomial coefficients
 *
 * @return Operator
 */
Operator<double> pidxpj( unsigned n)
{
    Operator<double> op(n, 0);
    for( unsigned i=0; i<n; i++)
        for( unsigned j=0; j<n; j++)
        {
            if( i < j)
            {
                if( (i+j)%2 != 0)
                    op( i, j) = 2;
            }
        }
    return op;
}
/**
 * @brief Create the R-matrix
 *
 * @param n # of polynomial coefficients
 *
 * @return Operator
 */
Operator<double> rirj( unsigned n)
{
    return Operator<double>( n, 1.);
}
/**
 * @brief Create the RL-matrix
 *
 * @param n # of polynomial coefficients
 *
 * @return Operator
 */
Operator<double> rilj( unsigned n)
{
    Operator<double> op( n, -1.);
    for( unsigned i=0; i<n; i++)
        for( unsigned j=0; j<n; j++)
            if( j%2 == 0)
                op( i,j) = 1.;
    return op;
}
/**
 * @brief Create the LR-matrix
 *
 * @param n # of polynomial coefficients
 *
 * @return Operator
 */
Operator<double> lirj( unsigned n) {return rilj( n).transpose();}
/**
 * @brief Create the L-matrix
 *
 * @param n # of polynomial coefficients
 *
 * @return Operator
 */
Operator<double> lilj( unsigned n) 
{
    Operator<double> op( n, -1.);
    for( unsigned i=0; i<n; i++)
        for( unsigned j=0; j<n; j++)
            if( ((i+j)%2) == 0)
                op( i,j) = 1.;
    return op;
}
///@}
}//namespace create



} //namespace dg

#endif //_DG_OPERATORS_DYN_
