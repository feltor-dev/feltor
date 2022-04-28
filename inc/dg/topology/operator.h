#ifndef _DG_OPERATORS_DYN_
#define _DG_OPERATORS_DYN_

#include <vector>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <cassert>
#include "dlt.h"
#include "../blas1.h" // reproducible DOT products

namespace dg{

/**
* @brief A square nxn matrix
*
* @ingroup densematrix
* An enhanced square dynamic matrix
* for which arithmetic operators are overloaded.
* It is not meant for performance critical code but is very convenient
* for example for the assembly of matrices.
* @sa \c dg::create::inverse, \c dg::create::lu_pivot, \c dg::create::lu_solve
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
    /**
     * @brief allocate storage for nxn matrix
     *
     * @param n size
     */
    explicit Operator( const unsigned n): n_(n), data_(n_*n_){}
    /**
    * @brief Initialize elements.
    *
    * @param n matrix is of size n x n
    * @param value Every element is initialized to.
    */
    Operator( const unsigned n, const T& value): n_(n), data_(n_*n_, value) {}
    /**
     * @brief Construct from iterators
     *
     * @tparam InputIterator
     * @param first
     * @param last
     */
    template< class InputIterator>
    Operator( InputIterator first, InputIterator last, std::enable_if_t<!std::is_integral<InputIterator>::value>* = 0): data_(first, last)
    {
        unsigned n = std::distance( first, last);
        n_ = (unsigned)sqrt( (value_type)n);
        if( n_*n_!=n) throw Error( Message(_ping_)<<"Too few elements "<<n<<" need "<<n_*n_<<"\n");
    }
    /**
     * @brief Copy from existing data
     *
     * @param src size must be a square number
     */
    Operator( const std::vector<T>& src): data_(src)
    {
        unsigned n = src.size();
        n_ = (unsigned)sqrt( (value_type)n);
        if( n_*n_!=n) throw Error( Message(_ping_)<<"Wrong number of elements "<<n<<" need "<<n_*n_<<"\n");
    }

    /**
     * @brief Assign zero to all elements
     */
    void zero() {
        for( unsigned i=0; i<n_*n_; i++)
            data_[i] = 0;
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
        if(!(i<n_&&j<n_)) throw Error( Message(_ping_) << "You tried to access out of range "<<i<<" "<<j<<" size is "<<n_<<"\n");
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
        if(!(i<n_&&j<n_)) throw Error( Message(_ping_) << "You tried to access out of range "<<i<<" "<<j<<" size is "<<n_<<"\n");
#endif
        return data_[ i*n_+j];
    }

    /**
     * @brief Size n of the Operator
     *
     * @return n
     */
    unsigned size() const { return n_;}
    /**
     * @brief Resize
     *
     * @param m new size
     * @param val if m is greater than the current size new elements are initialized with val
     */
    void resize( unsigned m, T val = T()) {
        n_ = m;
        data_.resize( m*m, val);
    }

    /**
     * @brief access underlying data
     *
     * @return
     */
    const std::vector<value_type>& data() const {return data_;}

    /**
     * @brief Swap two lines in the square matrix
     *
     * @param i first line
     * @param k second line
     */
    void swap_lines( const size_t i, const size_t k)
    {
        if(!( i< n_ && k<n_)) throw Error( Message(_ping_) << "Out of range "<<i<<" "<<k<<" range is "<<n_<<"\n");
        for( size_t j = 0; j<n_; j++)
        {
            std::swap( data_[i*n_+j], data_[k*n_+j]);
        }
    }

    /**
    * @brief Transposition
    *
    * @return  A newly generated Operator containing the transpose.
    */
    Operator transpose() const
    {
        Operator o(*this);
        for( unsigned i=0; i<n_; i++)
            for( unsigned j=0; j<i; j++)
            {
                std::swap( o.data_[i*n_+j], o.data_[j*n_+i]);
            }
        return o;
    }

    /*! @brief two Matrices are considered equal if elements are equal
     *
     * @param rhs Matrix to be compared to this
     * @return true if rhs does not equal this
     */
    bool operator!=( const Operator& rhs) const{
        for( size_t i = 0; i < n_*n_; i++)
            if( data_[i] != rhs.data_[i])
                return true;
        return false;
    }

    /*! @brief two Matrices are considered equal if elements are equal
     *
     * @param rhs Matrix to be compared to this
     * @return true if rhs equals this
     */
    bool operator==( const Operator& rhs) const {return !((*this != rhs));}

    /**
     * @brief subtract
     *
     * @return
     */
    Operator operator-() const
    {
        Operator temp(n_, 0.);
        for( unsigned i=0; i<n_*n_; i++)
            temp.data_[i] = -data_[i];
        return temp;
    }
    /**
     * @brief add
     *
     * @param op
     *
     * @return
     */
    Operator& operator+=( const Operator& op)
    {
#ifdef DG_DEBUG
        assert( op.size() == this->size());
#endif//DG_DEBUG
        for( unsigned i=0; i<n_*n_; i++)
            data_[i] += op.data_[i];
        return *this;
    }
    /**
     * @brief subtract
     *
     * @param op
     *
     * @return
     */
    Operator& operator-=( const Operator& op)
    {
#ifdef DG_DEBUG
        assert( op.size() == this->size());
#endif//DG_DEBUG
        for( unsigned i=0; i<n_*n_; i++)
            data_[i] -= op.data_[i];
        return *this;
    }
    /**
     * @brief scalar multiply
     *
     * @param value
     *
     * @return
     */
    Operator& operator*=( const T& value )
    {
        for( unsigned i=0; i<n_*n_; i++)
            data_[i] *= value;
        return *this;
    }
    /**
     * @brief add
     *
     * @param lhs
     * @param rhs
     *
     * @return
     */
    friend Operator operator+( const Operator& lhs, const Operator& rhs)
    {
        Operator temp(lhs);
        temp+=rhs;
        return temp;
    }
    /**
     * @brief subtract
     *
     * @param lhs
     * @param rhs
     *
     * @return
     */
    friend Operator operator-( const Operator& lhs, const Operator& rhs)
    {
        Operator temp(lhs);
        temp-=rhs;
        return temp;
    }
    /**
     * @brief scalar multiplication
     *
     * @param value
     * @param rhs
     *
     * @return
     */
    friend Operator operator*( const T& value, const Operator& rhs )
    {
        Operator temp(rhs);
        temp*=value;
        return temp;
    }

    /**
     * @brief scalar multiplication
     *
     * @param lhs
     * @param value
     *
     * @return
     */
    friend Operator operator*( const Operator& lhs, const T& value)
    {
        return  value*lhs;
    }

    /**
     * @brief matrix multiplication
     *
     * @param lhs
     * @param rhs
     *
     * @return
     */
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
/*! @brief LU Decomposition with partial pivoting
 *
 * @tparam T value type
 * @throw std::runtime_error if the matrix is singular
 * @param m  contains lu decomposition of input on output (inplace transformation)
 * @param p contains the pivot elements on output
 * @return determinant of \c m
 * @note uses extended accuracy of \c dg::exblas which makes it quite robust
 * against almost singular matrices
 * @ingroup densematrix
 * @sa \c dg::create::lu_solve
 */
template< class T>
T lu_pivot( dg::Operator<T>& m, std::vector<unsigned>& p)
{
    //from numerical recipes
    T pivot, determinant=(T)1;
    unsigned pivotzeile, numberOfSwaps=0;
    const size_t n = m.size();
    p.resize( n);
    for( size_t j = 0; j < n; j++) //gehe Spalten /Diagonale durch
    {
        //compute upper matrix except for the diagonal element (the pivot)
        for( size_t i = 0; i< j; i++)
        {
            thrust::host_vector<T> mik(i), mkj(i);
            for( size_t k=0; k<i; k++)
                mik[k] = m(i,k), mkj[k] = m(k,j);
            m(i,j) -= dg::blas1::dot( mik, mkj);
        }
        //compute canditates for pivot elements
        for( size_t i = j; i< n; i++)
        {
            thrust::host_vector<T> mik(j), mkj(j);
            for( size_t k=0; k<j; k++)
                mik[k] = m(i,k), mkj[k] = m(k,j);
            m(i,j) -= dg::blas1::dot( mik, mkj);
        }
        //search for absolute maximum of pivot candidates
        pivot = m(j,j);
        pivotzeile = j;
        for( size_t i = j+1; i < n; i++)
            if( fabs( m(i,j)) > fabs(pivot))
            {
                pivot = m(i,j), pivotzeile = i;
            }

        if( fabs(pivot) > 1e-15 )
        {
            if( pivotzeile != j)
            {
                m.swap_lines( pivotzeile, j);
                numberOfSwaps++;
            }
            p[j] = pivotzeile;
            //divide all elements below the diagonal by the pivot to get the lower matrix
            for( size_t i=j+1; i<n; i++)
                m(i,j) /= pivot;
            determinant*=m(j,j);

        }
        else
            throw std::runtime_error( "Matrix is singular!!");
    }
    if( numberOfSwaps % 2 != 0)
        determinant*=-1.;
    return determinant;

}

/**
 * @brief Solve the linear system with the LU decomposition
 *
 * @tparam T value type
 * @param lu result of \c dg::create::lu_pivot
 * @param p pivot vector from \c dg::create::lu_pivot
 * @param b right hand side (contains solution on output)
 * @ingroup densematrix
 */
template<class T>
void lu_solve( const dg::Operator<T>& lu, const std::vector<unsigned>& p, std::vector<T>& b)
{
    assert(p.size() == lu.size() && p.size() == b.size());
    const size_t n = p.size();
    // Vorwärtseinsetzen
    for( size_t i = 0; i<n; i++)
    {
        //mache Zeilentausch
        std::swap( b[ p[i] ], b[i]);
        thrust::host_vector<T> lui(i), bi(i);
        for( size_t j = 0; j < i; j++)
            lui[j] = lu(i,j), bi[j] = b[j];
        b[i] -= dg::blas1::dot( lui, bi);
    }
    // Rückwärtseinsetzen
    for( int i = n-1; i>=0; i--)
    {
        thrust::host_vector<T> lui(n-(i+1)), bi(n-(i+1));
        for( size_t j = i+1; j < n; j++)
            lui[j-(i+1)] = lu(i,j), bi[j-(i+1)] = b[j];
        b[i] -= dg::blas1::dot( lui, bi);
        b[i] /= lu(i,i);
    }
}

//
/**
 * @brief Compute the inverse of a square matrix
 *
 * using lu decomposition in combination with our accurate scalar products
 * @tparam T value type
 * @param in input matrix
 *
 * @ingroup densematrix
 * @return the inverse of in if it exists
 * @throw std::runtime_error if in is singular
 */
template<class T>
dg::Operator<T> inverse( const dg::Operator<T>& in)
{
    dg::Operator<T> out(in);
    const unsigned n = in.size();
    std::vector<unsigned> pivot( n);
    dg::Operator<T> lu(in);
    T determinant = lu_pivot( lu, pivot);
    if( fabs(determinant ) < 1e-14)
        throw std::runtime_error( "Determinant zero!");
    for( unsigned i=0; i<n; i++)
    {
        std::vector<T> unit(n, 0);
        unit[i] = 1;
        lu_solve( lu, pivot, unit);
        for( unsigned j=0; j<n; j++)
            out(j,i) = unit[j];
    }
    return out;
}

///@addtogroup lowlevel
///@{

/**
 * @brief Create the unit matrix
 *
 * @param n # of polynomial coefficients
 *
 * @return Operator
 */
template<class real_type>
Operator<real_type> delta( unsigned n)
{
    Operator<real_type> op(n, 0);
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
template<class real_type>
Operator<real_type> pipj( unsigned n)
{
    Operator<real_type> op(n, 0);
    for( unsigned i=0; i<n; i++)
        op( i,i) = 2./(real_type)(2*i+1);
    return op;
}
/**
 * @brief Create the T-matrix
 *
 * @param n # of polynomial coefficients
 *
 * @return Operator
 */
template<class real_type>
Operator<real_type> pipj_inv( unsigned n)
{
    Operator<real_type> op(n, 0);
    for( unsigned i=0; i<n; i++)
        op( i,i) = (real_type)(2*i+1)/2.;
    return op;
}
/**
 * @brief Create the D-matrix
 *
 * @param n # of polynomial coefficients
 *
 * @return Operator
 */
template<class real_type>
Operator<real_type> pidxpj( unsigned n)
{
    Operator<real_type> op(n, 0);
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
template<class real_type>
Operator<real_type> rirj( unsigned n)
{
    return Operator<real_type>( n, 1.);
}
/**
 * @brief Create the RL-matrix
 *
 * @param n # of polynomial coefficients
 *
 * @return Operator
 */
template<class real_type>
Operator<real_type> rilj( unsigned n)
{
    Operator<real_type> op( n, -1.);
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
template<class real_type>
Operator<real_type> lirj( unsigned n) {
    return rilj<real_type>( n).transpose();
}
/**
 * @brief Create the L-matrix
 *
 * @param n # of polynomial coefficients
 *
 * @return Operator
 */
template<class real_type>
Operator<real_type> lilj( unsigned n)
{
    Operator<real_type> op( n, -1.);
    for( unsigned i=0; i<n; i++)
        for( unsigned j=0; j<n; j++)
            if( ((i+j)%2) == 0)
                op( i,j) = 1.;
    return op;
}

/**
 * @brief Create the N-matrix
 *
 * @param n # of polynomial coefficients
 *
 * @return Operator
 */
template<class real_type>
Operator<real_type> ninj( unsigned n)
{
    Operator<real_type> op( n, 0.);
    for( int i=0; i<(int)n; i++)
        for( int j=0; j<(int)n; j++)
        {
            if( i == j+1)
                op( i,j) = 2./(2*i+1)/(2*j+1);
            if( i == j-1)
                op( i,j) = -2./(2*i+1)/(2*j+1);
        }
    op(0,0) = 2;
    return op;
}


/**
 * @brief Construct a diagonal operator with weights
 *
 * @param dlt
 *
 * @return new operator
 */
template<class real_type>
Operator<real_type> weights( const DLT<real_type>& dlt)
{
    unsigned n = dlt.weights().size();
    Operator<real_type> op( n, 0);
    for( unsigned i=0; i<n; i++)
        op(i,i) = dlt.weights()[i];
    return op;
}
/**
 * @brief Construct a diagonal operator with inverse weights
 *
 * @param dlt
 *
 * @return new operator
 */
template<class real_type>
Operator<real_type> precond( const DLT<real_type>& dlt)
{
    unsigned n = dlt.weights().size();
    Operator<real_type> op( n, 0);
    for( unsigned i=0; i<n; i++)
        op(i,i) = 1./dlt.weights()[i];
    return op;
}
///@}
}//namespace create



} //namespace dg

#endif //_DG_OPERATORS_DYN_
