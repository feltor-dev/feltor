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
#include "../blas2.h" // Matrix tensor traits

namespace dg{


/**
* @brief A square nxn matrix
*
* An enhanced square dynamic matrix for which arithmetic operators are
* overloaded.  It is not meant for performance critical code but is very
* convenient for example for the assembly of matrices or testing code.
* @sa There are some direct inversion routines that make use of the
* extended accuracy of \c dg::exblas and are thus quite robust agains almost
* singular matrices: \c dg::invert, \c dg::create::lu_pivot, \c dg::lu_solve
* @tparam T value type
* @ingroup densematrix
*/
template< class T>
class SquareMatrix
{
  public:
    typedef T value_type; //!< typically double or float
    /**
    * @brief Construct empty SquareMatrix
    */
    SquareMatrix() = default;
    /**
     * @brief allocate storage for nxn matrix
     *
     * @param n size
     */
    explicit SquareMatrix( const unsigned n): n_(n), data_(n_*n_){}
    /**
    * @brief Initialize elements.
    *
    * @param n matrix is of size n x n
    * @param value Every element is initialized to.
    */
    SquareMatrix( const unsigned n, const T& value): n_(n), data_(n_*n_, value) {}
    /**
     * @brief Construct from iterators
     *
     * @tparam InputIterator
     * @param first
     * @param last
     */
    template< class InputIterator>
    SquareMatrix( InputIterator first, InputIterator last, std::enable_if_t<!std::is_integral<InputIterator>::value>* = 0): data_(first, last)
    {
        unsigned n = std::distance( first, last);
        n_ = (unsigned)sqrt( (double)n);
        if( n_*n_!=n) throw Error( Message(_ping_)<<"Too few elements "<<n<<" need "<<n_*n_<<"\n");
    }
    /**
     * @brief Copy from existing data
     *
     * @param src size must be a square number
     */
    SquareMatrix( const std::vector<T>& src): data_(src)
    {
        unsigned n = src.size();
        n_ = (unsigned)sqrt( (double)n);
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
     * @param i row index
     * @param j column index
     * @return reference to value at that location
     */
    T& operator()(const size_t i, const size_t j)
    {
        return data_[ i*n_+j];
    }
    /*! @brief const access operator
     *
     * @param i row index
     * @param j column index
     * @return const value at that location
     */
    const T& operator()(const size_t i, const size_t j) const {
        return data_[ i*n_+j];
    }

    /**
     * @brief Size n of the SquareMatrix
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
    * @return  A newly generated SquareMatrix containing the transpose.
    */
    SquareMatrix transpose() const
    {
        SquareMatrix o(*this);
        for( unsigned i=0; i<n_; i++)
            for( unsigned j=0; j<i; j++)
            {
                std::swap( o.data_[i*n_+j], o.data_[j*n_+i]);
            }
        return o;
    }

    /*! @brief Matrix vector multiplication \f$ y = S x\f$
     *
     * This makes SquareMatrix usable in \c dg::blas2::symv
     * @tparam ContainerType Any container with <tt>operator[]</tt>
     * @param x input vector
     * @param y contains the solution on output (may not alias \p x)
     */
    template<class ContainerType1, class ContainerType2>
    void symv( const ContainerType1& x, ContainerType2& y) const
    {
        for( unsigned j=0; j<n_; j++)
        {
            y[j] = 0;
            for( unsigned k=0; k<n_; k++)
                y[j] += data_[j*n_+k]*x[k];
        }
    }
    /*! @brief Matrix vector multiplication \f$ y = \alpha S x + \beta y\f$
     *
     * This makes SquareMatrix usable in \c dg::blas2::symv
     * @tparam ContainerType Any container with <tt>operator[]</tt>
     * @param alpha A scalar
     * @param x input vector
     * @param beta A scalar
     * @param y contains the solution on output (may not alias \p x)
     */
    template<class value_type1, class ContainerType1, class value_type2, class ContainerType2>
    void symv( value_type1 alpha, const ContainerType1& x, value_type2 beta, ContainerType2& y) const
    {
        for( unsigned j=0; j<n_; j++)
        {
            y[j] *= beta;
            for( unsigned k=0; k<n_; k++)
                y[j] += alpha*data_[j*n_+k]*x[k];
        }
    }

    /*! @brief two Matrices are considered equal if elements are equal
     *
     * @param rhs Matrix to be compared to this
     * @return true if rhs does not equal this
     */
    bool operator!=( const SquareMatrix& rhs) const{
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
    bool operator==( const SquareMatrix& rhs) const {return !((*this != rhs));}

    /**
     * @brief subtract
     *
     * @return
     */
    SquareMatrix operator-() const
    {
        SquareMatrix temp(n_, 0.);
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
    SquareMatrix& operator+=( const SquareMatrix& op)
    {
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
    SquareMatrix& operator-=( const SquareMatrix& op)
    {
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
    SquareMatrix& operator*=( const T& value )
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
    friend SquareMatrix operator+( const SquareMatrix& lhs, const SquareMatrix& rhs)
    {
        SquareMatrix temp(lhs);
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
    friend SquareMatrix operator-( const SquareMatrix& lhs, const SquareMatrix& rhs)
    {
        SquareMatrix temp(lhs);
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
    friend SquareMatrix operator*( const T& value, const SquareMatrix& rhs )
    {
        SquareMatrix temp(rhs);
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
    friend SquareMatrix operator*( const SquareMatrix& lhs, const T& value)
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
    friend SquareMatrix operator*( const SquareMatrix& lhs, const SquareMatrix& rhs)
    {
        unsigned n_ = lhs.n_;
        SquareMatrix temp(n_, 0.);
        for( unsigned i=0; i< n_; i++)
            for( unsigned j=0; j<n_; j++)
            {
                temp.data_[i*n_+j] = (T)0;
                for( unsigned k=0; k<n_; k++)
                    temp.data_[i*n_+j] += lhs.data_[i*n_+k]*rhs.data_[k*n_+j];
            }
        return temp;
    }

    /**
     * @brief matrix-vector multiplication  \f$  y = S x\f$
     *
     * @snippet{trimleft} operator_t.cpp matvec
     * @param S Matrix
     * @param x Vector
     *
     * @return Vector
     */
    template<class ContainerType>
    friend ContainerType operator*( const SquareMatrix& S, const ContainerType& x)
    {
        ContainerType out(x);
        S.symv( x, out);
        return out;
    }

    /*! @brief puts a matrix linewise in output stream
     *
     * @tparam Ostream The stream e.g. std::cout
     * @param os the outstream
     * @param mat the matrix to output
     * @return the outstream
     */
    template< class Ostream>
    friend Ostream& operator<<(Ostream& os, const SquareMatrix& mat)
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
    friend Istream& operator>> ( Istream& is, SquareMatrix<T>& mat){
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

///@ingroup traits
template<class T>
struct TensorTraits<SquareMatrix<T>>
{
    using value_type  = T;
    using execution_policy = SerialTag;
    using tensor_category = SelfMadeMatrixTag;
};


namespace create
{
///@addtogroup invert
///@{

/*! @brief LU Decomposition with partial pivoting
 *
 * For example
 * @snippet{trimleft} operator_t.cpp det
 * @tparam T value type
 * @throw std::runtime_error if the matrix is singular
 * @param m  contains lu decomposition of input on output (inplace transformation)
 * @param p contains the pivot elements on output (will be resized)
 * @return determinant of \c m
 * @note uses extended accuracy of \c dg::exblas which makes it quite robust
 * against almost singular matrices
 * @sa \c dg::lu_solve
 */
template< class T>
T lu_pivot( dg::SquareMatrix<T>& m, std::vector<unsigned>& p)
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
 * @brief Invert a square matrix
 *
 * using lu decomposition in combination with our accurate scalar products
 *
 * For example
 * @snippet{trimleft} operator_t.cpp invert
 * @tparam T value type
 * @param in input matrix
 *
 * @return the inverse of in if it exists
 * @throw std::runtime_error if in is singular
 * @note uses extended accuracy of \c dg::exblas which makes it quite robust
 * against almost singular matrices
 */
template<class T>
dg::SquareMatrix<T> inverse( const dg::SquareMatrix<T>& in)
{
    dg::SquareMatrix<T> out(in);
    const unsigned n = in.size();
    std::vector<unsigned> pivot( n);
    dg::SquareMatrix<T> lu(in);
    T determinant = lu_pivot( lu, pivot);
    if( fabs(determinant ) == 0)
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

///@cond

/**
 * @brief Create the unit matrix
 *
 * @param n # of polynomial coefficients
 *
 * @return SquareMatrix
 */
template<class real_type>
SquareMatrix<real_type> delta( unsigned n)
{
    SquareMatrix<real_type> op(n, 0);
    for( unsigned i=0; i<n; i++)
        op( i,i) = 1.;
    return op;
}


// Not sure why we called this dg::create::lu_solve instead of dg::lu_solve
template<class T>
void lu_solve( const dg::SquareMatrix<T>& lu, const std::vector<unsigned>& p, std::vector<T>& b)
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


// S-matrix
template<class real_type>
SquareMatrix<real_type> pipj( unsigned n)
{
    SquareMatrix<real_type> op(n, 0);
    for( unsigned i=0; i<n; i++)
        op( i,i) = 2./(real_type)(2*i+1);
    return op;
}
// T-matrix
template<class real_type>
SquareMatrix<real_type> pipj_inv( unsigned n)
{
    SquareMatrix<real_type> op(n, 0);
    for( unsigned i=0; i<n; i++)
        op( i,i) = (real_type)(2*i+1)/2.;
    return op;
}
// D-matrix
template<class real_type>
SquareMatrix<real_type> pidxpj( unsigned n)
{
    SquareMatrix<real_type> op(n, 0);
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
// R-matrix
template<class real_type>
SquareMatrix<real_type> rirj( unsigned n)
{
    return SquareMatrix<real_type>( n, 1.);
}
// RL-matrix
template<class real_type>
SquareMatrix<real_type> rilj( unsigned n)
{
    SquareMatrix<real_type> op( n, -1.);
    for( unsigned i=0; i<n; i++)
        for( unsigned j=0; j<n; j++)
            if( j%2 == 0)
                op( i,j) = 1.;
    return op;
}
// LR-matrix
template<class real_type>
SquareMatrix<real_type> lirj( unsigned n) {
    return rilj<real_type>( n).transpose();
}
// L-matrix
template<class real_type>
SquareMatrix<real_type> lilj( unsigned n)
{
    SquareMatrix<real_type> op( n, -1.);
    for( unsigned i=0; i<n; i++)
        for( unsigned j=0; j<n; j++)
            if( ((i+j)%2) == 0)
                op( i,j) = 1.;
    return op;
}

// N-matrix
template<class real_type>
SquareMatrix<real_type> ninj( unsigned n)
{
    SquareMatrix<real_type> op( n, 0.);
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
///@endcond


///@}
}//namespace create

/**
 * @brief Solve the linear system with the LU decomposition
 *
 * @tparam T value type
 * @param lu result of \c dg::create::lu_pivot
 * @param p pivot vector from \c dg::create::lu_pivot
 * @param b right hand side (contains solution on output)
 * @ingroup invert
 * @sa dg::create::lu_pivot
 */
template<class T>
void lu_solve( const dg::SquareMatrix<T>& lu, const std::vector<unsigned>& p, std::vector<T>& b)
{
    dg::create::lu_solve( lu, p, b);
}

///@brief Compute inverse of square matrix (alias for \c dg::create::inverse)
///@copydetails dg::create::inverse(const dg::SquareMatrix<T>&)
///@ingroup invert
template<class T>
dg::SquareMatrix<T> invert( const dg::SquareMatrix<T>& in)
{
    return dg::create::inverse(in);
}
///
/// @brief The old name for SquareMatrix was Operator
/// @ingroup densematrix
template<class T>
using Operator = SquareMatrix<T>;

} //namespace dg

#endif //_DG_OPERATORS_DYN_
