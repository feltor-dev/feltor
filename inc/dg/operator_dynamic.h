#ifndef _DG_OPERATORS_DYN_
#define _DG_OPERATORS_DYN_

#include <vector>
#include <stdexcept>
#include "dlt.h"

namespace dg{

/**
* @brief Helper class main_ly for the assembly of Matrices
*
* @in_group lowlevel
* In_ prin_ciple it's an_ en_han_ced quadratic dyn_amic matrix
* but it's n_ot mean_t for performan_ce critical code. 
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
        Operator temp;
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
        Operator temp;
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
Operator<double> delta( unsigned n)
{
    Operator<double> op(n, 0);
    for( unsigned i=0; i<n; i++)
        op( i,i) = 1.;
    return op;
}
Operator<double> pipj( unsigned n)
{
    Operator<double> op(n, 0);
    for( unsigned i=0; i<n; i++)
        op( i,i) = 2./(double)(2*i+1);
    return op;
}
Operator<double> pipj_inv( unsigned n)
{
    Operator<double> op(n, 0);
    for( unsigned i=0; i<n; i++)
        op( i,i) = (double)(2*i+1)/2.;
    return op;
}
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
Operator<double> rirj( unsigned n)
{
    return Operator<double>( n, 1.);
}
Operator<double> rilj( unsigned n)
{
    Operator<double> op( n, -1.);
    for( unsigned i=0; i<n; i++)
        for( unsigned j=0; j<n; j++)
            if( j%2 == 0)
                op( i,j) = 1.;
    return op;
}
Operator<double> lirj( unsigned n) {return rilj( n).transpose();}
Operator<double> lilj( unsigned n) 
{
    Operator<double> op( n, -1.);
    for( unsigned i=0; i<n; i++)
        for( unsigned j=0; j<n; j++)
            if( ((i+j)%2) == 0)
                op( i,j) = 1.;
    return op;
}

Operator<double> weights( unsigned n)
{
    Operator<double> w(n ,0.);
    switch( n)
    {
        case( 1): 
            for( unsigned i=0; i<n; i++)
                w(i,i) = DLT<1>::weight[i];
            break;
        case( 2): 
            for( unsigned i=0; i<n; i++)
                w(i,i) = DLT<2>::weight[i];
            break;
        case( 3): 
            for( unsigned i=0; i<n; i++)
                w(i,i) = DLT<3>::weight[i];
            break;
        case( 4): 
            for( unsigned i=0; i<n; i++)
                w(i,i) = DLT<4>::weight[i];
            break;
        case( 5): 
            for( unsigned i=0; i<n; i++)
                w(i,i) = DLT<5>::weight[i];
            break;
        default:
            throw std::out_of_range( "not implemented yet");
    }
    return w;
}

Operator<double> weights_inv( unsigned n)
{
    Operator<double> op = weights( n);
    for( unsigned i=0; i<n; i++)
        op(i,i) = 1./ op(i,i);
    return op;
}


Operator<double> forward( unsigned n)
{
    Operator<double> op( n);
    switch( n)
    {
        case( 1): 
            for( unsigned i=0; i<n; i++)
                for( unsigned j=0; j<n; j++)
                    op( i,j) = DLT<1>::forward[i][j];
                  break;
        case(2): 
            for( unsigned i=0; i<n; i++)
                for( unsigned j=0; j<n; j++)
                    op( i,j) = DLT<2>::forward[i][j];
            break;
        case(3): 
            for( unsigned i=0; i<n; i++)
                for( unsigned j=0; j<n; j++)
                    op( i,j) = DLT<3>::forward[i][j];
            break;
        case(4): 
            for( unsigned i=0; i<n; i++)
                for( unsigned j=0; j<n; j++)
                    op( i,j) = DLT<4>::forward[i][j];
            break;
        case(5): 
            for( unsigned i=0; i<n; i++)
                for( unsigned j=0; j<n; j++)
                    op( i,j) = DLT<5>::forward[i][j];
            break;
        default:
            throw std::out_of_range("not implemented yet");
    }
    return op;
}

Operator<double> backward( unsigned n)
{
    Operator<double> op( n);
    switch( n)
    {
        case( 1): 
            for( unsigned i=0; i<n; i++)
                for( unsigned j=0; j<n; j++)
                    op( i,j) = DLT<1>::backward[i][j];
                  break;
        case(2): 
            for( unsigned i=0; i<n; i++)
                for( unsigned j=0; j<n; j++)
                    op( i,j) = DLT<2>::backward[i][j];
            break;
        case(3): 
            for( unsigned i=0; i<n; i++)
                for( unsigned j=0; j<n; j++)
                    op( i,j) = DLT<3>::backward[i][j];
            break;
        case(4): 
            for( unsigned i=0; i<n; i++)
                for( unsigned j=0; j<n; j++)
                    op( i,j) = DLT<4>::backward[i][j];
            break;
        case(5): 
            for( unsigned i=0; i<n; i++)
                for( unsigned j=0; j<n; j++)
                    op( i,j) = DLT<5>::backward[i][j];
            break;
        default:
            throw std::out_of_range("not implemented yet");
    }
    return op;
}
Operator<double> backwardEQ( unsigned n)
{
    Operator<double> op( n);
    switch( n)
    {
        case( 1): 
            for( unsigned i=0; i<n; i++)
                for( unsigned j=0; j<n; j++)
                    op( i,j) = DLT<1>::backwardEQ[i][j];
                  break;
        case(2): 
            for( unsigned i=0; i<n; i++)
                for( unsigned j=0; j<n; j++)
                    op( i,j) = DLT<2>::backwardEQ[i][j];
            break;
        case(3): 
            for( unsigned i=0; i<n; i++)
                for( unsigned j=0; j<n; j++)
                    op( i,j) = DLT<3>::backwardEQ[i][j];
            break;
        case(4): 
            for( unsigned i=0; i<n; i++)
                for( unsigned j=0; j<n; j++)
                    op( i,j) = DLT<4>::backwardEQ[i][j];
            break;
        case(5): 
            for( unsigned i=0; i<n; i++)
                for( unsigned j=0; j<n; j++)
                    op( i,j) = DLT<5>::backwardEQ[i][j];
            break;
        default:
            throw std::out_of_range("not implemented yet");
    }
    return op;
}


}//namespace create



} //namespace dg

#endif //_DG_OPERATORS_DYN_
