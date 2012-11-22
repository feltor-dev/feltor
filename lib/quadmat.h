/*!
 * @file
 * @brief Implementation of the QuadMat class and the invert function
 * @author Matthias Wiesenberger
 * @email Matthias.Wiesenberger@uibk.ac.at
 *
 */
#ifndef _QUADMAT_
#define _QUADMAT_
#include <iostream>
#include "exceptions.h"

namespace toefl{
    /*! @brief Container for quadratic fixed size matrices
     *
     * The QuadMat stores its values inside the object.
     * i.e. a QuadMat<double, 2> stores four double variables continously
     * in memory. Therefore it is well suited for the use in the Matrix
     * class (because memcpy and memset correctly work on this type)
     * \note T and n should be of small size to reduce object size.
     * \note QuadMat is an aggregate so you can use initializer lists in c++11..
     * @tparam T tested with double and std::complex<double> 
     * @tparam n size of the Matrix, assumed to be small (either 2 or 3)
     */
    template <class T, size_t n>
    class QuadMat
    {
      private:
        T data[n*n];
      public:
        /*! @brief no values are assigned*/
        QuadMat() = default;
        /*! @brief Initialize elements to a value
         *
         * @param value The initial value
         */
        QuadMat( const T& value)
        {
            for( unsigned i=0; i<n*n; i++)
                data[i] = value;
        }
        /*! @brief Use c++0x new feature*/
        QuadMat( std::initializer_list<T> l)
        {
            if( l.size() != n*n)
                throw Message( "Initializer list has wrong size", ping);
            unsigned i=0;
            for( auto& s: l)
                data[i++] = s;
        }
        /*! @brief copies values of src into this*/
        QuadMat( const QuadMat& src) = default;
        /*! @brief Copies values of src into this
         * 
         * implicitly defined
        */
        QuadMat& operator=( const QuadMat& rhs) = default;
    
        /*! @brief set memory to 0
         */
        void zero()
        {
            for( size_t i = 0; i < n*n; i++)
                data[i] = 0;
        }
        /*! @brief access operator
         *
         * Performs a range check if TL_DEBUG is defined
         * @param i row index
         * @param j column index
         * @return reference to value at that location
         */
        T& operator()(const size_t i, const size_t j){
    #ifdef TL_DEBUG
            if( i >= n || j >= n)
                throw BadIndex( i, n, j, n, ping);
    #endif
            return data[ i*n+j];
        }
        /*! @brief const access operator
         *
         * Performs a range check if TL_DEBUG is defined
         * @param i row index
         * @param j column index
         * @return const value at that location
         */
        const T& operator()(const size_t i, const size_t j) const {
    #ifdef TL_DEBUG
            if( i >= n || j >= n)
                throw BadIndex( i, n, j, n, ping);
    #endif
            return data[ i*n+j];
        }
        /*! @brief two Matrices are considered equal if elements are equal
         *
         * @param rhs Matrix to be compared to this
         * @return true if rhs does not equal this
         */
        const bool operator!=( const QuadMat& rhs) const{
            for( size_t i = 0; i < n*n; i++)
                if( data[i] != rhs.data[i])
                    return true;
            return false;
        }
        /*! @brief two Matrices are considered equal if elements are equal
         *
         * @param rhs Matrix to be compared to this
         * @return true if rhs equals this
         */
        const bool operator==( const QuadMat& rhs) const {return !((*this != rhs));}
        /*! @brief puts a matrix linewise in output stream
         *
         * @param os the outstream
         * @param mat the matrix to output
         * @return the outstream
         */
        friend std::ostream& operator<<(std::ostream& os, const QuadMat<T,n>& mat)
        {
            for( size_t i=0; i < n ; i++)
            {
                for( size_t j = 0;j < n; j++)
                    os << mat(i,j) << " ";
                os << "\n";
            }
            return os;
        }
    };

    /*! @brief Return the One Matrix
     * @return Matrix containing ones on the diagonal and zeroes elsewhere
     */
    template< size_t n>
    QuadMat<double, n> Eins()
    {
        QuadMat<double, n> E(0);
        for( unsigned i=0; i<n; i++)
            E(i,i) = 1;
        return E;
    }
    /*! @brief Return the Zero Matrix
     * @return Matrix containing only zeroes 
     */
    template< size_t n>
    QuadMat<double, n> Zero()
    {
        QuadMat<double, n> E(0);
        return E;
    }

    /*! @brief inverts a 2x2 matrix of given type
     *
     * \note throws a Message if Determinant is zero. 
     * @tparam T The type must support basic algorithmic functionality (i.e. +, -, * and /)
     * @param in The input matrix 
     * @param out The output matrix contains the invert of in on output.
     *  Inversion is inplace if in and out dereference the same object.
     */
    template<class T>
    void invert(const QuadMat<T, 2>& in, QuadMat< T,2>& out);
    template<class T>
    void invert(const QuadMat<T, 2>& m, QuadMat<T,2>& m1) 
    {
        T det, temp;
        det = m(0,0)*m(1,1) - m(0,1)*m(1,0);
        if( det== (T)0) throw Message("Determinant is Zero\n", ping);
        temp = m(0,0);
        m1(0,0) = m(1,1)/det;
        m1(0,1) /=-det;
        m1(1,0) /=-det;
        m1(1,1) = temp/det;
    }
    
    /*! @brief inverts a 3x3 matrix of given type
     *
     * (overloads the 2x2 version)
     * \note throws a Message if Determinant is zero. 
     * @tparam The type must support basic algorithmic functionality (i.e. +, -, * and /)
     * @param in The input matrix 
     * @param out The output matrix contains the invert of in on output.
     *  Inversion is inplace if in and out dereference the same object.
     */
    template< typename T>
    void invert( const QuadMat< T, 3>& in, QuadMat<T,3>& out );
    template< typename T>
    void invert( const QuadMat< T, 3>& m, QuadMat<T,3>& m1 ) 
    {
        T det, temp00, temp01, temp02, temp10, temp11, temp20;
        det = m(0,0)*(m(1,1)*m(2,2)-m(2,1)*m(1,2))+m(0,1)*(m(1,2)*m(2,0)-m(1,0)*m(2,2))+m(0,2)*(m(1,0)*m(2,1)-m(2,0)*m(1,1));
        if( det== (T)0) throw Message("Determinant is Zero\n", ping);
        temp00 = m(0,0);
        temp01 = m(0,1);
        temp02 = m(0,2);
        m1(0,0) = (m(1,1)*m(2,2) - m(1,2)*m(2,1))/det;
        m1(0,1) = (m(0,2)*m(2,1) - m(0,1)*m(2,2))/det;
        m1(0,2) = (temp01*m(1,2) - m(0,2)*m(1,1))/det;
    
        temp10 = m(1,0);
        temp11 = m(1,1);
        m1(1,0) = (m(1,2)*m(2,0) - m(1,0)*m(2,2))/det;
        m1(1,1) = (temp00*m(2,2) - temp02*m(2,0))/det;
        m1(1,2) = (temp02*temp10 - temp00*m(1,2))/det;
    
        temp20 = m(2,0);
        m1(2,0) = (temp10*m(2,1) - temp11*m(2,0))/det;
        m1(2,1) = (temp01*temp20 - temp00*m(2,1))/det;
        m1(2,2) = (temp00*temp11 - temp10*temp01)/det;
    }
}
#endif //_QUADMAT_
