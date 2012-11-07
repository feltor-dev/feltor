#ifndef _TL_GHOSTMATRIX_
#define _TL_GHOSTMATRIX_


#include "matrix.h"

namespace toefl{


    /*! @brief Mimic a Matrix with ghostcells
     *
     * The idea is that the access to boundary values doesn't need to be optimized
     * since typically only a small fraction (0.1 - 1%) of the 
     * values in a matrix are boundary values. This means it doesn't hurt to 
     * pay some if() clauses in the access operator to get an easy way to implement
     * e.g. the arakawa scheme for various boundary conditions. 
     * A GhostMatrix is a Matrix with the additional access operator at()
     * with which boundary (including ghost) values can be manipulated.
     */
    template< typename T, enum Padding P>
    class GhostMatrix: public Matrix<T,P>
    {
      private:
        Matrix<T,TL_NONE> ghostRows;
        Matrix<T,TL_NONE> ghostCols;
      public:
        /*! @brief Same as Matrix Constructor.
         *
         * Like any other Matrix a GhostMatrix can be void, padded etc. 
         * Note however that the ghostCells are always allocated, regardless of
         * whether the Matrix itself is void or not.
         */
        GhostMatrix( const size_t rows, const size_t cols, const bool allocate = true);
        /*! @brief Access Operator for boundary values
         *
         * @param i row index (may equal -1 and rows)
         * @param j column index (may equal -1 and col)
         * @return material value or ghost value
         */
        inline T& at( const int i, const int j);
        /*! @brief Access Operator for const boundary values
         *
         * @param i row index (may equal -1 and rows)
         * @param j column index (may equal -1 and col)
         * @return material value or ghost value
         */
        inline const T& at( const int i, const int j) const;
    };
    
    template< typename T, enum Padding P>
    GhostMatrix<T,P>::GhostMatrix( const size_t rows, const size_t cols, const bool allocate): Matrix<T,P>(rows, cols, allocate), ghostRows( 2, cols + 2), ghostCols( rows, 2)
    {
    }
    
    template< typename T, enum Padding P>
    T& GhostMatrix<T,P>::at( const int i, const int j)
    {
        const int n = this-> n, m = this->m;
    #ifdef TL_DEBUG
       if( i < -1 || i > n || j < -1|| j > m)
           throw BadIndex( i,n, j,m, ping);
       if( this->ptr == NULL) 
           throw Message( "Trying to access a void matrix!", ping);
    #endif
        if( i == -1 ) //j = -1,...,m
            return ghostRows( 0, j + 1); 
        if( i == n ) 
            return ghostRows( 1, j + 1);
        if( j == -1) //i = 0,...,n-1
            return ghostCols( i, 0);
        if( j == m) 
            return ghostCols( i, 1);
        return (*this)(i,j);
    }
    
    template< typename T, enum Padding P>
    const T& GhostMatrix<T,P>::at( const int i, const int j) const 
    {
        const int n = this-> n, m = this->m;
    #ifdef TL_DEBUG
       if( i < -1 || i > n || j < -1|| j > m)
           throw BadIndex( i,n, j,m, ping);
       if( this->ptr == NULL) 
           throw Message( "Trying to access a void matrix!", ping);
    #endif
        if( i == -1 ) //j = -1,...,m
            return ghostRows( 0, j + 1); 
        if( i == n ) 
            return ghostRows( 1, j + 1);
        if( j == -1) //i = 0,...,n-1
            return ghostCols( i, 0);
        if( j == m) 
            return ghostCols( i, 1);
        return (*this)(i,j);
    }

}



#endif //_TL_GHOSTMATRIX_
