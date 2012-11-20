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
     * @attention Be careful with the use of the allocate and resize function,
     * after use of the swap_fields function. 
     * The swap_fields function only swaps matrices without(!) the ghostcells. 
     * It might then happen that the ghostcells are allocated but the matrix is void.
     * Then the use of the allocate or resize function will throw an exception. 
     * The ghostcells are only allocatable or resizable together with a void matrix. 
     */
    template< typename T, enum Padding P = TL_NONE>
    class GhostMatrix: public Matrix<T,P>
    {
      private:
        Matrix<T,TL_NONE> ghostRows;
        Matrix<T,TL_NONE> ghostCols;
        void allocate_virtual(){//Allocates not only parent matrix but also ghostMatrices
            this->allocate_();
            ghostRows.allocate();
            ghostCols.allocate();
        }
        void resize_virtual( const size_t new_rows, const size_t new_cols){
            this->resize_( new_rows, new_cols);
            ghostRows.resize( 2, new_cols + 2);
            ghostCols.resize( new_rows, 2);
        }
            
      public:
        /*! @brief Construct an empty void GhostMatrix*/
        GhostMatrix( );
        /*! @brief Same as Matrix Constructor.
         *
         * Like any other Matrix a GhostMatrix can be void, padded etc. 
         * If the Matrix is void then the ghostcells are also void!
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


        /*! @brief Display Matrix including ghostcells
         *
         * The use of the operator<< function of the Matrix class doesn't
         * show the ghostcells.
         * @param os The outstream to be used.
         */
        void display( std::ostream& os = std::cout);

        template< size_t row>
        inline T& ghostRow( const int col) {return ghostRow( row,col);}
        template< size_t col>
        inline T& ghostCol( const int row) { return ghostCol( row, col);}

    };
    template< typename T, enum Padding P>
    GhostMatrix<T,P>::GhostMatrix():Matrix<T,P>(), ghostRows(), ghostCols(){}
    
    template< typename T, enum Padding P>
    GhostMatrix<T,P>::GhostMatrix( const size_t rows, const size_t cols, const bool alloc): 
                        Matrix<T,P>(rows, cols, alloc), ghostRows( 2, cols + 2, alloc), ghostCols( rows, 2, alloc)
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
    template< typename T, enum Padding P>
    void GhostMatrix<T,P>::display( std::ostream& os)
    {
#ifdef TL_DEBUG
       if( this->ptr == NULL) 
           throw Message( "Trying to access a void matrix!", ping);
#endif
        for(int i = -1; i < (int)this->n + 1; i++)
        {
            for ( int j = -1; j < (int)this->m + 1; j++)
                os << (this->at)(i,j) << " ";
            os << "\n";
        }
    }

}



#endif //_TL_GHOSTMATRIX_
