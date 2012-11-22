#ifndef _TL_GHOSTMATRIX_
#define _TL_GHOSTMATRIX_


#include "matrix.h"

namespace toefl{

    enum bc { TL_PERIODIC, //!< Periodic boundary
              TL_DST00, //!< dst 1
              TL_DST01, //!< dst 2
              TL_DST10, //!< dst 3
              TL_DST11 //!<  dst 4
    }; 

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
        /*
        void resize_virtual( const size_t new_rows, const size_t new_cols){
            this->resize_( new_rows, new_cols);
            ghostRows.resize( 2, new_cols + 2);
            ghostCols.resize( new_rows, 2);
        }
        */
            
      public:
        /*! @brief Construct an empty void GhostMatrix*/
        GhostMatrix( );
        /*! @brief Allocate memory.
         *
         * Like any other Matrix a GhostMatrix can be void, padded etc. 
         * If the Matrix is void then the ghostcells are also void!
         * @param rows Rows of the matrix
         * @param cols Columsn of the Matrix
         * @param allocate Whether memory shall be allocated or not
         */
        GhostMatrix( const size_t rows, const size_t cols, const bool allocate = true);
        /*! @brief Allocate and init memory.
         *
         * Like any other Matrix a GhostMatrix can be void, padded etc. 
         * If the Matrix is void then the ghostcells are also void!
         * @param rows Rows of the matrix
         * @param cols Columsn of the Matrix
         * @param value 
         *  Value the memory (including ghostcells) shall be initialized to
         */
        GhostMatrix( const size_t rows, const size_t cols, const T& value );
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

        /*! @brief Initialize ghost cells according to given boundary conditions
         *
         * @param bc_rows Condition for the ghost rows. 
         * @param bc_cols Condition for the ghost columns.
         */
        void initGhostCells( enum bc bc_rows, enum bc bc_cols);

    };
    template< typename T, enum Padding P>
    GhostMatrix<T,P>::GhostMatrix():Matrix<T,P>(), ghostRows(), ghostCols(){}
    
    template< typename T, enum Padding P>
    GhostMatrix<T,P>::GhostMatrix( const size_t rows, const size_t cols, const bool alloc): 
                        Matrix<T,P>(rows, cols, alloc), ghostRows( 2, cols + 2, alloc), ghostCols( rows, 2, alloc) {}

    template< typename T, enum Padding P>
    GhostMatrix<T,P>::GhostMatrix( const size_t rows, const size_t cols, const T& value): 
                        Matrix<T,P>(rows, cols, value), ghostRows( 2, cols + 2, value), ghostCols( rows, 2, value) { }
    
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
    void GhostMatrix<T,P>::initGhostCells( enum bc b_rows, enum bc b_cols)
    {
        const unsigned cols = ghostRows.cols(); 
        const unsigned rows = ghostCols.rows(); 
        switch(b_cols)
        {
            case( TL_PERIODIC): 
                for( unsigned i=0; i<rows; i++)
                {
                    ghostCols( i, 0) = (*this)(i, cols-3);
                    ghostCols( i, 1) = (*this)(i, 0);
                }
                break;
            case( TL_DST00):
                for( unsigned i=0; i<rows; i++)
                {
                    ghostCols( i, 0) = 0;
                    ghostCols( i, 1) = 0;
                }
                break;
            case( TL_DST01):
                for( unsigned i=0; i<rows; i++)
                {
                    ghostCols( i, 0) = -(*this)(i, 0);
                    ghostCols( i, 1) = -(*this)(i, cols-3);
                }
                break;
            case( TL_DST10):
                for( unsigned i=0; i<rows; i++)
                {
                    ghostCols( i, 0) = 0;
                    ghostCols( i, 1) = (*this)(i, cols-4);
                }
                break;
            case( TL_DST11):
                for( unsigned i=0; i<rows; i++)
                {
                    ghostCols( i, 0) = -(*this)(i, 0);
                    ghostCols( i, 1) = (*this)(i, cols-3);
                }
                break;
        }
        switch( b_rows)
        {
            case( TL_PERIODIC):
                ghostRows(0,0) = ghostCols( rows-1, 0);
                for( unsigned i=0; i<cols-2; i++)
                {
                    ghostRows( 0,i+1) = (*this)( rows-1, i);
                }
                ghostRows(0, cols-1) = ghostCols( rows-1, 1);
                ghostRows(1,0) = ghostCols( 0, 0);
                for( unsigned i=0; i<cols-2; i++)
                {
                    ghostRows( 1,i+1) = (*this)( 0, i);
                }
                ghostRows(1, cols-1) = ghostCols(0 , 1);
                break;
            case( TL_DST00):
                for( unsigned i=0; i<cols; i++)
                    ghostRows(0,i) = 0;
                for( unsigned i=0; i<cols; i++)
                    ghostRows(1,i) = 0;
                break;
            case( TL_DST01):
                ghostRows(0,0) = -ghostCols( 0, 0);
                for( unsigned i=0; i<cols-2; i++)
                {
                    ghostRows( 0,i+1) = -(*this)( 0, i);
                }
                ghostRows(0, cols-1) = -ghostCols( 0, 1);
                ghostRows(1,0) = -ghostCols( rows-1, 0);
                for( unsigned i=0; i<cols-2; i++)
                {
                    ghostRows( 1,i+1) = -(*this)( rows-1, i);
                }
                ghostRows(1, cols-1) = -ghostCols( rows-1 , 1);
                break;
            case( TL_DST10):
                for( unsigned i=0; i<cols; i++)
                    ghostRows(0,i) = 0;
                ghostRows(1,0) = ghostCols( rows-2, 0);
                for( unsigned i=0; i<cols-2; i++)
                {
                    ghostRows( 1,i+1) = (*this)( rows-2, i);
                }
                ghostRows(1, cols-1) = ghostCols( rows-2 , 1);
                break;
            case( TL_DST11):
                ghostRows(0,0) = -ghostCols( 0, 0);
                for( unsigned i=0; i<cols-2; i++)
                {
                    ghostRows( 0,i+1) = -(*this)( 0, i);
                }
                ghostRows(0, cols-1) = -ghostCols( 0, 1);
                ghostRows(1,0) = ghostCols( rows-1, 0);
                for( unsigned i=0; i<cols-2; i++)
                {
                    ghostRows( 1,i+1) = (*this)( rows-1, i);
                }
                ghostRows(1, cols-1) = ghostCols( rows-1 , 1);
                break;
        }
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
