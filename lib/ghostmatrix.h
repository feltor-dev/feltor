#ifndef _TL_GHOSTMATRIX_
#define _TL_GHOSTMATRIX_


#include "matrix.h"
#include <array>

namespace toefl{

/*! @brief Possible boundary conditions for the ghostmatrix class.
 * Naming follows FFTW R2R kind naming.
 *
 * @sa http://en.wikipedia.org/wiki/Discrete_sine_transform
 */
enum bc { TL_PERIODIC = 0, //!< Periodic boundary
          TL_DST00, //!< dst (discrete sine transform) 1
          TL_DST10, //!< dst 2
          TL_DST01, //!< dst 3
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
 * @attention  
 * The swap_fields function only swaps matrices without(!) the ghostcells. 
 */
template< typename T, enum Padding P = TL_NONE>
class GhostMatrix: public Matrix<T,P>
{
  public:
    /*! @brief Allocate memory for the Matrix and the ghostcells.
     *
     * Like any other Matrix a GhostMatrix can be void, padded etc. 
     * @param rows Rows of the matrix
     * @param cols Columns of the Matrix
     * @param bc_rows The boundary condition for the rows, i.e. for the first
     *  and the last line of the Matrix.
     * @param bc_cols The boundary condition for the columns, i.e. the first and 
     *  the last column. 
     * @param allocate Whether memory shall be allocated or not. Ghostcells
     * are always allocated.
     * @attention The horizontal boundary condition for e.g. the r2r transforms
     *  in DRT_DRT corresponds to the boundary condition for columns. 
     *  Analogously the vertical boundary conditions correspond to bc_rows.
     */
    GhostMatrix( const size_t rows, const size_t cols, const enum bc bc_rows = TL_PERIODIC, const enum bc bc_cols = TL_PERIODIC, const bool allocate = true);
    /*! @brief Allocate and init memory.
     *
     * Like any other Matrix a GhostMatrix can be void, padded etc. 
     * The ghostcells are always allocated
     * @param rows Rows of the matrix
     * @param cols Columns of the Matrix
     * @param bc_rows The boundary condition for the rows, i.e. for the first
     *  and the last line of the Matrix.
     * @param bc_cols The boundary condition for the columns, i.e. the first and 
     *  the last column. 
     * @param value 
     *  Value the memory (including ghostcells) shall be initialized to
     */
    GhostMatrix( const size_t rows, const size_t cols, const T& value, const enum bc bc_rows = TL_PERIODIC,  const enum bc bc_cols = TL_PERIODIC);
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
     */
    inline void initGhostCells( );
  private:
    enum bc bc_rows, bc_cols;
    Matrix<T,TL_NONE> ghostRows;
    Matrix<T,TL_NONE> ghostCols;
        

};


////////////////////////////////////////////DEFINITIONS////////////////////////////////////////////////
template< typename T, enum Padding P>
GhostMatrix<T,P>::GhostMatrix( const size_t rows, const size_t cols, const enum bc bc_rows, const enum bc bc_cols , const bool alloc):
                    Matrix<T,P>(rows, cols, alloc), bc_rows( bc_rows), bc_cols( bc_cols), ghostRows( 2, cols + 2), ghostCols( rows, 2) {}

template< typename T, enum Padding P>
GhostMatrix<T,P>::GhostMatrix( const size_t rows, const size_t cols, const T& value, const enum bc bc_rows,  const enum bc bc_cols):  Matrix<T,P>(rows, cols, value),bc_rows(bc_rows), bc_cols(bc_cols), ghostRows( 2, cols + 2, value), ghostCols( rows, 2, value) { }

template< typename T, enum Padding P>
T& GhostMatrix<T,P>::at( const int i, const int j)
{
    const int n = this->rows(), m = this->cols();
#ifdef TL_DEBUG
   if( i < -1 || i > n || j < -1|| j > m)
       throw BadIndex( i,n, j,m, ping);
   if( this->isVoid()) 
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
    const int n = this->rows(), m = this->cols();
#ifdef TL_DEBUG
   if( i < -1 || i > n || j < -1|| j > m)
       throw BadIndex( i,n, j,m, ping);
   if( this->isVoid()) 
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
void GhostMatrix<T,P>::initGhostCells()
{
    const unsigned cols = ghostRows.cols(); 
    const unsigned rows = ghostCols.rows(); 
    switch(bc_cols)
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
        case( TL_DST10):
            for( unsigned i=0; i<rows; i++)
            {
                ghostCols( i, 0) = -(*this)(i, 0);
                ghostCols( i, 1) = -(*this)(i, cols-3);
            }
            break;
        case( TL_DST01):
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
    switch( bc_rows)
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
        case( TL_DST10):
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
        case( TL_DST01):
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
   if( this->isVoid()) 
       throw Message( "Trying to access a void matrix!", ping);
#endif
    for(int i = -1; i < (int)this->rows() + 1; i++)
    {
        for ( int j = -1; j < (int)this->cols() + 1; j++)
            os << (this->at)(i,j) << " ";
        os << "\n";
    }
}
} //namespace toefl



#endif //_TL_GHOSTMATRIX_
