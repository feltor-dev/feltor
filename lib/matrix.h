#ifndef _TL_MATRIX_
#define _TL_MATRIX_

#include <iostream>
#include <cstring> //memcpy and memset
#include "fftw3.h"
#include "exceptions.h"

namespace toefl{


    enum Padding{ TL_NONE, TL_DFT_1D, TL_DFT_2D};
    enum Allocate{ TL_VOID = false};

    template <class T, enum Padding P>
    class Matrix;
    
    /*! @brief swap memories of two equally sized matrices
     *
         * Performs a range check if TL_DEBUG is defined
         * The sizes of the actually allocated memory, which depend on padding and the value type, have to be equal.
 
     */
    template<class T1, enum Padding P1, class T2, enum Padding P2>
    void swap_fields( Matrix<T1, P1>& lhs, Matrix<T2, P2>& rhs);

    /*! @brief permute memory of matrices with the same type
     *
     * Permutation is done clockwise i.e. 
     * third becomes first, second becomes third and first becomes second
     */
    template <class T, enum Padding P>
    void permute_fields( Matrix<T, P>& first, Matrix<T, P>& second, Matrix<T, P>& third);
    
    template <class T, enum Padding P>
    std::ostream& operator<< ( std::ostream& os, const Matrix<T, P>& mat); 
    template <class T, enum Padding P>
    std::istream& operator>> ( std::istream& is, Matrix<T, P>& mat); 

    template< typename T>
    void transpose( Matrix< T, TL_NONE>& inout, Matrix< T, TL_NONE>& swap);
    
    
    /*! @brief Matrix class of constant size that provides fftw compatible 2D fields
     *
     * The primary goal of the Matrix class is to provide a way to have 
     * a dynamically continuously allocated 2d field for either real or complex values. 
     * Once allocated the size (i.e. the number of rows and columns) of a Matrix cannot be changed any more. 
     * The following example code should be pretty obvious:
     * \code
      Matrix<double> m(3, 3);
      m.zero();
      m( 4,2) = 3;
      std::cout << m << std::endl; 
      \endcode
     *
     * Once you want to fourier transform a real matrix inplace you will be confronted
     * with two issues: The first thing is that the result is of complex type and 
     * you probably want to calculate with complex numbers. 
     * Therefore there should be a way for a matrix to somehow change its type
     * from real to complex. This is solved here by the globally defined swap_fields routine that exchanges the pointers to memory of two Matrices, even if the types are different. 
     * So if you swap pointers between a real and a complex matrix you effectively made the complex matrix real and the real matrix complex. 
     * If the additional allocated memory is of concern to you, there is the 
     * allocate flag in the constructor that you can set to TL_VOID. Then no memory is allocated, the only way for such a matrix to get memory is by swapping it in from 
     * another matrix by the swap_fields routine. See the example code:
     * \code
     Matrix<double> m(5, 10);
     Matrix<complex<double>> cm( 5,5, TL_VOID); //complex numbers have twice the size
     m.zero()
     m(0,1) = 5;
     swap_fields( m, cm); //now cm has an imaginary value at cm(0,0);
     \endcode
     * The second issue is that for an inplace transform 
     * the input array needs to be padded regardless of whether you want to perform many
     * 1d (linewise) transformations or one 2d transformation.  Therefore there
     * needs to be a way to allocate a padded 2d field. This padded field then 
     * behaves like an unpadded field in all situations except that it is fourier
     * transformable. See the fft.h file for further information on that topic. 
     *
     * @tparam T either double, complex<double> 
     * @tparam P one of TL_NONE(default), TL_DFT_1D or TL_DFT_2D
     */
    template <class T, enum Padding P = TL_NONE>
    class Matrix
    {
      private:
    
      protected:
          //maybe an id (static int id) wouldn't be bad to identify in errors
        const size_t n; 
        const size_t m;
        T *ptr;
        inline void swap( Matrix& rhs);
      public:
        /*! @brief allocates continous memory on the heap
         *
         * @param rows number of rows
         * @param cols number of columns
         * @param allocate determines whether memory should actually be allocated.
         *  Use TL_VOID if matrix should be empty!
         */
        Matrix( const size_t rows, const size_t cols, const bool allocate = true);
        ~Matrix();
        Matrix( const Matrix& src);
        const Matrix& operator=( const Matrix& src);
    
        const size_t rows() const {return n;}
        const size_t cols() const {return m;}
        T* getPtr() const{ return ptr;}
        inline void zero();

        const bool operator!=( const Matrix& rhs) const; 
        const bool operator==( const Matrix& rhs) const {return !((*this != rhs));}
        //hier sollte kein overhead für vektoren liegen weil der Compiler den 
        //Zugriff m(0,i) auf ptr[i] optimieren sollte
        /* !@brief access operator
         *
         * Performs a range check if TL_DEBUG is defined
         * @param i row index
         * @param j column index
         * @return reference value at that location
         */
        inline T& operator()( const size_t i, const size_t j);
        /* !@brief const access operator
         *
         * Performs a range check if TL_DEBUG is defined
         * @param i row index
         * @param j column index
         * @return const value at that location
         */
        inline const T& operator()( const size_t i, const size_t j) const;
    
        template<class T1, enum Padding P1, class T2, enum Padding P2>
        friend void swap_fields( Matrix<T1, P1>& lhs, Matrix<T2, P2>& rhs);
        bool isVoid() const { return (ptr == NULL) ? true : false;}

        friend void permute_fields<T, P>( Matrix& first, Matrix& second, Matrix& third);
        friend std::ostream& operator<< <T, P> ( std::ostream& os, const Matrix& mat); 	//Ausgabe der Matrix 		 			cout << setw(5) << a;
        friend std::istream& operator>> <T, P> ( std::istream& is, Matrix& mat); 
    };
    template <enum Padding P>
    struct TotalNumberOf
    {
        static inline size_t cols( const size_t m){return m;}
        static inline size_t elements( const size_t n, const size_t m){return n*m;}
    };
    
    template <>
    struct TotalNumberOf<TL_DFT_1D>
    {
        static inline size_t cols( const size_t m){ return m - m%2 + 2;}
        static inline size_t elements( const size_t n, const size_t m){return n*(m - m%2 + 2);}
    };
    
    template <>
    struct TotalNumberOf<TL_DFT_2D>
    {
        static inline size_t cols( const size_t m){ return m;}
        static inline size_t elements( const size_t n, const size_t m){return n*(m - m%2 + 2);}
    };

    template<class T1, enum Padding P1, class T2, enum Padding P2>
    void swap_fields( Matrix<T1, P1>& lhs, Matrix<T2, P2>& rhs){

#ifdef TL_DEBUG
        if( TotalNumberOf<P1>::elements(lhs.n, lhs.m)*sizeof(T1) != TotalNumberOf<P2>::elements(rhs.n, rhs.m)*sizeof(T2)) 
            throw Message( "Swap not possible. Sizes not equal\n", ping);
        if( lhs.n != rhs.n)
            throw Message( "Swap not possible! Shape not equal!\n", ping);
#endif
        T1 * ptr = lhs.ptr;
        lhs.ptr = reinterpret_cast<T1*>(rhs.ptr);
        rhs.ptr = reinterpret_cast<T2*>(ptr); 
    }

template <class T, enum Padding P>
Matrix<T, P>::Matrix( const size_t n, const size_t m, const bool allocate): n(n), m(m)
{
    if( allocate){
        ptr = (T*)fftw_malloc( TotalNumberOf<P>::elements(n, m)*sizeof(T));
        if (ptr == NULL) //might be done by fftw_malloc
            throw AllocationError(n, m, ping);
    } else {
        ptr = NULL;
    }
}

template <class T, enum Padding P>
Matrix<T, P>::~Matrix()
{
    if( ptr!= NULL)
        fftw_free( ptr);
}

template <class T, enum Padding P>
Matrix<T, P>::Matrix( const Matrix& src):n(src.n), m(src.m){
    if( src.ptr != NULL){
        ptr = (T*)fftw_malloc( TotalNumberOf<P>::elements(n, m)*sizeof(T));
        if( ptr == NULL) 
            throw AllocationError(n, m, ping);
        memcpy( ptr, src.ptr, TotalNumberOf<P>::elements(n, m)*sizeof(T)); //memcpy here!!!!
    } else {
        ptr = NULL;
    }
}

template <class T, enum Padding P>
const Matrix<T, P>& Matrix<T, P>::operator=( const Matrix& src)
{
#ifdef TL_DEBUG
    if( n!=src.n || m!=src.m)
        throw  Message( "Assignment error! Sizes not equal!", ping);
#endif
    Matrix temp( src);
    swap( temp);
    return *this;
}

template <class T, enum Padding P>
T& Matrix<T, P>::operator()( const size_t i, const size_t j)
{
#ifdef TL_DEBUG
    if( i >= n || j >= m)
        throw BadIndex( i,n, j,m, ping);
    if( ptr == NULL) 
        throw Message( "Trying to access a void matrix!", ping);
#endif
    return ptr[ i*TotalNumberOf<P>::cols(m) + j];
}

template <class T, enum Padding P>
const T&  Matrix<T, P>::operator()( const size_t i, const size_t j) const
{
#ifdef TL_DEBUG
    if( i >= n || j >= m)
        throw BadIndex( i,n, j,m, ping);
    if( ptr == NULL) 
        throw Message( "Trying to access a void matrix!", ping);
#endif
    return ptr[ i*TotalNumberOf<P>::cols(m) + j];
}

template <class T, enum Padding P>
void Matrix<T, P>::zero(){
#ifdef TL_DEBUG
    if( ptr == NULL) 
        throw  Message( "Trying to zero a void matrix!", ping);
#endif
    memset( ptr, 0, TotalNumberOf<P>::elements(n, m)*sizeof( T));
}

template <class T, enum Padding P>
void Matrix<T, P>::swap( Matrix& rhs)
{
#ifdef TL_DEBUG
    if( TotalNumberOf<P>::elements(this->n, this->m)*sizeof(T) != TotalNumberOf<P>::elements(rhs.n, rhs.m)*sizeof(T)) 
        throw Message( "Swap not possible! Sizes not equal!\n", ping);
    if( this->n != rhs.n)
        throw Message( "Swap not possible! Shape not equal!\n", ping);
#endif
    T * ptr = this->ptr;
    this->ptr = reinterpret_cast<T*>(rhs.ptr);
    rhs.ptr = reinterpret_cast<T*>(ptr); 
}

template <class T, enum Padding P>
void permute_fields( Matrix<T, P>& first, Matrix<T, P>& second, Matrix<T, P>& third)
{
#ifdef TL_DEBUG
    if( first.n!=second.n || first.m!=second.m || first.n != third.n || first.m != third.m)
        throw  Message( "Permutation error! Sizes not equal!", ping);
#endif
    T * ptr = first.ptr;
    first.ptr = third.ptr; 
    third.ptr = second.ptr;
    second.ptr = ptr;
}

template <class T, enum Padding P>
std::ostream& operator<< ( std::ostream& os, const Matrix<T, P>& mat)
{
#ifdef TL_DEBUG
    if( mat.ptr == NULL)
        throw  Message( "Trying to output a void matrix!\n", ping);
#endif
     int w = os.width();
     for( size_t i=0; i<mat.n; i++)
     {
         for( size_t j=0; j<mat.m; j++)
         {
             os.width(w); 
             os << mat(i,j)<<" ";	//(Feldbreite gilt immmer nur bis zur nächsten Ausgabe)
         }
         os << "\n";
     }
     return os;
}

template <class T, enum Padding P>
std::istream& operator>>( std::istream& is, Matrix<T, P>& mat)
{
#ifdef TL_DEBUG
    if( mat.ptr == NULL)
        throw  Message( "Trying to write in a void matrix!\n", ping);
#endif
    for( size_t i=0; i<mat.n; i++)
        for( size_t j=0; j<mat.m; j++)
            is >> mat(i, j);
    return is;
}

//certainly optimizable transposition algorithm for inplace Matrix transposition
template <class T>
void transpose( Matrix< T, TL_NONE>& inout, Matrix< T, TL_NONE>& swap)
{
#ifdef TL_DEBUG
    if( swap.isVoid() == false) throw Message("Swap Matrix is not void in transpose algorithm!", ping);
    if( swap.rows() != inout.cols()|| swap.cols() != inout.rows()) throw Message("Swap Matrix has wrong size for transposition!", ping);
#endif
    T temp;
    for (size_t i = 0; i < inout.rows; i ++) {
        for (size_t j = i; j < inout.cols; j ++) {
            temp = inout(i,j);
            inout(i,j) = inout( j,i);
            inout(j,i) = temp;
        }
    }
    swap_fields( inout, swap);
}

    template< class T, enum Padding P>
    const bool Matrix<T,P>::operator!= ( const Matrix& rhs) const
    {
#ifdef TL_DEBUG
        if( n != rhs.n || m != rhs.m)
            throw Message( "Comparison not possible! Sizes not equal!\n", ping);
#endif
        for( size_t i = 0; i < n; i++)
            for( size_t j = 0; j < m; j++)
                if( (*this)( i, j) != rhs( i, j))  
                    return true;
        return false;
    }



}


#endif //_TL_MATRIX_
