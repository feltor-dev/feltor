#ifndef _TL_MATRIX_
#define _TL_MATRIX_

#include <iostream>
#include <array>
#include <vector>
#include "fftw3.h"
#include "exceptions.h"
#include "padding.h"

namespace toefl{

/*! @brief enum for telling to not allocate memory 
  @ingroup containers
  */
enum Void{ TL_VOID = false //!< Use for not allocating memory in the matrix
            };

// forward declare friend functions of Matrix class
template <class T, enum Padding P>
class Matrix;

template<class T1, enum Padding P1, class T2, enum Padding P2>
void swap_fields( Matrix<T1, P1>& lhs, Matrix<T2, P2>& rhs);
template <class T, enum Padding P>
void permute_fields( Matrix<T, P>& first, Matrix<T, P>& second, Matrix<T, P>& third);
/*! @brief print a Matrix to the given outstream
 *
 * Matrix is printed linewise with a newline after each line.
 * @param os the outstream
 * @param mat the matrix to be printed
 * @return the outstream
 */
template <class T, enum Padding P>
std::ostream& operator<< ( std::ostream& os, const Matrix<T, P>& mat); 
template <class T, enum Padding P>
std::istream& operator>> ( std::istream& is, Matrix<T, P>& mat); 


/*! @brief Matrix class of constant size that provides fftw compatible dynamically allocated 2D fields 
 *
 * @ingroup containers
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
 * allocate flag in the constructor that you can set to TL_VOID. Then no memory is allocated, the only way for such a matrix to get memory is by swap_ping_ it in from 
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
 * @tparam T This class is designed for primitive or trivial datatypes
 * ( the ones  that are "memcpy - able" ). This is important because 
 * we want to swap pointers between different types if the total amount
 * of allocated memory is the same. T may in no case have dynamic allocated
 * memory itself.
 * @tparam P One of the values of the Padding enum,  TL_NONE(default)
 * \note The fftw_complex type does not work as a template paramter. (Mainly due to the comparison and istream and outstream methods)
 * However std::complex<double> should be byte compatible so you can use reinterpret_cast<fftw_complex*>() on the pointer you get with getPtr() to use fftw routines!
 * \note No errors are thrown if the macro TL_DEBUG is not defined
 */
template <class T, enum Padding P = TL_NONE>
class Matrix
{
  public:
    /*! @brief Construct an empty matrix*/
    //Matrix(): n(0), m(0), ptr( NULL){ }
    /*! @brief Allocate continous memory on the heap
     *
     * @param rows logical number of rows (cannot be changed as long as memory is allocated for that object)
     * @param cols logical number of columns (cannot be changed as long as memory is allocated for that object)
     * @param allocate determines whether memory should actually be allocated.
     *  Use TL_VOID if matrix should be empty! Then only the swap_fields function can make
     *  the matrix usable. 
     *  \note The physical size of the actually allocated memory depends on the padding type. 
     *  (In the case that memory is allocated)
     */
    Matrix( const size_t rows, const size_t cols, const bool allocate = true);

    /*! @brief Allocate and assign memory on the heap
     *
     * @param rows logical number of rows (cannot be changed as long as memory is allocated for that object)
     * @param cols logical number of columns (cannot be changed as long as memory is allocated for that object)
     * @param value Use operator= of type T to assign values
     */
    Matrix( const size_t rows, const size_t cols, const T& value);
    /*! @brief Free all allocated memory
     */
    ~Matrix();
    /*! @brief deep copy of an existing Matrix 
     *
     * Copy of 1e6 double takes less than 0.01s
     * @param src the source matrix. 
     * \note 
     * If src is void then so will be this.
     */
    Matrix( const Matrix& src);

    /*! @brief Experimental move constructor in the new standard
     *
     * Does the same thing as the normal copy constructor.
     * @param temporary_src The temporary source matrix
     */
    Matrix( Matrix&& temporary_src);
    /*! @brief Deep assignment 
     *
     * Copy every (including padded) value of the source Matrix
     * to the existing (non void) Matrix with equal numbers of rows and columns.
     * Copy of 1e6 double takes less than 0.01s
     * @param src the right hand side
     * @return this
     * @throws A Message if src is void or doesn't have the same size as this.
     */
    Matrix& operator=( const Matrix& src);

    /*! @brief Deep move assignment
     *
     * Effectively the same as deep assignment but modifies its source
     * @param temporary_src The temporary source matrix
     * @return this
     * @throws A Message if src is void or doesn't have the same size as this
     */
    Matrix& operator=( Matrix&& temporary_src);

    /*! @brief Allocate memory for void matrices
     *
     * This function uses the current values of n and m to 
     * allocate the right amount of memory! 
     * @throws A Message when called on non-void Matrices.
     */
    void allocate(){ allocate_();}

    /*! @brief number of rows
     *
     * Return the  number of rows the object manages (the one you specified in the constructor)
     * even if 
     * no memory (or more in the padded case) is allocated. 
     * This number doesn't change as long as memory is allocated for that object.
     * @return number of columns
     */
    const size_t rows() const {return n;}
    /*! @brief number of columns
     *
     * Return the number of columns the object manages (the one you specified in the constructor), even if 
     * no memory is allocated. 
     * This number doesn't change as long as memory is allocated for that object.
     * @return number of columns
     */
    const size_t cols() const {return m;}
    /*! @brief get the address of the first element
     *
     * Replaces the use of &m(0,0) which is kind of clumsy!
     * @return pointer to allocated memory
     * \attention DO NOT DELETE THIS POINTER!! 
     *  This class manages the memory it allocates by itself.
     */
    T* getPtr() { return ptr;}

    /*! @brief Get the address of the first element for const reference
     *
     * Replaces the use of &m(0,0) which is kind of clumsy!
     * @return read only pointer to allocated memory
     * \attention DO NOT DELETE THIS POINTER!! 
     *  This class manages the memory it allocates by itself.
     */
    T const * getPtr()const {return ptr;}

    /*! @brief Copy the data linearly and without padding to a std vector
     *
     * @return newly instantiated vector holding a copy of the matrix data
     */
    std::vector<T> copy() const
    {
        std::vector<T> vec( n*m);
        for( unsigned i=0; i<n; i++)
            for( unsigned j=0; j<m; j++)
                vec[i*m+j] = ptr[i*TotalNumberOf<P>::columns(m) + j];
        return vec;
    }

    /*! @brief uses operator= to set memory to 0
     *
     * takes less than 0.01s for 1e6 elements
     */
    inline void zero();
    /*! @brief checks whether matrix is empty i.e. no memory is allocated
     *
     * @return true if memory isn't allocated 
     */
    bool isVoid() const { return (ptr == NULL) ? true : false;}

    /*! @brief two Matrices are considered equal if elements are equal
     *
     * @param rhs Matrix to be compared to this
     * @return true if rhs does not equal this
     */
    const bool operator!=( const Matrix& rhs) const; 
    /*! @brief two Matrices are considered equal if elements are equal
     *
     * @param rhs Matrix to be compared to this
     * @return true if rhs equals this
     */
    const bool operator==( const Matrix& rhs) const {return !((*this != rhs));}
    //hier sollte kein overhead für vektoren liegen weil der Compiler den 
    //Zugriff m(0,i) auf ptr[i] optimieren sollte

    /*! @brief access operator
     *
     * Performs a range check if TL_DEBUG is defined
     * @param i row index
     * @param j column index
     * @return reference to value at that location
     */
    inline T& operator()( const size_t i, const size_t j);

    /*! @brief const access operator
     *
     * Performs a range check if TL_DEBUG is defined
     * @param i row index
     * @param j column index
     * @return const value at that location
     */
    inline const T& operator()( const size_t i, const size_t j) const;

    // friend functions
    /*! @brief swap memories of two equally sized matrices of arbitrary type
     *
     * Performs a range check if TL_DEBUG is defined. 
     * The sizes of the actually allocated memory, which depend on padding and the value type, have to be equal.
     * @param lhs changes memory with rhs
     * @param rhs changes memory with lhs
     */
    template<class T1, enum Padding P1, class T2, enum Padding P2>
    friend void swap_fields( Matrix<T1, P1>& lhs, Matrix<T2, P2>& rhs);
    /*! @brief permute memory of matrices with the same type
     *
     * @param first contains third on output
     * @param second contains first on output
     * @param third contains second on output
     */
    friend void permute_fields<T, P>( Matrix& first, Matrix& second, Matrix& third);
    /*! @brief print a Matrix to the given outstream
     *
     * Matrix is printed linewise with a newline after each line.
     * @param os the outstream
     * @param mat the matrix to be printed
     * @return the outstream
     */
    friend std::ostream& operator<< <T,P> ( std::ostream& os, const Matrix& mat);
    /*! @brief read values into a Matrix from given istream
     *
     * The values are filled linewise into the matrix. Values are seperated by 
     * whitespace charakters. (i.e. newline, blank, etc)
     * @param is the istream
     * @param mat the Matrix into which the values are written
     * @return the istream
     */
    friend std::istream& operator>> <T, P> ( std::istream& is, Matrix& mat); 
  protected:
      //maybe an id (static int id) wouldn't be bad to identify in errors
  private:
    void allocate_(); //normal allocate function of Matrix class (called by constructors)
    size_t n; //!< # of columns
    size_t m; //!< # of rows
    T *ptr; //!< pointer to allocated memory
};

/////////////////////////////////////DEFINITIONS///////////////////////////////////////////////////////////////////////////////
template<class T1, enum Padding P1, class T2, enum Padding P2>
void swap_fields( Matrix<T1, P1>& lhs, Matrix<T2, P2>& rhs)
{
#ifdef TL_DEBUG
    if( TotalNumberOf<P1>::elements(lhs.n, lhs.m)*sizeof(T1) != TotalNumberOf<P2>::elements(rhs.n, rhs.m)*sizeof(T2)) 
        throw Message( "Swap not possible. Sizes not equal\n", _ping_);
#endif
    //test for self swap not necessary (not an error)
    T1 * ptr = lhs.ptr;
    lhs.ptr = reinterpret_cast<T1*>(rhs.ptr);
    rhs.ptr = reinterpret_cast<T2*>(ptr); 
}

template <class T, enum Padding P>
Matrix<T, P>::Matrix( const size_t n, const size_t m, const bool allocate): n(n), m(m), ptr(NULL)
{
#ifdef TL_DEBUG
    if( n==0|| m==0)
        throw Message("Use TL_VOID to not allocate any memory!\n", _ping_);
#endif
    if( allocate)
        allocate_();
}

template< class T, enum Padding P>
Matrix<T,P>::Matrix( const size_t n, const size_t m, const T& value):n(n),m(m),ptr(NULL)
{
#ifdef TL_DEBUG
    if( n==0|| m==0)
        throw Message("Use TL_VOID to not allocate any memory!\n", _ping_);
#endif
    allocate_();
    for( unsigned i=0; i<TotalNumberOf<P>::elements(n,m); i++)
        ptr[i] = value;
}



template <class T, enum Padding P>
Matrix<T, P>::~Matrix()
{
    if( ptr!= NULL/*NULL*/)
        fftw_free( ptr);
}

template <class T, enum Padding P>
Matrix<T, P>::Matrix( const Matrix& src):n(src.n), m(src.m), ptr(NULL){
    if( src.ptr != NULL)
    {
        allocate_();
        for( size_t i =0; i < TotalNumberOf<P>::elements(n, m); i++)
            ptr[i] = src.ptr[i];
    }
}
template <class T, enum Padding P>
Matrix<T, P>::Matrix(  Matrix&& src):n(src.n), m(src.m), ptr(src.ptr){
    src.ptr = NULL;
}

template <class T, enum Padding P>
Matrix<T, P>& Matrix<T, P>::operator=( const Matrix& src)
{
    if( &src != this)
    {
#ifdef TL_DEBUG
        if( n!=src.n || m!=src.m)
            throw  Message( "Assignment error! Sizes not equal!", _ping_);
        if( ptr == NULL || src.ptr == NULL)
            throw Message( "Assigning to or from a void matrix!", _ping_);
#endif
        for( size_t i =0; i < TotalNumberOf<P>::elements(n, m); i++)
            ptr[i] = src.ptr[i];
    }
    return *this;
}

template <class T, enum Padding P>
Matrix<T, P>& Matrix<T, P>::operator=( Matrix&& src)
{
    if( &src != this)
    {
#ifdef TL_DEBUG
        if( n!=src.n || m!=src.m)
            throw  Message( "Assignment error! Sizes not equal!", _ping_);
        if( ptr == NULL || src.ptr == NULL)
            throw Message( "Assigning to or from a void matrix!", _ping_);
#endif
        ptr = src.ptr; 
        src.ptr = NULL;
    }
    return *this;
}

template <class T, enum Padding P>
void Matrix<T, P>::allocate_()
{
    if( ptr == NULL) //allocate only if matrix is void 
    {
        ptr = (T*)fftw_malloc( TotalNumberOf<P>::elements(n, m)*sizeof(T));
        if( ptr == NULL) 
            throw AllocationError(n, m, _ping_);
    }
    else 
        throw Message( "Memory already exists!", _ping_);
}



template <class T, enum Padding P>
T& Matrix<T, P>::operator()( const size_t i, const size_t j)
{
#ifdef TL_DEBUG
    if( i >= n || j >= m)
        throw BadIndex( i,n, j,m, _ping_);
    if( ptr == NULL) 
        throw Message( "Trying to access a void matrix!", _ping_);
#endif
    return ptr[ i*TotalNumberOf<P>::columns(m) + j];
}

template <class T, enum Padding P>
const T&  Matrix<T, P>::operator()( const size_t i, const size_t j) const
{
#ifdef TL_DEBUG
    if( i >= n || j >= m)
        throw BadIndex( i,n, j,m, _ping_);
    if( ptr == NULL) 
        throw Message( "Trying to access a void matrix!", _ping_);
#endif
    return ptr[ i*TotalNumberOf<P>::columns(m) + j];
}

template <class T, enum Padding P>
void Matrix<T, P>::zero(){
#ifdef TL_DEBUG
    if( ptr == NULL) 
        throw  Message( "Trying to zero a void matrix!", _ping_);
#endif
    for( size_t i =0; i < TotalNumberOf<P>::elements(n, m); i++)
        ptr[i] = (T)0;
    //memset( ptr, 0, TotalNumberOf<P>::elements(n, m)*sizeof( T));
}

template <class T, enum Padding P>
void permute_fields( Matrix<T, P>& first, Matrix<T, P>& second, Matrix<T, P>& third)
{
#ifdef TL_DEBUG
    if( first.n!=second.n || first.m!=second.m || first.n != third.n || first.m != third.m)
        throw  Message( "Permutation error! Sizes not equal!", _ping_);
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
        throw  Message( "Trying to output a void matrix!\n", _ping_);
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
        throw  Message( "Trying to write in a void matrix!\n", _ping_);
#endif
    for( size_t i=0; i<mat.n; i++)
        for( size_t j=0; j<mat.m; j++)
            is >> mat(i, j);
    return is;
}

template< class T, enum Padding P>
const bool Matrix<T,P>::operator!= ( const Matrix& rhs) const
{
#ifdef TL_DEBUG
    if( n != rhs.n || m != rhs.m)
        throw Message( "Comparison not possible! Sizes not equal!\n", _ping_);
#endif
    for( size_t i = 0; i < n; i++)
        for( size_t j = 0; j < m; j++)
            if( (*this)( i, j) != rhs( i, j) )  
                return true;
    return false;
}



} //namespace toefl


#endif //_TL_MATRIX_
