#ifndef _TL_MATRIX_
#define _TL_MATRIX_

#include <iostream>
#include <array>
//#include <cstring> //memcpy and memset
#include "fftw3.h"
#include "exceptions.h"
#include "padding.h"

namespace toefl{


    enum Allocate{ TL_VOID = false};

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
    //template< typename T>
    //void transpose( Matrix< T, TL_NONE>& inout, Matrix< T, TL_NONE>& swap);
    
    
    /*! @brief Matrix class of constant size that provides fftw compatible dynamically allocated 2D fields
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
     * @tparam T tested with double and std::complex<double> 
     * @tparam P one of TL_NONE(default), TL_DFT_1D, TL_DFT_2D, TL_DST_DFT, TL_DFT_DFT
     * \note The fftw_complex type does not work as a template paramter. (Mainly due to the comparison and istream and outstream methods)
     * However std::complex<double> should be byte compatible so you can use reinterpret_cast<fftw_complex*>() on the pointer you get with getPtr() to use fftw routines!
     * \note No errors are thrown if the macro TL_DEBUG is not defined
     * \note In order to use arrays of Matrices there are the two member functions
     *  resize and allocate that are usable for void matrices!
     */
    template <class T, enum Padding P = TL_NONE>
    class Matrix
    {
      private:
      protected:
        void allocate_(); //normal allocate function of Matrix class (called by constructors)
        virtual void allocate_virtual(){ allocate_();}
        void resize_( const size_t new_rows, const size_t new_cols){
            if( ptr == nullptr)
                n = new_rows, m = new_cols;
            else
                throw Message( "Non void Matrix may not be resized!", ping);
        }
        virtual void resize_virtual( const size_t new_rows, const size_t new_cols ){ resize_( new_rows, new_cols);}
          //maybe an id (static int id) wouldn't be bad to identify in errors
        size_t n; //!< # of columns
        size_t m; //!< # of rows
        T *ptr; //!< pointer to allocated memory
        //inline void swap( Matrix& rhs);
      public:
        /*! @brief Construct an empty matrix*/
        Matrix(): n(0), m(0), ptr( nullptr){ }
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
         * @param cols logical number of columns (cannot be changed as long as memory is allocated for that object)
         * @param allocate determines whether memory should actually be allocated.
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
         * \note throws an error if src is void or doesn't have the same size.
         * If src is void then so will be this.
         */
        Matrix( const Matrix& src);
        /*! @brief Deep assignment 
         *
         * Copy every (including padded) value of the source Matrix
         * to the existing (non void) Matrix with equal numbers of rows and columns.
         * Copy of 1e6 double takes less than 0.01s
         * @param src the right hand side
         * @return this
         * \note throws an error if src is void or doesn't have the same size.
         */
        const Matrix& operator=( const Matrix& src);

        /*! @brief Allocate memory for void matrices
         *
         * This function uses the current values of n and m to 
         * allocate the right amount of memory! It
         * is useful in connection with arrays of Matrices.
         * Throws an exception when called on non-void Matrices.
         */
        void allocate(){ allocate_virtual();}

        /*! @brief resize void matrices
         *
         * No new memory is allocated! Just usable for void matrices!
         * @param new_rows new number of rows
         * @param new_cols new number of columns
         */
        void resize( const size_t new_rows, const size_t new_cols)
        {
            resize_virtual( new_rows, new_cols);
        }

        /*! @brief Resize and allocate memory for void matrices
         *
         * @param new_rows new number of rows
         * @param new_cols new number of columns
         * @param value Value the elements are initialized to using operator= of type T
         */
        void allocate( const size_t new_rows, const size_t new_cols, const T& value = T())
        {
            resize_virtual( new_rows, new_cols);
            allocate_virtual();
            for( size_t i=0; i < TotalNumberOf<P>::elements(n, m); i++)
                ptr[i] = value;
        }
    
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
         * \note DO NOT DELETE THIS POINTER!! This class manages the memory it allocates by itself.
         */
        T* getPtr(){ return ptr;}
        /*! @brief uses operator= to set memory to 0
         *
         * takes less than 0.01s for 1e6 elements
         */
        inline void zero();
        /*! @brief checks whether matrix is empty i.e. no memory is allocated
         *
         * @return true if memory isn't allocated 
         */
        bool isVoid() const { return (ptr == nullptr) ? true : false;}

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
         * @param first becomes second
         * @param second becomes third
         * @param third becomes first
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
    };

    /////////////////////////////////////DEFINITIONS///////////////////////////////////////////////////////////////////////////////
    template<class T1, enum Padding P1, class T2, enum Padding P2>
    void swap_fields( Matrix<T1, P1>& lhs, Matrix<T2, P2>& rhs)
    {
#ifdef TL_DEBUG
        if( TotalNumberOf<P1>::elements(lhs.n, lhs.m)*sizeof(T1) != TotalNumberOf<P2>::elements(rhs.n, rhs.m)*sizeof(T2)) 
            throw Message( "Swap not possible. Sizes not equal\n", ping);
#endif
        //test for self swap not necessary (not an error)
        T1 * ptr = lhs.ptr;
        lhs.ptr = reinterpret_cast<T1*>(rhs.ptr);
        rhs.ptr = reinterpret_cast<T2*>(ptr); 
    }

    template <class T, enum Padding P>
    Matrix<T, P>::Matrix( const size_t n, const size_t m, const bool allocate): n(n), m(m), ptr(nullptr)
    {
    #ifdef TL_DEBUG
        if( n==0|| m==0)
            throw Message("Use TL_VOID to not allocate any memory!\n", ping);
    #endif
        if( allocate)
            allocate_();
    }

    template< class T, enum Padding P>
    Matrix<T,P>::Matrix( const size_t n, const size_t m, const T& value):n(n),m(m),ptr(nullptr)
    {
    #ifdef TL_DEBUG
        if( n==0|| m==0)
            throw Message("Use TL_VOID to not allocate any memory!\n", ping);
    #endif
        allocate_();
        for( unsigned i=0; i<TotalNumberOf<P>::elements(n,m); i++)
            ptr[i] = value;
    }


    
    template <class T, enum Padding P>
    Matrix<T, P>::~Matrix()
    {
        if( ptr!= nullptr)
            fftw_free( ptr);
    }
    
    template <class T, enum Padding P>
    Matrix<T, P>::Matrix( const Matrix& src):n(src.n), m(src.m), ptr(nullptr){
        if( src.ptr != nullptr)
        {
            allocate_();
            for( size_t i =0; i < TotalNumberOf<P>::elements(n, m); i++)
                ptr[i] = src.ptr[i];
        }
    }
    
    template <class T, enum Padding P>
    const Matrix<T, P>& Matrix<T, P>::operator=( const Matrix& src)
    {
        if( &src != this)
        {
    #ifdef TL_DEBUG
            if( n!=src.n || m!=src.m)
                throw  Message( "Assignment error! Sizes not equal!", ping);
            if( ptr == nullptr || src.ptr == nullptr)
                throw Message( "Assigning to or from a void matrix!", ping);
    #endif
            for( size_t i =0; i < TotalNumberOf<P>::elements(n, m); i++)
                ptr[i] = src.ptr[i];
        }
        return *this;
    }

    template <class T, enum Padding P>
    void Matrix<T, P>::allocate_()
    {
        if( ptr == nullptr) //allocate only if matrix is void 
        {
            ptr = (T*)fftw_malloc( TotalNumberOf<P>::elements(n, m)*sizeof(T));
            if( ptr == nullptr) 
                throw AllocationError(n, m, ping);
        }
        else 
            throw Message( "Memory already exists!", ping);
    }


    
    template <class T, enum Padding P>
    T& Matrix<T, P>::operator()( const size_t i, const size_t j)
    {
    #ifdef TL_DEBUG
        if( i >= n || j >= m)
            throw BadIndex( i,n, j,m, ping);
        if( ptr == nullptr) 
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
        if( ptr == nullptr) 
            throw Message( "Trying to access a void matrix!", ping);
    #endif
        return ptr[ i*TotalNumberOf<P>::cols(m) + j];
    }
    
    template <class T, enum Padding P>
    void Matrix<T, P>::zero(){
    #ifdef TL_DEBUG
        if( ptr == nullptr) 
            throw  Message( "Trying to zero a void matrix!", ping);
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
        if( mat.ptr == nullptr)
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
        if( mat.ptr == nullptr)
            throw  Message( "Trying to write in a void matrix!\n", ping);
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
            throw Message( "Comparison not possible! Sizes not equal!\n", ping);
#endif
        for( size_t i = 0; i < n; i++)
            for( size_t j = 0; j < m; j++)
                if( (*this)( i, j) != rhs( i, j) )  
                    return true;
        return false;
    }


    template < class T, enum Padding P = TL_NONE>
    inline std::array< Matrix<T,P>,2> void_matrix_array( const size_t rows, const size_t cols)
    {
        return std::array< Matrix<T,P>,2>{{ Matrix<T,P>(rows, cols, TL_VOID), Matrix<T,P>(rows, cols, TL_VOID)}};
    }
    template < class T, enum Padding P = TL_NONE>
    inline std::array< Matrix<T,P>,2> matrix_array( const size_t rows, const size_t cols)
    {
        return std::array< Matrix<T,P>,2>{{ Matrix<T,P>(rows, cols), Matrix<T,P>(rows, cols)}};
    }
    






}


#endif //_TL_MATRIX_
