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
    
    template<class T1, enum Padding P1, class T2, enum Padding P2>
    void swap_fields( Matrix<T1, P1>& lhs, Matrix<T2, P2>& rhs);

    template <class T, enum Padding P>
    void permute( Matrix<T, P>& first, Matrix<T, P>& second, Matrix<T, P>& third);
    
    template <class T, enum Padding P>
    std::ostream& operator<< ( std::ostream& os, const Matrix<T, P>& mat); 	//Ausgabe der Matrix 		 			cout << setw(5) << a;
    template <class T, enum Padding P>
    std::istream& operator>> ( std::istream& is, Matrix<T, P>& mat); 
    
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
    
    /*! @brief Matrix class of constant size that provides fftw compatible 2D fields
     *
     * The logical size of the Matrix cannot be changed once it is constructed. 
     * Nevertheless the memory can be either not allocated, not padded, or padded for 
     * linewise in-place fourier transformation or 2d inplace fourier 
     * transformations. 
     * Padding needs to be enabled for in-place r2c fourier transforms. 
     * (s. fftw documentation )
     *
     * @tparam T either double, complex<double> or fftw_complex
     * @tparam P one of TL_NONE, TL_DFT_1D or TL_DFT_2D
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
      public:
        /*! @brief constructor can allocate a void (i.e. empty matrix)
         *
         *
         * @param rows number of rows
         * @param cols number of columns
         * @param allocate determines whether memory should actually be allocated
         */
        Matrix( const size_t rows, const size_t cols, const bool allocate = true);
        ~Matrix();
        Matrix( const Matrix& src);
        const Matrix& operator=( const Matrix& src);
    
        const size_t rows() const {return n;}
        const size_t cols() const {return m;}
        T* getPtr() const{ return ptr;}
        inline void zero();
        inline void swap( Matrix& rhs);

        const bool operator!=( const Matrix& rhs) const; 
        const bool operator==( const Matrix& rhs) const {return !((*this != rhs));}
        //hier sollte kein overhead für vektoren liegen weil der Compiler den 
        //Zugriff m(0,i) auf ptr[i] optimieren sollte
        inline T& operator()( const size_t i, const size_t j);
        inline const T& operator()( const size_t i, const size_t j) const;
    
        template<class T1, enum Padding P1, class T2, enum Padding P2>
        friend void swap_fields( Matrix<T1, P1>& lhs, Matrix<T2, P2>& rhs);
        bool isVoid() const { return (ptr == NULL) ? true : false;}

        friend void permute<T, P>( Matrix& first, Matrix& second, Matrix& third);
        friend std::ostream& operator<< <T, P> ( std::ostream& os, const Matrix& mat); 	//Ausgabe der Matrix 		 			cout << setw(5) << a;
        friend std::istream& operator>> <T, P> ( std::istream& is, Matrix& mat); 
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
#include "matrix.cpp"
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
