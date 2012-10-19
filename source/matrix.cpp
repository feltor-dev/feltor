
#include "numerics/fehler.h"

class AllocationError: public Fehler
{
  private:
    size_t n, m;
  public:
    AllocationError( size_t n, size_t m, const char *d, const int l): Fehler( "Memory couldn't be allocated for: ", d, l), n(n), m(m){}
    void anzeigen() const
    {
        Fehler::anzeigen();
        std::cerr << "# of rows " << n << " # of cols "<<m << std::endl;
    }
};

class BadIndex: public Fehler
{
  private:
    size_t i, j;
    size_t i_max, j_max;
  public:
    BadIndex( size_t i, size_t i_max, size_t j, size_t j_max, const char *d, const int l): Fehler( "Access out of bounds", d, l), i(i), j(j), i_max(i_max), j_max(j_max){}
    void anzeigen() const
    {
        Fehler::anzeigen();
        std::cerr << " in row index " << i << " of " <<i_max<< " rows and column index "<<j <<" of "<<j_max<< "columns\n";
    }
};
    

template <class T>
Matrix<T>::Matrix( const size_t n, const size_t m): n(n), m(m)
{
    ptr = new T* [n];
    ptr[0] = (T*)fftw_malloc( n*m*sizeof(T));
    if (ptr[0] == 0) 
        throw AllocationError(n, m, ping);
    else 
        for (size_t i = 1; i < n; ++i) 
            ptr[i] = ptr[i-1] + m;
}
    



template <class T>
Matrix<T>::Matrix( const size_t n, const size_t m, const T value): n(n), m(m)
{
    //same as in above constructor
    ptr = new T* [n];
    ptr[0] = (T*)fftw_malloc( n*m*sizeof(T));
    if (ptr[0] == 0) 
        throw AllocationError(n, m, ping);
    else 
        for (size_t i = 1; i < n; ++i) 
            ptr[i] = ptr[i-1] + m;
    //assign the value
    for( size_t i=0; i<n; i++)
        for( size_t j=0; j<m; j++)
            ptr[i][j] = value;
}

template <class T>
Matrix<T>::~Matrix()
{
    fftw_free(ptr[0]);
    delete ptr;
}

template <class T>
Matrix<T>::Matrix( const Matrix& src):n(src.n), m(src.m){
    ptr = new T* [n];
    ptr[0] = (T*)fftw_malloc(n*m*sizeof(T));
    if (ptr[0] == 0) 
        throw AllocationError(n, m, ping);
    else 
        for (size_t i = 1; i < n; ++i) 
            ptr[i] = ptr[i-1] + m;
    //copy by value
    for( size_t i=0; i<n; i++)
        for( size_t j=0; j<m; j++)
            ptr[i][j] = src.ptr[i][j];
}



template <class T>
Matrix<T>& Matrix<T>::operator=( const Matrix& src)
{
#ifdef TL_DEBUG
    if( n!=src.n || m!=src.m)
        throw Fehler( "Assignment error! Sizes not equal!", ping);
#endif
    Matrix temp( src);
    swap( temp);
    return *this;
}

template <class T>
Matrix<T>& Matrix<T>::operator=( const T value)
{
    for( size_t i=0; i<n; i++)
        for( size_t j=0; j<m; j++)
            ptr[i][j] = value;
    return *this;
}

template <class T>
T& Matrix<T>::operator()( const size_t i, const size_t j)
{
#ifdef TL_DEBUG
    if( i >= n || j >= m)
        throw BadIndex( i,n, j,m, ping);
#endif
    return ptr[i][j];
}

template <class T>
const T&  Matrix<T>::operator()( const size_t i, const size_t j) const
{
#ifdef TL_DEBUG
    if( i >= n || j >= m)
        throw BadIndex( i,n, j,m, ping);
#endif
    return ptr[i][j];
}

template <class T>
void Matrix<T>::swap( Matrix& rhs)
{
#ifdef TL_DEBUG
    if( n!=rhs.n || m!=rhs.m)
        throw Fehler( "Swap error! Sizes not equal!", ping);
#endif
    T ** ptr = this->ptr;
    this->ptr = rhs.ptr;
    rhs.ptr = ptr; 
}

template <class T>
void permute_cw( Matrix<T>& first, Matrix<T>& second, Matrix<T>& third)
{
#ifdef TL_DEBUG
    if( first.n!=second.n || first.m!=second.m || first.n != third.n || first.m != third.m)
        throw Fehler( "Permutation error! Sizes not equal!", ping);
#endif
    T ** ptr = first.ptr;
    first.ptr = third.ptr; 
    third.ptr = second.ptr;
    second.ptr = ptr;
}

template <class T>
std::ostream& operator<< ( std::ostream& os, const Matrix<T>& mat)
{
     int w = os.width();
     for( size_t i=0; i<mat.n; i++)
     {
         for( size_t j=0; j<mat.m; j++)
         {
             os.width(w); 
             os << mat.ptr[i][j]<<" ";	//(Feldbreite gilt immmer nur bis zur nächsten Ausgabe)
         }
         os << "\n";
     }
     return os;
}
template <class T>
std::istream& operator>>( std::istream& is, Matrix<T>& mat)
{
    for( size_t i=0; i<mat.n; i++)
        for( size_t j=0; j<mat.m; j++)
            is >> mat.ptr[i][j];
    return is;
}





