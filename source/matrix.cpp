

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






