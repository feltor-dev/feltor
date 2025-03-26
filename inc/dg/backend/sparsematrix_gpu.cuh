#pragma once

#include <cusparse.h>
#include <exception>
#include <complex.h>
#include <thrust/complex.h>

namespace dg
{
namespace detail
{
template<class value_type>
inline cudaDataType_t getCudaDataType(){ assert( false && "CUDA Type not supported!\n" ); return CUDA_R_64F; }

template<> inline cudaDataType_t getCudaDataType<int>(){ return CUDA_R_32I;}
template<> inline cudaDataType_t getCudaDataType<float>() { return CUDA_R_32F;}
template<> inline cudaDataType_t getCudaDataType<double>(){ return CUDA_R_64F;}
template<> inline cudaDataType_t getCudaDataType<std::complex<float>>(){ return CUDA_C_32F;}
template<> inline cudaDataType_t getCudaDataType<std::complex<double>>(){ return CUDA_C_64F;}
template<> inline cudaDataType_t getCudaDataType<thrust::complex<float>>(){ return CUDA_C_32F;}
template<> inline cudaDataType_t getCudaDataType<thrust::complex<double>>(){ return CUDA_C_64F;}

template<class value_type>
inline cusparseIndexType_t getCudaIndexType(){ assert( false && "CUDA Type not supported!\n" ); return CUSPARSE_INDEX_32I; }
template<> inline cusparseIndexType_t getCudaIndexType<int>(){ return CUSPARSE_INDEX_32I;}
template<> inline cusparseIndexType_t getCudaIndexType<signed long int>(){ return CUSPARSE_INDEX_64I;}


struct CusparseError : public std::exception
{
    CusparseError( cusparseStatus_t error): m_error( error) {}

    cusparseStatus_t error() const { return m_error;}
    cusparseStatus_t& error() { return m_error;}
    char const* what() const noexcept{
        return cusparseGetErrorString(m_error);}
  private:
    cusparseStatus_t m_error;
};

struct CusparseErrorHandle
{
    CusparseErrorHandle operator=( cusparseStatus_t err)
    {
        CusparseErrorHandle h;
        return h(err);
    }
    CusparseErrorHandle operator()( cusparseStatus_t err)
    {
        if( err != CUSPARSE_STATUS_SUCCESS)
            throw CusparseError( err);
        return *this;
    }
};


// Singleton
// https://stackoverflow.com/questions/1008019/how-do-you-implement-the-singleton-design-pattern
struct CusparseHandle
{
    static CusparseHandle& getInstance()
    {
        // Exists only once even in different translation units
        // https://stackoverflow.com/questions/50609921/singleton-translation-unit-confusion
        static CusparseHandle instance;
        return instance;
    }
    private:
    cusparseHandle_t m_handle;
    CusparseHandle()
    {
        cusparseCreate( &m_handle);
    }
    ~CusparseHandle()
    {
        cusparseDestroy(m_handle);
    }
    public:
    cusparseHandle_t handle() const { return m_handle;}
    CusparseHandle( const CusparseHandle&) = delete;
    void operator=( const CusparseHandle&) = delete;
};


inline bool cusparse_is_initialized = false;

// https://docs.nvidia.com/cuda/cusparse/#cusparsespmv
struct CSRCache_gpu
{
    CSRCache_gpu() = default;
    template<class I, class V>
    CSRCache_gpu(
        size_t num_rows, size_t num_cols, size_t nnz,
        const I* pos , const I* idx, const V* val)
    {
        update( num_rows, num_cols, nnz, pos, idx, val);
    }
    CSRCache_gpu( const CSRCache_gpu& src)
    {
        // Copying makes the cache inactive
    }
    CSRCache_gpu( CSRCache_gpu&& src)
    {
        src.swap(*this);
    }
    CSRCache_gpu& operator=( const CSRCache_gpu& src){
        if( &src != this)
        {
            CSRCache_gpu tmp(src);
            tmp.swap( *this);
        }
        return *this;
    }
    CSRCache_gpu& operator=( CSRCache_gpu&& src){
        CSRCache_gpu tmp( std::move(src));
        tmp.swap(*this);
        return *this;
    }
    ~CSRCache_gpu( )
    {
        if ( m_dBuffer != nullptr)
            cudaFree( m_dBuffer);
        if( m_active)
            cusparseDestroySpMat( m_matA);
    }
    void swap( CSRCache_gpu& src)
    {
        std::swap( m_active, src.m_active);
        std::swap( m_matA, src.m_matA);
        std::swap( m_dBuffer, src.m_dBuffer);
        std::swap( m_bufferSize, src.m_bufferSize);
    }
    void forget() { m_active = false;}
    bool isUpToDate() const { return m_active;}
    template<class I, class V>
    void update(
        size_t num_rows, size_t num_cols, size_t nnz,
        const I* pos , const I* idx, const V* val)
    {
        // cusparse practically only supports uniform data-types (e.g. double Matrix, complex vector is not supported)
        CusparseErrorHandle err;
        cusparseDnVecDescr_t vecX;
        cusparseDnVecDescr_t vecY;
        err = cusparseCreateCsr( &m_matA, num_rows, num_cols, nnz,
            const_cast<I*>(pos), const_cast<I*>(idx), const_cast<V*>(val),
	    getCudaIndexType<I>(), getCudaIndexType<I>(),
            CUSPARSE_INDEX_BASE_ZERO, getCudaDataType<V>() );
        V * x_ptr;
        cudaMalloc( &x_ptr, num_cols*sizeof(V));
        err = cusparseCreateDnVec( &vecX, num_cols, x_ptr, getCudaDataType<V>());
        V * y_ptr;
        cudaMalloc( &y_ptr, num_cols*sizeof(V));
        err = cusparseCreateDnVec( &vecY, num_rows, y_ptr, getCudaDataType<V>());
        size_t bufferSize = 0;
        V alpha = 1, beta = 0;
        //ALG1 is not reproducible, ALG2 is bitwise reproducible
        err = cusparseSpMV_bufferSize( CusparseHandle::getInstance().handle(),
            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, m_matA, vecX, &beta, vecY,
            getCudaDataType<V>(), CUSPARSE_SPMV_CSR_ALG1, &bufferSize);
        // Re-allocate buffer
        if ( m_dBuffer != nullptr)
            cudaFree( m_dBuffer);
        cudaMalloc( &m_dBuffer, bufferSize);

        m_active = true;
#if (CUDART_VERSION >= 12040) // _preprocess only exists as of 12.4 
        err = cusparseSpMV_preprocess( CusparseHandle::getInstance().handle(),
            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, m_matA, vecX, &beta, vecY,
            getCudaDataType<V>(), CUSPARSE_SPMV_CSR_ALG1, m_dBuffer);
        // m_buffer is now associated to m_matA
#endif
        err = cusparseDestroyDnVec( vecX);
        err = cusparseDestroyDnVec( vecY);
        cudaFree( x_ptr);
        cudaFree( y_ptr);
    }
    size_t getBufferSize() { return m_bufferSize;}
    void * getBuffer() { return m_dBuffer;}
    cusparseSpMatDescr_t getSpMat() const { return m_matA;}

    private:
    bool m_active = false;
    cusparseSpMatDescr_t m_matA;
    void * m_dBuffer = nullptr;
    size_t m_bufferSize = 0;
};

//y = alpha A*x + beta y
template<class I, class V, class value_type, class C1, class C2>
void spmv_gpu_kernel(
    CSRCache_gpu& cache,
    size_t A_num_rows, size_t A_num_cols, size_t A_nnz,
    const I* A_pos , const I* A_idx, const V* A_val,
    value_type alpha, value_type beta, const C1* x_ptr, C2* y_ptr
)
{
    CusparseErrorHandle err;
    // Assume here that the descriptors are lightweight structures ...
    cusparseDnVecDescr_t vecX;
    cusparseDnVecDescr_t vecY;
    err = cusparseCreateDnVec( &vecX, A_num_cols, const_cast<C1*>(x_ptr), getCudaDataType<C1>());
    err = cusparseCreateDnVec( &vecY, A_num_rows, y_ptr, getCudaDataType<C2>());

    if( not cache.isUpToDate())
        cache.update<I,V>( A_num_rows, A_num_cols, A_nnz, A_pos, A_idx, A_val);

    err = cusparseSpMV( CusparseHandle::getInstance().handle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, cache.getSpMat(), vecX, &beta, vecY, getCudaDataType<V>(),
            CUSPARSE_SPMV_CSR_ALG1, cache.getBuffer());

    err = cusparseDestroyDnVec( vecX);
    err = cusparseDestroyDnVec( vecY);
}

} // namespace detail
} // namespace dg
