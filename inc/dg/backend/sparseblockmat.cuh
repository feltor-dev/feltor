#pragma once

#include <thrust/device_vector.h>
//#include <cusp/system/cuda/utils.h>
#include "sparseblockmat.h"

namespace dg
{

/**
* @brief Ell Sparse Block Matrix format device version
*
* @ingroup sparsematrix
* This class holds a copy of a EllSparseBlockMat on the device, which may 
be gpu or omp depending on the THRUST_DEVICE_SYSTEM macro. It can be applied
to device vectors and does the same thing as the host version
*/
template<class value_type>
struct EllSparseBlockMatDevice
{
    EllSparseBlockMatDevice(){}
    /**
    * @brief Allocate storage
    *
    * A device matrix has to be constructed from a host matrix. It simply
        copies all internal data of the host matrix to the device
        @param src  source on the host
    */
    template< class OtherValueType>
    EllSparseBlockMatDevice( const EllSparseBlockMat<OtherValueType>& src)
    {
        data = src.data;
        cols_idx = src.cols_idx, data_idx = src.data_idx;
        num_rows = src.num_rows, num_cols = src.num_cols, blocks_per_line = src.blocks_per_line;
        n = src.n, left_size = src.left_size, right_size = src.right_size;
        right_range = src.right_range;
        m_trivial = is_trivial();
        m_directional = is_directional();
    }
    /**
    * @brief Display internal data to a stream
    *
    * @param os the output stream
    */
    void display( std::ostream& os = std::cout) const;
    
    /**
    * @brief Apply the matrix to a vector
    *
    * same as symv( 1., x,0.,y);
    * @tparam Vector a valid Vector type
    * @param x input
    * @param y output may not equal input
    */
    template<class Vector>
    void symv(const Vector& x, Vector& y) const {symv( 1., x, 0., y);}
    /**
    * @brief Apply the matrix to a vector
    * \f[  y= \alpha M x + \beta y\f]
    * @tparam Vector a valid Vector type
    * @param alpha multiplies input
    * @param x input
    * @param beta premultiplies output
    * @param y output may not alias input
    */
    template<class Vector>
    void symv(value_type alpha, const Vector& x, value_type beta, Vector& y) const
    {
        symv( get_vector_category<Vector>(), get_execution_policy<Vector>(), alpha, x,beta,y);
    }
    private:
    template<class Vector>
    void symv(VectorVectorTag, CudaTag, value_type alpha, const Vector& x, value_type beta, Vector& y) const
    {
        for(unsigned i=0; i<x.size(); i++)
            symv( get_vector_category<typename Vector::value_type>(), get_execution_policy<typename Vector::value_type>(), alpha, x[i], beta, y[i]);
    }
    template<class Vector>
    void symv(SharedVectorTag, CudaTag, value_type alpha, const Vector& x, value_type beta, Vector& y) const;
#ifdef _OPENMP
    template<class Vector>
    void symv(VectorVectorTag, OmpTag, value_type alpha, const Vector& x, value_type beta, Vector& y) const
    {
        if( !omp_in_parallel())
        {
            #pragma omp parallel
            {
                for(unsigned i=0; i<x.size(); i++)
                    symv( get_vector_category<typename Vector::value_type>(), OmpTag(), alpha, x[i], beta, y[i]);
            }
        }
        else
            for(unsigned i=0; i<x.size(); i++)
                symv( get_vector_category<typename Vector::value_type>(), OmpTag(), alpha, x[i], beta, y[i]);
    }
    template<class Vector>
    void symv(SharedVectorTag, OmpTag, value_type alpha, const Vector& x, value_type beta, Vector& y) const;
#endif //_OPENMP
    bool is_trivial() const;
    bool is_directional() const;
    using IVec = thrust::device_vector<int>;
    void launch_multiply_kernel(value_type alpha, const value_type* x, value_type beta, value_type* y) const;
    
    thrust::device_vector<value_type> data;
    IVec cols_idx, data_idx; 
    int num_rows, num_cols, blocks_per_line;
    int n;
    int left_size, right_size;
    IVec right_range;
    bool m_trivial, m_directional;
};


/**
* @brief Coo Sparse Block Matrix format device version
*
* @ingroup sparsematrix
* This class holds a copy of a CooSparseBlockMat on the device, which may 
be gpu or omp depending on the THRUST_DEVICE_SYSTEM macro. It does the same thing as the host version with the difference that it applies to device vectors.
*/
template<class value_type>
struct CooSparseBlockMatDevice
{
    CooSparseBlockMatDevice(){}
    /**
    * @brief Allocate storage
    *
    * A device matrix has to be constructed from a host matrix. It simply
        copies all internal data of the host matrix to the device
        @param src  source on the host
    */
    template<class OtherValueType>
    CooSparseBlockMatDevice( const CooSparseBlockMat<OtherValueType>& src)
    {
        data = src.data;
        rows_idx = src.rows_idx, cols_idx = src.cols_idx, data_idx = src.data_idx;
        num_rows = src.num_rows, num_cols = src.num_cols, num_entries = src.num_entries;
        n = src.n, left_size = src.left_size, right_size = src.right_size;
    }
    /**
    * @brief Display internal data to a stream
    *
    * @param os the output stream
    */
    void display(std::ostream& os = std::cout) const;
    
    /**
    * @brief Apply the matrix to a vector
    *
    * @param alpha multiplies input
    * @param x input
    * @param beta premultiplies output
    * @param y output may not equal input
    */
    template<class Vector>
    void symv(value_type alpha, const Vector& x, value_type beta, Vector& y) const
    {
        symv( get_vector_category<Vector>(), alpha, x,beta,y);
    }
    private:
    template<class Vector>
    void symv(VectorVectorTag, value_type alpha, const Vector& x, value_type beta, Vector& y) const
    {
        for(unsigned i=0; i<x.size(); i++)
            symv( get_vector_category<typename Vector::value_type>(), get_execution_policy<typename Vector::value_type>(), alpha, x[i], beta, y[i]);
    }
    template<class Vector>
    void symv(SharedVectorTag, value_type alpha, const Vector& x, value_type beta, Vector& y) const;
    using IVec = thrust::device_vector<int>;

    void launch_multiply_kernel(value_type alpha, const value_type* x, value_type beta, value_type* y) const;
    
    thrust::device_vector<value_type> data;
    IVec cols_idx, rows_idx, data_idx; 
    int num_rows, num_cols, num_entries;
    int n, left_size, right_size;
};

///@cond
template<class value_type>
template<class Vector>
inline void EllSparseBlockMatDevice<value_type>::symv(SharedVectorTag, CudaTag, value_type alpha, const Vector& x, value_type beta, Vector& y) const
{
    if( y.size() != (unsigned)num_rows*n*left_size*right_size) {
        throw Error( Message(_ping_)<<"y has the wrong size "<<(unsigned)y.size()<<" and not "<<(unsigned)num_rows*n*left_size*right_size);
    }
    if( x.size() != (unsigned)num_cols*n*left_size*right_size) {
        throw Error( Message(_ping_)<<"x has the wrong size "<<(unsigned)x.size()<<" and not "<<(unsigned)num_cols*n*left_size*right_size);
    }
    const value_type * x_ptr = thrust::raw_pointer_cast(x.data());
          value_type * y_ptr = thrust::raw_pointer_cast(y.data());
    launch_multiply_kernel( alpha, x_ptr, beta, y_ptr);
}
#ifdef _OPENMP
template<class value_type>
template<class Vector>
inline void EllSparseBlockMatDevice<value_type>::symv(SharedVectorTag, OmpTag, value_type alpha, const Vector& x, value_type beta, Vector& y) const
{
    if( !omp_in_parallel())
    {
        if( y.size() != (unsigned)num_rows*n*left_size*right_size) {
            throw Error( Message(_ping_)<<"y has the wrong size "<<(unsigned)y.size()<<" and not "<<(unsigned)num_rows*n*left_size*right_size);
        }
        if( x.size() != (unsigned)num_cols*n*left_size*right_size) {
            throw Error( Message(_ping_)<<"x has the wrong size "<<(unsigned)x.size()<<" and not "<<(unsigned)num_cols*n*left_size*right_size);
        }
        #pragma omp parallel
        {
            const value_type * x_ptr = thrust::raw_pointer_cast(x.data());
                  value_type * y_ptr = thrust::raw_pointer_cast(y.data());
            launch_multiply_kernel(alpha, x_ptr, beta, y_ptr);
        }
        return;
    }
    const value_type * x_ptr = thrust::raw_pointer_cast(x.data());
          value_type * y_ptr = thrust::raw_pointer_cast(y.data());
    launch_multiply_kernel(alpha, x_ptr, beta, y_ptr);
}
#endif //_OPENMP
template<class value_type>
bool EllSparseBlockMatDevice<value_type>::is_trivial() const{
    bool trivial = true;
    for( int i=1; i<num_rows-1; i++)
        for( int d=0; d<3; d++)
        {
            if( data_idx[i*3+d] != d) trivial = false;
            //if( cols_idx[i*3+d] != i+d-1) trivial = false;
        }
    return trivial;
}
template<class value_type>
bool EllSparseBlockMatDevice<value_type>::is_directional() const{
    bool forward = true, backward = true;
    for( int i=1; i<num_rows-1; i++)
        for( int d=0; d<2; d++)
        {
            if( data_idx[i*2+d] != d) forward = backward = false;
            //if( cols_idx[i*2+d] != i+d-1) backward = false;
            //if( cols_idx[i*2+d] != i+d) forward = false;
        }
    return forward  || backward;

    //bool forward = true, backward = true;
    //for( int i=1; i<num_rows-1; i++)
    //    for( int d=0; d<2; d++)
    //    {
    //        if( data_idx[i*2+d] != d) forward = backward = false;
    //        //if( cols_idx[i*2+d] != i+d-1) backward = false;
    //        if( cols_idx[i*2+d] != cols_idx[i*2]+d) forward = backward = false;
    //    }
}

template<class value_type>
template<class Vector>
inline void CooSparseBlockMatDevice<value_type>::symv(SharedVectorTag, value_type alpha, const Vector& x, value_type beta, Vector& y) const
{
    static_assert( std::is_same<get_execution_policy<Vector>, OmpTag>::value ||
                   std::is_same<get_execution_policy<Vector>, CudaTag>::value, "Either OmpTag or CudTag required");
    if( y.size() != (unsigned)num_rows*n*left_size*right_size) {
        throw Error( Message(_ping_)<<"y has the wrong size "<<(unsigned)y.size()<<" and not "<<(unsigned)num_rows*n*left_size*right_size);
    }
    if( x.size() != (unsigned)num_cols*n*left_size*right_size) {
        throw Error( Message(_ping_)<<"x has the wrong size "<<(unsigned)x.size()<<" and not "<<(unsigned)num_cols*n*left_size*right_size);
    }
    const value_type * x_ptr = thrust::raw_pointer_cast(x.data());
          value_type * y_ptr = thrust::raw_pointer_cast(y.data());
    launch_multiply_kernel( alpha, x_ptr, beta, y_ptr);
}

template<class value_type>
void EllSparseBlockMatDevice<value_type>::display( std::ostream& os) const
{
    os << "Data array has   "<<data.size()/n/n<<" blocks of size "<<n<<"x"<<n<<"\n";
    os << "num_rows         "<<num_rows<<"\n";
    os << "num_cols         "<<num_cols<<"\n";
    os << "blocks_per_line  "<<blocks_per_line<<"\n";
    os << "n                "<<n<<"\n";
    os << "left_size             "<<left_size<<"\n";
    os << "right_size            "<<right_size<<"\n";
    os << "right_range_0         "<<right_range[0]<<"\n";
    os << "right_range_1         "<<right_range[1]<<"\n";
    os << " Columns: \n";
    for( int i=0; i<num_rows; i++)
    {
        for( int d=0; d<blocks_per_line; d++)
            os << cols_idx[i*blocks_per_line + d] <<" ";
        os << "\n";
    }
    os << "\n Data: \n";
    for( int i=0; i<num_rows; i++)
    {
        for( int d=0; d<blocks_per_line; d++)
            os << data_idx[i*blocks_per_line + d] <<" ";
        os << "\n";
    }
    os << std::endl;
    
}
template<class value_type>
void CooSparseBlockMatDevice<value_type>::display( std::ostream& os) const
{
    os << "Data array has   "<<data.size()/n/n<<" blocks of size "<<n<<"x"<<n<<"\n";
    os << "num_rows         "<<num_rows<<"\n";
    os << "num_cols         "<<num_cols<<"\n";
    os << "num_entries      "<<num_entries<<"\n";
    os << "n                "<<n<<"\n";
    os << "left_size             "<<left_size<<"\n";
    os << "right_size            "<<right_size<<"\n";
    os << " Columns: \n";
    for( int i=0; i<num_entries; i++)
        os << cols_idx[i] <<" ";
    os << "\n Rows: \n";
    for( int i=0; i<num_entries; i++)
        os << rows_idx[i] <<" ";
    os << "\n Data: \n";
    for( int i=0; i<num_entries; i++)
        os << data_idx[i] <<" ";
    os << std::endl;
    
}


template <class T>
struct MatrixTraits<EllSparseBlockMatDevice<T> >
{
    using value_type      = T;
    using matrix_category = SelfMadeMatrixTag;
};
template <class T>
struct MatrixTraits<const EllSparseBlockMatDevice<T> >
{
    using value_type      = T;
    using matrix_category = SelfMadeMatrixTag;
};
template <class T>
struct MatrixTraits<CooSparseBlockMatDevice<T> >
{
    using value_type      = T;
    using matrix_category = SelfMadeMatrixTag;
};
template <class T>
struct MatrixTraits<const CooSparseBlockMatDevice<T> >
{
    using value_type      = T;
    using matrix_category = SelfMadeMatrixTag;
};
///@endcond
} //namespace dg
#if THRUST_DEVICE_SYSTEM!=THRUST_DEVICE_SYSTEM_CUDA
#include "sparseblockmat_omp_kernels.h"
#else
#include "sparseblockmat_gpu_kernels.cuh"
#endif
