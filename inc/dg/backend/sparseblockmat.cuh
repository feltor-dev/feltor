#pragma once

#include <thrust/device_vector.h>
//#include <cusp/system/cuda/utils.h>
#include "sparseblockmat.h"

namespace dg
{

/**
* @brief Ell Sparse Block Matrix format device version
*
* @ingroup lowlevel
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
        n = src.n, left = src.left, right = src.right;
    }
    
    /**
    * @brief Apply the matrix to a vector
    *
    * @param x input
    * @param y output may not equal input
    */
    template <class deviceContainer>
    void symv(const deviceContainer& x, deviceContainer& y) const;
    /**
    * @brief Display internal data to a stream
    *
    * @param os the output stream
    */
    void display( std::ostream& os = std::cout) const;
    private:
    typedef thrust::device_vector<int> IVec;
    template <class deviceContainer>
    void launch_multiply_kernel(const deviceContainer& x, deviceContainer& y) const;
    
    thrust::device_vector<value_type> data;
    IVec cols_idx, data_idx; 
    int num_rows, num_cols, blocks_per_line;
    int n;
    int left, right;
};


/**
* @brief Coo Sparse Block Matrix format device version
*
* @ingroup lowlevel
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
        n = src.n, left = src.left, right = src.right;
    }
    
    /**
    * @brief Apply the matrix to a vector
    *
    * @param alpha multiplies input
    * @param x input
    * @param beta premultiplies output
    * @param y output may not equal input
    */
    template<class Device>
    void symv(value_type alpha, const Device& x, value_type beta, Device& y) const;
    /**
    * @brief Display internal data to a stream
    *
    * @param os the output stream
    */
    void display(std::ostream& os = std::cout) const;
    private:
    typedef thrust::device_vector<int> IVec;
    template<class Device>
    void launch_multiply_kernel(value_type alpha, const Device& x, value_type beta, Device& y) const;
    
    thrust::device_vector<value_type> data;
    IVec cols_idx, rows_idx, data_idx; 
    int num_rows, num_cols, num_entries;
    int n, left, right;
};

///@cond
template<class value_type>
void EllSparseBlockMatDevice<value_type>::display( std::ostream& os) const
{
    os << "Data array has   "<<data.size()/n/n<<" blocks of size "<<n<<"x"<<n<<"\n";
    os << "num_rows         "<<num_rows<<"\n";
    os << "num_cols         "<<num_cols<<"\n";
    os << "blocks_per_line  "<<blocks_per_line<<"\n";
    os << "n                "<<n<<"\n";
    os << "left             "<<left<<"\n";
    os << "right            "<<right<<"\n";
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
    os << "left             "<<left<<"\n";
    os << "right            "<<right<<"\n";
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
template<class value_type>
template<class DeviceContainer>
inline void EllSparseBlockMatDevice<value_type>::symv( const DeviceContainer& x, DeviceContainer& y) const
{
    launch_multiply_kernel( x,y);
}
template<class value_type>
template<class DeviceContainer>
inline void CooSparseBlockMatDevice<value_type>::symv( value_type alpha, const DeviceContainer& x, value_type beta, DeviceContainer& y) const
{
    launch_multiply_kernel(alpha, x, beta, y);
}


template <class T>
struct MatrixTraits<EllSparseBlockMatDevice<T> >
{
    typedef T value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
template <class T>
struct MatrixTraits<const EllSparseBlockMatDevice<T> >
{
    typedef T value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
template <class T>
struct MatrixTraits<CooSparseBlockMatDevice<T> >
{
    typedef T value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
template <class T>
struct MatrixTraits<const CooSparseBlockMatDevice<T> >
{
    typedef T value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
///@endcond
} //namespace dg

#if THRUST_DEVICE_SYSTEM!=THRUST_DEVICE_SYSTEM_CUDA
#include "sparseblockmat_omp_kernels.h"
#else
#include "sparseblockmat_gpu_kernels.cuh"
#endif


