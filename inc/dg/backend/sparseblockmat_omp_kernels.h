
namespace dg{

// multiply kernel
template<class value_type>
void ell_multiply_kernel(
         const value_type* data, const int* cols_idx, const int* data_idx, 
         const int num_rows, const int num_cols, const int blocks_per_line,
         const int n, 
         const int left, const int right, 
         const value_type* x, value_type *y
         )
{
    //simplest implementation
#pragma omp parallel for collapse(4)
    for( int s=0; s<left; s++)
    for( int i=0; i<num_rows; i++)
    for( int k=0; k<n; k++)
    for( int j=0; j<right; j++)
    {
        int I = ((s*num_rows + i)*n+k)*right+j;
        y[I] =0;
        for( int d=0; d<blocks_per_line; d++)
        {
            int B = (data_idx[i*blocks_per_line+d]*n+k)*n;
            int J = (s*num_cols+cols_idx[i*blocks_per_line+d])*n;
            for( int q=0; q<n; q++) //multiplication-loop
                y[I] += data[ B+q]*
                    x[(J+q)*right+j];
        }
    }
}

// multiply kernel n=3, 3 blocks per line
template<class value_type>
void ell_multiply_kernel33(
         const value_type* data, const int* cols_idx, const int* data_idx, 
         const int num_rows, const int num_cols, 
         const int left, const int right, 
         const value_type* x, value_type *y
         )
{
#pragma omp parallel for collapse(2)
    for( int s=0; s<left; s++)
    for( int i=0; i<num_rows; i++)
    for( int k=0; k<3; k++)
    for( int j=0; j<right; j++)
    {
        value_type temp = 0;
        int B0 = (data_idx[i*3+0]*3+k)*3;
        int B1 = (data_idx[i*3+1]*3+k)*3;
        int B2 = (data_idx[i*3+2]*3+k)*3;
        int J0 = (s*num_cols+cols_idx[i*3+0])*3;
        int J1 = (s*num_cols+cols_idx[i*3+1])*3;
        int J2 = (s*num_cols+cols_idx[i*3+2])*3;
        temp +=data[ B0+0]* x[(J0+0)*right+j];
        temp +=data[ B0+1]* x[(J0+1)*right+j];
        temp +=data[ B0+2]* x[(J0+2)*right+j];

        temp +=data[ B1+0]* x[(J1+0)*right+j];
        temp +=data[ B1+1]* x[(J1+1)*right+j];
        temp +=data[ B1+2]* x[(J1+2)*right+j];

        temp +=data[ B2+0]* x[(J2+0)*right+j];
        temp +=data[ B2+1]* x[(J2+1)*right+j];
        temp +=data[ B2+2]* x[(J2+2)*right+j];
        int I = ((s*num_rows + i)*3+k)*right+j;
        y[I]=temp;
    }
}

// multiply kernel, n=3, 2 blocks per line
template<class value_type>
void ell_multiply_kernel32(
         const value_type* data, const int* cols_idx, const int* data_idx, 
         const int num_rows, const int num_cols,
         const int left, const int right, 
         const value_type* x, value_type *y
         )
{
#pragma omp parallel for collapse(2)
    for( int s=0; s<left; s++)
    for( int i=0; i<num_rows; i++)
    for( int k=0; k<3; k++)
    for( int j=0; j<right; j++)
    {
        value_type temp = 0;
        int B0 = (data_idx[i*2+0]*3+k)*3;
        int B1 = (data_idx[i*2+1]*3+k)*3;
        int J0 = (s*num_cols+cols_idx[i*2+0])*3;
        int J1 = (s*num_cols+cols_idx[i*2+1])*3;
        temp +=data[ B0+0]* x[(J0+0)*right+j];
        temp +=data[ B0+1]* x[(J0+1)*right+j];
        temp +=data[ B0+2]* x[(J0+2)*right+j];
        temp +=data[ B1+0]* x[(J1+0)*right+j];
        temp +=data[ B1+1]* x[(J1+1)*right+j];
        temp +=data[ B1+2]* x[(J1+2)*right+j];
        int I = ((s*num_rows + i)*3+k)*right+j;
        y[I]=temp;
    }

}
// multiply kernel, n=3, 3 blocks per line, right = 1
template<class value_type>
void ell_multiply_kernel33x(
         const value_type* data, const int* cols_idx, const int* data_idx, 
         const int num_rows, const int num_cols,
         const int left,
         const value_type* x, value_type *y
         )
{
    bool trivial = true;
    for( int i=1; i<num_rows-1; i++)
        for( int d=0; d<3; d++)
        {
            if( data_idx[i*3+d] != d) trivial = false;
            if( cols_idx[i*3+d] != i+d-1) trivial = false;
        }
    if( trivial)
    {
#pragma omp parallel for
    for( int s=0; s<left; s++)
    for( int i=0; i<1; i++)
    for( int k=0; k<3; k++)
    {
        value_type temp = 0;
        int B0 = (data_idx[i*3+0]*3+k)*3;
        int J0 = (s*num_cols+cols_idx[i*3+0])*3;
        temp +=data[ B0+0]* x[(J0+0)];
        temp +=data[ B0+1]* x[(J0+1)];
        temp +=data[ B0+2]* x[(J0+2)];

        int B1 = (data_idx[i*3+1]*3+k)*3;
        int J1 = (s*num_cols+cols_idx[i*3+1])*3;
        temp +=data[ B1+0]* x[(J1+0)];
        temp +=data[ B1+1]* x[(J1+1)];
        temp +=data[ B1+2]* x[(J1+2)];

        int B2 = (data_idx[i*3+2]*3+k)*3;
        int J2 = (s*num_cols+cols_idx[i*3+2])*3;
        temp +=data[ B2+0]* x[(J2+0)];
        temp +=data[ B2+1]* x[(J2+1)];
        temp +=data[ B2+2]* x[(J2+2)];
        int I = ((s*num_rows + i)*3+k);
        y[I]=temp;
    }

#pragma omp parallel for// collapse(2)
    for( int s=0; s<left; s++)
    for( int i=1; i<num_rows-1; i++)
    for( int k=0; k<3; k++)
    {
        value_type temp = 0;
        int B0 = (0*3+k)*3;
        int J0 = (s*num_cols+i+0-1)*3;
        temp +=data[ B0+0]* x[(J0+0)];
        temp +=data[ B0+1]* x[(J0+1)];
        temp +=data[ B0+2]* x[(J0+2)];
        int B1 = (1*3+k)*3;
        int J1 = (s*num_cols+i+1-1)*3;
        temp +=data[ B1+0]* x[(J1+0)];
        temp +=data[ B1+1]* x[(J1+1)];
        temp +=data[ B1+2]* x[(J1+2)];
        int B2 = (2*3+k)*3;
        int J2 = (s*num_cols+i+2-1)*3;
        temp +=data[ B2+0]* x[(J2+0)];
        temp +=data[ B2+1]* x[(J2+1)];
        temp +=data[ B2+2]* x[(J2+2)];

        int I = ((s*num_rows + i)*3+k);
        y[I]=temp;
    }
#pragma omp parallel for
    for( int s=0; s<left; s++)
    for( int i=num_rows-1; i<num_rows; i++)
    for( int k=0; k<3; k++)
    {
        value_type temp = 0;
        int B0 = (data_idx[i*3+0]*3+k)*3;
        int J0 = (s*num_cols+cols_idx[i*3+0])*3;
        temp +=data[ B0+0]* x[(J0+0)];
        temp +=data[ B0+1]* x[(J0+1)];
        temp +=data[ B0+2]* x[(J0+2)];

        int B1 = (data_idx[i*3+1]*3+k)*3;
        int J1 = (s*num_cols+cols_idx[i*3+1])*3;
        temp +=data[ B1+0]* x[(J1+0)];
        temp +=data[ B1+1]* x[(J1+1)];
        temp +=data[ B1+2]* x[(J1+2)];

        int B2 = (data_idx[i*3+2]*3+k)*3;
        int J2 = (s*num_cols+cols_idx[i*3+2])*3;
        temp +=data[ B2+0]* x[(J2+0)];
        temp +=data[ B2+1]* x[(J2+1)];
        temp +=data[ B2+2]* x[(J2+2)];

        int I = ((s*num_rows + i)*3+k);
        y[I]=temp;
    }
    }
    else 
        ell_multiply_kernel( data, cols_idx, data_idx, num_rows, num_cols, 3, 3, left, 1, x, y);
}

// multiply kernel, n=3, 2 blocks per line, right = 1
template<class value_type>
void ell_multiply_kernel32x(
         const value_type* data, const int* cols_idx, const int* data_idx, 
         const int num_rows, const int num_cols,
         const int left, 
         const value_type* x, value_type *y
         )
{
#pragma omp parallel for// collapse(2)
    for( int s=0; s<left; s++)
    for( int i=0; i<num_rows; i++)
    for( int k=0; k<3; k++)
    {
        value_type temp = 0;
        int B0 = (data_idx[i*2+0]*3+k)*3;
        int B1 = (data_idx[i*2+1]*3+k)*3;
        int J0 = (s*num_cols+cols_idx[i*2+0])*3;
        int J1 = (s*num_cols+cols_idx[i*2+1])*3;
        temp +=data[ B0+0]* x[(J0+0)];
        temp +=data[ B0+1]* x[(J0+1)];
        temp +=data[ B0+2]* x[(J0+2)];
        temp +=data[ B1+0]* x[(J1+0)];
        temp +=data[ B1+1]* x[(J1+1)];
        temp +=data[ B1+2]* x[(J1+2)];
        int I = ((s*num_rows + i)*3+k);
        y[I]=temp;
    }

}

template<class value_type>
template<class DeviceContainer>
void EllSparseBlockMatDevice<value_type>::launch_multiply_kernel( const DeviceContainer& x, DeviceContainer& y) const
{
    assert( y.size() == (unsigned)num_rows*n*left*right);
    assert( x.size() == (unsigned)num_cols*n*left*right);

    const value_type* data_ptr = thrust::raw_pointer_cast( &data[0]);
    const int* cols_ptr = thrust::raw_pointer_cast( &cols_idx[0]);
    const int* block_ptr = thrust::raw_pointer_cast( &data_idx[0]);
    const value_type* x_ptr = thrust::raw_pointer_cast( &x[0]);
    value_type* y_ptr = thrust::raw_pointer_cast( &y[0]);
    if( n == 3)
    {
        if( blocks_per_line == 3)
        {
            if( right == 1)
                ell_multiply_kernel33x<value_type> ( data_ptr, cols_ptr, block_ptr, num_rows, num_cols, left, x_ptr,y_ptr);
            else
                ell_multiply_kernel33<value_type> ( data_ptr, cols_ptr, block_ptr, num_rows, num_cols, left, right, x_ptr,y_ptr);
        }
        else if( blocks_per_line == 2)
        {
            if( right == 1)
                ell_multiply_kernel32x<value_type> ( data_ptr, cols_ptr, block_ptr, num_rows, num_cols, left, x_ptr,y_ptr);
            else
                ell_multiply_kernel32<value_type> ( data_ptr, cols_ptr, block_ptr, num_rows, num_cols, left, right, x_ptr,y_ptr);
        }
        else
            ell_multiply_kernel<value_type>( 
                data_ptr, cols_ptr, block_ptr, num_rows, num_cols, blocks_per_line, 3, left, right, x_ptr,y_ptr);
    }
    else
        ell_multiply_kernel<value_type>  ( 
            data_ptr, cols_ptr, block_ptr, num_rows, num_cols, blocks_per_line, n, left, right, x_ptr,y_ptr);
}

template<class value_type>
template<class DeviceContainer>
void CooSparseBlockMatDevice<value_type>::launch_multiply_kernel( value_type alpha, const DeviceContainer& x, value_type beta, DeviceContainer& y) const
{
    assert( y.size() == (unsigned)num_rows*n*left*right);
    assert( x.size() == (unsigned)num_cols*n*left*right);
    assert( beta == 1);

    for( int i=0; i<num_entries; i++)
#pragma omp parallel for collapse(3)
    for( int s=0; s<left; s++)
    for( int k=0; k<n; k++)
    for( int j=0; j<right; j++)
    {
        int I = ((s*num_rows + rows_idx[i])*n+k)*right+j;
        value_type temp=0;
        for( int q=0; q<n; q++) //multiplication-loop
            temp+= data[ (data_idx[i]*n + k)*n+q]*
                x[((s*num_cols + cols_idx[i])*n+q)*right+j];
        y[I] += alpha*temp;
    }
}

}//namespace dg
