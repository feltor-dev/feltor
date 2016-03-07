
namespace dg{
template<class value_type>
 void ell_multiply_kernel(
         const thrust::device_vector<value_type>& data, const thrust::device_vector<value_type>& cols_idx, const thrust::device_vector<int> data_idx, 
         const int num_rows, const int num_cols, const int blocks_per_line,
         const int n, const int size,
         const int left, const int right, 
         const thrust::device_vector<value_type>& x, thrust::device_vector<value_type> y
         )
{

}

template<class value_type>
 void ell_multiply_kernel33(
         const thrust::device_vector<value_type>& data, const thrust::device_vector<value_type>& cols_idx, const thrust::device_vector<int> data_idx, 
         const int num_rows, const int num_cols,
         const int n, const int size,
         const int left, const int right, 
         const thrust::device_vector<value_type>& x, thrust::device_vector<value_type> y
         )
{

}
template<class value_type>
 void ell_multiply_kernel33x(
         const thrust::device_vector<value_type>& data, const thrust::device_vector<value_type>& cols_idx, const thrust::device_vector<int> data_idx, 
         const int num_rows, const int num_cols, 
         const int n, const int size,
         const int left,
         const thrust::device_vector<value_type>& x, thrust::device_vector<value_type> y
         )
{

}
template<class value_type>
 void ell_multiply_kernel32(
         const thrust::device_vector<value_type>& data, const thrust::device_vector<value_type>& cols_idx, const thrust::device_vector<int> data_idx, 
         const int num_rows, const int num_cols, 
         const int n, const int size,
         const int left, const int right, 
         const thrust::device_vector<value_type>& x, thrust::device_vector<value_type> y
         )
{

}
template<class value_type>
 void ell_multiply_kernel32x(
         const thrust::device_vector<value_type>& data, const thrust::device_vector<value_type>& cols_idx, const thrust::device_vector<int> data_idx, 
         const int num_rows, const int num_cols, 
         const int n, const int size,
         const int left, 
         const thrust::device_vector<value_type>& x, thrust::device_vector<value_type> y
         )
{

}

template<class value_type>
void EllSparseBlockMatDevice<value_type>::launch_multiply_kernel( const thrust::device_vector<value_type>& x, thrust::device_vector<value_type>& y) const
{
    if( thrust::detail::is_same<value_type, float>::value )
        std::cout << "Value type is float! "<<std::endl;
    else
        std::cout << "Value type is double! "<<std::endl;

    
    assert( y.size() == (unsigned)num_rows*n*left*right);
    assert( x.size() == (unsigned)num_cols*n*left*right);

    int offset[blocks_per_line];
    for( int d=0; d<blocks_per_line; d++)
        offset[d] = cols_idx[blocks_per_line+d]-1;
if(right==1) //alle dx Ableitungen
{
#pragma omp parallel for 
    for( int s=0; s<left; s++)
    for( int i=0; i<1; i++)
    for( int k=0; k<n; k++)
    {
        value_type temp=0;
        for( int d=0; d<blocks_per_line; d++)
            for( int q=0; q<n; q++) //multiplication-loop
                temp += data[ (data_idx[i*blocks_per_line+d]*n + k)*n+q]*
                    x[((s*num_cols + cols_idx[i*blocks_per_line+d])*n+q)];
        y[(s*num_rows+i)*n+k]=temp;
    }
#pragma omp parallel for 
    for( int s=0; s<left; s++)
    for( int i=1; i<num_rows-1; i++)
    for( int k=0; k<n; k++)
    {
        value_type temp=0;
        for( int d=0; d<blocks_per_line; d++)
            for( int q=0; q<n; q++) //multiplication-loop
                temp+=data[(d*n + k)*n+q]*x[((s*num_cols + i+cols_idx[i*blocks_per_line + d])*n+q)];
        y[(s*num_rows+i)*n+k]=temp;
    }
#pragma omp parallel for 
    for( int s=0; s<left; s++)
    for( int i=num_rows-1; i<num_rows; i++)
    for( int k=0; k<n; k++)
    {
        value_type temp=0;
        for( int d=0; d<blocks_per_line; d++)
            for( int q=0; q<n; q++) //multiplication-loop
                temp += data[ (data_idx[i*blocks_per_line+d]*n + k)*n+q]*
                    x[((s*num_cols + cols_idx[i*blocks_per_line+d])*n+q)];
        y[(s*num_rows+i)*n+k]=temp;
    }
    return;
} //if right==1



#pragma omp parallel for
    for( unsigned  i=0; i<y.size(); i++)
    {
        y[i] =0;
    }
#pragma omp parallel for collapse(4)
    for( int s=0; s<left; s++)
    for( int i=0; i<1; i++)
    for( int k=0; k<n; k++)
    for( int j=0; j<right; j++)
    {
        value_type temp=0;
        int I = ((s*num_rows + i)*n+k)*right+j;
        for( int d=0; d<blocks_per_line; d++)
            for( int q=0; q<n; q++) //multiplication-loop
                temp += data[ (data_idx[i*blocks_per_line+d]*n + k)*n+q]*
                    x[((s*num_cols + cols_idx[i*blocks_per_line+d])*n+q)*right+j];
        y[I] = temp;
    }

if(left > 1)
{
    for( int d=0; d<blocks_per_line; d++)
    {
#pragma omp parallel for collapse(2)
    for( int s=0; s<left; s++)
    for( int i=1; i<num_rows-1; i++)
    {
        int J = i+offset[d];
        for( int k=0; k<n; k++)
        for( int j=0; j<right; j++)
        {
            int I = ((s*num_rows + i)*n+k)*right+j;
            for( int q=0; q<n; q++) //multiplication-loop
                y[I] += data[ (d*n+k)*n+q]*x[((s*num_cols + J)*n+q)*right+j];
            //value_type temp=0;
            //for( int d=0; d<blocks_per_line; d++)
            //    for( int q=0; q<n; q++) //multiplication-loop
            //        temp+=data[(d*n + k)*n+q]*
            //        x[((s*num_cols + i + offset[d])*n+q)*right+j];
            //y[((s*num_rows+i)*n+k)*right+j]=temp;
        }
    }
    }
}
else
{
    for( int d=0; d<blocks_per_line; d++)
    {
#pragma omp parallel for 
    for( int i=1; i<num_rows-1; i++)
    {
        int J = i+offset[d];
        for( int k=0; k<n; k++)
        for( int j=0; j<right; j++)
        {
            int I = (i*n+k)*right+j;
            for( int q=0; q<n; q++) //multiplication-loop
                y[I] += data[ (d*n+k)*n+q]*x[(J*n+q)*right+j];
        }
    }
    }
} //endif left > 1
#pragma omp parallel for collapse(4)
    for( int s=0; s<left; s++)
    for( int i=num_rows-1; i<num_rows; i++)
    for( int k=0; k<n; k++)
    for( int j=0; j<right; j++)
    {
        int I = ((s*num_rows + i)*n+k)*right+j;
        value_type temp=0; 
        for( int d=0; d<blocks_per_line; d++)
        for( int q=0; q<n; q++) //multiplication-loop
            temp += data[ (data_idx[i*blocks_per_line+d]*n + k)*n+q]*
                x[((s*num_cols + cols_idx[i*blocks_per_line+d])*n+q)*right+j];
        y[I] = temp; //do not add here because of the case num_rows==1
    }
}

template<class value_type>
void EllSparseBlockMatDevice<value_type>::launch_multiply_kernel3( const thrust::device_vector<value_type>& x, thrust::device_vector<value_type>& y) const
{
    std::cout << "In specialized version for n=3\n";
    
    assert( y.size() == (unsigned)num_rows*3*left*right);
    assert( x.size() == (unsigned)num_cols*3*left*right);
    int offset[blocks_per_line];
    for( int d=0; d<blocks_per_line; d++)
        offset[d] = cols_idx[blocks_per_line+d]-1;
if(right==1) //alle dx Ableitungen
{
#pragma omp parallel for 
    for( int s=0; s<left; s++)
    for( int i=0; i<1; i++)
    for( int k=0; k<3; k++)
    {
        value_type temp=0;
        for( int d=0; d<blocks_per_line; d++)
        {
            temp += data[ (data_idx[i*blocks_per_line+d]*3 + k)*3+0]*
                x[((s*num_cols + cols_idx[i*blocks_per_line+d])*3+0)];
            temp += data[ (data_idx[i*blocks_per_line+d]*3 + k)*3+1]*
                x[((s*num_cols + cols_idx[i*blocks_per_line+d])*3+1)];
            temp += data[ (data_idx[i*blocks_per_line+d]*3 + k)*3+2]*
                x[((s*num_cols + cols_idx[i*blocks_per_line+d])*3+2)];
        }
        y[(s*num_rows+i)*3+k]=temp;
    }
#pragma omp parallel for 
    for( int s=0; s<left; s++)
    for( int i=1; i<num_rows-1; i++)
    for( int k=0; k<3; k++)
    {
        value_type temp=0;
        for( int d=0; d<blocks_per_line; d++)
        {
            temp+=data[(d*3 + k)*3+0]*x[((s*num_cols + i+offset[d])*3+0)];
            temp+=data[(d*3 + k)*3+1]*x[((s*num_cols + i+offset[d])*3+1)];
            temp+=data[(d*3 + k)*3+2]*x[((s*num_cols + i+offset[d])*3+2)];
        }
        y[(s*num_rows+i)*3+k]=temp;
    }
#pragma omp parallel for 
    for( int s=0; s<left; s++)
    for( int i=num_rows-1; i<num_rows; i++)
    for( int k=0; k<3; k++)
    {
        value_type temp=0;
        for( int d=0; d<blocks_per_line; d++)
        {
            temp += data[ (data_idx[i*blocks_per_line+d]*3 + k)*3+0]*
                x[((s*num_cols + cols_idx[i*blocks_per_line+d])*3+0)];
            temp += data[ (data_idx[i*blocks_per_line+d]*3 + k)*3+1]*
                x[((s*num_cols + cols_idx[i*blocks_per_line+d])*3+1)];
            temp += data[ (data_idx[i*blocks_per_line+d]*3 + k)*3+2]*
                x[((s*num_cols + cols_idx[i*blocks_per_line+d])*3+2)];
        }
        y[(s*num_rows+i)*3+k]=temp;
    }
    return;
} //if right==1



#pragma omp parallel for
    for( unsigned  i=0; i<y.size(); i++)
    {
        y[i] =0;
    }
#pragma omp parallel for collapse(4)
    for( int s=0; s<left; s++)
    for( int i=0; i<1; i++)
    for( int k=0; k<3; k++)
    for( int j=0; j<right; j++)
    {
        value_type temp=0;
        int I = ((s*num_rows + i)*3+k)*right+j;
        for( int d=0; d<blocks_per_line; d++)
            for( int q=0; q<3; q++) //multiplication-loop
                temp += data[ (data_idx[i*blocks_per_line+d]*3 + k)*3+q]*
                    x[((s*num_cols + cols_idx[i*blocks_per_line+d])*3+q)*right+j];
        y[I] = temp;
    }

if(left > 1)
{
    for( int d=0; d<blocks_per_line; d++)
    {
#pragma omp parallel for collapse(2)
    for( int s=0; s<left; s++)
    for( int i=1; i<num_rows-1; i++)
    {
        int J = i+offset[d];
        for( int k=0; k<3; k++)
        for( int j=0; j<right; j++)
        {
            int I = ((s*num_rows + i)*3+k)*right+j;
            for( int q=0; q<3; q++) //multiplication-loop
                y[I] += data[ (d*3+k)*3+q]*x[((s*num_cols + J)*3+q)*right+j];
            //value_type temp=0;
            //for( int d=0; d<blocks_per_line; d++)
            //    for( int q=0; q<n; q++) //multiplication-loop
            //        temp+=data[(d*n + k)*n+q]*
            //        x[((s*num_cols + i + offset[d])*n+q)*right+j];
            //y[((s*num_rows+i)*n+k)*right+j]=temp;
        }
    }
    }
}
else
{
    for( int d=0; d<blocks_per_line; d++)
    {
#pragma omp parallel for 
    for( int i=1; i<num_rows-1; i++)
    {
        int J = i+offset[d];
        for( int k=0; k<3; k++)
        for( int j=0; j<right; j++)
        {
            int I = (i*3+k)*right+j;
            for( int q=0; q<3; q++) //multiplication-loop
                y[I] += data[ (d*3+k)*3+q]*x[(J*3+q)*right+j];
        }
    }
    }
} //endif left > 1
#pragma omp parallel for collapse(4)
    for( int s=0; s<left; s++)
    for( int i=num_rows-1; i<num_rows; i++)
    for( int k=0; k<3; k++)
    for( int j=0; j<right; j++)
    {
        int I = ((s*num_rows + i)*3+k)*right+j;
        value_type temp=0; 
        for( int d=0; d<blocks_per_line; d++)
        for( int q=0; q<3; q++) //multiplication-loop
            temp += data[ (data_idx[i*blocks_per_line+d]*3 + k)*3+q]*
                x[((s*num_cols + cols_idx[i*blocks_per_line+d])*3+q)*right+j];
        y[I] = temp; //do not add here because of the case num_rows==1
    }
}

template<class value_type>
void CooSparseBlockMatDevice<value_type>::launch_multiply_kernel( value_type alpha, const thrust::device_vector<value_type>& x, value_type beta, thrust::device_vector<value_type>& y) const
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
