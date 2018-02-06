#include "vector_categories.h"

namespace dg{

// multiply kernel
template<class value_type>
void ell_multiply_kernel( value_type alpha, value_type beta,
         const value_type * RESTRICT data, const int * RESTRICT cols_idx, const int * RESTRICT data_idx, 
         const int num_rows, const int num_cols, const int blocks_per_line,
         const int n, 
         const int left_size, const int right_size, 
         const int * RESTRICT right_range,
         const value_type * RESTRICT x, value_type * RESTRICT y
         )
{
#pragma omp parallel for collapse(2)
    for( int s=0; s<left_size; s++)
    for( int i=0; i<num_rows; i++)
    {
        int J[blocks_per_line];
        for( int d=0; d<blocks_per_line; d++)
            J[d] = (s*num_cols+cols_idx[i*blocks_per_line+d])*n;
        for( int k=0; k<n; k++)
        {
            int B[blocks_per_line];
            for( int d=0; d<blocks_per_line; d++)
                B[d] = (data_idx[i*blocks_per_line+d]*n+k)*n;
            for( int j=right_range[0]; j<right_range[1]; j++)
            {
                value_type temp = 0;
                for( int d=0; d<blocks_per_line; d++)
                    for( int q=0; q<n; q++) //multiplication-loop
                        temp += data[ B[d]+q]*
                            x[(J[d]+q)*right_size+j];
                int I = ((s*num_rows + i)*n+k)*right_size+j;
                y[I] = alpha*temp + beta*y[I];
            }
        }
    }
}

template<class value_type>
void ell_multiply_kernel3( value_type alpha, value_type beta,
         const value_type * RESTRICT data, const int * RESTRICT cols_idx, const int * RESTRICT data_idx, 
         const int num_rows, const int num_cols, const int blocks_per_line,
         const int left_size, const int right_size, 
         const int * RESTRICT right_range,
         const value_type * RESTRICT x, value_type * RESTRICT y
         )
{
    if (blocks_per_line == 1)
    {
        if( right_size==1)
        {
            #pragma omp parallel for 
            for( int s=0; s<left_size; s++)
            for( int i=0; i<num_rows; i++)
            {
                int J = (s*num_cols+cols_idx[i])*3;
                for( int k=0; k<3; k++)
                {
                    value_type temp = 0;
                    int B = (data_idx[i]*3+k)*3;
                    temp +=data[ B+0]* x[(J+0)];
                    temp +=data[ B+1]* x[(J+1)];
                    temp +=data[ B+2]* x[(J+2)];
                    int I = (s*num_rows + i)*3+k;
                    y[I] = alpha*temp + beta*y[I];
                }
            }
        }
        else
        {
            #pragma omp parallel for collapse(2)
            for( int s=0; s<left_size; s++)
            for( int i=0; i<num_rows; i++)
            for( int k=0; k<3; k++)
            {
                int B = (data_idx[i]*3+k)*3;
                int J = (s*num_cols+cols_idx[i])*3;
                for( int j=right_range[0]; j<right_range[1]; j++)
                {
                    value_type temp = 0;
                    temp +=data[ B+0]* x[(J+0)*right_size+j];
                    temp +=data[ B+1]* x[(J+1)*right_size+j];
                    temp +=data[ B+2]* x[(J+2)*right_size+j];
                    int I = ((s*num_rows + i)*3+k)*right_size+j;
                    y[I] = alpha*temp + beta*y[I];
                }
            }
        }
    }
    else
    {
#pragma omp parallel for collapse(2)
        for( int s=0; s<left_size; s++)
        for( int i=0; i<num_rows; i++)
        for( int k=0; k<3; k++)
        {
            int B[blocks_per_line], J[blocks_per_line];
            for( int d=0; d<blocks_per_line; d++)
            {
                B[d] = (data_idx[i*blocks_per_line+d]*3+k)*3;
                J[d] = (s*num_cols+cols_idx[i*blocks_per_line+d])*3;
            }
            for( int j=right_range[0]; j<right_range[1]; j++)
            {
                value_type temp = 0;
                for( int d=0; d<blocks_per_line; d++)
                {
                    temp +=data[ B[d]+0]* x[(J[d]+0)*right_size+j];
                    temp +=data[ B[d]+1]* x[(J[d]+1)*right_size+j];
                    temp +=data[ B[d]+2]* x[(J[d]+2)*right_size+j];
                }
                int I = ((s*num_rows + i)*3+k)*right_size+j;
                y[I] = alpha*temp + beta*y[I];
            }
        }
    }
}

// multiply kernel n=3, 3 blocks per line
template<class value_type>
void ell_multiply_kernel33( value_type alpha, value_type beta,
         const value_type * RESTRICT data, const int * RESTRICT cols_idx, const int * RESTRICT data_idx, 
         const int num_rows, const int num_cols, 
         const int left_size, const int right_size, 
         const int* right_range,
         const value_type * RESTRICT x, value_type * RESTRICT y
         )
{
    bool trivial = true;
    for( int i=1; i<num_rows-1; i++)
        for( int d=0; d<3; d++)
        {
            if( data_idx[i*3+d] != d) trivial = false;
            //if( cols_idx[i*3+d] != i+d-1) trivial = false;
        }
    if( trivial)
    {
	#pragma omp parallel for collapse(2)
	for( int s=0; s<left_size; s++)
	{
		for( int i=0; i<num_rows; i++)
		{
            //int J0 = (i==0 || i==num_rows-1)?(s*num_cols+cols_idx[i*3+0])*3:(s*num_cols+i+0-1)*3;
            //int J1 = (i==0 || i==num_rows-1)?(s*num_cols+cols_idx[i*3+1])*3:(s*num_cols+i+1-1)*3;
            //int J2 = (i==0 || i==num_rows-1)?(s*num_cols+cols_idx[i*3+2])*3:(s*num_cols+i+2-1)*3;
            int J0 = (i==0 || i==num_rows-1)?(s*num_cols+cols_idx[i*3+0])*3:(s*num_cols+cols_idx[i*3+0])*3;
            int J1 = (i==0 || i==num_rows-1)?(s*num_cols+cols_idx[i*3+1])*3:(s*num_cols+cols_idx[i*3+1])*3;
            int J2 = (i==0 || i==num_rows-1)?(s*num_cols+cols_idx[i*3+2])*3:(s*num_cols+cols_idx[i*3+2])*3;
			for( int k=0; k<3; k++)
			{
				int B0 = (i==0 || i==num_rows-1)?(data_idx[i*3+0]*3+k)*3:(0*3+k)*3;
                int B1 = (i==0 || i==num_rows-1)?(data_idx[i*3+1]*3+k)*3:(1*3+k)*3;
                int B2 = (i==0 || i==num_rows-1)?(data_idx[i*3+2]*3+k)*3:(2*3+k)*3;
#if defined(__INTEL_COMPILER)
				#pragma vector nontemporal(y)
#endif //__INTEL_COMPILER
				for( int j=right_range[0]; j<right_range[1]; j++)
				{
				    value_type temp = 0;
				    temp +=data[ B0+0]* x[(J0+0)*right_size+j];
				    temp +=data[ B0+1]* x[(J0+1)*right_size+j];
				    temp +=data[ B0+2]* x[(J0+2)*right_size+j];
				
				    temp +=data[ B1+0]* x[(J1+0)*right_size+j];
				    temp +=data[ B1+1]* x[(J1+1)*right_size+j];
				    temp +=data[ B1+2]* x[(J1+2)*right_size+j];
				
				    temp +=data[ B2+0]* x[(J2+0)*right_size+j];
				    temp +=data[ B2+1]* x[(J2+1)*right_size+j];
				    temp +=data[ B2+2]* x[(J2+2)*right_size+j];
				    int I = ((s*num_rows + i)*3+k)*right_size+j;
                    y[I]=alpha*temp+beta*y[I];
				}
			}
		}
	}
    }
    else 
        ell_multiply_kernel3(alpha,beta, data, cols_idx, data_idx, num_rows, num_cols, 3, left_size, right_size, right_range,  x, y);
}

// multiply kernel, n=3, 2 blocks per line
template<class value_type>
void ell_multiply_kernel32( value_type alpha, value_type beta,
         const value_type * RESTRICT data, const int * RESTRICT cols_idx, const int * RESTRICT data_idx, 
         const int num_rows, const int num_cols,
         const int left_size, const int right_size, 
         const int * RESTRICT right_range,
         const value_type * RESTRICT x, value_type * RESTRICT y
         )
{
    bool forward = true, backward = true;
    for( int i=1; i<num_rows-1; i++)
        for( int d=0; d<2; d++)
        {
            if( data_idx[i*2+d] != d) forward = backward = false;
            //if( cols_idx[i*2+d] != i+d-1) backward = false;
            //if( cols_idx[i*2+d] != i+d) forward = false;
        }
    //int diff = -1;
    //if(forward ) diff = 0;
    if( forward || backward )
    {
#pragma omp parallel for 
    for( int s=0; s<left_size; s++)
    for( int i=0; i<1; i++)
    for( int k=0; k<3; k++)
    for( int j=right_range[0]; j<right_range[1]; j++)
    {
        value_type temp = 0;
        int B0 = (data_idx[i*2+0]*3+k)*3;
        int B1 = (data_idx[i*2+1]*3+k)*3;
        int J0 = (s*num_cols+cols_idx[i*2+0])*3;
        int J1 = (s*num_cols+cols_idx[i*2+1])*3;
        temp +=data[ B0+0]* x[(J0+0)*right_size+j];
        temp +=data[ B0+1]* x[(J0+1)*right_size+j];
        temp +=data[ B0+2]* x[(J0+2)*right_size+j];
        temp +=data[ B1+0]* x[(J1+0)*right_size+j];
        temp +=data[ B1+1]* x[(J1+1)*right_size+j];
        temp +=data[ B1+2]* x[(J1+2)*right_size+j];
        int I = ((s*num_rows + i)*3+k)*right_size+j;
        y[I]=alpha*temp+beta*y[I];
    }
#pragma omp parallel for collapse(2)
    for( int s=0; s<left_size; s++)
    for( int i=1; i<num_rows-1; i++)
    for( int k=0; k<3; k++)
    {
        int B0 = (0*3+k)*3;
        int B1 = (1*3+k)*3;
        //int J0 = (s*num_cols+i+0+diff)*3;
        //int J1 = (s*num_cols+i+1+diff)*3;
        int J0 = (s*num_cols+cols_idx[i*2+0])*3;
        int J1 = (s*num_cols+cols_idx[i*2+1])*3;
        for( int j=right_range[0]; j<right_range[1]; j++)
        {
            value_type temp = 0;
            temp +=data[ B0+0]* x[(J0+0)*right_size+j];
            temp +=data[ B0+1]* x[(J0+1)*right_size+j];
            temp +=data[ B0+2]* x[(J0+2)*right_size+j];
            temp +=data[ B1+0]* x[(J1+0)*right_size+j];
            temp +=data[ B1+1]* x[(J1+1)*right_size+j];
            temp +=data[ B1+2]* x[(J1+2)*right_size+j];
            int I = ((s*num_rows + i)*3+k)*right_size+j;
            y[I]=alpha*temp+beta*y[I];
        }
    }
#pragma omp parallel for 
    for( int s=0; s<left_size; s++)
    for( int i=num_rows-1; i<num_rows; i++)
    for( int k=0; k<3; k++)
    for( int j=right_range[0]; j<right_range[1]; j++)
    {
        value_type temp = 0;
        int B0 = (data_idx[i*2+0]*3+k)*3;
        int B1 = (data_idx[i*2+1]*3+k)*3;
        int J0 = (s*num_cols+cols_idx[i*2+0])*3;
        int J1 = (s*num_cols+cols_idx[i*2+1])*3;
        temp +=data[ B0+0]* x[(J0+0)*right_size+j];
        temp +=data[ B0+1]* x[(J0+1)*right_size+j];
        temp +=data[ B0+2]* x[(J0+2)*right_size+j];
        temp +=data[ B1+0]* x[(J1+0)*right_size+j];
        temp +=data[ B1+1]* x[(J1+1)*right_size+j];
        temp +=data[ B1+2]* x[(J1+2)*right_size+j];
        int I = ((s*num_rows + i)*3+k)*right_size+j;
        y[I]=alpha*temp+beta*y[I];
    }
    }
    else
        ell_multiply_kernel3(alpha,beta, data, cols_idx, data_idx, num_rows, num_cols, 2, left_size, right_size, right_range,  x, y);

}
// multiply kernel, n=3, 3 blocks per line, right_size = 1
template<class value_type>
void ell_multiply_kernel33x( value_type alpha, value_type beta,
         const value_type * RESTRICT data, const int * RESTRICT cols_idx, const int * RESTRICT data_idx, 
         const int num_rows, const int num_cols,
         const int left_size,
         const value_type * RESTRICT x, value_type * RESTRICT y
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
        value_type data_[27];
        for( int k=0; k<3; k++)
        for( int i=0; i<3; i++)
        for( int j=0; j<3; j++)
            data_[(i*3+k)*3+j] = data[(k*3+i)*3+j];
#pragma omp parallel for
    for( int s=0; s<left_size; s++)
    {
    for( int i=0; i<1; i++)
    for( int k=0; k<3; k++)
    {
        value_type temp = 0;
        int B0 = (data_idx[i*3+0]*3+k)*3;
        int B1 = (data_idx[i*3+1]*3+k)*3;
        int B2 = (data_idx[i*3+2]*3+k)*3;
        int J0 = (s*num_cols+cols_idx[i*3+0])*3;
        int J1 = (s*num_cols+cols_idx[i*3+1])*3;
        int J2 = (s*num_cols+cols_idx[i*3+2])*3;
        temp +=data[ B0+0]* x[(J0+0)];
        temp +=data[ B0+1]* x[(J0+1)];
        temp +=data[ B0+2]* x[(J0+2)];

        temp +=data[ B1+0]* x[(J1+0)];
        temp +=data[ B1+1]* x[(J1+1)];
        temp +=data[ B1+2]* x[(J1+2)];

        temp +=data[ B2+0]* x[(J2+0)];
        temp +=data[ B2+1]* x[(J2+1)];
        temp +=data[ B2+2]* x[(J2+2)];
        int I = ((s*num_rows + i)*3+k);
        y[I]=alpha*temp+beta*y[I];
    }
    for( int i=1; i<num_rows-1; i++)
    {
        //int J0 = (s*num_cols+cols_idx[i*3+0])*3;
        int J0 = (s*num_cols+i+0-1)*3;
        for( int k=0; k<3; k++)
        {
            value_type temp = 0;
            int B0 = 9*k;
            temp +=data_[ B0+0]* x[(J0+0)];
            temp +=data_[ B0+1]* x[(J0+1)];
            temp +=data_[ B0+2]* x[(J0+2)];
            temp +=data_[ B0+3]* x[(J0+3)];
            temp +=data_[ B0+4]* x[(J0+4)];
            temp +=data_[ B0+5]* x[(J0+5)];
            temp +=data_[ B0+6]* x[(J0+6)];
            temp +=data_[ B0+7]* x[(J0+7)];
            temp +=data_[ B0+8]* x[(J0+8)];

            int I = ((s*num_rows + i)*3+k);
            y[I]=alpha*temp+beta*y[I];
        }
    }
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
        y[I]=alpha*temp+beta*y[I];
    }
    }
    }
    else 
    {
        int right_range[2] = {0,1};
        ell_multiply_kernel3(alpha,beta, data, cols_idx, data_idx, num_rows, num_cols, 3, left_size, 1, right_range,  x, y);
    }
}

// multiply kernel, n=3, 2 blocks per line, right_size = 1
template<class value_type>
void ell_multiply_kernel32x( value_type alpha, value_type beta,
         const value_type * RESTRICT data, const int * RESTRICT cols_idx, const int * RESTRICT data_idx, 
         const int num_rows, const int num_cols,
         const int left_size, 
         const value_type * RESTRICT x, value_type * RESTRICT y
         )
{
    bool forward = true, backward = true;
    for( int i=1; i<num_rows-1; i++)
        for( int d=0; d<2; d++)
        {
            if( data_idx[i*2+d] != d) forward = backward = false;
            //if( cols_idx[i*2+d] != i+d-1) backward = false;
            if( cols_idx[i*2+d] != cols_idx[i*2]+d) forward = backward = false;
        }
    //int diff = -1;
    //if(forward ) {diff = 0; }
    if( forward || backward )
    {
        value_type data_[18];
        for( int k=0; k<2; k++)
        for( int i=0; i<3; i++)
        for( int j=0; j<3; j++)
            data_[(i*2+k)*3+j] = data[(k*3+i)*3+j];
#pragma omp parallel for
    for( int s=0; s<left_size; s++)
    for( int i=0; i<1; i++)
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
        y[I]=alpha*temp+beta*y[I];
    }
#pragma omp parallel for
    for( int s=0; s<left_size; s++)
    for( int i=1; i<num_rows-1; i++)
    {
        //int J0 = (s*num_cols+i+0+diff)*3;
        //int J1 = (s*num_cols+i+1+diff)*3;
        int J0 = (s*num_cols+cols_idx[i*2+0])*3;
        for( int k=0; k<3; k++)
        {
            value_type temp = 0;
            int B0 = k*6;
            temp +=data_[ B0+0]* x[(J0+0)];
            temp +=data_[ B0+1]* x[(J0+1)];
            temp +=data_[ B0+2]* x[(J0+2)];
            temp +=data_[ B0+3]* x[(J0+3)];
            temp +=data_[ B0+4]* x[(J0+4)];
            temp +=data_[ B0+5]* x[(J0+5)];
            int I = ((s*num_rows + i)*3+k);
            y[I]=alpha*temp+beta*y[I];
        }
    }
#pragma omp parallel for
    for( int s=0; s<left_size; s++)
    for( int i=num_rows-1; i<num_rows; i++)
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
        y[I]=alpha*temp+beta*y[I];
    }
    }
    else
    {
        int right_range[2] = {0,1};
        ell_multiply_kernel3(alpha,beta, data, cols_idx, data_idx, num_rows, num_cols, 2, left_size, 1, right_range, x, y);
    }


}

template<class value_type>
template<class DeviceContainer>
void EllSparseBlockMatDevice<value_type>::launch_multiply_kernel( value_type alpha, const DeviceContainer& x, value_type beta, DeviceContainer& y) const
{
    const value_type* data_ptr = thrust::raw_pointer_cast( &data[0]);
    const int* cols_ptr = thrust::raw_pointer_cast( &cols_idx[0]);
    const int* block_ptr = thrust::raw_pointer_cast( &data_idx[0]);
    const value_type* x_ptr = thrust::raw_pointer_cast( &x[0]);
    value_type* y_ptr = thrust::raw_pointer_cast( &y[0]);
    const int* right_range_ptr = thrust::raw_pointer_cast( &right_range[0]);
    if( n == 3)
    {
        if( blocks_per_line == 3)
        {
            if( right_size == 1)
                ell_multiply_kernel33x<value_type> ( alpha, beta, data_ptr, cols_ptr, block_ptr, num_rows, num_cols, left_size, x_ptr,y_ptr);
            else
                ell_multiply_kernel33<value_type> ( alpha, beta, data_ptr, cols_ptr, block_ptr, num_rows, num_cols, left_size, right_size, right_range_ptr,  x_ptr,y_ptr);
        }
        else if( blocks_per_line == 2)
        {
            if( right_size == 1)
                ell_multiply_kernel32x<value_type> ( alpha, beta, data_ptr, cols_ptr, block_ptr, num_rows, num_cols, left_size,  x_ptr,y_ptr);
            else
                ell_multiply_kernel32<value_type> ( alpha, beta, data_ptr, cols_ptr, block_ptr, num_rows, num_cols, left_size, right_size, right_range_ptr,  x_ptr,y_ptr);
        }
        else
            ell_multiply_kernel3<value_type>(alpha, beta,  
                data_ptr, cols_ptr, block_ptr, num_rows, num_cols, blocks_per_line, left_size, right_size, right_range_ptr,  x_ptr,y_ptr);
    }
    else
        ell_multiply_kernel<value_type>  (alpha, beta,  
            data_ptr, cols_ptr, block_ptr, num_rows, num_cols, blocks_per_line, n, left_size, right_size, right_range_ptr,  x_ptr,y_ptr);
}

template<class value_type>
template<class DeviceContainer>
void CooSparseBlockMatDevice<value_type>::launch_multiply_kernel( value_type alpha, const DeviceContainer& x, value_type beta, DeviceContainer& y) const
{
    for( int i=0; i<num_entries; i++)
#pragma omp parallel for collapse(3)
    for( int s=0; s<left_size; s++)
    for( int k=0; k<n; k++)
    for( int j=0; j<right_size; j++)
    {
        int I = ((s*num_rows + rows_idx[i])*n+k)*right_size+j;
        value_type temp=0;
        for( int q=0; q<n; q++) //multiplication-loop
            temp+= data[ (data_idx[i]*n + k)*n+q]*
                x[((s*num_cols + cols_idx[i])*n+q)*right_size+j];
        y[I] = alpha*temp + beta*y[I];
    }
}

}//namespace dg
