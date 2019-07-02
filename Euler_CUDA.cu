#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <float.h>
#include <magma.h>
#include <magma_types.h>
#include <magma_lapack.h>

// CUDA kernels

// Matrix value assignment kernel

__global__ void value(int N1, int N2, double* a, double* a_copy){
  if (blockIdx.x < N1 && blockIdx.y < N2){
    a_copy[blockIdx.x+blockIdx.y*gridDim.x] = a[blockIdx.x+blockIdx.y*gridDim.x];
  }
}

// X major to Y major kernel

__global__ void change_format(int m, int n, double *in, double *out){
   int xIndex = blockDim.x*blockIdx.x + blockIdx.x;
   int yIndex = blockDim.y*blockIdx.y + blockIdx.y;

   if (xIndex < n && yIndex < m/4){
       int index_in  = xIndex + n*yIndex;
       int index_out = yIndex + m/4*xIndex;
       out[index_out] = in[index_in];
   }

   if (xIndex < n && yIndex < m/2 && yIndex >= m/4){
      int index_in  = xIndex + n*yIndex;
      int index_out = yIndex + m/2*xIndex;
       out[index_out] = in[index_in];
   }

   if (xIndex < n && yIndex < 3*m/4 && yIndex >= m/2){
      int index_in  = xIndex + n*yIndex;
      int index_out = yIndex + 3*m/4*xIndex;
       out[index_out] = in[index_in];
   }
   if (xIndex < n && yIndex < m && yIndex >= 3*m/4){
      int index_in  = xIndex + n*yIndex;
      int index_out = yIndex + m*xIndex;
       out[index_out] = in[index_in];
   }
}

// Matrix multiplication kernel

__global__ void mult(int m, int n, int k, double *A, double *B, double *C){
    int row = blockIdx.y*blockDim.y + blockIdx.y;
    int col = blockIdx.x*blockDim.x + blockIdx.x;
    double sum = 0;
    if( col < k && row < m){
        for(int i = 0; i < n; i++){
            sum += A[row * n + i]*B[i * k + col];
        }
        C[row*k + col] = sum;
    }
}

// Matrix transpose kernel

__global__ void transpose(int m, int n, double* in, double* out){
   int xIndex = blockDim.x*blockIdx.x + blockIdx.x;
   int yIndex = blockDim.y*blockIdx.y + blockIdx.y;

   if (xIndex < n && yIndex < m){
       int index_in  = xIndex + n*yIndex;
       int index_out = yIndex + m*xIndex;
       out[index_out] = in[index_in];
   }
}

// START MAIN PROGRAM

int main(int argc, char* argv[]){

    // Set up runtime clock, start

    clock_t start_time = clock();

    // Set up to be taken as inputs

    int N = atoi(argv[1]);
    int M = atoi(argv[2]);


    // Define parameter values

    double gamma = 1.4;
    double T = 2.0;
    double dt = (double) T/M;
    double dx = (double) 2.0/N;

    // Allocate memory on host (CPU)

    double* MLX = (double *)malloc((4*N)*(4*N)*sizeof(double));
    double* MLY = (double *)malloc((4*N)*(4*N)*sizeof(double));
    double* MRX = (double *)malloc((4*N)*(4*N)*sizeof(double));
    double* MRY = (double *)malloc((4*N)*(4*N)*sizeof(double));
    double* soln = (double *)malloc((4*N)*N*sizeof(double));
    double* rho = (double *)malloc(N*N*sizeof(double));
    double* u =   (double *)malloc(N*N*sizeof(double));
    double* v =   (double *)malloc(N*N*sizeof(double));
    double* p =   (double *)malloc(N*N*sizeof(double));
    double* X =   (double *)malloc(N*N*sizeof(double));
    double* Y =   (double *)malloc(N*N*sizeof(double));
    double* I =   (double *)malloc(N*N*sizeof(double));
    double* D =   (double *)malloc(N*N*sizeof(double));
    double* ND =   (double *)malloc(N*N*sizeof(double));
    double* NGD =   (double *)malloc(N*N*sizeof(double));
    double* GD =   (double *)malloc(N*N*sizeof(double));
    double* V =   (double *)malloc(4*N*N*sizeof(double));


    // Allocate memory on device (GPU)

    double *MLX_cu, *MLY_cu, *MRX_cu, *MRY_cu;
    cudaMalloc((void**)&MLX_cu, (4*N)*(4*N)*sizeof(double));
    cudaMalloc((void**)&MLY_cu, (4*N)*(4*N)*sizeof(double));
    cudaMalloc((void**)&MRX_cu, (4*N)*(4*N)*sizeof(double));
    cudaMalloc((void**)&MRY_cu, (4*N)*(4*N)*sizeof(double));

    double *V_cu, *V_out, *W, *W_out;
    cudaMalloc((void**)&V_cu, (4*N)*N*sizeof(double));
    cudaMalloc((void**)&V_out, (4*N)*N*sizeof(double));
    cudaMalloc((void**)&W, (4*N)*N*sizeof(double));
    cudaMalloc((void**)&W_out, (4*N)*N*sizeof(double));



    // Initialize rho, u, v, p

    for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){

			X[i*N+j] = -1 + dx * i;
			Y[i*N+j] = -1 + dx * j;
			rho[i*N + j] = 2/gamma * exp(-100 * X[i*N+j]*X[i*N+j] + Y[i*N+j]*Y[i*N+j]);
			p[i*N + j] = 2 * exp(-100 *(X[i*N+j]*X[i*N+j] + Y[i*N+j]*Y[i*N+j]));
		}
	}





  free(X);
  free(Y);

  // Segmentation fault occurs in following loop

	// Store matrices in V, block form

	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
				V[i*N+j] = rho[i*N+j];
				V[(i+N)*N+j] = 0;
				V[(i+2*N)*N+j] = 0;
				V[(i+3*N)*N+j] = p[i*N+j];
		}
	}


	// Free rho and p

	free(rho);
	free(p);



	// Populate identity matrix (NxN)

	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			if (i == j){
				I[i*N+j] = 1;
			}else{
				I[i*N+j] = 0;
			}
		}
	}



	// Populate D matrix

	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){

			if (i == j-1){
				D[i*N+j] = -dt/(4*dx);
			}else if (i == j+1){
				D[i*N+j] = +dt/(4*dx);
			}else if (i == N-1 && j == 0){
				D[i*N+j] = -dt/(4*dx);
			}else if (i == 0 && j == N-1){
				D[i*N+j] = +dt/(4*dx);
			}else{
				D[i*N+j] = 0;
      }
    }
  }



	// Populate -D matrix

	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			ND[i*N+j]= -1* D[i*N+j];
		}
	}

	// Populate product matrix GD = (gamma*D)

	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			GD[i*N+j]= gamma * D[i*N+j];
		}
	}

	// Populate product matrix, NGD = -GD

	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			NGD[i*N+j]= - GD[i*N+j];
		}
	}



	// Populate MRX matrix

	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
            MRX[i*4*N+j] = I[i+N*j];
            MRX[i*4*N+(N+j)] = 0;
            MRX[i*4*N+(2*N+j)] = 0;
            MRX[i*4*N+(3*N+j)] = 0;
            MRX[(N+i)*4*N+j] = ND[i*N+j];
            MRX[(N+i)*4*N+(N+j)] = I[i*N+j];
            MRX[(N+i)*4*N+(2*N+j)] = 0;
            MRX[(N+i)*4*N+(3*N+j)] = NGD[i*N+j];
            MRX[(2*N+i)*4*N+j]=0;
            MRX[(2*N+i)*4*N+(N+j)]=0;
            MRX[(2*N+i)*4*N+(2*N+j)]=I[i*N+j];
            MRX[(2*N+i)*4*N+(3*N+j)]=0;
            MRX[(3*N+i)*4*N+j]=0;
            MRX[(3*N+i)*4*N+(N+j)]=ND[i*N+j];
            MRX[(3*N+i)*4*N+(2*N+j)]=0;
            MRX[(3*N+i)*4*N+(3*N+j)]=I[i*N+j];
		}
	}



	// Populate MLX matrix

	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
            MLX[i*4*N+j]=I[i*N+j];
            MLX[i*4*N+(N+j)]=0;
            MLX[i*4*N+(2*N+j)]=0;
            MLX[i*4*N+(3*N+j)]=0;
            MLX[(N+i)*4*N+j]=D[i*N+j];
            MLX[(N+i)*4*N+(N+j)]=I[i*N+j];
            MLX[(N+i)*4*N+(2*N+j)]=0;
            MLX[(N+i)*4*N+(3*N+j)]=GD[i*N+j];
            MLX[(2*N+i)*4*N+j]=0;
            MLX[(2*N+i)*4*N+(N+j)]=0;
            MLX[(2*N+i)*4*N+(2*N+j)]=I[i*N+j];
            MLX[(2*N+i)*4*N+(3*N+j)]=0;
            MLX[(3*N+i)*4*N+j]=0;
            MLX[(3*N+i)*4*N+(N+j)]=D[i*N+j];
            MLX[(3*N+i)*4*N+(2*N+j)]=0;
            MLX[(3*N+i)*4*N+(3*N+j)]=I[i*N+j];
		}
	}



	// Populate MRY matrix

	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
            MRY[i*4*N+j]=I[i*N+j];
            MRY[i*4*N+(N+j)]=0;
            MRY[i*4*N+(2*N+j)]=0;
            MRY[i*4*N+(3*N+j)]=0;
            MRY[(N+i)*4*N+j]=0;
            MRY[(N+i)*4*N+(N+j)]=I[i*N+j];
            MRY[(N+i)*4*N+(2*N+j)]=0;
            MRY[(N+i)*4*N+(3*N+j)]=0;
            MRY[(2*N+i)*4*N+j]=ND[i*N+j];
            MRY[(2*N+i)*4*N+(N+j)]=0;
            MRY[(2*N+i)*4*N+(2*N+j)]=I[i*N+j];
            MRY[(2*N+i)*4*N+(3*N+j)]=NGD[i*N+j];
            MRY[(3*N+i)*4*N+j]=0;
            MRY[(3*N+i)*4*N+(N+j)]=0;
            MRY[(3*N+i)*4*N+(2*N+j)]=ND[i*N+j];
            MRY[(3*N+i)*4*N+(3*N+j)]=I[i*N+j];
		}
	}



	// Populate MLY matrix

	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
            MLY[i*4*N+j] = I[i*N+j];
            MLY[i*4*N+(N+j)]=0;
            MLY[i*4*N+(2*N+j)]=0;
            MLY[i*4*N+(3*N+j)]=0;
            MLY[(N+i)*4*N+j]=0;
            MLY[(N+i)*4*N+(N+j)]=I[i*N+j];
            MLY[(N+i)*4*N+(2*N+j)]=0;
            MLY[(N+i)*4*N+(3*N+j)]=0;
            MLY[(2*N+i)*4*N+j]=D[i*N+j];
            MLY[(2*N+i)*4*N+(N+j)]=0;
            MLY[(2*N+i)*4*N+(2*N+j)]=I[i*N+j];
            MLY[(2*N+i)*4*N+(3*N+j)]=GD[i*N+j];
            MLY[(3*N+i)*4*N+j]=0;
            MLY[(3*N+i)*4*N+(N+j)]=0;
            MLY[(3*N+i)*4*N+(2*N+j)]=D[i*N+j];
            MLY[(3*N+i)*4*N+(3*N+j)]=I[i*N+j];
        }
    }


    // Free all RHS matrices

    free(I);
    free(D);
	free(ND);
	free(NGD);
	free(GD);

  // Copy all from host to device

  cudaMemcpy(MLX_cu, MLX, (4*N)*(4*N)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(MLY_cu, MLY, (4*N)*(4*N)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(MRX_cu, MRX, (4*N)*(4*N)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(MRY_cu, MRY, (4*N)*(4*N)*sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(V_cu, V, (4*N)*(4*N)*sizeof(double), cudaMemcpyHostToDevice);

  dim3 dimGrid(4*N,4*N);

  /////////// LU decomposition

  // Magma initialization

  magma_init();
  magma_int_t *pivot;
  magma_int_t *pivot_a;
  magma_int_t info;
  magma_int_t n_magma = N;
  magma_int_t m_magma = 4*N;


  pivot = (magma_int_t*)malloc(m_magma*sizeof(magma_int_t));
  pivot_a = (magma_int_t*)malloc(m_magma*sizeof(magma_int_t));

  magma_dsetmatrix(m_magma, m_magma, MLX, m_magma, MLX_cu, m_magma);
  magma_dsetmatrix(m_magma, m_magma, MLY, m_magma, MLY_cu, m_magma);

  magma_dgetrf_gpu(m_magma, m_magma, MLX_cu, m_magma, pivot, &info);
  magma_dgetrf_gpu(m_magma, m_magma, MLY_cu, m_magma, pivot_a, &info);



  // Target files

  FILE *rho_file= fopen("Euler_R.out","w");
  FILE *u_file = fopen("Euler_U.out","w");
  FILE *v_file = fopen("Euler_V.out","w");
  FILE *p_file = fopen("Euler_P.out","w");


  ///////////// START MAIN LOOP

  // initialize the output counter

  int output_count = 0;



  //////// Explicit in x, matrix multiplication

  for (int n = 0; n < M+1; ++n){

    // matrix multiplication

    mult<<<dimGrid, 1>>>(4*N, N, N, MRX_cu, V_cu, V_out);


    /////// Implicit in y

    // Switch to y major format

    change_format<<<dimGrid, 1>>>(4*N, N, V_out, W);

    // Transpose to send to magma

    transpose<<<dimGrid, 1>>>(4*N, N, W, W_out);

    // Linear solve

    magma_dgetrs_gpu(MagmaNoTrans, m_magma, n_magma, MLY_cu, m_magma, pivot_a, W_out, m_magma, &info);


    // Transpose again

    transpose<<<dimGrid, 1>>>(N, 4*N, W_out, W);

    //////// Explicit in y, matrix multiplication

    mult<<<dimGrid, 1>>>(4*N, N, N, MRY_cu, W, W_out);

    ////////// Implicit in x

    // Change back to x-major

    change_format<<<dimGrid, 1>>>(4*N, N, W_out, V_cu);

    // Transpose to send to magma

    transpose<<<dimGrid, 1>>>(4*N, N, V_cu, V_out);

    // Linear solve

    magma_dgetrs_gpu(MagmaNoTrans, m_magma, n_magma, MLX_cu, m_magma, pivot, V_out, m_magma, &info);

    // Transpose again

    transpose<<<dimGrid, 1>>>(N, 4*N, V_out, V_cu);



    // Store solutions

    if ((int)((double)n*dt/0.2) == output_count){
      cudaMemcpy(soln, V_cu, (4*N)*N*sizeof(double), cudaMemcpyDeviceToHost);
      for (int i = 0; i < N; ++i){
        for (int j = 0; j < N; ++j){
          fwrite(&soln[i*N + j], sizeof(double), 1, rho_file);
          fwrite(&soln[(i+N)*N + j], sizeof(double) ,1 ,u_file);
          fwrite(&soln[(i+2*N)*N + j], sizeof(double), 1, v_file);
          fwrite(&soln[(i+3*N)*N + j], sizeof(double), 1, p_file);
        }
      }
      ++output_count;
    }

  }

  // END MAIN LOOP

  cudaThreadSynchronize();



  // Free all

    free(MLX);
    free(MRX);
    free(MLY);
    free(MRY);
    free(soln);
    cudaFree(MLX_cu);
    cudaFree(MRX_cu);
    cudaFree(MLY_cu);
    cudaFree(MRY_cu);
    cudaFree(V);
    cudaFree(V_out);
    cudaFree(W);
    cudaFree(W_out);

  // Close target files

    fclose(rho_file);
    fclose(u_file);
    fclose(v_file);
    fclose(p_file);

  // Output runtime

    clock_t end_time = clock();
    printf("Runtime: %gs\n.", (double)(end_time - start_time)/CLOCKS_PER_SEC);

  return 0;
}








