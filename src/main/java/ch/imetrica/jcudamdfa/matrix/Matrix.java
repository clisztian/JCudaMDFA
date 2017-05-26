package ch.imetrica.jcudamdfa.matrix;

import java.io.Serializable;
import java.util.Locale;
import java.util.Random;


import jcuda.Pointer;
import jcuda.Sizeof;
import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.jcublas.JCublas2.cublasDestroy;
import static jcuda.jcublas.JCublas2.cublasGetMatrix;
import static jcuda.jcublas.JCublas2.cublasGetVector;
import static jcuda.jcublas.JCublas2.cublasIsamax;
import static jcuda.jcublas.JCublas2.cublasSetMatrix;
import static jcuda.jcublas.JCublas2.cublasSetVector;
import static jcuda.jcublas.JCublas2.cublasSgemm;
import static jcuda.jcublas.JCublas2.cublasSgemv;
import static jcuda.jcublas.JCublas2.cublasSger;
import static jcuda.jcublas.JCublas2.cublasSscal;
import static jcuda.jcublas.JCublas2.cublasSswap;
import static jcuda.jcublas.JCublas2.cublasStrmv;
import static jcuda.jcublas.cublasFillMode.CUBLAS_FILL_MODE_UPPER;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice;

import static jcuda.jcurand.JCurand.curandDestroyGenerator;
import static jcuda.runtime.JCuda.cudaDeviceSynchronize;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;
import static jcuda.jcublas.JCublas2.cublasSdot;
import static jcuda.jcublas.JCublas2.cublasSetPointerMode;
import static jcuda.jcublas.cublasPointerMode.CUBLAS_POINTER_MODE_DEVICE;
import static jcuda.jcublas.cublasPointerMode.CUBLAS_POINTER_MODE_HOST;
import static jcuda.jcurand.curandRngType.CURAND_RNG_PSEUDO_DEFAULT;
import static jcuda.jcurand.JCurand.curandCreateGenerator;
import static jcuda.jcurand.JCurand.curandSetPseudoRandomGeneratorSeed;
import static jcuda.jcurand.JCurand.curandGenerateNormal;
import static jcuda.jcurand.JCurand.curandGenerateNormalDouble;
import static jcuda.jcurand.JCurand.curandGenerateUniformDouble;
import static jcuda.jcublas.JCublas2.cublasDaxpy;

import jcuda.jcurand.JCurand;
import jcuda.jcurand.curandGenerator;
import java.util.Arrays;
import jcuda.jcublas.JCublas;
import jcuda.jcublas.JCublas2;

import jcuda.jcublas.cublasHandle;
import jcuda.runtime.JCuda;


public class Matrix implements Serializable {
	
	private static final long serialVersionUID = 1L;
	public int rows;
	public int cols;
	public int size;
	
	//----- JCuda pointers to floats ---------
	public Pointer w;
	public double[] data;


	public Matrix(int rows, int cols) {
		
		this.rows = rows;
		this.cols = cols;
		this.size = rows*cols;

		w = new Pointer();
		data = new double[this.size];
		
		cudaMalloc(w, this.size * Sizeof.DOUBLE);
        zerosFromHost();
       
	}
	
	public static Matrix rand(int rows, int cols, double initParamsStdDev, curandGenerator generator) 
	{		
		Matrix result = new Matrix(rows, cols);		
		result.rand(initParamsStdDev, generator); 		
		return result;
	}
	
	
	
	
	public Matrix(int dim) {

		this.rows = dim;
		this.cols = 1;
		this.size = dim; 
		
		w = new Pointer();
		data = new double[this.size];
		cudaMalloc(w, this.size * Sizeof.DOUBLE);
        zerosFromHost();        
	}
	
	public Matrix(double[] v) {
		
		this.rows = v.length;
		this.cols = 1;
		this.size = v.length;
		
		w = new Pointer();
		cudaMalloc(w, this.size * Sizeof.DOUBLE);
        cudaMemcpy(w, Pointer.to(v), this.size * Sizeof.DOUBLE,cudaMemcpyHostToDevice);
        
        zerosFromHost();
	}
	
	public Matrix(double[] v, int rows) {
		
		this.rows = rows;
		this.cols = v.length/rows;
		this.size = v.length*rows;
		
		w = new Pointer();
		data = new double[this.size];
		
		cudaMalloc(w, this.size * Sizeof.DOUBLE);
        cudaMemcpy(w, Pointer.to(v), this.size * Sizeof.DOUBLE,
    	        cudaMemcpyHostToDevice);
       
        zerosFromHost();

	}
	
	public Matrix(double[] v, int rows, int nbatch) throws Exception {
		
        if(v.length != rows*nbatch) { 
			
			throw new Exception("matrix dimension mismatch: this vs copy = " + v.length + " " + rows*nbatch);
		}
				
		this.rows = rows;
		this.cols = nbatch;
		this.size = v.length;
		
		w = new Pointer();
		data = v;

		
		cudaMalloc(w, this.size * Sizeof.DOUBLE);	
        cudaMemcpy(w, Pointer.to(v), this.size * Sizeof.DOUBLE,
    	        cudaMemcpyHostToDevice);
        
        zerosFromHost();
	}
	
	
	public Matrix(){
		
	}
	
	
	public void copy(Matrix copyMe) throws Exception {
		
		if(this.size != copyMe.size) { 
			
			throw new Exception("matrix dimension mismatch: this vs copy = " + this.size + " " + copyMe.size);
		}
		
		JCublas.cublasDcopy(this.size, copyMe.w, 1, this.w, 1);		
	}
	
	public static Matrix copyMatrix(final Matrix copyMe) throws Exception {
		
		Matrix result = new Matrix(copyMe.rows, copyMe.cols);		 
		result.copy(copyMe);
		
		return result;
	}
	
	public void set(double[] v)
	{
		data = v;
		cudaMemcpy(w, Pointer.to(v), this.size * Sizeof.DOUBLE,
    	        cudaMemcpyHostToDevice);
	}
	
	
    public void rand(double initParamsStdDev, curandGenerator generator) 
    {
    	  	
        // Create pseudo-random number generator 
        curandCreateGenerator(generator, CURAND_RNG_PSEUDO_DEFAULT);

        // Set seed 
        curandSetPseudoRandomGeneratorSeed(generator, 1);

        // Generate n floats on device 
    	curandGenerateNormalDouble(generator, w, this.size, 0, initParamsStdDev);
		
	}
    
    public void urand(curandGenerator generator) 
    {
    	

        // Create pseudo-random number generator 
        curandCreateGenerator(generator, CURAND_RNG_PSEUDO_DEFAULT);

        // Set seed 
        curandSetPseudoRandomGeneratorSeed(generator, 1);

        // Generate n floats on device 
        curandGenerateUniformDouble(generator, w, this.size);
    		    
	}
	
    public static Matrix ones(int rows, int cols) 
    {
    	Matrix result = new Matrix(rows, cols);		
		result.ones();		
		return result;
    }
    
    public static Matrix zeros(int rows, int cols) 
    {
    	
    	Matrix result = new Matrix(rows, cols);		
	    Arrays.fill(result.data,  0.0);    
	    cudaMemcpy(result.w, Pointer.to(result.data), result.size * Sizeof.DOUBLE,
	        cudaMemcpyHostToDevice);  

		return result;
    }
    

    public static void mdfaMatrixSet(final Matrix m, int i, int j, double v) {
   
    	m.data[m.cols*i + j] = v;
    }
    
    public static double mdfaMatrixGet(final Matrix m, int i, int j) {
    	   
    	return m.data[m.cols*i + j];
    }
    
    
    public void resetToSmall() 
    {
    	
    	double hostData[] = new double[this.size];
	    Arrays.fill(hostData, -Double.MAX_VALUE);
	    
	    cudaMemcpy(this.w, Pointer.to(hostData), this.size * Sizeof.DOUBLE,
	        cudaMemcpyHostToDevice);  
    }
    
    
    
    public static Matrix zeros(int rows) 
    {
    	Matrix result = new Matrix(rows, 1);		
		
    	double hostData[] = new double[result.size];
	    Arrays.fill(hostData,  0.0);
	    
	    cudaMemcpy(result.w, Pointer.to(hostData), result.size * Sizeof.DOUBLE,
	        cudaMemcpyHostToDevice);  
    	
		return result;
    }    
    
    
    public static Matrix negones(int rows, int cols) 
    {
    	Matrix result = new Matrix(rows, cols);				
    	double[] temp = new double[result.size];
    	for(int i = 0; i < result.size; i++) {temp[i] = -1.0;}
    	   	    	
    	cudaMemcpy(result.w, Pointer.to(temp), result.size*Sizeof.DOUBLE,
    	        cudaMemcpyHostToDevice);   
    	
		return result;
    }
    
    
    public void ones() 
    {
    	
    	for(int i = 0; i < this.size; i++) {this.data[i] = 1.0;}
    	   	    	
    	cudaMemcpy(w, Pointer.to(this.data), this.size*Sizeof.DOUBLE,
    	        cudaMemcpyHostToDevice);     	   	   	
	}
    
        
     
    
    public void zerosFromHost()
    {    
	    cudaMemcpy(w, Pointer.to(this.data), this.size * Sizeof.DOUBLE,
	        cudaMemcpyHostToDevice);    
    }
     
    
    public void destroyMatrix()
    {
    	cudaFree(this.w);
    }
    
    
    public void setTarget(int target)
    {
    	double[] temp = new double[this.size];
    	if(target < this.size && target >= 0)
    	{
    		temp[target] = 1.0;
    		cudaMemcpy(w, Pointer.to(temp), this.size*Sizeof.DOUBLE,
        	        cudaMemcpyHostToDevice);
    	}		
    }
    

    
    
    public static void testRandom()
    {
    	    // Enable exceptions and omit all subsequent error checks
            JCuda.setExceptionsEnabled(true);
            JCurand.setExceptionsEnabled(true);

            int n = 100;
            curandGenerator generator = new curandGenerator();

            // Allocate n floats on host 
            float hostData[] = new float[n];

            // Allocate n floats on device 
            Pointer deviceData = new Pointer();
            cudaMalloc(deviceData, n * Sizeof.FLOAT);

            // Create pseudo-random number generator 
            curandCreateGenerator(generator, CURAND_RNG_PSEUDO_DEFAULT);

            // Set seed 
            curandSetPseudoRandomGeneratorSeed(generator, 1234);

            // Generate n floats on device 
            float mean = (float) 0; 
            float std = (float) 3.0;
            
            curandGenerateNormal(generator, deviceData, n, mean, std);
            //curandGenerateUniform(generator, deviceData, n);

            
            // Copy device memory to host 
            cudaMemcpy(Pointer.to(hostData), deviceData, 
                n * Sizeof.FLOAT, cudaMemcpyDeviceToHost);

            // Show result
            System.out.println(Arrays.toString(hostData));

            // Cleanup 
            curandDestroyGenerator(generator);
            cudaFree(deviceData);
        
    }
    
    
    
    
    public static void main(String[] args)
    {

        
        // Create a CUBLAS handle
        cublasHandle handle = new cublasHandle();
        cublasCreate(handle);

        // Create the input matrix
        int size = 200;
        float A[] = createRandomFloatData(size * size);

        // Invert the matrix
        float invA[] = A.clone();
        invertMatrix(handle, size, invA);

        // Compute A*invA, which should yield the identity matrix
        float identity[] = new float[size * size];
        multiply(handle, size, A, invA, identity);

        // Print the results
//        System.out.println("A:");
//        System.out.println(toString2D(A, size));
//        System.out.println("invA:");
//        System.out.println(toString2D(invA, size));
//        System.out.println("identity:");
//        System.out.println(toString2D(identity, size));
        
        // Verify the result
        System.out.println("Done...");
        boolean passed = true;
        final float epsilon = 1e-4f;
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                int index = i * size + j;
                float value = identity[index];
                if (i == j)
                {
                    passed &= Math.abs(value - 1.0f) <= epsilon;
                }
                else
                {
                    passed &= Math.abs(value) <= epsilon;
                }
            }
        }
        System.out.println((passed ? "PASSED" : "FAILED"));

        // Clean up
        cublasDestroy(handle);

        testPointer();
        
        
        System.out.println("Matrix destroyed");
        System.out.println("Vector addition");
        
        testVectorAddition();
        
        System.out.println("Test Kernel vector addition");

        
    }

    /**
     * Copies the given n x n matrix into device memory, inverts it by calling
     * {@link #invertMatrix(cublasHandle, int, Pointer)}, and copies it back 
     * into the given array.
     * 
     * @param handle The CUBLAS handle
     * @param n The size of the matrix
     * @param A The matrix
     */
    public static void invertMatrix(cublasHandle handle, int n, float A[])
    {
        Pointer dA = new Pointer();
        cudaMalloc(dA, n * n * Sizeof.FLOAT);
        cublasSetMatrix(n, n, Sizeof.FLOAT, Pointer.to(A), n, dA, n);

        invertMatrix(handle, n, dA);

        cublasGetMatrix(n, n, Sizeof.FLOAT, dA, n, Pointer.to(A), n);
        cudaFree(dA);
    }

    /**
     * Invert the n x n matrix that is given in device memory.
     * 
     * @param n The size of the matrix
     * @param dA The matrix
     */
    public static void invertMatrix(cublasHandle handle, int n, Pointer dA)
    {
        // Perform LU factorization
        int[] pivots = cudaSgetrfSquare(handle, n, dA);

        // Perform inversion on factorized matrix
        cudaSgetri(handle, n, dA, pivots);
    }

    /**
     * Convenience method that returns a pointer with the given offset (in
     * number of 4-byte float elements) from the given pointer.
     * 
     * @param p The pointer
     * @param floatOffset The offset, in number of float elements
     * @return The new pointer
     */
    private static Pointer at(Pointer p, int floatOffset)
    {
        return p.withByteOffset(floatOffset * Sizeof.FLOAT);
    }

    /**
     * cudaSgetrf performs an in-place LU factorization on a square matrix. 
     * Uses the unblocked BLAS2 approach
     * 
     * @param n The matrix size
     * @param dA The pointer to the matrix (in device memory)
     * @return The pivots
     */
    private static int[] cudaSgetrfSquare(
        cublasHandle handle, int n, Pointer dA)
    {
        int[] pivots = new int[n];
        for (int i = 0; i < n; i++)
        {
            pivots[i] = i;
        }

        Pointer minusOne = Pointer.to(new float[] { -1.0f });
        float[] factor = { 0.0f };
        Pointer pFactor = Pointer.to(factor);
        for (int i = 0; i < n - 1; i++)
        {
            Pointer offset = at(dA, i * n + i);

            int max[] = { 0 };
            cublasIsamax(handle, n - i, offset, 1, Pointer.to(max));
            int pivot = i - 1 + max[0];
            if (pivot != i)
            {
                pivots[i] = pivot;
                cublasSswap(handle, n, at(dA, pivot), n, at(dA, i), n);
            }

            cublasGetVector(1, Sizeof.FLOAT, offset, 1, pFactor, 1);
            factor[0] = 1 / factor[0];
            cublasSscal(handle, n - i - 1, pFactor, at(offset, 1), 1);
            cublasSger(handle, n - i - 1, n - i - 1, minusOne, at(offset, 1), 
                1, at(offset, n), n, at(offset, n + 1), n);
        }
        return pivots;
    }

    /***
     * cudaSgetri Computes the inverse of an LU-factorized square matrix
     * 
     * @param n The matrix size
     * @param dA The matrix in device memory
     * @param pivots The pivots
     */
    private static void cudaSgetri(
        cublasHandle handle, int n, Pointer dA, int[] pivots)
    {
        // Perform inv(U)
        cudaStrtri(handle, n, dA);

        // Solve inv(A)*L = inv(U)
        Pointer dWork = new Pointer();
        cudaMalloc(dWork, (n - 1) * Sizeof.FLOAT);

        Pointer zero = Pointer.to(new float[]{ 0.0f });
        Pointer one = Pointer.to(new float[]{ 1.0f });
        Pointer minusOne = Pointer.to(new float[]{ -1.0f });
        for (int i = n - 1; i > 0; i--)
        {
            Pointer offset = at(dA, ((i - 1) * n + i));
            cudaMemcpy(dWork, offset, (n - 1) * Sizeof.FLOAT,
                cudaMemcpyDeviceToDevice);
            cublasSscal(handle, n - i, zero, offset, 1);
            cublasSgemv(handle, CUBLAS_OP_N, n, n - i, minusOne, 
                at(dA, i * n), n, dWork, 1, one, at(dA, ((i - 1) * n)), 1);
        }

        cudaFree(dWork);

        // Pivot back to original order
        for (int i = n - 1; i >= 0; i--)
        {
            if (i != pivots[i])
            {
                cublasSswap(handle, n, at(dA, i * n), 1, 
                    at(dA, pivots[i] * n), 1);
            }
        }

    }

    /***
     * cudaStrtri Computes the inverse of an upper triangular matrix in place
     * Uses the unblocked BLAS2 approach
     * 
     * @param n The size of the matrix
     * @param dA The matrix
     */
    private static void cudaStrtri(cublasHandle handle, int n, Pointer dA)
    {
        float[] factor = { 0.0f };
        Pointer pFactor = Pointer.to(factor);
        for (int i = 0; i < n; i++)
        {
            Pointer offset = at(dA, i * n);
            cublasGetVector(1, Sizeof.FLOAT, at(offset, i), 1, pFactor, 1);
            factor[0] = 1 / factor[0];
            cublasSetVector(1, Sizeof.FLOAT, pFactor, 1, at(offset, i), 1);

            factor[0] = -factor[0];
            cublasStrmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                CUBLAS_OP_N, i, dA, n, offset, 1);
            cublasSscal(handle, i, pFactor, offset, 1);
        }
    }

    // === Utility methods for this sample ====================================

    /**
     * Multiplies the matrices A and B and writes the result into C.
     * 
     * @param size The size of the matrices
     * @param A Matrix A
     * @param B Matrix B
     * @param C Matrix C
     */
    private static void multiply(cublasHandle handle, int size, float A[],
        float B[], float C[])
    {
        Pointer dA = new Pointer();
        Pointer dB = new Pointer();
        Pointer dC = new Pointer();

        cudaMalloc(dA, size * size * Sizeof.FLOAT);
        cudaMalloc(dB, size * size * Sizeof.FLOAT);
        cudaMalloc(dC, size * size * Sizeof.FLOAT);
        cublasSetVector(size * size, Sizeof.FLOAT, Pointer.to(A), 1, dA, 1);
        cublasSetVector(size * size, Sizeof.FLOAT, Pointer.to(B), 1, dB, 1);

        Pointer zero = Pointer.to(new float[]{ 0.0f });
        Pointer one = Pointer.to(new float[]{ 1.0f });
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, one, 
            dA, size, dB, size, zero, dC, size);

        cublasGetVector(size * size, Sizeof.FLOAT, dC, 1, Pointer.to(C), 1);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }

    
    /**
     * Creates an array of the specified size, containing float values from
     * the range [0.0f, 1.0f)
     * 
     * @param n The size of the array
     * @return The array of random values
     */
    public static float[] createRandomFloatData(int n)
    {
        Random random = new Random(0);
        float a[] = new float[n];
        for (int i = 0; i < n; i++)
        {
            a[i] = random.nextFloat();
        }
        return a;
    }
    
    /**
     * Creates a string representation of the given array as a matrix with 
     * with given number of columns.
     * 
     * @param a The array
     * @param columns The number of columns
     * @return The string representation
     */
    public static String toString2D(float[] a, int columns)
    {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < a.length; i++)
        {
            if ((i > 0) && (i % columns == 0))
            {
                sb.append("\n");
            }
            sb.append(String.format(Locale.ENGLISH, "%7.4f ", a[i]));
        }
        return sb.toString();
    }
    
    
    
    public static void testPointer()
    {
    	
    	
	    // Enable exceptions and omit subsequent error checks
	    JCublas2.setExceptionsEnabled(true);
	    JCuda.setExceptionsEnabled(true);
	
	    // Create the input data: A vector containing the
	    // value 1.0 exactly n times.
	    int n = 1000000;
	    float hostData[] = new float[n];
	    Arrays.fill(hostData,  1.0f);
	
	    // Allocate device memory, and copy the input data to the device
	    Pointer deviceData = new Pointer();
	    cudaMalloc(deviceData, n * Sizeof.FLOAT);
	    cudaMemcpy(deviceData, Pointer.to(hostData), n * Sizeof.FLOAT,
	        cudaMemcpyHostToDevice);
	
	    // Create a CUBLAS handle
	    cublasHandle handle = new cublasHandle();
	    cublasCreate(handle);
	
	
	    // Execute the 'dot' function in HOST pointer mode:
	    // The result will be written to a pointer that
	    // points to host memory.
	
	    // Set the pointer mode to HOST
	    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
	
	    // Prepare the pointer for the result in HOST memory
	    float hostResult[] = { -1.0f };
	    Pointer hostResultPointer = Pointer.to(hostResult);
	
	    // Execute the 'dot' function
	    long beforeHostCall = System.nanoTime();
	    cublasSdot(handle, n, deviceData, 1, deviceData, 1, hostResultPointer);
	    long afterHostCall = System.nanoTime();
	
	    // Print the result and timing information
	    double hostDuration = (afterHostCall - beforeHostCall) / 1e6;
	    System.out.println("Host call duration: " + hostDuration + " ms");
	    System.out.println("Result: " + hostResult[0]);
	
	
	    // Execute the 'dot' function in DEVICE pointer mode:
	    // The result will be written to a pointer that
	    // points to device memory.
	
	    // Set the pointer mode to DEVICE
	    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
	
	    // Prepare the pointer for the result in DEVICE memory
	    Pointer deviceResultPointer = new Pointer();
	    cudaMalloc(deviceResultPointer, Sizeof.FLOAT);
	
	    // Execute the 'dot' function
	    long beforeDeviceCall = System.nanoTime();
	    cublasSdot(handle, n, deviceData, 1, deviceData, 1,
	        deviceResultPointer);
	    long afterDeviceCall = System.nanoTime();
	
	    // Synchronize in order to wait for the result to
	    // be available (note that this is done implicitly
	    // when cudaMemcpy is called)
	    cudaDeviceSynchronize();
	    long afterDeviceSync = System.nanoTime();
	
	    // Copy the result from the device to the host
	    float deviceResult[] = { -1.0f };
	    cudaMemcpy(Pointer.to(deviceResult), deviceResultPointer, 
	        Sizeof.FLOAT, cudaMemcpyDeviceToHost);
	
	    // Print the result and timing information
	    double deviceCallDuration = (afterDeviceCall - beforeDeviceCall) / 1e6;
	    double deviceFullDuration = (afterDeviceSync - beforeDeviceCall) / 1e6;
	    System.out .println(
	        "Device call duration: " + deviceCallDuration + " ms");
	    System.out.println(
	        "Device full duration: " + deviceFullDuration + " ms");
	    System.out.println("Result: " + deviceResult[0]);
	
	    // Clean up
	    cudaFree(deviceData);
	    cublasDestroy(handle);
	  
    }    
    
    public static void testVectorAddition()
    {
    	
    	
    	JCublas2.setExceptionsEnabled(true);
	    JCuda.setExceptionsEnabled(true);
    	
	    // Create a CUBLAS handle
	    cublasHandle handle = new cublasHandle();
	    cublasCreate(handle);
	    
	    int n = 15;
	    double hostData[] = new double[n];
	    Arrays.fill(hostData,  1.0);
	    
	    // cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
	    // Allocate device memory, and copy the input data to the device
	    Pointer deviceData = new Pointer();
	    cudaMalloc(deviceData, n * Sizeof.DOUBLE);
	    cudaMemcpy(deviceData, Pointer.to(hostData), n * Sizeof.DOUBLE,
	        cudaMemcpyHostToDevice);  
	    
	    Pointer alpha = Pointer.to(new double[] {3.0});
	    
    	cublasDaxpy(handle,n, alpha, deviceData, 1, deviceData, 1);
    	
    	//cudaMemcpy(Pointer.to(hostData), deviceData, n*Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
    	
    	JCublas.cublasGetVector(n, Sizeof.DOUBLE, deviceData, 1, Pointer.to(hostData), 1);
    	
//    	for(int i = 0; i < n; i++)
//    	{System.out.println(hostData[i]);}
    	
    	cudaFree(deviceData);
    	cublasDestroy(handle);
    }
    
    
    
    

    
    
    public void printMatrix() {
    	
    	//-- Copy matrix to host
        double hostOutputW[] = new double[this.size];        
        cudaMemcpy(Pointer.to(hostOutputW), this.w,  this.size * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);

        for(int i = 0; i < this.rows; i++) {
        	
        	for(int j = 0; j < this.cols; j++) System.out.print(hostOutputW[i*this.cols + j] + ", ");
        	System.out.println("");
        }
        
    }
    

    
    public void copyDataToDevice() {
    	
    	cudaMemcpy(w, Pointer.to(this.data), this.size * Sizeof.DOUBLE,
		        cudaMemcpyHostToDevice); 
    }
	
	
	public void identity()
	{
	    for(int i = 0; i < cols; i++) {this.data[i*cols + i] = 1.0;}  
	    cudaMemcpy(w, Pointer.to(this.data), this.size * Sizeof.DOUBLE,
		        cudaMemcpyHostToDevice); 		
	}
	
	
	public void clone(double[] start) {
		
		this.data = start;
		cudaMemcpy(this.w, Pointer.to(start), this.size * Sizeof.DOUBLE,
		        cudaMemcpyHostToDevice);
	}

	public static void mdfaMatrixScale(Matrix m, double d) {
		for(int i = 0; i < m.size; i++) m.data[i] *= d;	
	}	
	

}