package ch.imetrica.jcudamdfa.matrix;

import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoadData;
import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.jcublas.JCublas2.cublasDgeam;
import static jcuda.jcublas.JCublas2.cublasDgemm;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
import static jcuda.nvrtc.JNvrtc.nvrtcCompileProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcCreateProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcDestroyProgram;
import static jcuda.nvrtc.JNvrtc.nvrtcGetPTX;
import static jcuda.nvrtc.JNvrtc.nvrtcGetProgramLog;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.jcublas.JCublas;
import jcuda.jcublas.cublasHandle;
import jcuda.nvrtc.nvrtcProgram;

public class JCublasInterface {

	
	public static cublasHandle handle;
	private static nvrtcProgram program;
	private String[] programLog;
	private String[] ptx;
	private static CUmodule module; 
	private static CUfunction function;
	
	static int blockSizeX = 100;
	int blockSizeY = 100;
	
	private static String updateSourceCode = 
				        
	        "extern \"C\"" + "\n" +
	        "__global__ void add(int n, double *m1, double *m2)" + "\n" +
	        "{" + "\n" +
	        "    int i = blockIdx.x * blockDim.x + threadIdx.x;" + "\n" +
	        "    if (i<n)" + "\n" +
	        "    {" + "\n" +
	        "        m1[i] += m2[i];" + "\n" +
	        "    }" + "\n" +
	        "}";
	
	
	
	
	
	public JCublasInterface() {
		
		handle = new cublasHandle();
	    cublasCreate(handle);
	    
	    program = new nvrtcProgram();
        nvrtcCreateProgram(program, updateSourceCode, null, 0, null, null);
        nvrtcCompileProgram(program, 0, null);
                
        // Print the compilation log (for the case there are any warnings)
        programLog = new String[1];
        nvrtcGetProgramLog(program, programLog);
        System.out.println("Nonlinear Backprob Program compilation log me:\n" + programLog[0]); 
    	    	
        // Obtain the PTX ("CUDA Assembler") code of the compiled program
        ptx = new String[1];
        nvrtcGetPTX(program, ptx);
        nvrtcDestroyProgram(program);

        // Create a CUDA module from the PTX code
        module = new CUmodule();
        cuModuleLoadData(module, ptx[0]);

        // Obtain the function pointer to the "add" function from the module
        function = new CUfunction();	
	    
	}
	
	public void matrixmultip(int hA, int wA, int wB, Pointer dA, Pointer dB, Pointer out)
	{
		Pointer zero = Pointer.to(new double[]{ 0.0 });
        Pointer one = Pointer.to(new double[]{ 1.0 }); 
        Pointer temp = new Pointer();
        
        cudaMalloc(temp, hA*wB*Sizeof.DOUBLE);

        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, hA, wB, wA, one, dA, wA, dB, wB, zero, temp, hA);
        cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, wB, hA, one, temp, hA, zero, temp, hA, out, wB);	
        
        cudaFree(temp);
	}
	
	public static void matrixmultdw1(int hA, int wA, int wB, Pointer dA, Pointer dB, Pointer out)
	{
//		Pointer one = Pointer.to(new double[]{ 1.0 }); 
//        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, hA, wB, wA, one, dA, hA, dB, wB, one, out, hA);

        JCublas.cublasDgemm('N', 'T', hA, wB, wA, 1.0, dA, hA, dB, wB, 1.0, out, hA);
        
	}
	
	
	public static void add(int n, Pointer a, Pointer b) 
	{
		cuModuleGetFunction(function, module, "add");
		Pointer kernelParameters = Pointer.to(
                Pointer.to(new int[]{n}),
                Pointer.to(a),
                Pointer.to(b)
        );
		
		int gridSizeX = (n + blockSizeX - 1) / blockSizeX;
		cuLaunchKernel(function,
	            gridSizeX,  1, 1,      // Grid dimension
	            blockSizeX, 1, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            kernelParameters, null // Kernel- and extra parameters
	        );
	    cuCtxSynchronize();		
	}
	
	
}
