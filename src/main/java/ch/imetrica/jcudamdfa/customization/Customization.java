package ch.imetrica.jcudamdfa.customization;

import ch.imetrica.jcudamdfa.matrix.Matrix;

public class Customization {

	
	
	Matrix gramm;
	
	
	
	public void constructGrammMatrix(int nrep, double[] Gamma, int L, double lambda, double lag) {
		
		int K1 = Gamma.length;
		int K = K1-1;
		int i,j,l;
		int Lag = (int)lag;
		double M_PI = Math.PI;
		
		for(i=0;i<nrep;i++) 
		{
			for(l=1;l<=L;l++) 
			{      
		       for(j=0;j<K1;j++)
		       {
		    	   cexp(I*(l-1-Lag)*M_PI*j/K)

		           Xc = creal(cexp(I*(l-1-Lag)*M_PI*j/K)*eweight[j][i]) + Math.sqrt(1 + Gamma[j]*lambda)*I*(cimag(cexp(I*(l-1-Lag)*M_PI*j/K)*eweight[j][i]));
		           
		           gsl_matrix_set(REX, j, L*i +(l-1), creal(Xc));  gsl_matrix_set(IMX, j, L*i +(l-1), cimag(Xc)); 			   
	            }
		      }
		}
		
		
	}
	
	
	
}
