package ch.imetrica.jcudamdfa.regularization;

import ch.imetrica.jcudamdfa.matrix.Matrix;

public class Regularization {

	
	
	double smooth; 
	double decay_strength;
	double decay_length;
	double cross_cor; 
	double shift_constraint;
	double lag; 
	
	int L; 
	int i1;
	int i2; 
	
	int n_rep;
	
	Matrix Q_smooth; 
	Matrix Q_decay; 
	Matrix Q_cross;
	
	Regularization(int n_rep, int L, int i1, int i2, double lag, double smooth, 
			double decay_strength, double decay_length, double cross_cor, double shift_constraint) {
		
		this.cross_cor = 100*Math.tan(Math.min(cross_cor,0.999999)*Math.PI/2.0);
		this.decay_length = 100*Math.tan(Math.min(decay_length,0.999999)*Math.PI/2.0);
		this.decay_strength = 100*Math.tan(Math.min(decay_strength,0.999999)*Math.PI/2.0);
		this.smooth = 100*Math.tan(Math.min(smooth,0.999999)*Math.PI/2.0);
		this.shift_constraint = shift_constraint;
				
		this.L = L; 
		this.i1 = i1;
		this.i2 = i2; 
		this.lag = lag; 
		this.n_rep = n_rep;
		
		
		
		
		
		
	}
	
	
	private void computeRegularizationMatrices() {
		
		
		//---  create dimensions of regularization matrices
		int i,j,k,start;
		int nrep = n_rep;
		int ncols2 = L*nrep;
		int ncols = L*nrep;
		
		if(L > 2) {
		  if(i2 == 1) {
		     if(i1 == 0) ncols2 = (L-1)*nrep;
		     else ncols2 = (L-2)*nrep;
		  }
		  else {
		     if(i1 == 0) ncols2 = L*nrep;
		     else ncols2 = (L-1)*nrep;
		  }
		}

		Matrix _Q_smooth = new Matrix(L,L);                
		Matrix _Q_decay = new Matrix(L,L); 
		Matrix cross_dev = new Matrix(ncols, ncols2);
		Matrix Qdev2 = new Matrix(ncols, ncols);
		Matrix des = new Matrix(ncols, ncols2);
		
		Q_smooth = new Matrix(ncols, ncols);
		Q_decay = new Matrix(ncols,ncols);          
		Q_cross = new Matrix(ncols,ncols);    
		  

	    //--- set initial values -------
		if(L > 2) {
			
			Matrix.mdfaMatrixSet(_Q_smooth,0,0,  1.0*smooth);  
		    Matrix.mdfaMatrixSet(_Q_smooth,0,1, -2.0*smooth);  
		    Matrix.mdfaMatrixSet(_Q_smooth,0,2,  1.0*smooth);  
		    Matrix.mdfaMatrixSet(_Q_decay, 0, 0, decay_strength*Math.pow(1.0 + decay_length,  (2.0*Math.abs(0.0-lag)))); 
		 
		    Matrix.mdfaMatrixSet(_Q_smooth,1,0,  -2.0*smooth); 
		    Matrix.mdfaMatrixSet(_Q_smooth,1,1,   5.0*smooth);
		    Matrix.mdfaMatrixSet(_Q_smooth,1,2,  -4.0*smooth);
		    Matrix.mdfaMatrixSet(_Q_smooth,1,3,   1.0*smooth);
		    Matrix.mdfaMatrixSet(_Q_decay, 1, 1, decay_strength*Math.pow(1.0 + decay_length,  (2.0*Math.abs(1.0-lag)))); 

		    i=L-1;
		    Matrix.mdfaMatrixSet(_Q_smooth,i,i-2,  1.0*smooth);       
		    Matrix.mdfaMatrixSet(_Q_smooth,i,i-1, -2.0*smooth);     
		    Matrix.mdfaMatrixSet(_Q_smooth,i,i,    1.0*smooth);    
		    Matrix.mdfaMatrixSet(_Q_decay, i, i, decay_strength*Math.pow(1.0 + decay_length, (2.0*Math.abs(i-lag))));        

		    i=L-2;
		    Matrix.mdfaMatrixSet(_Q_smooth,i,i-2,  1.0*smooth); 
		    Matrix.mdfaMatrixSet(_Q_smooth,i,i-1, -4.0*smooth);
		    Matrix.mdfaMatrixSet(_Q_smooth,i,i,    5.0*smooth);
		    Matrix.mdfaMatrixSet(_Q_smooth,i,i+1, -2.0*smooth);
		    Matrix.mdfaMatrixSet(_Q_decay, i, i, decay_strength*Math.pow(1.0 + decay_length,  (2.0*Math.abs(i-lag))));

		    
		    //------------ Now do the rest -------------------------
		    for(i=2;i<L-2;i++)
		    {    
		      //Q_smooth[i,(i-2):(i+2)]<-lambda_smooth*c(1,-4,6,-4,1)    
		      Matrix.mdfaMatrixSet(_Q_decay, i, i, decay_strength*Math.pow(1.0 + decay_length,  (2.0*Math.abs(1.0*i-lag))));                  
		      Matrix.mdfaMatrixSet(_Q_smooth,i,i-2,  1.0*smooth); 
		      Matrix.mdfaMatrixSet(_Q_smooth,i,i-1, -4.0*smooth);
		      Matrix.mdfaMatrixSet(_Q_smooth,i,i,    6.0*smooth);
		      Matrix.mdfaMatrixSet(_Q_smooth,i,i+1, -4.0*smooth);
		      Matrix.mdfaMatrixSet(_Q_smooth,i,i+2, 1.0*smooth);
		    }


		    
		    
		    for(j=0;j<nrep;j++)
		    {
		      start = j*L;
		      for(i=0;i<L;i++)
		      {
		        for(k=0;k<L;k++)
		        {
		           Matrix.mdfaMatrixSet(Q_smooth, start + i, start + k, Matrix.mdfaMatrixGet(_Q_smooth,i,k));
		           Matrix.mdfaMatrixSet(Q_decay, start + i, start + k, Matrix.mdfaMatrixGet(_Q_decay,i,k));
		        }
		      }
		    }
		 }
		
		
		//---- set cross -------------------------------------------------------------
	    if(nrep > 1) {
	      for(i = 0; i < ncols; i++) Matrix.mdfaMatrixSet(Q_cross, i, i, 1.0);	    
	    }
	    
	    for(i=0;i<nrep;i++) {
	      for(j=0;j<L;j++) {
		    for(k=0;k<nrep;k++) {
		    	double val = Matrix.mdfaMatrixGet(Q_cross, i*L+j, k*L+j) - 1.0/(1.0*nrep);   
		    	Matrix.mdfaMatrixSet(Q_cross, i*L+j, j + k*L, val);
		    }
	      }
	    }
	    
	    //----------- 
	    double trace=0.0; double strace = 0.0; double ctrace = 0.0;
	    for(i=0;i<L;i++) 
	    {
	      trace = trace + Matrix.mdfaMatrixGet(Q_decay,i,i);  
	      strace = strace + Matrix.mdfaMatrixGet(Q_smooth,i,i);
	      ctrace = ctrace + Matrix.mdfaMatrixGet(Q_cross,i,i); //printf("%lf\n",Matrix.mdfaMatrixGet(Q_cross,i,i)/cross);
	    }   

	    //printf("decay_trace = %le, smooth_trace = %le, cross_trace = %le\n", trace, strace, ctrace); 
	    //printf("nrep = %u\n", nrep);
	    if(decay_strength > 0) 
	    {Matrix.mdfaMatrixScale(Q_decay,decay_strength/(nrep*trace));}// printf("decay = %le\n",decay2/(nrep*trace));}    
	    if(smooth > 0) 
	    {Matrix.mdfaMatrixScale(Q_smooth,smooth/(nrep*strace));}//  printf("smooth = %le\n",smooth/(nrep*strace));} //3.472222e-03
	    if(cross_cor > 0.0) 
	    {Matrix.mdfaMatrixScale(Q_cross, cross_cor/(nrep*ctrace));}//  printf("cross = %le\n",cross/(nrep*ctrace));}
	 
	    Matrix.mdfaMatrixScale(Q_cross, cross_cor);		
		
		
		
		
		
		
		
	}
	
	
}
