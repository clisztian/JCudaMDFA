package ch.imetrica.jcudamdfa.regularization;

import ch.imetrica.jcudamdfa.matrix.Matrix;

public class Regularization {

	
	
	double smooth; 
	double decay_strength;
	double decay_length;
	double cross_cor; 
	double shift_constraint;
	double lag; 
	double[] weight_constraint;
	
	int L; 
	int i1;
	int i2; 
	
	int n_rep;
	
	Matrix Q_smooth; 
	Matrix Q_decay; 
	Matrix Q_cross;
	Matrix des_mat; 
	Matrix w_eight;
	Matrix Q_cdev;
	
	Regularization(int n_rep, int L, int i1, int i2, double lag, double smooth, 
			double decay_strength, double decay_length, double cross_cor, double shift_constraint, 
			double[] weight_constraint) {
		
		this.cross_cor = 100*Math.tan(Math.min(cross_cor,0.999999)*Math.PI/2.0);
		this.decay_length = 100*Math.tan(Math.min(decay_length,0.999999)*Math.PI/2.0);
		this.decay_strength = 100*Math.tan(Math.min(decay_strength,0.999999)*Math.PI/2.0);
		this.smooth = 100*Math.tan(Math.min(smooth,0.999999)*Math.PI/2.0);
		this.shift_constraint = shift_constraint;
		this.weight_constraint = weight_constraint;
		
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
		
		Q_smooth = new Matrix(ncols, ncols);
		Q_decay = new Matrix(ncols,ncols);          
		Q_cross = new Matrix(ncols,ncols);    
		des_mat = new Matrix(ncols2,ncols);   

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
		
			
	    start = 0;
	    
		//-------- create general weight matrix (w_eight vector)
	    if(i1 == 1) {
	    	
	    	if(i2 == 1) {	
	    		
	    		for(j=0; j<nrep; j++) {
	              
	    		  if(lag < 1) {	  
	    			  Matrix.mdfaMatrixSet(w_eight, j*L,    -(lag-1.0)*weight_constraint[j] - shift_constraint);
		              Matrix.mdfaMatrixSet(w_eight, j*L + 1,  1.0*lag*weight_constraint[j] + shift_constraint); 
		          } 
	    		  else {
	    			  Matrix.mdfaMatrixSet(w_eight, (int)lag + j*L, weight_constraint[j] - shift_constraint);
	    			  Matrix.mdfaMatrixSet(w_eight, (int)lag + j*L + 1, shift_constraint);     
	              }
		        }
	        }
	        else {
	        	for(j=0;j<nrep; j++) {
	        		if (lag<1) {
		              Matrix.mdfaMatrixSet(w_eight, j*L, weight_constraint[j]); 
	                } 
	                else {
	                  Matrix.mdfaMatrixSet(w_eight, (int)lag + j*L,weight_constraint[j]); 
	                }
	            }
	        }
	   } 
	   else {
	    
		   if(i2 == 1) { 
			   for(j=0;j<nrep; j++) {
	            if (lag<1) {
	            	Matrix.mdfaMatrixSet(w_eight,L*j,0); 
	            	Matrix.mdfaMatrixSet(w_eight,L*j+1, shift_constraint/(1.0-lag));
	            }
	            else {
	            	Matrix.mdfaMatrixSet(w_eight, (int)lag + L*j, 0); 
	            	Matrix.mdfaMatrixSet(w_eight, (int)lag + L*j+1, shift_constraint);
	            }        
	       }
	     } 
	  }
	    
	 
	  int il; double im; double in;
	  if(i2 == 1) {    
		
		  if(i1 == 1) {
	      for(i=0;i<L-2;i++) {       
	       
	       if(lag<1) {
	         
	          for(j=0;j<nrep; j++) {          
		       start = j*L;
	           Matrix.mdfaMatrixSet(des_mat,i, i + 2 + start, 1.0); //des_mat[i,i+2+(0:(length(weight_h_exp[1,])-1))*L]<-1
	           Matrix.mdfaMatrixSet(des_mat,i, start        , i+1); //des_mat[i,1+(0:(length(weight_h_exp[1,])-1))*L]<-i
	           Matrix.mdfaMatrixSet(des_mat,i, 1 + start   ,-(i+2)); //des_mat[i,2+(0:(length(weight_h_exp[1,])-1))*L]<--(i+1)
	          }         
	       }
	       else {
	    	   if(i >= lag) {il=i+2; im = i-lag+1.0;  in = -1.0*(i-lag+2.0);} 
	    	   else {il = i; im = -(lag+2.0-(i+1)); in = 1.0*(lag+2.0-i);}
		  
	           for(j=0;j<nrep; j++) {        
		         start = j*L; 
	             Matrix.mdfaMatrixSet(des_mat,i, il + start     , 1.0); //des_mat[i,ifelse(i<lag+1,i,i+2)+(0:(length(weight_h_exp[1,])-1))*L]<-1
	             Matrix.mdfaMatrixSet(des_mat,i, (int)lag+ start     , im);  //des_mat[i,lag+1+(0:(length(weight_h_exp[1,])-1))*L]<-ifelse(i<lag+1,-(lag+2-i),i-lag)
	             Matrix.mdfaMatrixSet(des_mat,i, (int)lag + 1 + start, in);  //des_mat[i,lag+2+(0:(length(weight_h_exp[1,])-1))*L]<-ifelse(i<lag+1,(lag+1-i),-(i-lag+1))  
	          }        
	       }
	     }
	     for(j=1;j<nrep;j++) {
	       start = j*(L-2); 
	       for (i=0;i<L-2;i++)  {
		   if (lag<1) {
		        
		         Matrix.mdfaMatrixSet(des_mat,i + start, i + 2,  -1.0);  //des_mat[i+j*(L-2),i+2]<--1
	             Matrix.mdfaMatrixSet(des_mat,i + start, 0,    -(i+1));  //
	             Matrix.mdfaMatrixSet(des_mat,i + start, 1,    (i+2));   //des_mat[i+j*(L-2),2]<-(i+1)                
	       } 
	       else {
		     
		        if(i >= lag) {il=i+2; im = i-lag+1.0; in = -1.0*(i-lag+2.0);} 
		        else {il = i; im = -(lag+2.0-(i+1.0)); in = lag+1.0-(i+1.0);}
		       
		        Matrix.mdfaMatrixSet(des_mat, i + start, il, -1.0); //des_mat[i+j*(L-2),ifelse(i<lag+1,i,i+2)]<--1
		        Matrix.mdfaMatrixSet(des_mat, i + start, (int)lag, -im);  //des_mat[i+j*(L-2),lag+1]<--ifelse(i<lag+1,-(lag+2-i),i-lag)
		        Matrix.mdfaMatrixSet(des_mat, i + start, (int)lag+1, -in); //des_mat[i+j*(L-2),lag+2]<--ifelse(i<lag+1,(lag+1-i),-(i-lag+1))  
	       }
           if (lag<1) {
		        
		         Matrix.mdfaMatrixSet(des_mat,i + start, i + 2 + j*L, 1.0);  //des_mat[i+j*(L-2),i+2]<--1
	             Matrix.mdfaMatrixSet(des_mat,i + start, j*L,   (i+1.0));  //
	             Matrix.mdfaMatrixSet(des_mat,i + start, 1 + j*L, -(i+2.0));   //des_mat[i+j*(L-2),2]<-(i+1)                
	       } 
	       else {
		     
		        if(i >= lag) {il=i+2; im = i-lag+1.0; in = -1.0*(i-lag+2.0);} 
		        else {il = i; im = -(lag+2.0-(i+1.0)); in = lag+1.0-(i+1.0);}
		       
		        Matrix.mdfaMatrixSet(des_mat, i + start, il + j*L, -1.0); //des_mat[i+j*(L-2),ifelse(i<lag+1,i,i+2)]<--1
		        Matrix.mdfaMatrixSet(des_mat, i + start, (int)lag + j*L, im);  //des_mat[i+j*(L-2),lag+1]<--ifelse(i<lag+1,-(lag+2-i),i-lag)
		        Matrix.mdfaMatrixSet(des_mat, i + start, (int)lag+1 + j*L, in); //des_mat[i+j*(L-2),lag+2]<--ifelse(i<lag+1,(lag+1-i),-(i-lag+1))  
	       }  
	     }
	   }
	  }
	  else {
	    for (i=0;i<L-1;i++) {

		  for(j=0;j<nrep;j++)
		  {
		    	    	    
		    if(lag<1)  {
		      
		      if(i < 1) {il=i; im = lag/(1.0-lag);} else {il=i+1; im = (lag-(i+1.0))/(1.0-lag);} 
		  
		      Matrix.mdfaMatrixSet(des_mat, i, il + j*L, 1.0); 
		      Matrix.mdfaMatrixSet(des_mat, i, 1 + j*L, im); 

		    } 
		    else {
		      
		      if(i < lag+1) {il=i; im = lag + 1.0 - (i+1.0);} 
		      else {il=i+1; im = lag - (i+1);} 
		  
		      Matrix.mdfaMatrixSet(des_mat, i, il + j*L, 1.0); 
		      Matrix.mdfaMatrixSet(des_mat, i, (int)lag+1 + j*L, im);  
		     
	        }
	      }
	   }
	      
	   for (j=1;j<nrep;j++) {
		for (i=0;i<L-1;i++)  {    
	       
		    if (lag<1) {
		        if(i < 1) {il=i; im = lag/(1.0-lag);} else {il=i+1; im = (lag-(i+1.0))/(1.0-lag);}
		        Matrix.mdfaMatrixSet(des_mat, i + j*(L-1), il, -1.0); //des_mat[i+j*(L-1),ifelse(i<2,i,i+1)]<--1
		        Matrix.mdfaMatrixSet(des_mat, i + j*(L-1), 1, -im);   //des_mat[i+j*(L-1),1+1]<--ifelse(i==1,lag,lag-i)/(1-lag)
	        } 
	        else {
			  if(i < lag+1) {il=i; im = lag+1.0-(i+1.0);} else {il=i+1; im = (lag-(i+1.0));}
			  Matrix.mdfaMatrixSet(des_mat, i + j*(L-1), il, -1.0);
			  Matrix.mdfaMatrixSet(des_mat, i + j*(L-1), (int)lag+1, -im); 
	        }
 
		    if (lag<1) {
			
		        if(i < 1) {il=i; im = lag/(1.0-lag);} else {il=i+1; im = (lag-(i+1.0))/(1.0-lag);} 
		        Matrix.mdfaMatrixSet(des_mat, i+j*(L-1), il + j*L, 1.0); //des_mat[i+j*(L-1),ifelse(i<2,i,i+1)+j*L]<-1
		        Matrix.mdfaMatrixSet(des_mat, i+j*(L-1), 1 + j*L, im);  //des_mat[i+j*(L-1),1+1+j*L]<-ifelse(i==1,lag,lag-i)/(1-lag)
			           
	        } 
	        else {
			  
	        	if(i < lag+1) {il=i; im = lag+1.0-(i+1.0);} else {il=i+1; im = (lag-(i+1.0));}
			    Matrix.mdfaMatrixSet(des_mat, i+j*(L-1), il + j*L, 1.0); //des_mat[i+j*(L-1),ifelse(i<2,i,i+1)+j*L]<-1
		        Matrix.mdfaMatrixSet(des_mat, i+j*(L-1), (int)lag+1 + j*L, im);  //des_mat[i+j*(L-1),1+1+j*L]<-ifelse(i==1,lag,lag-i)/(1-lag)
			
	        }                        
	      }
	    }
	  }
	}
	else {
	  if (i1==1)  {
	      
	      for(i=0; i < L-1; i++)  {
		    for(j=0;j<nrep;j++) {

	         if (lag<1) {
		      Matrix.mdfaMatrixSet(des_mat, i, i + 1 + j*L, 1.0); //des_mat[i,i+1+(0:(length(weight_h_exp[1,])-1))*L]<-1
		      Matrix.mdfaMatrixSet(des_mat, i, j*L, -1.0);     //des_mat[i,1+(0:(length(weight_h_exp[1,])-1))*L]<--1
	         } 
	         else {
		      if(i >= lag) {il = i+1;} else {il = i;} 
		      Matrix.mdfaMatrixSet(des_mat, i, il + j*L, 1.0); 
		      Matrix.mdfaMatrixSet(des_mat, i, (int)lag + j*L, -1.0);     
	         }
		   }
	     }
	     for (j=1;j<nrep;j++) {
		    start = j*(L-1);
	        for (i=0;i<L-1;i++) {
	          
	          if (lag<1) {    
		      
	              Matrix.mdfaMatrixSet(des_mat, i+start, i+1, -1.0);  //des_mat[i+j*(L-1),i+1]<--1
	              Matrix.mdfaMatrixSet(des_mat, i+start, 0, 1.0);  //des_mat[i+j*(L-1),1]<-1
	              Matrix.mdfaMatrixSet(des_mat, i+start, i+1+j*L, 1.0);  //des_mat[i+j*(L-1),i+1+j*L]<-1
	              Matrix.mdfaMatrixSet(des_mat, i+start, j*L, -1.0); //des_mat[i+j*(L-1),1+j*L]<--1
	          } 
	          else {
		      
		       if(i >= lag) {il = i+1;} else {il = i;} 
	            Matrix.mdfaMatrixSet(des_mat, i+start, il, -1.0); //des_mat[i+j*(L-1),ifelse(i<lag+1,i,i+1)]<--1
	            Matrix.mdfaMatrixSet(des_mat, i+start, (int)lag, 1.0); //des_mat[i+j*(L-1),lag+1]<-1
	            Matrix.mdfaMatrixSet(des_mat, i+start, il + j*L, 1.0);  //des_mat[i+j*(L-1),ifelse(i<lag+1,i,i+1)+j*L]<-1
	            Matrix.mdfaMatrixSet(des_mat, i+start, (int)lag + j*L, -1.0); //des_mat[i+j*(L-1),lag+1+j*L]<--1
	          }
	         }
	     }             
	    }
	    else {
	     for(j=0;j<nrep;j++) { 
	      start = j*L; 
	      for (i=0;i<L;i++) {
	        Matrix.mdfaMatrixSet(des_mat, i, i + j*L, 1.0); // des_mat[i,i+(0:(length(weight_h_exp[1,])-1))*L]<-1
	      }
	     }
	     for(j=1;j<nrep;j++) {
	       start = j*L;
	       for (i=0;i<L;i++) {
	           Matrix.mdfaMatrixSet(des_mat, i+start, i, -1.0);  //des_mat[i+(j)*(L),i]<--1
	           Matrix.mdfaMatrixSet(des_mat, i+start, i+start, 1.0); 
	       } 
	     }
	    }      
	  }
	    
	  if(nrep > 1){
	        
	        //A*At
		    Matrix.mdfaMatrixMult(Q_cdev, des_mat, cross_dev);  

	        //res = gsl_matrix_transpose_memcpy(des_mat, cross_dev);
	        //reg_t<-(Q_smooth+Q_decay+Q_cross)
	        Matrix.mdfaMatrixAdd(Q_smooth, Q_decay);
	        Matrix.mdfaMatrixAdd(Q_smooth, Q_cross);
	        
	  }
	  else 
	  {Matrix.mdfaMatrixAdd(Q_smooth, Q_decay);}  
	    
	    
	    
	    
	    
	}
	
	
}
