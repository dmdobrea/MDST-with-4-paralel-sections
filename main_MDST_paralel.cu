#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <math.h>

__global__ void MDST4 (double *xa, double *xb, double *Ysb, double *Ycb);
void init(void);

#define myPI 3.14159265358979323846
#define N1   34
#define N    17   //N = N1/2 

/*================================================
 *   MDST kernel with four sections
 =================================================*/

__global__ void MDST4 (double *xa, double *xb, double *Ysb, double *Ycb)
{
	int i;
	
	__shared__ double xcb[N]; 
	__shared__ double xcc[N];

	__shared__ double PvA[5];
	__shared__ double PvB[5]; 
	__shared__ double vA[4];    
	__shared__ double vB[4];

	__shared__ double Tsb1a [4];
	__shared__ double Tsb1b [4];		
	__shared__ double Tcb1a [4];		
	__shared__ double Tcb1b [4];		
	
	/* ================= xcb, xcc ================= */

    xcb[N-1] = xb[N-1];				
	xcc[N-1] = xa[N-1];

	for(i=N-2;i>=0;i--)
		{
        xcb[i] = xb[i]-xcb[i+1];
        xcc[i] = xa[i]-xcc[i+1];
		}

	// =============================> Starting from here we go in paralele mode
	int tid = blockIdx.x;     // for <<<4, 1>>>

	//================= PROCESS 1
	switch (tid)
	{
	/* Tsb1a = C*2*D^-1*diag(P*cA)*P*vA + C*2*D^-1*diag(P*cB)*P*vB */
	/*															   */
	/*           cA = c2546                      cB = c7013        */
	/*	Tm_c2546 = 2*D^-1*diag(P*cA)  Tm_7013 = 2*D^-1*diag(P*cB)  */
		case 0:
			vA[0] = xcc[3] + xcc[14]; vA[1] = xcc[7] + xcc[10]; vA[2] = xcc[5] + xcc[12]; vA[3] = xcc[6] + xcc[11];
			vB[0] = xcc[8] + xcc[9];  vB[1] = xcc[4] + xcc[13]; vB[2] = xcc[2] + xcc[15]; vB[3] = xcc[1] + xcc[16];
			
			//first half
			PvA[0] = ( vA[0] + vA[1] + vA[2] + vA[3]                 ) * (-0.6403882032);  //Tm_2546[0];
			PvA[1] = ( vA[0] - vA[1] + vA[2] - vA[3]                 ) * (-0.8124635689);  //Tm_2546[1];
			PvA[2] = ( vA[0]         - vA[2]                         ) *   0.1237912497;   //Tm_2546[2];	
			PvA[3] = ( vA[0] + vA[1] - vA[2] - vA[3] + vA[1] - vA[3] ) * (-0.5956100962);  //Tm_2546[3];
			PvA[4] = (         vA[1]         - vA[3]                 ) * (-0.7194013458);  //Tm_2546[4];	
			
			// second half
			PvB[0] = ( vB[0] + vB[1] + vB[2] + vB[3]                 ) * 0.3903882032;   //Tm_7013[0];
			PvB[1] = ( vB[0] - vB[1] + vB[2] - vB[3]                 ) * 0.6343523857;   //Tm_7013[1];
			PvB[2] = ( vB[0]         - vB[2]                         ) * 0.4201019350;   //Tm_7013[2]; 	
			PvB[3] = ( vB[0] + vB[1] - vB[2] - vB[3] + vB[1] - vB[3] ) * 2.1420839519;   //Tm_7013[3];
			PvB[4] = (         vB[1]         - vB[3]                 ) * 1.7219820169;   //Tm_7013[4];	
			
			Tsb1a[0] = PvB[0] + PvB[1] + PvB[2] +        - PvB[4] + PvB[2] + 
			           PvA[0] + PvA[1] + PvA[2]          - PvA[4] + PvA[2];
			Tsb1a[1] = PvB[0] - PvB[1] - PvB[2] + PvB[3] - PvB[4] - PvB[4] + 
				       PvA[0] - PvA[1] - PvA[2] + PvA[3] - PvA[4] - PvA[4];
			Tsb1a[2] = PvB[0] + PvB[1] - PvB[2] +        + PvB[4] - PvB[2] + 
			           PvA[0] + PvA[1] - PvA[2]          + PvA[4] - PvA[2];
			Tsb1a[3] = PvB[0] - PvB[1] + PvB[2] - PvB[3] + PvB[4] + PvB[4] +
			           PvA[0] - PvA[1] + PvA[2] - PvA[3] + PvA[4] + PvA[4];
	
			Ysb[0]  = (xcc[0] + Tsb1a[3]) * 0.9829730996839018; // -0.9829730996839018 = cos(2*8*alpha1)
			Ysb[1]  = (xcc[0] + Tsb1a[0]) * 0.9324722294043558; // cos(2*1*alpha1)
			Ysb[3]  = (xcc[0] + Tsb1a[1]) * 0.7390089172206591; // cos(2*2*alpha1);
			Ysb[7]  = (xcc[0] + Tsb1a[2]) * 0.0922683594633020; // cos(2*4*alpha1);
		break;
	
	//================= PROCESS 2
		/* Tsb1a = C*2*D^-1*diag(P*cA)*P*vA + C*2*D^-1*diag(P*cB)*P*vB */
		/*															   */
		/*           cA = c0137                      cB = c2546        */
		/*	Tm_0137 = 2*D^-1*diag(P*cA)	  Tm_2546 = 2*D^-1*diag(P*cB)  */	
		case 1:
			vA[0] = xcc[3] + xcc[14]; vA[1] = xcc[7] + xcc[10]; vA[2] = xcc[5] + xcc[12]; vA[3] = xcc[6] + xcc[11];
			vB[0] = xcc[8] + xcc[9];  vB[1] = xcc[4] + xcc[13]; vB[2] = xcc[2] + xcc[15]; vB[3] = xcc[1] + xcc[16];


			//first half
			PvA[0] = ( vA[0] + vA[1] + vA[2] + vA[3]                 ) *   0.3903882032;  //Tm_0137[0];
			PvA[1] = ( vA[0] - vA[1] + vA[2] - vA[3]                 ) * (-0.6343523857); //Tm_0137[1];
			PvA[2] = ( vA[0]         - vA[2]                         ) *   0.8609910085;  //Tm_0137[2];	
			PvA[3] = ( vA[0] + vA[1] - vA[2] - vA[3] + vA[1] - vA[3] ) *   0.0207871385;  //Tm_0137[3];
			PvA[4] = (         vA[1]         - vA[3]                 ) * (-0.8402038699); //Tm_0137[4];	
			
			// second half
			PvB[0] = ( vB[0] + vB[1] + vB[2] + vB[3]                 ) * (-0.6403882032);  //Tm_2546[0];
			PvB[1] = ( vB[0] - vB[1] + vB[2] - vB[3]                 ) * (-0.8124635689);  //Tm_2546[1];
			PvB[2] = ( vB[0]         - vB[2]                         ) *   0.1237912497;   //Tm_2546[2];	
			PvB[3] = ( vB[0] + vB[1] - vB[2] - vB[3] + vB[1] - vB[3] ) * (-0.5956100962);  //Tm_2546[3];
			PvB[4] = (         vB[1]         - vB[3]                 ) * (-0.7194013458);  //Tm_2546[4];	

			Tsb1b[0] = PvB[0] + PvB[1] + PvB[2] +        - PvB[4] + PvB[2] +
			           PvA[0] + PvA[1] + PvA[2] +        - PvA[4] + PvA[2];
			Tsb1b[1] = PvB[0] - PvB[1] - PvB[2] + PvB[3] - PvB[4] - PvB[4] +
			  	       PvA[0] - PvA[1] - PvA[2] + PvA[3] - PvA[4] - PvA[4];
			Tsb1b[2] = PvB[0] + PvB[1] - PvB[2] +        + PvB[4] - PvB[2] +
			           PvA[0] + PvA[1] - PvA[2] +        + PvA[4] - PvA[2];
			Tsb1b[3] = PvB[0] - PvB[1] + PvB[2] - PvB[3] + PvB[4] + PvB[4] +
				       PvA[0] - PvA[1] + PvA[2] - PvA[3] + PvA[4] + PvA[4]; 	
	
			/* Calcul Ysb */
			Ysb[2] = (xcc[0] + Tsb1b[2]) * 0.8502171357296142; // -0.8502171357296142 = cos(2.0 * 7.0 * alpha1);
			Ysb[4] = (xcc[0] + Tsb1b[0]) * 0.6026346363792563; // -0.6026346363792563 = cos(2.0 * 6.0 * alpha1);
			Ysb[5] = (xcc[0] + Tsb1b[3]) * 0.4457383557765383; // cos(2.0 * 3.0 * alpha1);
			Ysb[6] = (xcc[0] + Tsb1b[1]) * 0.2736629900720829; //-0.2736629900720829 = cos(2.0 * 5.0 * alpha1);
		break;
	
	//================= PROCESS 3
	/* Tsb1a = C*2*D^-1*diag(P*cA)*P*vA + C*2*D^-1*diag(P*cB)*P*vB */
	/*															   */
	/*           cA = c2546                      cB = c7013        */
	/*	Tm_c2546 = 2*D^-1*diag(P*cA)  Tm_7013 = 2*D^-1*diag(P*cB)  */		
		case 2:
			vA[0] = xcb[3] + xcb[14]; vA[1] = xcb[7] + xcb[10]; vA[2] = xcb[5] + xcb[12]; vA[3] = xcb[6] + xcb[11];
			vB[0] = xcb[8] + xcb[9];  vB[1] = xcb[4] + xcb[13]; vB[2] = xcb[2] + xcb[15]; vB[3] = xcb[1] + xcb[16];

			//first half
			PvA[0] = ( vA[0] + vA[1] + vA[2] + vA[3]                 ) * -0.6403882032; //Tm_2546[0];
			PvA[1] = ( vA[0] - vA[1] + vA[2] - vA[3]                 ) * -0.8124635689; //Tm_2546[1];
			PvA[2] = ( vA[0]         - vA[2]                         ) *  0.1237912497; //Tm_2546[2];	
			PvA[3] = ( vA[0] + vA[1] - vA[2] - vA[3] + vA[1] - vA[3] ) * -0.5956100962; //Tm_2546[3];
			PvA[4] = (         vA[1]         - vA[3]                 ) * -0.7194013458; //Tm_2546[4];	
			
			// second half
			PvB[0] = ( vB[0] + vB[1] + vB[2] + vB[3]                 ) *  0.3903882032; //Tm_7013[0];
			PvB[1] = ( vB[0] - vB[1] + vB[2] - vB[3]                 ) *  0.6343523857; //Tm_7013[1];
			PvB[2] = ( vB[0]         - vB[2]                         ) *  0.4201019350; //Tm_7013[2];	
			PvB[3] = ( vB[0] + vB[1] - vB[2] - vB[3] + vB[1] - vB[3] ) *  2.1420839519; //Tm_7013[3];
			PvB[4] = (         vB[1]         - vB[3]                 ) *  1.7219820169; //Tm_7013[4];	

			Tcb1a[0] = PvB[0] + PvB[1] + PvB[2] +        - PvB[4] + PvB[2] +
				       PvA[0] + PvA[1] + PvA[2] +        - PvA[4] + PvA[2];
			Tcb1a[1] = PvB[0] - PvB[1] - PvB[2] + PvB[3] - PvB[4] - PvB[4] +
			           PvA[0] - PvA[1] - PvA[2] + PvA[3] - PvA[4] - PvA[4];
			Tcb1a[2] = PvB[0] + PvB[1] - PvB[2] +        + PvB[4] - PvB[2] +
			           PvA[0] + PvA[1] - PvA[2] +        + PvA[4] - PvA[2];
			Tcb1a[3] = PvB[0] - PvB[1] + PvB[2] - PvB[3] + PvB[4] + PvB[4] + 
				       PvA[0] - PvA[1] + PvA[2] - PvA[3] + PvA[4] + PvA[4];

			Ycb[0] = (xcb[0] + Tcb1a[3]) * 0.9829730996839018; // -0.9829730996839018 = cos(2*8*alpha1)		
			Ycb[1] = (xcb[0] + Tcb1a[0]) * 0.9324722294043558; // cos(2*1*alpha1)  
			Ycb[3] = (xcb[0] + Tcb1a[1]) * 0.7390089172206591; // cos(2*2*alpha1); 
			Ycb[7] = (xcb[0] + Tcb1a[2]) * 0.0922683594633020; // cos(2*4*alpha1);	  
		break;

	//================= PROCESS 4
	/* Tsb1a = C*2*D^-1*diag(P*cA)*P*vA + C*2*D^-1*diag(P*cB)*P*vB */
	/*															   */
	/*           cA = c0137                      cB = c2546        */
	/*	Tm_0137 = 2*D^-1*diag(P*cA)	  Tm_2546 = 2*D^-1*diag(P*cB)  */		
		case 3:
			vA[0] = xcb[3] + xcb[14]; vA[1] = xcb[7] + xcb[10]; vA[2] = xcb[5] + xcb[12]; vA[3] = xcb[6] + xcb[11];
			vB[0] = xcb[8] + xcb[9];  vB[1] = xcb[4] + xcb[13]; vB[2] = xcb[2] + xcb[15]; vB[3] = xcb[1] + xcb[16];

			//first half
			PvA[0] = ( vA[0] + vA[1] + vA[2] + vA[3]                 ) *  0.3903882032; //Tm_0137[0];
			PvA[1] = ( vA[0] - vA[1] + vA[2] - vA[3]                 ) * -0.6343523857; //Tm_0137[1];
			PvA[2] = ( vA[0]         - vA[2]                         ) *  0.8609910085; //Tm_0137[2];	
			PvA[3] = ( vA[0] + vA[1] - vA[2] - vA[3] + vA[1] - vA[3] ) *  0.0207871385; //Tm_0137[3];
			PvA[4] = (         vA[1]         - vA[3]	             ) * -0.8402038699; //Tm_0137[4];
			
			// second half
			PvB[0] = ( vB[0] + vB[1] + vB[2] + vB[3]                 ) * -0.6403882032; //Tm_2546[0];
			PvB[1] = ( vB[0] - vB[1] + vB[2] - vB[3]                 ) * -0.8124635689; //Tm_2546[1];
			PvB[2] = ( vB[0]         - vB[2]                         ) *  0.1237912497; //Tm_2546[2];	
			PvB[3] = ( vB[0] + vB[1] - vB[2] - vB[3] + vB[1] - vB[3] ) * -0.5956100962; //Tm_2546[3];
			PvB[4] = (         vB[1]         - vB[3]                 ) * -0.7194013458; //Tm_2546[4];	
			
			Tcb1b[0] = PvB[0] + PvB[1] + PvB[2] +        - PvB[4] + PvB[2] +
			           PvA[0] + PvA[1] + PvA[2] +        - PvA[4] + PvA[2];
			Tcb1b[1] = PvB[0] - PvB[1] - PvB[2] + PvB[3] - PvB[4] - PvB[4] +
			           PvA[0] - PvA[1] - PvA[2] + PvA[3] - PvA[4] - PvA[4];
			Tcb1b[2] = PvB[0] + PvB[1] - PvB[2] +        + PvB[4] - PvB[2] + 
			           PvA[0] + PvA[1] - PvA[2] +        + PvA[4] - PvA[2];
			Tcb1b[3] = PvB[0] - PvB[1] + PvB[2] - PvB[3] + PvB[4] + PvB[4] +
			           PvA[0] - PvA[1] + PvA[2] - PvA[3] + PvA[4] + PvA[4];				

			Ycb[2] = (xcb[0] + Tcb1b[2]) * 0.8502171357296142; // -0.8502171357296142 = cos(2.0 * 7.0 * alpha1);   
			Ycb[4] = (xcb[0] + Tcb1b[0]) * 0.6026346363792563; // -0.6026346363792563 = cos(2.0 * 6.0 * alpha1);  
			Ycb[5] = (xcb[0] + Tcb1b[3]) * 0.4457383557765383; // = cos(2.0 * 3.0 * alpha1); 
			Ycb[6] = (xcb[0] + Tcb1b[1]) * 0.2736629900720829; // -0.2736629900720829 = cos(2.0 * 5.0 * alpha1); 		
		break;
	}	
}

#define myPI 3.14159265358979323846
#define N1   34
#define N    17   //N = N1/2 

// input
	double x[N1];  

// output
	double Y2[N] = {0.0};
	double Yreal[N];

double xs[N1];
double xa[N], xb[N];

double Ysb[8],  Ycb[8];		//with these I build T => Y  |  8 = (N-1)/2

double T[N];
double Y[N];

double alpha;
double alpha1;

double sin_xs[N1];

int main (void)
{
	int i, k;
	
//==================> random generation of data ======================
    /* random init*/
	srand(time(NULL));
	
	for (i = 0; i < N1; i++)	// N1 = 34
		x[i] = (double)rand() / RAND_MAX;   //Random(-1.0, 1.0); 
	
//==================> implementare new paralel MDST 
	init();

	// create events
	cudaEvent_t start, stop;

	cudaEventCreate (&start);
	cudaEventCreate (&stop);

	// device (GPU) arrays
	double *in_gpu_xa, *in_gpu_xb, *out_gpu_Ysb, *out_gpu_Ycb;

	//alloc vectors on the GPU memory
	cudaMalloc((void**) &in_gpu_xa,  N * sizeof(double));
	cudaMalloc((void**) &in_gpu_xb,  N * sizeof(double));

	cudaMalloc((void**) &out_gpu_Ysb, 8 * sizeof(double));
	cudaMalloc((void**) &out_gpu_Ycb, 8 * sizeof(double));
	
	cudaEventRecord (start);

	// ==================================> the new algorithm START HERE	

	for(i = 0; i < N1; i++)
	  	xs[i] = x[i] * sin_xs[i];		

	/* ================= c => xs => xa, xb ================= */
	int semn = -1;
	double diff;
	
	for(i=0; i<N; i++)
		{
		xa[i]=xs[i]+xs[N1-1-i];
		
		//xb[i]=pow(-1.0,i+1)*(xs[i]-xs[N1-1-i]);	//asa era inainte
		diff  = xs[i]-xs[N1-1-i];
		xb[i] = (semn < 0) ? -diff : diff;
		semn  = -semn;
		}

	// send data to GPU
	cudaMemcpy( in_gpu_xa, xa, 136, cudaMemcpyHostToDevice );	// 17 (N) * 8 (sizeof(double)) = 126
	cudaMemcpy( in_gpu_xb, xb, 136, cudaMemcpyHostToDevice );
	
	// start computation
	MDST4 <<<4, 1>>> ( in_gpu_xa, in_gpu_xb, out_gpu_Ysb, out_gpu_Ycb );

	cudaMemcpy( Ysb, out_gpu_Ysb, 64, cudaMemcpyDeviceToHost );  	// 8 * 8 (sizeof(double)) = 64
	cudaMemcpy( Ycb, out_gpu_Ycb, 64, cudaMemcpyDeviceToHost );
	
	// T
	double ta, tb;
	semn = -1;
	for ( k = 1; k <= (N-1)/2; k++)     // N = 17   k = 1..8 Ysb,Ycb[0, 7]
	{
		//T[2*k - 1]   = pow(-1.0,k) * 2.0 * Ysb [k-1];
		//T[N-2*k - 1] = pow(-1.0,k) * 2.0 * Ycb [k-1];

		ta = Ysb [k-1] + Ysb [k-1];				// 2.0 * Ysb [k-1];
		tb = Ycb [k-1] + Ycb [k-1];				// 2.0 * Ycb [k-1]
		T[2*k - 1]   = (semn < 0) ? -ta : ta;
		T[N-2*k - 1] = (semn < 0) ? -tb : tb;

		semn  = -semn;
	}	
	
	// Y
	Y[0] = 0.0;
	for ( i = 0; i < N; i++)
		Y[0] += xa[i];
		
	// get the output values
	for (k = 1; k < N; k++) 
		Y[k] = T[k-1] + Y[k-1];
	
	// ==================================> the new algorithm END HERE
	cudaEventRecord (stop);
	cudaEventSynchronize(stop);
		
	float milliseconds = 0;
	cudaEventElapsedTime (&milliseconds, start, stop);

//==================> END!!!!		
	getchar();
	return 0;
}

void init(void)
{
	alpha   = myPI / (2.0 * N1);
	alpha1  = 4.0 * alpha;
	
	/* ========= constants computation ========== */
	// sin_xs[i] = sin((2.0*i + 1.0 + N1/2.0) * alpha);
	double sin_cst[N1] = { 0.7390089172,   0.7980172273,   0.8502171357,   0.8951632914,   0.9324722294,   
	 	                   0.9618256432,   0.9829730997,   0.9957341763,   1.0000000000,   0.9957341763,     
		                   0.9829730997,   0.9618256432,   0.9324722294,   0.8951632914,   0.8502171357,   
		                   0.7980172273,   0.7390089172,   0.6736956436,   0.6026346364,   0.5264321629,   
		                   0.4457383558,   0.3612416662,   0.2736629901,   0.1837495178,   0.0922683595,   
		                   0.0000000000,  -0.0922683595,  -0.1837495178,  -0.2736629901,  -0.3612416662,  
		                  -0.4457383558,  -0.5264321629,  -0.6026346364,  -0.6736956436};
	for (int i=0; i<N1; i++)
		sin_xs[i] = sin_cst[i];
}
