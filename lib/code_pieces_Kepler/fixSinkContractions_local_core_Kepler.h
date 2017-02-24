int sid = blockIdx.x*blockDim.x + threadIdx.x;
int cacheIndex = threadIdx.x;
__shared__ FLOAT2 shared_cache[16*THREADS_PER_BLOCK];
int c_stride_spatial = c_stride/c_localL[3];
register FLOAT2 accum[16];

int x_id, y_id , z_id;
int r1,r2;

r1 = sid / c_localL[0];
x_id = sid - r1 * c_localL[0];
r2 = r1 / c_localL[1];
y_id = r1 - r2*c_localL[1];
z_id = r2;

int x,y,z;

x = x_id + c_procPosition[0] * c_localL[0] - x0;
y = y_id + c_procPosition[1] * c_localL[1] - y0;
z = z_id + c_procPosition[2] * c_localL[2] - z0;

FLOAT phase;
FLOAT2 expon;

FLOAT2 gammaM[4][4];
int i;
#define PROP(tex,mu,nu,a,b) ( FETCH_FLOAT2(tex,sid + it*c_stride_spatial + ( (mu*4+nu)*3*3 + a*3 + b ) * c_stride) )



for(int i = 0 ; i < 16 ; i++){
  accum[i].x = 0.; accum[i].y = 0.;
 }

if (sid < c_threads/c_localL[3]){
  for(int iop = 0 ; iop < 16 ; iop++){
    get_Operator(gammaM,iop,TESTPARTICLE,partflag);
    for(int nu = 0 ; nu < c_nSpin ; nu++)
      for(int rho = 0 ; rho < c_nSpin ; rho++)
	for(int mup = 0 ; mup < c_nSpin ; mup++)
	  for(int b = 0 ; b < c_nColor ; b++)
	    for(int ap = 0 ; ap < c_nColor ; ap++){
	      accum[iop] = accum[iop] + gammaM[nu][rho] * PROP(fwdTex,rho,mup,b,ap) * PROP(seqTex,nu,mup,b,ap);
	    }  
  }
 }

__syncthreads();

for(int imom = 0 ; imom < c_Nmoms ; imom++){
  phase = ( ((FLOAT) c_moms[imom][0]*x)/c_totalL[0] + ((FLOAT) c_moms[imom][1]*y)/c_totalL[1] + ((FLOAT) c_moms[imom][2]*z)/c_totalL[2] ) * 2. * PI;
  expon.x = cos(phase);
  expon.y = sin(phase);
  for(int iop = 0 ; iop < 16 ; iop++){
    shared_cache[iop*THREADS_PER_BLOCK + cacheIndex] = accum[iop] * expon; 
  }
  __syncthreads();
  i = blockDim.x/2;
  while (i != 0){
    if(cacheIndex < i){
      for(int iop = 0 ; iop < 16 ; iop++){
	shared_cache[iop*THREADS_PER_BLOCK + cacheIndex].x += shared_cache[iop*THREADS_PER_BLOCK + cacheIndex + i].x;
	shared_cache[iop*THREADS_PER_BLOCK + cacheIndex].y += shared_cache[iop*THREADS_PER_BLOCK + cacheIndex + i].y;
      }
    }
    __syncthreads();
    i /= 2;
  }
  
  if(cacheIndex == 0){
    for(int iop = 0 ; iop < 16 ; iop++)
      block[imom*16*gridDim.x + iop*gridDim.x + blockIdx.x] = shared_cache[iop*THREADS_PER_BLOCK + cacheIndex + 0];	
  }
  
 } // close momentum
