int sid = blockIdx.x*blockDim.x + threadIdx.x;
int c_stride_spatial = c_stride/c_localL[3];
register FLOAT2 accum[16];

FLOAT2 gammaM[4][4];

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

if (sid < c_threads/c_localL[3]){
  for(int iop=0;iop<16;iop++)  block[iop + 16*sid] = accum[iop];  
 }
__syncthreads();
