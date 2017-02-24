int sid = blockIdx.x*blockDim.x + threadIdx.x;                                                                                                                                
if (sid >= c_threads) return;

Float2 spinor[4][3];

for(int mu = 0 ; mu < c_nSpin ; mu++)
  for(int c1 = 0 ; c1 < c_nColor ; c1++)
    spinor[mu][c1] = inOut[(mu*c_nColor+c1)*c_stride + sid];

for(int c1 = 0 ; c1 < c_nColor ; c1++){
  inOut[(0*c_nColor+c1)*c_stride + sid] = spinor[2][c1];
  inOut[(1*c_nColor+c1)*c_stride + sid] = spinor[3][c1];
  inOut[(2*c_nColor+c1)*c_stride + sid] = spinor[0][c1];
  inOut[(3*c_nColor+c1)*c_stride + sid] = spinor[1][c1];
 }

