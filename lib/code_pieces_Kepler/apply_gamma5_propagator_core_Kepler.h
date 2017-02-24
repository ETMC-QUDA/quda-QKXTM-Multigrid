int sid = blockIdx.x*blockDim.x + threadIdx.x;                                                                                                                                
if (sid >= c_threads) return;

Float2 prop[4][4][3][3];

for(int mu = 0 ; mu < c_nSpin ; mu++)
  for(int nu = 0 ; nu < c_nSpin ; nu++)
    for(int c1 = 0 ; c1 < c_nColor ; c1++)
      for(int c2 = 0 ; c2 < c_nColor ; c2++)
        prop[mu][nu][c1][c2] = inOut[(mu*c_nSpin*c_nColor*c_nColor + nu*c_nColor*c_nColor + c1*c_nColor + c2)*c_stride + sid];


for(int nu = 0 ; nu < c_nSpin ; nu++)
  for(int c1 = 0 ; c1 < c_nColor ; c1++)
    for(int c2 = 0 ; c2 < c_nColor ; c2++){
      inOut[(0*c_nSpin*c_nColor*c_nColor + nu*c_nColor*c_nColor + c1*c_nColor + c2)*c_stride + sid] = prop[2][nu][c1][c2];
      inOut[(1*c_nSpin*c_nColor*c_nColor + nu*c_nColor*c_nColor + c1*c_nColor + c2)*c_stride + sid] = prop[3][nu][c1][c2];
      inOut[(2*c_nSpin*c_nColor*c_nColor + nu*c_nColor*c_nColor + c1*c_nColor + c2)*c_stride + sid] = prop[0][nu][c1][c2];
      inOut[(3*c_nSpin*c_nColor*c_nColor + nu*c_nColor*c_nColor + c1*c_nColor + c2)*c_stride + sid] = prop[1][nu][c1][c2];
    }
