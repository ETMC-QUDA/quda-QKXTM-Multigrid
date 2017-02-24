  for(int mu = 0 ; mu < 4 ; mu++)
    for(int nu = 0 ; nu < 4 ; nu++){
      projector[mu][nu].x = 0 ; projector[mu][nu].y = 0;
    }

  if( PID == G4 ){
    if(PARTICLE == PROTON){
      projector[0][0].x=0.25; projector[1][1].x=0.25; projector[2][2].x=-0.25;  projector[3][3].x=-0.25;
      projector[0][2].y=0.25; projector[1][3].y=0.25; projector[2][0].y=0.25;   projector[3][1].y=0.25;
    }
    else if(PARTICLE == NEUTRON){
      projector[0][0].x=0.25; projector[1][1].x=0.25; projector[2][2].x=-0.25;  projector[3][3].x=-0.25;
      projector[0][2].y=-0.25; projector[1][3].y=-0.25; projector[2][0].y=-0.25; projector[3][1].y=-0.25;
    }
  }
  else if( PID == G5G123 ){
    if(PARTICLE == PROTON){
      projector[0][0].x = 0.25;
      projector[0][1].x = 0.25; projector[0][1].y = -0.25;
      projector[0][2].y = 0.25;
      projector[0][3].x = 0.25; projector[0][3].y = 0.25;
      projector[1][0].x = 0.25; projector[1][0].y = 0.25;
      projector[1][1].x = -0.25;
      projector[1][2].x = -0.25; projector[1][2].y = 0.25;
      projector[1][3].y = -0.25;
      projector[2][0].y = 0.25;
      projector[2][1].x = 0.25; projector[2][1].y = 0.25;
      projector[2][2].x = -0.25;
      projector[2][3].x = -0.25; projector[2][3].y = 0.25;
      projector[3][0].x = -0.25; projector[3][0].y = 0.25;
      projector[3][1].y = -0.25;
      projector[3][2].x = -0.25; projector[3][2].y = -0.25;
      projector[3][3].x = 0.25;
    }
    else if(PARTICLE == NEUTRON){
      projector[0][0].x = 0.25;
      projector[0][1].x = 0.25; projector[0][1].y = -0.25;
      projector[0][2].y = -0.25;
      projector[0][3].x = -0.25; projector[0][3].y = -0.25;
      projector[1][0].x = 0.25; projector[1][0].y = 0.25;
      projector[1][1].x = -0.25;
      projector[1][2].x = 0.25; projector[1][2].y = -0.25;
      projector[1][3].y = 0.25;
      projector[2][0].y = -0.25;
      projector[2][1].x = -0.25; projector[2][1].y = -0.25;
      projector[2][2].x = -0.25;
      projector[2][3].x = -0.25; projector[2][3].y = 0.25;
      projector[3][0].x = 0.25; projector[3][0].y = -0.25;
      projector[3][1].y = 0.25;
      projector[3][2].x = -0.25; projector[3][2].y = -0.25;
      projector[3][3].x = 0.25;
    }
  }
  else if( PID == G5G1 ){
    if(PARTICLE == PROTON){
      projector[0][1].x = +0.25; projector[1][0].x = +0.25; projector[2][3].x = -0.25; projector[3][2].x = -0.25;
      projector[0][3].y = +0.25; projector[1][2].y = +0.25; projector[2][1].y = +0.25; projector[3][0].y = +0.25;
    }
    else if(PARTICLE == NEUTRON){
      projector[0][1].x = +0.25; projector[1][0].x = +0.25; projector[2][3].x = -0.25; projector[3][2].x = -0.25;
      projector[0][3].y = -0.25; projector[1][2].y = -0.25; projector[2][1].y = -0.25; projector[3][0].y = -0.25;
    }
  }
  else if( PID == G5G2 ){
    if(PARTICLE == PROTON){
      projector[0][3].x = +0.25; projector[1][2].x = -0.25; projector[2][1].x = +0.25; projector[3][0].x = -0.25;
      projector[0][1].y = -0.25; projector[1][0].y = +0.25; projector[2][3].y = +0.25; projector[3][2].y = -0.25;
    }
    else if(PARTICLE == NEUTRON){
      projector[0][3].x = -0.25; projector[1][2].x = +0.25; projector[2][1].x = -0.25; projector[3][0].x = +0.25;
      projector[0][1].y = -0.25; projector[1][0].y = +0.25; projector[2][3].y = +0.25; projector[3][2].y = -0.25;
    }
  }
  else if( PID == G5G3 ){
    if(PARTICLE == PROTON){
      projector[0][0].x = +0.25; projector[1][1].x = -0.25; projector[2][2].x = -0.25; projector[3][3].x = +0.25;
      projector[0][2].y = +0.25; projector[1][3].y = -0.25; projector[2][0].y = +0.25; projector[3][1].y = -0.25;
    }
    else if(PARTICLE == NEUTRON){

      projector[0][0].x = +0.25; projector[1][1].x = -0.25; projector[2][2].x = -0.25; projector[3][3].x = +0.25;
      projector[0][2].y = -0.25; projector[1][3].y = +0.25; projector[2][0].y = -0.25; projector[3][1].y = +0.25;
    }
  }

