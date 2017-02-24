  for(int mu = 0 ; mu < 4 ; mu++)
    for(int nu = 0 ; nu < 4 ; nu++){
      gamma[mu][nu].x = 0 ; gamma[mu][nu].y = 0;
    }

switch( flag ){
 case 0: // 1 -> ig5 or -ig5
   if(partFlag == 1){
     if(TESTPARTICLE == PROTON){
       gamma[0][2].y=1.; gamma[1][3].y=1.; gamma[2][0].y=1.; gamma[3][1].y=1.;}
     else{
       gamma[0][2].y=-1.; gamma[1][3].y=-1.; gamma[2][0].y=-1.; gamma[3][1].y=-1.;}
   }
   else{
     if(TESTPARTICLE == PROTON){
       gamma[0][2].y=-1.; gamma[1][3].y=-1.; gamma[2][0].y=-1.; gamma[3][1].y=-1.;}
     else{
       gamma[0][2].y=1.; gamma[1][3].y=1.; gamma[2][0].y=1.; gamma[3][1].y=1.;}
   }
   break;
 case 1: // g1
   gamma[3][0].y=-1.; gamma[2][1].y=-1.; gamma[1][2].y=1.; gamma[0][3].y=1.;
   break;
 case 2: // g2
   gamma[3][0].x=1.; gamma[2][1].x=-1.; gamma[1][2].x=-1.; gamma[0][3].x=1.;
   break;
 case 3: // g3
   gamma[0][2].y=1.; gamma[1][3].y=-1.; gamma[2][0].y=-1.; gamma[3][1].y=1.;
   break;
 case 4: // g4 
   gamma[0][0].x=1.; gamma[1][1].x=1.; gamma[2][2].x=-1.; gamma[3][3].x=-1.;
   break;
 case 5: // g5 -> i or -i
   if(partFlag == 1){
     if(TESTPARTICLE == PROTON){
       gamma[0][0].y=1.; gamma[1][1].y=1.; gamma[2][2].y=1.; gamma[3][3].y=1.;}
     else{
       gamma[0][0].y=-1.; gamma[1][1].y=-1.; gamma[2][2].y=-1.; gamma[3][3].y=-1.;}
   }
   else{
     if(TESTPARTICLE == PROTON){
       gamma[0][0].y=-1.; gamma[1][1].y=-1.; gamma[2][2].y=-1.; gamma[3][3].y=-1.;}
     else{
       gamma[0][0].y=1.; gamma[1][1].y=1.; gamma[2][2].y=1.; gamma[3][3].y=1.;}
   }
   break;
 case 6: // g5g1
   gamma[0][1].y=-1.; gamma[1][0].y=-1.; gamma[2][3].y=1.; gamma[3][2].y=1.;
   break;
 case 7: // g5g2
   gamma[0][1].x=-1.; gamma[1][0].x=1.; gamma[2][3].x=1.; gamma[3][2].x=-1.;
   break;
 case 8: // g5g3
   gamma[0][0].y=-1.; gamma[1][1].y=1.; gamma[2][2].y=1.; gamma[3][3].y=-1.;
   break;
 case 9: // g5g4
   gamma[0][2].x=-1.; gamma[1][3].x=-1.; gamma[2][0].x=1.; gamma[3][1].x=1.;
   break;
 case 10: // 0.5*g5[g1,g2]
   if(partFlag == 1){
     if(TESTPARTICLE == PROTON){
       gamma[0][2].y=1.; gamma[1][3].y=-1.; gamma[2][0].y=1.; gamma[3][1].y=-1.;}
     else{
       gamma[0][2].y=-1.; gamma[1][3].y=1.; gamma[2][0].y=-1.; gamma[3][1].y=+1.;}
   }
   else{
     if(TESTPARTICLE == PROTON){
       gamma[0][2].y=-1.; gamma[1][3].y=1.; gamma[2][0].y=-1.; gamma[3][1].y=+1.;}
     else{
       gamma[0][2].y=1.; gamma[1][3].y=-1.; gamma[2][0].y=1.; gamma[3][1].y=-1.;}
   }
   break;
 case 11: // 0.5*g5[g1,g3]
   if(partFlag == 1){
     if(TESTPARTICLE == PROTON){
       gamma[0][3].x=-1.; gamma[1][2].x=1.; gamma[2][1].x=-1.; gamma[3][0].x=1.;}
     else{
       gamma[0][3].x=1.; gamma[1][2].x=-1.; gamma[2][1].x=1.; gamma[3][0].x=-1.;}
   }
   else{
     if(TESTPARTICLE == PROTON){
       gamma[0][3].x=1.; gamma[1][2].x=-1.; gamma[2][1].x=1.; gamma[3][0].x=-1.;}
     else{
       gamma[0][3].x=-1.; gamma[1][2].x=1.; gamma[2][1].x=-1.; gamma[3][0].x=1.;}
   }
   break;

 case 12: // 0.5*g5[g2,g3]
   if(partFlag == 1){
     if(TESTPARTICLE == PROTON){
       gamma[0][3].y=1.; gamma[1][2].y=1.; gamma[2][1].y=1.; gamma[3][0].y=1.;}
     else{
       gamma[0][3].y=-1.; gamma[1][2].y=-1.; gamma[2][1].y=-1.; gamma[3][0].y=-1.;}
   }
   else{
     if(TESTPARTICLE == PROTON){
       gamma[0][3].y=-1.; gamma[1][2].y=-1.; gamma[2][1].y=-1.; gamma[3][0].y=-1.;}
     else{
       gamma[0][3].y=1.; gamma[1][2].y=1.; gamma[2][1].y=1.; gamma[3][0].y=1.;}
   }
   break;

 case 13: // 0.5*g5[g4,g1]
   if(partFlag == 1){
     if(TESTPARTICLE == PROTON){
       gamma[0][1].y=1.; gamma[1][0].y=1.; gamma[2][3].y=1.; gamma[3][2].y=1.;}
     else{
       gamma[0][1].y=-1.; gamma[1][0].y=-1.; gamma[2][3].y=-1.; gamma[3][2].y=-1.;}
   }
   else{
     if(TESTPARTICLE == PROTON){
       gamma[0][1].y=-1.; gamma[1][0].y=-1.; gamma[2][3].y=-1.; gamma[3][2].y=-1.;}
     else{
       gamma[0][1].y=1.; gamma[1][0].y=1.; gamma[2][3].y=1.; gamma[3][2].y=1.;}
   }
   break;

 case 14: // 0.5*g5[g4,g2]
   if(partFlag == 1){
     if(TESTPARTICLE == PROTON){
       gamma[0][1].x=1.; gamma[1][0].x=-1.; gamma[2][3].x=1.; gamma[3][2].x=-1.;}
     else{
       gamma[0][1].x=-1.; gamma[1][0].x=1.; gamma[2][3].x=-1.; gamma[3][2].x=1.;}
   }
   else{
     if(TESTPARTICLE == PROTON){
       gamma[0][1].x=-1.; gamma[1][0].x=1.; gamma[2][3].x=-1.; gamma[3][2].x=1.;}
     else{
       gamma[0][1].x=1.; gamma[1][0].x=-1.; gamma[2][3].x=1.; gamma[3][2].x=-1.;}
   }
     
   break;

 case 15: // 0.5*g5[g4,g3]
   if(partFlag == 1){
     if(TESTPARTICLE == PROTON){
       gamma[0][0].y=1.; gamma[1][1].y=-1.; gamma[2][2].y=1.; gamma[3][3].y=-1.;}
     else{
       gamma[0][0].y=-1.; gamma[1][1].y=1.; gamma[2][2].y=-1.; gamma[3][3].y=1.;}
   }
   else{
     if(TESTPARTICLE == PROTON){
       gamma[0][0].y=-1.; gamma[1][1].y=1.; gamma[2][2].y=-1.; gamma[3][3].y=1.;}
     else{
       gamma[0][0].y=1.; gamma[1][1].y=-1.; gamma[2][2].y=1.; gamma[3][3].y=-1.;}
   }
   break;
 case 16: // 1 + g1
   gamma[0][0].x=1.; gamma[1][1].x=1.; gamma[2][2].x=1.; gamma[3][3].x=1.; gamma[0][3].y=1.;gamma[1][2].y=1.;gamma[2][1].y=-1.;gamma[3][0].y=-1.;
   break;
 case 17: // 1 + g2
   gamma[0][0].x=1.; gamma[1][1].x=1.; gamma[2][2].x=1.; gamma[3][3].x=1.; gamma[0][3].x=1.;gamma[1][2].x=-1.;gamma[2][1].x=-1.;gamma[3][0].x=1.;
   break;
 case 18: // 1 + g3
   gamma[0][0].x=1.; gamma[1][1].x=1.; gamma[2][2].x=1.; gamma[3][3].x=1.; gamma[0][2].y=1.;gamma[1][3].y=-1.;gamma[2][0].y=-1.;gamma[3][1].y=1.;
   break;
 case 19: // 1 + g4
   gamma[0][0].x=2.; gamma[1][1].x=2.;
   break;
 case 20: // 1 - g1
   gamma[0][0].x=1.; gamma[1][1].x=1.; gamma[2][2].x=1.; gamma[3][3].x=1.; gamma[0][3].y=-1.;gamma[1][2].y=-1.;gamma[2][1].y=1.;gamma[3][0].y=1.;
   break;
 case 21: // 1 - g2
   gamma[0][0].x=1.; gamma[1][1].x=1.; gamma[2][2].x=1.; gamma[3][3].x=1.; gamma[0][3].x=-1.;gamma[1][2].x=1.;gamma[2][1].x=1.;gamma[3][0].x=-1.;
   break;
 case 22: // 1 - g3
   gamma[0][0].x=1.; gamma[1][1].x=1.; gamma[2][2].x=1.; gamma[3][3].x=1.; gamma[0][2].y=-1.;gamma[1][3].y=1.;gamma[2][0].y=1.;gamma[3][1].y=-1.;
   break;
 case 23: // 1 - g4
   gamma[2][2].x=2.; gamma[3][3].x=2.;
   break;
 }
