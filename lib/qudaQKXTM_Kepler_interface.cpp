#include <qudaQKXTM_Kepler.cpp>
#include <sys/stat.h>
#include <unistd.h>
#define TIMING_REPORT
template  class QKXTM_Field_Kepler<double>;
template  class QKXTM_Gauge_Kepler<double>;
template  class QKXTM_Vector_Kepler<double>;
template  class QKXTM_Propagator_Kepler<double>;
template  class QKXTM_Propagator3D_Kepler<double>;
template  class QKXTM_Vector3D_Kepler<double>;

template  class QKXTM_Field_Kepler<float>;
template  class QKXTM_Gauge_Kepler<float>;
template  class QKXTM_Vector_Kepler<float>;
template  class QKXTM_Propagator_Kepler<float>;
template  class QKXTM_Propagator3D_Kepler<float>;
template  class QKXTM_Vector3D_Kepler<float>;

//QKXTM: DMH temporarily comment of all F_OK instances
// until its prupose is known

static bool exists_file (const char* name) {
  return ( access( name, F_OK ) != -1 );
}


void testPlaquette(void **gauge){
  QKXTM_Gauge_Kepler<float> *gauge_object = new QKXTM_Gauge_Kepler<float>(BOTH,GAUGE);
  gauge_object->printInfo();
  gauge_object->packGauge(gauge);
  gauge_object->loadGauge();
  gauge_object->calculatePlaq();
  delete gauge_object;

  QKXTM_Gauge_Kepler<double> *gauge_object_2 = new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  gauge_object_2->printInfo();
  gauge_object_2->packGauge(gauge);
  gauge_object_2->loadGauge();
  gauge_object_2->calculatePlaq();
  delete gauge_object_2;
}

void testGaussSmearing(void **gauge){

  QKXTM_Gauge_Kepler<double> *gauge_object = new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  gauge_object->printInfo();
  gauge_object->packGauge(gauge);
  gauge_object->loadGauge();
  gauge_object->calculatePlaq();

  QKXTM_Vector_Kepler<double> *vecIn = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<double> *vecOut = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);

  void *input_vector = malloc(GK_localVolume*4*3*2*sizeof(double));
  *((double*) input_vector) = 1.;
  vecIn->packVector((double*) input_vector);
  vecIn->loadVector();
  vecOut->gaussianSmearing(*vecIn,*gauge_object);
  vecOut->download();
  for(int mu = 0 ; mu < 4 ; mu++)
    for(int c1 = 0 ; c1 < 3 ; c1++)
      printf("%+e %+e\n",vecOut->H_elem()[mu*3*2+c1*2+0],vecOut->H_elem()[mu*3*2+c1*2+1]);

  delete vecOut;
  delete gauge_object;
}

void invertWritePropsNoApe_SL_v2_Kepler(void **gauge, void **gaugeAPE ,QudaInvertParam *param ,QudaGaugeParam *gauge_param,quda::qudaQKXTMinfo_Kepler info, char *prop_path){
  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);
  if (!initialized) errorQuda("QUDA not initialized");
  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);
  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);

  QKXTM_Gauge_Kepler<double> *K_gaugeAPE = new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  K_gaugeAPE->packGauge(gaugeAPE);
  K_gaugeAPE->loadGauge();
  K_gaugeAPE->calculatePlaq();

  bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) ||
    (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  bool pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE) ||
    (param->solve_type == QUDA_NORMOP_PC_SOLVE);
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) ||
    (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = (param->solve_type == QUDA_DIRECT_SOLVE) ||
    (param->solve_type == QUDA_DIRECT_PC_SOLVE);

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }

  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;
  
  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;
  // create the dirac operator                                                                     
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;
  profileInvert.TPSTART(QUDA_PROFILE_H2D);

  //QKXTM: DMH rewite for spinor field memalloc 
  ColorSpinorField *b = NULL;
  ColorSpinorField *x = NULL;
  ColorSpinorField *in = NULL;
  ColorSpinorField *out = NULL;
  const int *X = cudaGauge->X();

  // wrap CPU host side pointers
  void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  void *output_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  ColorSpinorParam cpuParam(input_vector, *param, X, pc_solution, param->input_location);
  ColorSpinorField *h_b = ColorSpinorField::Create(cpuParam);

  cpuParam.v = output_vector;
  cpuParam.location = param->output_location;
  ColorSpinorField *h_x = ColorSpinorField::Create(cpuParam);

  // dowload source
  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_COPY_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam);

  //Zero out the solution
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  x = new cudaColorSpinorField(cudaParam);

  profileInvert.TPSTOP(QUDA_PROFILE_H2D);
  setTuning(param->tune);

  if (pc_solution && !pc_solve) {
    errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");
  }
  if (!mat_solution && !pc_solution && pc_solve) {
    errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");
  }
  if( !(mat_solution == true && direct_solve == false) )errorQuda("qudaQKXTM package supports only mat solution and indirect solution");
  if( param->inv_type != QUDA_CG_INVERTER) errorQuda("qudaQKXTM package supports only cg inverter");

  DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
  SolverParam solverParam(*param);
  Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);

  QKXTM_Vector_Kepler<double> *K_vectorTmp = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR); 
  QKXTM_Vector_Kepler<double> *K_vectorGauss = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR); 
  
  char tempFilename[257];
  
  // for test source position will be 0,0,0,0 then I will make it general
  for(int ip = 0 ; ip < 12 ; ip++){
    
   memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

    // change to up quark
   b->changeTwist(QUDA_TWIST_PLUS);
   x->changeTwist(QUDA_TWIST_PLUS);

   b->Even().changeTwist(QUDA_TWIST_PLUS);
   b->Odd().changeTwist(QUDA_TWIST_PLUS);
   x->Even().changeTwist(QUDA_TWIST_PLUS);
   x->Odd().changeTwist(QUDA_TWIST_PLUS);

   // find where to put source
   int my_src[4];
   for(int i = 0 ; i < 4 ; i++)
     my_src[i] = info.sourcePosition[0][i] - comm_coords(default_topo)[i] * X[i];

   if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
     *( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + ip*2 ) = 1.;         //only real parts

   K_vectorTmp->packVector((double*) input_vector);
   K_vectorTmp->loadVector();
   K_vectorGauss->gaussianSmearing(*K_vectorTmp,*K_gaugeAPE);
   K_vectorGauss->uploadToCuda(b);

   //   K_vectorGauss->download();
   // K_vectorGauss->norm2Host();
   // exit(-1);

   //   K_vectorTmp->norm2Host();
   // K_vectorTmp->uploadToCuda(b);
   // double nb = norm2(*b);
   //  if(nb==0.0)errorQuda("Source has zero norm");

   blas::zero(*x);
   dirac.prepare(in, out, *x, *b, param->solution_type); // prepares the source vector 
   checkCudaError();
   cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);

   // indirect method needs apply of D^+ on source vector
   dirac.Mdag(*in, *tmp_up);                        
   (*solve)(*out, *in);      
   dirac.reconstruct(*x, *b, param->solution_type);

   K_vectorTmp->downloadFromCuda(x);
   if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     K_vectorTmp->scaleVector(2*param->kappa);
   }

   //   K_vectorGauss->gaussianSmearing(*K_vectorTmp,*K_gaugeAPE);
   //  K_vectorGauss->download();
   K_vectorTmp->download();

   sprintf(tempFilename,"%s_up.%04d",prop_path,ip);
   K_vectorTmp->write(tempFilename);
   delete tmp_up;
   printfQuda("Finish Inversion for up quark %d/12 \n",ip+1);
   
   // down
   K_vectorTmp->packVector((double*) input_vector);
   K_vectorTmp->loadVector();
   K_vectorGauss->gaussianSmearing(*K_vectorTmp,*K_gaugeAPE);


   K_vectorGauss->uploadToCuda(b);

   b->changeTwist(QUDA_TWIST_MINUS);
   x->changeTwist(QUDA_TWIST_MINUS);
   b->Even().changeTwist(QUDA_TWIST_MINUS);
   b->Odd().changeTwist(QUDA_TWIST_MINUS);
   x->Even().changeTwist(QUDA_TWIST_MINUS);
   x->Odd().changeTwist(QUDA_TWIST_MINUS);


   blas::zero(*x);
   dirac.prepare(in, out, *x, *b, param->solution_type); // prepares the source vector 
   cudaColorSpinorField *tmp_down = new cudaColorSpinorField(*in);
   dirac.Mdag(*in, *tmp_down);                        // indirect method needs apply of D^+ on source vector
   (*solve)(*out, *in);   
   dirac.reconstruct(*x, *b, param->solution_type);
   K_vectorTmp->downloadFromCuda(x);
   if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     K_vectorTmp->scaleVector(2*param->kappa);
   }

   //   K_vectorGauss->gaussianSmearing(*K_vectorTmp,*K_gaugeAPE);
   //  K_vectorGauss->download();
   K_vectorTmp->download();
   sprintf(tempFilename,"%s_down.%04d",prop_path,ip);
   K_vectorTmp->write(tempFilename);

   delete tmp_down;
   printfQuda("Finish Inversion for down quark %d/12 \n",ip+1);   

   }



 free(input_vector);
 free(output_vector);

 delete K_vectorTmp;
 delete K_vectorGauss;
 delete K_gaugeAPE;
 delete solve;
 delete h_b;
 delete h_x;
 delete b;
 delete x;
 
 delete d;
 delete dSloppy;
 delete dPre;

 popVerbosity();
 saveTuneCache();
 profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);


}


void invertWritePropsNoApe_SL_v2_Kepler_single(void **gauge, void **gaugeAPE ,QudaInvertParam *param ,QudaGaugeParam *gauge_param,quda::qudaQKXTMinfo_Kepler info, char *prop_path){
  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);
  if (!initialized) errorQuda("QUDA not initialized");
  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);
  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);

  QKXTM_Gauge_Kepler<float> *K_gaugeAPE = new QKXTM_Gauge_Kepler<float>(BOTH,GAUGE);
  K_gaugeAPE->packGauge(gaugeAPE);
  K_gaugeAPE->loadGauge();
  K_gaugeAPE->calculatePlaq();

  bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) ||
    (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  bool pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE) ||
    (param->solve_type == QUDA_NORMOP_PC_SOLVE);
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) ||
    (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = (param->solve_type == QUDA_DIRECT_SOLVE) ||
    (param->solve_type == QUDA_DIRECT_PC_SOLVE);

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }

  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;
  
  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;
  // create the dirac operator                                                                     
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;
  profileInvert.TPSTART(QUDA_PROFILE_H2D);

  //QKXTM: DMH rewite for spinor field memalloc
  ColorSpinorField *b = NULL;
  ColorSpinorField *x = NULL;
  ColorSpinorField *in = NULL;
  ColorSpinorField *out = NULL;
  const int *X = cudaGauge->X();

  void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(float));
  void *output_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(float));

  ColorSpinorParam cpuParam(input_vector, *param, X, pc_solution, param->input_location);
  ColorSpinorField *h_b = ColorSpinorField::Create(cpuParam);

  cpuParam.v = output_vector;
  cpuParam.location = param->output_location;
  ColorSpinorField *h_x = ColorSpinorField::Create(cpuParam);

  // download source
  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_COPY_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam);

  //Zero out the solution
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  x = new cudaColorSpinorField(cudaParam);

  profileInvert.TPSTOP(QUDA_PROFILE_H2D);
  setTuning(param->tune);

  if (pc_solution && !pc_solve) {
    errorQuda("Preconditioned (PC) solution_type requires a PC solve_type");
  }
  if (!mat_solution && !pc_solution && pc_solve) {
    errorQuda("Unpreconditioned MATDAG_MAT solution_type requires an unpreconditioned solve_type");
  }
  if( !(mat_solution == true && direct_solve == false) )errorQuda("qudaQKXTM package supports only mat solution and indirect solution");
  if( param->inv_type != QUDA_CG_INVERTER) errorQuda("qudaQKXTM package supports only cg inverter");

  DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
  SolverParam solverParam(*param);
  Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);

  QKXTM_Vector_Kepler<float> *K_vectorTmp = new QKXTM_Vector_Kepler<float>(BOTH,VECTOR); 
  QKXTM_Vector_Kepler<float> *K_vectorGauss = new QKXTM_Vector_Kepler<float>(BOTH,VECTOR); 

  char tempFilename[257];

   for(int ip = 0 ; ip < 12 ; ip++){     // for test source position will be 0,0,0,0 then I will make it general
   
   memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(float));

   // change to up quark
   b->changeTwist(QUDA_TWIST_PLUS);
   x->changeTwist(QUDA_TWIST_PLUS);

   b->Even().changeTwist(QUDA_TWIST_PLUS);
   b->Odd().changeTwist(QUDA_TWIST_PLUS);
   x->Even().changeTwist(QUDA_TWIST_PLUS);
   x->Odd().changeTwist(QUDA_TWIST_PLUS);

   // find where to put source
   int my_src[4];
   for(int i = 0 ; i < 4 ; i++)
     my_src[i] = info.sourcePosition[0][i] - comm_coords(default_topo)[i] * X[i];

   if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
     *( (float*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + ip*2 ) = 1.;         //only real parts

   K_vectorTmp->packVector((float*) input_vector);
   K_vectorTmp->loadVector();
   K_vectorGauss->gaussianSmearing(*K_vectorTmp,*K_gaugeAPE);
   K_vectorGauss->uploadToCuda(b);

   //   K_vectorGauss->download();
   // K_vectorGauss->norm2Host();
   // exit(-1);

   //   K_vectorTmp->norm2Host();
   // K_vectorTmp->uploadToCuda(b);
   // double nb = norm2(*b);
   //  if(nb==0.0)errorQuda("Source has zero norm");

   blas::zero(*x);
   dirac.prepare(in, out, *x, *b, param->solution_type); // prepares the source vector 
   checkCudaError();
   cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);

   // indirect method needs apply of D^+ on source vector
   dirac.Mdag(*in, *tmp_up);                        
   (*solve)(*out, *in);      
   dirac.reconstruct(*x, *b, param->solution_type);

   K_vectorTmp->downloadFromCuda(x);
   if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     K_vectorTmp->scaleVector(2*param->kappa);
   }

   //   K_vectorGauss->gaussianSmearing(*K_vectorTmp,*K_gaugeAPE);
   //  K_vectorGauss->download();
   K_vectorTmp->download();

   sprintf(tempFilename,"%s_up.%04d",prop_path,ip);
   K_vectorTmp->write(tempFilename);
   delete tmp_up;
   printfQuda("Finish Inversion for up quark %d/12 \n",ip+1);
   
   // down
   K_vectorTmp->packVector((float*) input_vector);
   K_vectorTmp->loadVector();
   K_vectorGauss->gaussianSmearing(*K_vectorTmp,*K_gaugeAPE);


   K_vectorGauss->uploadToCuda(b);

   b->changeTwist(QUDA_TWIST_MINUS);
   x->changeTwist(QUDA_TWIST_MINUS);
   b->Even().changeTwist(QUDA_TWIST_MINUS);
   b->Odd().changeTwist(QUDA_TWIST_MINUS);
   x->Even().changeTwist(QUDA_TWIST_MINUS);
   x->Odd().changeTwist(QUDA_TWIST_MINUS);


   blas::zero(*x);
   dirac.prepare(in, out, *x, *b, param->solution_type); // prepares the source vector 
   cudaColorSpinorField *tmp_down = new cudaColorSpinorField(*in);
   dirac.Mdag(*in, *tmp_down);                        // indirect method needs apply of D^+ on source vector
   (*solve)(*out, *in);   
   dirac.reconstruct(*x, *b, param->solution_type);
   K_vectorTmp->downloadFromCuda(x);
   if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
     K_vectorTmp->scaleVector(2*param->kappa);
   }

   //   K_vectorGauss->gaussianSmearing(*K_vectorTmp,*K_gaugeAPE);
   //  K_vectorGauss->download();
   K_vectorTmp->download();
   sprintf(tempFilename,"%s_down.%04d",prop_path,ip);
   K_vectorTmp->write(tempFilename);

   delete tmp_down;
   printfQuda("Finish Inversion for down quark %d/12 \n",ip+1);   

   }



 free(input_vector);
 free(output_vector);

 delete K_vectorTmp;
 delete K_vectorGauss;
 delete K_gaugeAPE;
 delete solve;
 delete h_b;
 delete h_x;
 delete b;
 delete x;
 
 delete d;
 delete dSloppy;
 delete dPre;

 popVerbosity();
 saveTuneCache();
 profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);


}

void checkReadingEigenVectors(int N_eigenVectors, char* pathIn, char *pathOut, char* pathEigenValues){
  QKXTM_Deflation_Kepler<float> *deflation = new QKXTM_Deflation_Kepler<float>(N_eigenVectors,false);
  deflation->printInfo();
  deflation->readEigenVectors(pathIn);
  deflation->writeEigenVectors_ASCII(pathOut);
  deflation->readEigenValues(pathEigenValues);
  for(int i = 0 ; i < N_eigenVectors; i++)
    printf("%e\n",deflation->EigenValues()[i]);
  delete deflation;
}

void checkDeflateVectorQuda(void **gauge,QudaInvertParam *param ,QudaGaugeParam *gauge_param,char *filename_eigenValues, char *filename_eigenVectors, char *filename_out,int NeV){
  bool flag_eo;
  if( (param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) && (param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) ) errorQuda("Only asymmetric operators are supported in deflation\n");
  if( param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC )
    flag_eo = true;
  else if(param->matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
    flag_eo = false;

  QKXTM_Deflation_Kepler<double> *deflation = new QKXTM_Deflation_Kepler<double>(NeV,flag_eo);

  deflation->readEigenValues(filename_eigenValues);
  deflation->readEigenVectors(filename_eigenVectors);
  deflation->rotateFromChiralToUKQCD();
  if(gauge_param->t_boundary == QUDA_ANTI_PERIODIC_T)deflation->multiply_by_phase();
  QKXTM_Vector_Kepler<double> *vecIn = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<double> *vecOut = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<double> *vecTest = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);

  // just for test set vec to all elements to be 1
  vecIn->zero_host();
  vecOut->zero_host();

  for(int iv = 0 ; iv < GK_localVolume ; iv++)
    for(int mu = 0 ; mu < 4 ; mu++)
      for(int c = 0 ; c < 3 ; c++)
	vecIn->H_elem()[iv*4*3*2+mu*c*2+c*2+0] = 1.;
  
  deflation->deflateVector(*vecOut,*vecIn);
  vecOut->download();
  vecTest->zero_host();

  std::complex<double> *cmplx_vecTest = NULL;
  std::complex<double> *cmplx_U = NULL;
  std::complex<double> *cmplx_b = NULL;


  for(int ie = 0 ; ie < NeV ; ie++){
    cmplx_vecTest = (std::complex<double>*) vecTest->H_elem();
    cmplx_U = (std::complex<double>*) &(deflation->H_elem()[ie*(GK_localVolume/2)*4*3*2]);
    cmplx_b = (std::complex<double>*) vecIn->H_elem();
    for(int alpha = 0 ; alpha < (GK_localVolume/2)*4*3 ; alpha++)
      for(int beta = 0 ; beta < (GK_localVolume/2)*4*3 ; beta++) {
	//QKXTM: DMH CAREFULL HERE
	//cmplx_vecTest[alpha] = cmplx_vecTest[alpha] + (*(cmplx_U+alpha)) * (1./deflation->EigenValues()[ie]) * conj(cmplx_U[beta]) * cmplx_b[beta]; 
    	cmplx_vecTest[alpha] = cmplx_vecTest[alpha] + (*(cmplx_U+alpha)) * (1.0/(deflation->EigenValues()[ie])) * (conj(cmplx_U[beta])) * (cmplx_b[beta]); 
      }
  }

  FILE *ptr_out = NULL;
  if(comm_rank() == 0){
    ptr_out=fopen(filename_out,"w"); 
    if(ptr_out == NULL)errorQuda("Error open file for writing\n");
    for(int iv = 0 ; iv < GK_localVolume ; iv++)
      for(int mu = 0 ; mu < 4 ; mu++)
	for(int c = 0 ; c < 3 ; c++)
	  fprintf(ptr_out,"%+e %+e \t %+e %+e \t %e %e\n",vecIn->H_elem()[iv*4*3*2+mu*c*2+c*2+0],vecIn->H_elem()[iv*4*3*2+mu*c*2+c*2+1],vecOut->H_elem()[iv*4*3*2+mu*c*2+c*2+0],vecOut->H_elem()[iv*4*3*2+mu*c*2+c*2+1],vecTest->H_elem()[iv*4*3*2+mu*c*2+c*2+0],vecTest->H_elem()[iv*4*3*2+mu*c*2+c*2+1]);
  }

  /*

  std::complex<float> *temp = (std::complex<float>*)malloc(NeV*2*sizeof(float));
  memset(temp,0,NeV*2*sizeof(float));
  for(int ie = 0 ; ie < NeV ; ie++){
    std::complex<float> *pointer = (std::complex<float>*) deflation->H_elem()[ie];
    std::complex<float> *cmplx_b = (std::complex<float>*) vecIn->H_elem();
    for(int iv = 0 ; iv < (GK_localVolume/2)*4*3 ; iv++)
      temp[ie] = temp[ie] + conj(pointer[iv]) * cmplx_b[iv];
    printfQuda("%+e %+e\n",temp[ie].real(),temp[ie].imag());
  }
  */
  delete vecTest;
  delete deflation;
  delete vecIn;
  delete vecOut;
}

void checkEigenVectorQuda(void **gauge,QudaInvertParam *param ,QudaGaugeParam *gauge_param,char *filename_eigenValues, char *filename_eigenVectors, char *filename_out,int NeV){
  bool flag_eo;
  if( (param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) && (param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) ) errorQuda("Only asymmetric operators are supported in deflation\n");
  if( param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC )
    flag_eo = true;
  else if(param->matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
    flag_eo = false;

  QKXTM_Deflation_Kepler<double> *deflation = new QKXTM_Deflation_Kepler<double>(NeV,flag_eo);

  deflation->readEigenValues(filename_eigenValues);
  deflation->readEigenVectors(filename_eigenVectors);
  deflation->rotateFromChiralToUKQCD();
  if(gauge_param->t_boundary == QUDA_ANTI_PERIODIC_T)deflation->multiply_by_phase();

  if (!initialized) errorQuda("QUDA not initialized");
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);

  if (cloverPrecise == NULL && ((param->dslash_type == QUDA_CLOVER_WILSON_DSLASH) || (param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH)))
    errorQuda("Clover field not allocated");
  //QKXTM: DMH cloverInvPrecise no longer exists in quda-0.9.0
  //if (cloverInvPrecise == NULL && param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH)
  //errorQuda("Clover field not allocated");
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  QKXTM_Gauge_Kepler<double> *qkxTM_gauge = new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  qkxTM_gauge->packGauge(gauge);
  qkxTM_gauge->loadGauge();
  qkxTM_gauge->calculatePlaq();


  bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) ||
    (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

  //  bool pc = (inv_param->solution_type == QUDA_MATPC_SOLUTION || inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

  //QKXTM: DMH rewite for spinor field memalloc
  ColorSpinorField *b = NULL;
  ColorSpinorField *x = NULL;
  ColorSpinorField *in = NULL;
  ColorSpinorField *out = NULL;
  const int *X = cudaGauge->X();

  double *input_vector =(double*) malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));  
  double *output_vector =(double*) malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));  
  // wrap CPU host side pointers
  ColorSpinorParam cpuParam(input_vector, *param, X, pc_solution, param->input_location);
  ColorSpinorField *h_b = ColorSpinorField::Create(cpuParam);

  cpuParam.v = output_vector;
  cpuParam.location = param->output_location;
  ColorSpinorField *h_x = ColorSpinorField::Create(cpuParam);

  // download source
  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_COPY_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam);

  //Zero out the solution
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  x = new cudaColorSpinorField(cudaParam);

  delete h_b;
  delete h_x;

  setTuning(param->tune);
  QKXTM_Vector_Kepler<double> *vec = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  DiracParam diracParam;
  setDiracParam(diracParam, param, true);
  Dirac *dirac = Dirac::create(diracParam);


  for(int i = 0 ; i < deflation->NeVs() ; i++){
    blas::zero(*b);
    blas::zero(*x);
    deflation->copyEigenVectorToQKXTM_Vector_Kepler(i,input_vector);
    vec->packVector(input_vector);
    vec->loadVector();
    vec->uploadToCuda(b,flag_eo);
    dirac->MdagM(*x,*b);
    vec->downloadFromCuda(x,flag_eo);
    vec->download();
    deflation->copyEigenVectorFromQKXTM_Vector_Kepler(i,vec->H_elem());
  }

  free(input_vector);
  free(output_vector);
  deflation->writeEigenVectors_ASCII(filename_out);

   delete dirac;  
   delete b;
   delete x;
   delete vec;
   delete qkxTM_gauge;
   delete deflation;

}

void checkDeflateAndInvert(void **gaugeToPlaquette, QudaInvertParam *param ,QudaGaugeParam *gauge_param, char *filename_eigenValues, char *filename_eigenVectors, char *filename_out,int NeV ){
  bool flag_eo;


  double t1,t2;


  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);
  if(param->solve_type != QUDA_NORMOP_PC_SOLVE) errorQuda("This function works only with even odd preconditioning");
  if(param->inv_type != QUDA_CG_INVERTER) errorQuda("This function works only with CG method");
  if( (param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) && (param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) ) errorQuda("Only asymmetric operators are supported in deflation\n");
  if( param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC )
    flag_eo = true;
  else if(param->matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
    flag_eo = false;

  QKXTM_Deflation_Kepler<double> *deflation = new QKXTM_Deflation_Kepler<double>(NeV,flag_eo);
  deflation->printInfo();
  t1 = MPI_Wtime();
  deflation->readEigenValues(filename_eigenValues);
  deflation->readEigenVectors(filename_eigenVectors);
  if(param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS) errorQuda("This function works only with ukqcd gamma basis\n");
  if(param->dirac_order != QUDA_DIRAC_ORDER) errorQuda("This function works only with colors inside the spins\n");
  deflation->rotateFromChiralToUKQCD();
  if(gauge_param->t_boundary == QUDA_ANTI_PERIODIC_T)deflation->multiply_by_phase();

  t2 = MPI_Wtime();

#ifdef TIMING_REPORT
  printfQuda("Timing report for read and transform eigenVectors is %f sec\n",t2-t1);
#endif

  if (!initialized) errorQuda("QUDA not initialized");
  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);

  QKXTM_Gauge_Kepler<double> *K_gauge = new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  K_gauge->packGauge(gaugeToPlaquette);
  K_gauge->loadGauge();
  K_gauge->calculatePlaq();

  bool pc_solution = false;
  bool pc_solve = true;
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) || (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = false;

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;
  
  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;
  // create the dirac operator                                                                                                                                                                         
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;
  profileInvert.TPSTART(QUDA_PROFILE_H2D);

  //QKXTM: DMH rewite for spinor field memalloc
  ColorSpinorField *b = NULL;
  ColorSpinorField *x = NULL;
  ColorSpinorField *in = NULL;
  ColorSpinorField *out = NULL;
  const int *X = cudaGauge->X();

  void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  void *output_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  memset(output_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  // wrap CPU host side pointers
  ColorSpinorParam cpuParam(input_vector, *param, X, pc_solution, param->input_location);
  ColorSpinorField *h_b = ColorSpinorField::Create(cpuParam);

  cpuParam.v = output_vector;
  cpuParam.location = param->output_location;
  ColorSpinorField *h_x = ColorSpinorField::Create(cpuParam);

  // download source
  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam);

  //Zero out the solution
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  x = new cudaColorSpinorField(cudaParam);

  profileInvert.TPSTOP(QUDA_PROFILE_H2D);
  setTuning(param->tune);


  blas::zero(*x);
  blas::zero(*b);

  DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
  SolverParam solverParam(*param);
  Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
  QKXTM_Vector_Kepler<double> *K_vector = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<double> *K_guess = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);

  t1 = MPI_Wtime();
  if(comm_rank() == 0)  *((double*) input_vector) = 1.;
  K_vector->packVector((double*) input_vector);
  K_vector->loadVector();
  K_vector->uploadToCuda(b,flag_eo);
  dirac.prepare(in,out,*x,*b,param->solution_type);

  // in is reference to the b but for a parity sinlet
  // out is reference to the x but for a parity sinlet
  cudaColorSpinorField *tmp = new cudaColorSpinorField(*in);
  dirac.Mdag(*in, *tmp);
  delete tmp;
  // now the the source vector b is ready to perform deflation and find the initial guess
  K_vector->downloadFromCuda(in,flag_eo);
  K_vector->download();
  deflation->deflateVector(*K_guess,*K_vector);
  K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
  t2 = MPI_Wtime();
#ifdef TIMING_REPORT
  printfQuda("Timing report for deflation procudure is %f sec\n",t2-t1);
#endif

  //  blas::zero(*out); // remove it later , just for test

  fflush(stdout);

  t1 = MPI_Wtime();
  (*solve)(*out,*in);
  t2 = MPI_Wtime();

#ifdef TIMING_REPORT
  printfQuda("Timing report inversion is %f sec\n",t2-t1);
#endif


  dirac.reconstruct(*x,*b,param->solution_type);
  K_vector->downloadFromCuda(x,flag_eo);
  if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
    K_vector->scaleVector(2*param->kappa);
  }
  K_vector->download();

  /*
  FILE *ptr_out = NULL;
  if(comm_rank() == 0){
    ptr_out=fopen(filename_out,"w"); 
    if(ptr_out == NULL)errorQuda("Error open file for writing\n");
    for(int iv = 0 ; iv < GK_localVolume ; iv++)
      for(int mu = 0 ; mu < 4 ; mu++)
	for(int c = 0 ; c < 3 ; c++)
	  fprintf(ptr_out,"%+e %+e\n",K_vector->H_elem()[iv*4*3*2+mu*c*2+c*2+0],K_vector->H_elem()[iv*4*3*2+mu*c*2+c*2+1]);
  }
  */

  /*
  K_guess->download();
  FILE *ptr_out = NULL;
  if(comm_rank() == 0){
    ptr_out=fopen(filename_out,"w"); 
    if(ptr_out == NULL)errorQuda("Error open file for writing\n");
    for(int iv = 0 ; iv < GK_localVolume ; iv++)
      for(int mu = 0 ; mu < 4 ; mu++)
	for(int c = 0 ; c < 3 ; c++)
	  fprintf(ptr_out,"%+e %+e\n",K_guess->H_elem()[iv*4*3*2+mu*c*2+c*2+0],K_guess->H_elem()[iv*4*3*2+mu*c*2+c*2+1]);
  }
  */

  free(input_vector);
  free(output_vector);
  delete solve;
  delete d;
  delete dSloppy;
  delete dPre;
  delete K_guess;
  delete K_vector;
  delete K_gauge;
  delete deflation;
  delete h_x;
  delete h_b;
  delete x;
  delete b;

  popVerbosity();
  saveTuneCache();
  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);

}


void DeflateAndInvert_twop(void **gaugeSmeared, QudaInvertParam *param ,QudaGaugeParam *gauge_param, char *filename_eigenValues_up, char *filename_eigenVectors_up, char *filename_eigenValues_down, char *filename_eigenVectors_down, char *filename_out,int NeV, qudaQKXTMinfo_Kepler info ){
  bool flag_eo;
  double t1,t2;

  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);
  if(param->solve_type != QUDA_NORMOP_PC_SOLVE) errorQuda("This function works only with even odd preconditioning");
  if(param->inv_type != QUDA_CG_INVERTER) errorQuda("This function works only with CG method");
  if( (param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) && (param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) ) errorQuda("Only asymmetric operators are supported in deflation\n");
  if( param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC )
    flag_eo = true;
  else if(param->matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
    flag_eo = false;

  QKXTM_Deflation_Kepler<double> *deflation_up = new QKXTM_Deflation_Kepler<double>(NeV,flag_eo);
  QKXTM_Deflation_Kepler<double> *deflation_down = new QKXTM_Deflation_Kepler<double>(NeV,flag_eo);
  void *input_vector = malloc(GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]*spinorSiteSize*sizeof(double));
  void *output_vector = malloc(GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]*spinorSiteSize*sizeof(double));
  QKXTM_Gauge_Kepler<double> *K_gauge = new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  QKXTM_Vector_Kepler<double> *K_vector = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<double> *K_guess = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<float> *K_temp = new QKXTM_Vector_Kepler<float>(BOTH,VECTOR);

  QKXTM_Propagator_Kepler<float> *K_prop_up = new QKXTM_Propagator_Kepler<float>(DEVICE,PROPAGATOR);
  QKXTM_Propagator_Kepler<float> *K_prop_down = new QKXTM_Propagator_Kepler<float>(DEVICE,PROPAGATOR);  
  QKXTM_Contraction_Kepler<float> *K_contract = new QKXTM_Contraction_Kepler<float>();
  printfQuda("Memory allocation was successfull\n");

  if(param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS) errorQuda("This function works only with ukqcd gamma basis\n");
  if(param->dirac_order != QUDA_DIRAC_ORDER) errorQuda("This function works only with colors inside the spins\n");

  deflation_up->printInfo();
  t1 = MPI_Wtime();
  deflation_up->readEigenValues(filename_eigenValues_up);
  deflation_up->readEigenVectors(filename_eigenVectors_up);
  deflation_up->rotateFromChiralToUKQCD();
  if(gauge_param->t_boundary == QUDA_ANTI_PERIODIC_T)deflation_up->multiply_by_phase();

  deflation_down->readEigenValues(filename_eigenValues_down);
  deflation_down->readEigenVectors(filename_eigenVectors_down);
  deflation_down->rotateFromChiralToUKQCD();
  if(gauge_param->t_boundary == QUDA_ANTI_PERIODIC_T)deflation_down->multiply_by_phase();
  t2 = MPI_Wtime();

#ifdef TIMING_REPORT
  printfQuda("Timing report for read and transform eigenVectors is %f sec\n",t2-t1);
#endif

  if (!initialized) errorQuda("QUDA not initialized");
  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);


  K_gauge->packGauge(gaugeSmeared);
  K_gauge->loadGauge();
  K_gauge->calculatePlaq();

  bool pc_solution = false;
  bool pc_solve = true;
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) || (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = false;

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;
  
  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;
  // create the dirac operator                                                                                                                                      
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;
  profileInvert.TPSTART(QUDA_PROFILE_H2D);

  //QKXTM: DMH rewite for spinor field memalloc
  ColorSpinorField *b = NULL;
  ColorSpinorField *x = NULL;
  ColorSpinorField *in = NULL;
  ColorSpinorField *out = NULL;
  const int *X = cudaGauge->X();

  memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  memset(output_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  // wrap CPU host side pointers
  ColorSpinorParam cpuParam(input_vector, *param, X, pc_solution, param->input_location);
  ColorSpinorField *h_b = ColorSpinorField::Create(cpuParam);

  cpuParam.v = output_vector;
  cpuParam.location = param->output_location;
  ColorSpinorField *h_x = ColorSpinorField::Create(cpuParam);

  // download source
  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam);

  //Zero out the solution
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  x = new cudaColorSpinorField(cudaParam);

  profileInvert.TPSTOP(QUDA_PROFILE_H2D);
  setTuning(param->tune);
  
  blas::zero(*x);
  blas::zero(*b);
  DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
  SolverParam solverParam(*param);
  Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);

  int my_src[4];
  char filename_mesons[257];
  char filename_baryons[257];

  for(int isource = 0 ; isource < info.Nsources ; isource++){
    sprintf(filename_mesons,"%s.mesons.SS.%02d.%02d.%02d.%02d.dat",filename_out,info.sourcePosition[isource][0],info.sourcePosition[isource][1],info.sourcePosition[isource][2],info.sourcePosition[isource][3]);
    sprintf(filename_baryons,"%s.baryons.SS.%02d.%02d.%02d.%02d.dat",filename_out,info.sourcePosition[isource][0],info.sourcePosition[isource][1],info.sourcePosition[isource][2],info.sourcePosition[isource][3]);
    bool checkMesons, checkBaryons;
    checkMesons = exists_file(filename_mesons);
    checkBaryons = exists_file(filename_baryons);
    if( (checkMesons == true) && (checkBaryons == true) ) continue;
    for(int isc = 0 ; isc < 12 ; isc++){
      t1 = MPI_Wtime();
      memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
      b->changeTwist(QUDA_TWIST_PLUS);
      x->changeTwist(QUDA_TWIST_PLUS);
      b->Even().changeTwist(QUDA_TWIST_PLUS);
      b->Odd().changeTwist(QUDA_TWIST_PLUS);
      x->Even().changeTwist(QUDA_TWIST_PLUS);
      x->Odd().changeTwist(QUDA_TWIST_PLUS);
      for(int i = 0 ; i < 4 ; i++)
	my_src[i] = info.sourcePosition[isource][i] - comm_coords(default_topo)[i] * X[i];

      if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
	*( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + isc*2 ) = 1.;

      K_vector->packVector((double*) input_vector);
      K_vector->loadVector();
      K_guess->gaussianSmearing(*K_vector,*K_gauge);
      K_guess->uploadToCuda(b,flag_eo);
      dirac.prepare(in,out,*x,*b,param->solution_type);
      // in is reference to the b but for a parity sinlet
      // out is reference to the x but for a parity sinlet
      cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);
      dirac.Mdag(*in, *tmp_up);
      delete tmp_up;
      // now the the source vector b is ready to perform deflation and find the initial guess
      K_vector->downloadFromCuda(in,flag_eo);
      K_vector->download();
      deflation_up->deflateVector(*K_guess,*K_vector);
      K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
      //  blas::zero(*out); // remove it later , just for test
      (*solve)(*out,*in);
      dirac.reconstruct(*x,*b,param->solution_type);
      K_vector->downloadFromCuda(x,flag_eo);
      if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	K_vector->scaleVector(2*param->kappa);
      }
      K_guess->gaussianSmearing(*K_vector,*K_gauge);
      K_temp->castDoubleToFloat(*K_guess);
      K_prop_up->absorbVectorToDevice(*K_temp,isc/3,isc%3);
      t2 = MPI_Wtime();
      printfQuda("Inversion up = %d,  for source = %d finished in time %f sec\n",isc,isource,t2-t1);
      //////////////////////////////////////////////////////////
      //////////////////////////////////////////////////////////
      /////////////////////////////////////////////////////////
      t1 = MPI_Wtime();
      memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
      b->changeTwist(QUDA_TWIST_MINUS);
      x->changeTwist(QUDA_TWIST_MINUS);
      b->Even().changeTwist(QUDA_TWIST_MINUS);
      b->Odd().changeTwist(QUDA_TWIST_MINUS);
      x->Even().changeTwist(QUDA_TWIST_MINUS);
      x->Odd().changeTwist(QUDA_TWIST_MINUS);
      for(int i = 0 ; i < 4 ; i++)
	my_src[i] = info.sourcePosition[isource][i] - comm_coords(default_topo)[i] * X[i];

      if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
	*( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + isc*2 ) = 1.;

      K_vector->packVector((double*) input_vector);
      K_vector->loadVector();
      K_guess->gaussianSmearing(*K_vector,*K_gauge);
      K_guess->uploadToCuda(b,flag_eo);
      dirac.prepare(in,out,*x,*b,param->solution_type);
      // in is reference to the b but for a parity sinlet
      // out is reference to the x but for a parity sinlet
      cudaColorSpinorField *tmp_down = new cudaColorSpinorField(*in);
      dirac.Mdag(*in, *tmp_down);
      delete tmp_down;
      // now the the source vector b is ready to perform deflation and find the initial guess
      K_vector->downloadFromCuda(in,flag_eo);
      K_vector->download();
      deflation_down->deflateVector(*K_guess,*K_vector);
      K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
      //  blas::zero(*out); // remove it later , just for test
      (*solve)(*out,*in);
      dirac.reconstruct(*x,*b,param->solution_type);
      K_vector->downloadFromCuda(x,flag_eo);
      if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	K_vector->scaleVector(2*param->kappa);
      }
      K_guess->gaussianSmearing(*K_vector,*K_gauge);
      K_temp->castDoubleToFloat(*K_guess);
      K_prop_down->absorbVectorToDevice(*K_temp,isc/3,isc%3);
      t2 = MPI_Wtime();
      printfQuda("Inversion down = %d,  for source = %d finished in time %f sec\n",isc,isource,t2-t1);
    } // close loop over 12 spin-color

    K_prop_up->rotateToPhysicalBase_device(+1);
    K_prop_down->rotateToPhysicalBase_device(-1);
    t1 = MPI_Wtime();
    K_contract->contractMesons(*K_prop_up,*K_prop_down,filename_mesons,isource);
    K_contract->contractBaryons(*K_prop_up,*K_prop_down,filename_baryons,isource);
    t2 = MPI_Wtime();
    printfQuda("Contractions for source = %d finished in time %f sec\n",isource,t2-t1);
  } // close loop over source positions


  free(input_vector);
  free(output_vector);
  delete K_temp;
  delete K_contract;
  delete K_prop_down;
  delete K_prop_up;
  delete solve;
  delete d;
  delete dSloppy;
  delete dPre;
  delete K_guess;
  delete K_vector;
  delete K_gauge;
  delete deflation_up;
  delete deflation_down;
  delete h_x;
  delete h_b;
  delete x;
  delete b;

  popVerbosity();
  saveTuneCache();
  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);

}

template <typename Float>
void getStochasticRandomSource(void *spinorIn, gsl_rng *rNum, SOURCE_T source_type){
  memset(spinorIn,0,GK_localVolume*12*2*sizeof(Float));
  for(int i = 0; i<GK_localVolume*12; i++){
    int randomNumber = gsl_rng_uniform_int(rNum, 4);

    if(source_type==UNITY){
      ((Float*) spinorIn)[i*2] = 1.0;
      ((Float*) spinorIn)[i*2+1] = 0.0;
    }
    else if(source_type==RANDOM){
      switch  (randomNumber){
      case 0:
	((Float*) spinorIn)[i*2] = 1.;
	break;
      case 1:
	((Float*) spinorIn)[i*2] = -1.;
	break;
      case 2:
	((Float*) spinorIn)[i*2+1] = 1.;
	break;
      case 3:
	((Float*) spinorIn)[i*2+1] = -1.;
	break;
      }
    }
    else{
      errorQuda("Source type not set correctly!! Aborting.\n");
    }
  }

}

template <typename Float>
void getStochasticRandomSource(void *spinorIn, gsl_rng *rNum){
  memset(spinorIn,0,GK_localVolume*12*2*sizeof(Float));

  for(int i = 0; i<GK_localVolume*12; i++){

    //- Unity sources
    //    ((Float*) spinorIn)[i*2] = 1.0;
    //    ((Float*) spinorIn)[i*2+1] = 0.0;

    //- Random sources
    int randomNumber = gsl_rng_uniform_int(rNum, 4);
    switch  (randomNumber)
      {
      case 0:
	((Float*) spinorIn)[i*2] = 1.;
	break;
      case 1:
	((Float*) spinorIn)[i*2] = -1.;
	break;
      case 2:
	((Float*) spinorIn)[i*2+1] = 1.;
	break;
      case 3:
	((Float*) spinorIn)[i*2+1] = -1.;
	break;
      }

  }//-for

}


void DeflateAndInvert_loop(void **gaugeToPlaquette, QudaInvertParam *param ,QudaGaugeParam *gauge_param, char *filename_eigenValues_down, char *filename_eigenVectors_down,char *filename_out , int NeV , int Nstoch, int seed ,int NdumpStep, qudaQKXTMinfo_Kepler info){
  bool flag_eo;
  double t1,t2;

  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);
  if(param->solve_type != QUDA_NORMOP_PC_SOLVE) errorQuda("This function works only with even odd preconditioning");
  if(param->inv_type != QUDA_CG_INVERTER) errorQuda("This function works only with CG method");
  if( (param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) && (param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) ) errorQuda("Only asymmetric operators are supported in deflation\n");
  if( param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC )
    flag_eo = true;
  else if(param->matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
    flag_eo = false;

  QKXTM_Deflation_Kepler<double> *deflation_down = new QKXTM_Deflation_Kepler<double>(NeV,flag_eo);
  deflation_down->printInfo();

  if(param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS) errorQuda("This function works only with ukqcd gamma basis\n");
  if(param->dirac_order != QUDA_DIRAC_ORDER) errorQuda("This function works only with colors inside the spins\n");

  t1 = MPI_Wtime();

  deflation_down->readEigenValues(filename_eigenValues_down);
  deflation_down->readEigenVectors(filename_eigenVectors_down);
  deflation_down->rotateFromChiralToUKQCD();
  if(gauge_param->t_boundary == QUDA_ANTI_PERIODIC_T)deflation_down->multiply_by_phase();
  t2 = MPI_Wtime();

#ifdef TIMING_REPORT
  printfQuda("Timing report for read and transform eigenVectors is %f sec\n",t2-t1);
#endif

  if (!initialized) errorQuda("QUDA not initialized");
  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);

  QKXTM_Gauge_Kepler<double> *K_gauge = new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  K_gauge->packGauge(gaugeToPlaquette);
  K_gauge->loadGauge();
  K_gauge->calculatePlaq();

  bool pc_solution = false;
  bool pc_solve = true;
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) || (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = false;

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;
  
  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;
  // create the dirac operator
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;
  profileInvert.TPSTART(QUDA_PROFILE_H2D);

  //QKXTM: DMH rewite for spinor field memalloc
  ColorSpinorField *b = NULL;
  ColorSpinorField *x = NULL;
  ColorSpinorField *in = NULL;
  ColorSpinorField *out = NULL;
  cudaColorSpinorField *tmp3 = NULL;
  cudaColorSpinorField *tmp4 = NULL;

  const int *X = cudaGauge->X();

  void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  void *output_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  memset(output_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  // wrap CPU host side pointers
  ColorSpinorParam cpuParam(input_vector, *param, X, pc_solution, param->input_location);
  ColorSpinorField *h_b = ColorSpinorField::Create(cpuParam);

  cpuParam.v = output_vector;
  cpuParam.location = param->output_location;
  ColorSpinorField *h_x = ColorSpinorField::Create(cpuParam);

  // download source
  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam);

  //Zero out the solution
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  x = new cudaColorSpinorField(cudaParam);

  tmp3 = new cudaColorSpinorField(cudaParam);
  tmp4 = new cudaColorSpinorField(cudaParam);

  profileInvert.TPSTOP(QUDA_PROFILE_H2D);
  setTuning(param->tune);
  
  blas::zero(*x);
  blas::zero(*b);
  DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
  SolverParam solverParam(*param);
  Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
  QKXTM_Vector_Kepler<double> *K_vector = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<double> *K_guess = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);

  void    *cnRes_vv;
  void    *cnRes_gv;

  void    *cnResTmp_vv;
  void    *cnResTmp_gv;

  if((cudaHostAlloc(&cnRes_vv, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("Error allocating memory cnRes_vv\n");
  if((cudaHostAlloc(&cnRes_gv, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("Error allocating memory cnRes_gv\n");

  cudaMemset      (cnRes_vv, 0, sizeof(double)*2*16*GK_localVolume);
  cudaMemset      (cnRes_gv, 0, sizeof(double)*2*16*GK_localVolume);

  if((cudaHostAlloc(&cnResTmp_vv, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("Error allocating memory cnRes_vv\n");
  if((cudaHostAlloc(&cnResTmp_gv, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("Error allocating memory cnRes_gv\n");

  cudaMemset      (cnResTmp_vv, 0, sizeof(double)*2*16*GK_localVolume);
  cudaMemset      (cnResTmp_gv, 0, sizeof(double)*2*16*GK_localVolume);

  gsl_rng *rNum = gsl_rng_alloc(gsl_rng_ranlux);
  gsl_rng_set(rNum, seed + comm_rank()*seed);

  if(info.source_type==RANDOM) printfQuda("Will use RANDOM stochastic sources\n");
  else if (info.source_type==UNITY) printfQuda("Will use UNITY stochastic sources\n");

  for(int is = 0 ; is < Nstoch ; is++){
    t1 = MPI_Wtime();
    memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
    getStochasticRandomSource<double>(input_vector,rNum,info.source_type);
    K_vector->packVector((double*) input_vector);
    K_vector->loadVector();
    K_vector->uploadToCuda(b,flag_eo);
    dirac.prepare(in,out,*x,*b,param->solution_type);
    // in is reference to the b but for a parity sinlet
    // out is reference to the x but for a parity sinlet
    cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);
    dirac.Mdag(*in, *tmp_up);
    delete tmp_up;
    // now the the source vector b is ready to perform deflation and find the initial guess
    K_vector->downloadFromCuda(in,flag_eo);
    K_vector->download();
    deflation_down->deflateVector(*K_guess,*K_vector);
    K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
    //  blas::zero(*out); // remove it later , just for test
    (*solve)(*out,*in);
    dirac.reconstruct(*x,*b,param->solution_type);
    oneEndTrick<double>(*x,*tmp3,*tmp4,param,cnRes_gv,cnRes_vv);
    t2 = MPI_Wtime();
    printfQuda("Stoch %d finished in %f sec\n",is,t2-t1);
    if( (is+1)%NdumpStep == 0){
      doCudaFFT<double>(cnRes_gv,cnRes_vv,cnResTmp_gv,cnResTmp_vv);
      dumpLoop<double>(cnResTmp_gv,cnResTmp_vv,filename_out,is+1,info.Q_sq);
    }
  } // close loop over source positions

  cudaFreeHost(cnRes_gv);
  cudaFreeHost(cnRes_vv);

  cudaFreeHost(cnResTmp_gv);
  cudaFreeHost(cnResTmp_vv);

  free(input_vector);
  free(output_vector);
  gsl_rng_free(rNum);
  delete solve;
  delete d;
  delete dSloppy;
  delete dPre;
  delete K_guess;
  delete K_vector;
  delete K_gauge;
  delete deflation_down;
  delete h_x;
  delete h_b;
  delete x;
  delete b;
  delete tmp3;
  delete tmp4;
  popVerbosity();
  saveTuneCache();
  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);

}

void DeflateAndInvert_loop_w_One_Der(void **gaugeToPlaquette, QudaInvertParam *param ,QudaGaugeParam *gauge_param, char *filename_eigenValues_down, char *filename_eigenVectors_down,char *filename_out , int NeV , int Nstoch, int seed ,int NdumpStep, qudaQKXTMinfo_Kepler info){
  bool flag_eo;
  double t1,t2;

  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);
  if(param->solve_type != QUDA_NORMOP_PC_SOLVE) errorQuda("This function works only with even odd preconditioning");
  if(param->inv_type != QUDA_CG_INVERTER) errorQuda("This function works only with CG method");
  if( (param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) && (param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) ) errorQuda("Only asymmetric operators are supported in deflation\n");
  if( param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC )
    flag_eo = true;
  else if(param->matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
    flag_eo = false;

  QKXTM_Deflation_Kepler<double> *deflation_down = new QKXTM_Deflation_Kepler<double>(NeV,flag_eo);
  deflation_down->printInfo();

  if(param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS) errorQuda("This function works only with ukqcd gamma basis\n");
  if(param->dirac_order != QUDA_DIRAC_ORDER) errorQuda("This function works only with colors inside the spins\n");

  t1 = MPI_Wtime();

  deflation_down->readEigenValues(filename_eigenValues_down);
  deflation_down->readEigenVectors(filename_eigenVectors_down);
  deflation_down->rotateFromChiralToUKQCD();
  if(gauge_param->t_boundary == QUDA_ANTI_PERIODIC_T)deflation_down->multiply_by_phase();
  t2 = MPI_Wtime();

#ifdef TIMING_REPORT
  printfQuda("Timing report for read and transform eigenVectors is %f sec\n",t2-t1);
#endif

  if (!initialized) errorQuda("QUDA not initialized");
  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);

  QKXTM_Gauge_Kepler<double> *K_gauge = new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  K_gauge->packGauge(gaugeToPlaquette);
  K_gauge->loadGauge();
  K_gauge->calculatePlaq();

  bool pc_solution = false;
  bool pc_solve = true;
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) || (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = false;

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;
  
  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;
  // create the dirac operator                                                                                                                                      
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;
  profileInvert.TPSTART(QUDA_PROFILE_H2D);

  //QKXTM: DMH rewite for spinor field memalloc
  ColorSpinorField *b = NULL;
  ColorSpinorField *x = NULL;
  ColorSpinorField *in = NULL;
  ColorSpinorField *out = NULL;
  ColorSpinorField *tmp3 = NULL;
  ColorSpinorField *tmp4 = NULL;
  const int *X = cudaGauge->X();

  void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  void *output_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  memset(output_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  // wrap CPU host side pointers
  ColorSpinorParam cpuParam(input_vector, *param, X, pc_solution, param->input_location);
  ColorSpinorField *h_b = ColorSpinorField::Create(cpuParam);

  cpuParam.v = output_vector;
  cpuParam.location = param->output_location;
  ColorSpinorField *h_x = ColorSpinorField::Create(cpuParam);

  //Zero out
  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  x = new cudaColorSpinorField(cudaParam);

  tmp3 = new cudaColorSpinorField(cudaParam);
  tmp4 = new cudaColorSpinorField(cudaParam);

  profileInvert.TPSTOP(QUDA_PROFILE_H2D);
  setTuning(param->tune);
  
  blas::zero(*x);
  blas::zero(*b);
  DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
  SolverParam solverParam(*param);
  Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
  QKXTM_Vector_Kepler<double> *K_vector = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<double> *K_guess = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);

  ////////////////////////// Allocate memory for local
  void    *cnRes_vv;
  void    *cnRes_gv;

  void    *cnTmp;

  if((cudaHostAlloc(&cnRes_vv, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("Error allocating memory cnRes_vv\n");
  if((cudaHostAlloc(&cnRes_gv, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("Error allocating memory cnRes_gv\n");

  cudaMemset      (cnRes_vv, 0, sizeof(double)*2*16*GK_localVolume);
  cudaMemset      (cnRes_gv, 0, sizeof(double)*2*16*GK_localVolume);

  if((cudaHostAlloc(&cnTmp, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("Error allocating memory cnTmp\n");

  cudaMemset      (cnTmp, 0, sizeof(double)*2*16*GK_localVolume);
  ///////////////////////////////////////////////////
  //////////// Allocate memory for one-Der and conserved current
  void    **cnD_vv;
  void    **cnD_gv;
  void    **cnC_vv;
  void    **cnC_gv;

  cnD_vv   = (void**) malloc(sizeof(double*)*2*4);
  cnD_gv   = (void**) malloc(sizeof(double*)*2*4);
  cnC_vv   = (void**) malloc(sizeof(double*)*2*4);
  cnC_gv   = (void**) malloc(sizeof(double*)*2*4);

  if(cnD_gv == NULL)errorQuda("Error allocating memory cnD_gv higher level\n");
  if(cnD_vv == NULL)errorQuda("Error allocating memory cnD_vv higher level\n");
  if(cnC_gv == NULL)errorQuda("Error allocating memory cnC_gv higher level\n");
  if(cnC_vv == NULL)errorQuda("Error allocating memory cnC_vv higher level\n");
  cudaDeviceSynchronize();

  for(int mu = 0; mu < 4 ; mu++){
    if((cudaHostAlloc(&(cnD_vv[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("Error allocating memory cnD_vv\n");
    if((cudaHostAlloc(&(cnD_gv[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("Error allocating memory cnD_gv\n");
    if((cudaHostAlloc(&(cnC_vv[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("Error allocating memory cnC_vv\n");
    if((cudaHostAlloc(&(cnC_gv[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("Error allocating memory cnC_gv\n");

    cudaMemset(cnD_vv[mu], 0, sizeof(double)*2*16*GK_localVolume);
    cudaMemset(cnD_gv[mu], 0, sizeof(double)*2*16*GK_localVolume);
    cudaMemset(cnC_vv[mu], 0, sizeof(double)*2*16*GK_localVolume);
    cudaMemset(cnC_gv[mu], 0, sizeof(double)*2*16*GK_localVolume);
  }
  cudaDeviceSynchronize();
  ///////////////////////////////////////////////////
  gsl_rng *rNum = gsl_rng_alloc(gsl_rng_ranlux);
  gsl_rng_set(rNum, seed + comm_rank()*seed);

  if(info.source_type==RANDOM) printfQuda("Will use RANDOM stochastic sources\n");
  else if (info.source_type==UNITY) printfQuda("Will use UNITY stochastic sources\n");

  for(int is = 0 ; is < Nstoch ; is++){
    t1 = MPI_Wtime();
    memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
    getStochasticRandomSource<double>(input_vector,rNum,info.source_type);
    K_vector->packVector((double*) input_vector);
    K_vector->loadVector();
    K_vector->uploadToCuda(b,flag_eo);
    dirac.prepare(in,out,*x,*b,param->solution_type);
    // in is reference to the b but for a parity sinlet
    // out is reference to the x but for a parity sinlet
    cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);
    dirac.Mdag(*in, *tmp_up);
    delete tmp_up;
    // now the the source vector b is ready to perform deflation and find the initial guess
    K_vector->downloadFromCuda(in,flag_eo);
    K_vector->download();
    deflation_down->deflateVector(*K_guess,*K_vector);
    K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
    //  blas::zero(*out); // remove it later , just for test
    (*solve)(*out,*in);
    dirac.reconstruct(*x,*b,param->solution_type);
    oneEndTrick_w_One_Der<double>(*x,*tmp3,*tmp4,param,cnRes_gv,cnRes_vv,cnD_gv,cnD_vv,cnC_gv,cnC_vv);
    t2 = MPI_Wtime();
    printfQuda("Stoch %d finished in %f sec\n",is,t2-t1);
    if( (is+1)%NdumpStep == 0){
      doCudaFFT_v2<double>(cnRes_vv,cnTmp);
      dumpLoop_ultraLocal<double>(cnTmp,filename_out,is+1,info.Q_sq,0); // Scalar
      doCudaFFT_v2<double>(cnRes_gv,cnTmp);
      dumpLoop_ultraLocal<double>(cnTmp,filename_out,is+1,info.Q_sq,1); // dOp
      for(int mu = 0 ; mu < 4 ; mu++){
	doCudaFFT_v2<double>(cnD_vv[mu],cnTmp);
	dumpLoop_oneD<double>(cnTmp,filename_out,is+1,info.Q_sq,mu,0); // Loops
	doCudaFFT_v2<double>(cnD_gv[mu],cnTmp);
	dumpLoop_oneD<double>(cnTmp,filename_out,is+1,info.Q_sq,mu,1); // LpsDw

	doCudaFFT_v2<double>(cnC_vv[mu],cnTmp);
	dumpLoop_oneD<double>(cnTmp,filename_out,is+1,info.Q_sq,mu,2); // LpsDw noether
	doCudaFFT_v2<double>(cnC_gv[mu],cnTmp);
	dumpLoop_oneD<double>(cnTmp,filename_out,is+1,info.Q_sq,mu,3); // LpsDw noether
      }
    } // close loop for dump loops

  } // close loop over source positions

  cudaFreeHost(cnRes_gv);
  cudaFreeHost(cnRes_vv);

  cudaFreeHost(cnTmp);

  for(int mu = 0 ; mu < 4 ; mu++){
    cudaFreeHost(cnD_vv[mu]);
    cudaFreeHost(cnD_gv[mu]);
    cudaFreeHost(cnC_vv[mu]);
    cudaFreeHost(cnC_gv[mu]);
  }
  
  free(cnD_vv);
  free(cnD_gv);
  free(cnC_vv);
  free(cnC_gv);

  free(input_vector);
  free(output_vector);
  gsl_rng_free(rNum);
  delete solve;
  delete d;
  delete dSloppy;
  delete dPre;
  delete K_guess;
  delete K_vector;
  delete K_gauge;
  delete deflation_down;
  delete h_x;
  delete h_b;
  delete x;
  delete b;
  delete tmp3;
  delete tmp4;
  popVerbosity();
  saveTuneCache();
  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);
}

void DeflateAndInvert_loop_w_One_Der_volumeSource(void **gaugeToPlaquette, QudaInvertParam *param ,QudaGaugeParam *gauge_param, char *filename_eigenValues_up, char *filename_eigenVectors_up, char *filename_eigenValues_down, char *filename_eigenVectors_down,char *filename_out , int NeV , int Nstoch, int seed ,int NdumpStep, qudaQKXTMinfo_Kepler info){
  bool flag_eo;
  double t1,t2;

  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);
  if(param->solve_type != QUDA_NORMOP_PC_SOLVE) errorQuda("This function works only with even odd preconditioning");
  if(param->inv_type != QUDA_CG_INVERTER) errorQuda("This function works only with CG method");
  if( (param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) && (param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) ) errorQuda("Only asymmetric operators are supported in deflation\n");
  if( param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC )
    flag_eo = true;
  else if(param->matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
    flag_eo = false;

  QKXTM_Deflation_Kepler<double> *deflation_up = new QKXTM_Deflation_Kepler<double>(NeV,flag_eo);
  QKXTM_Deflation_Kepler<double> *deflation_down = new QKXTM_Deflation_Kepler<double>(NeV,flag_eo);
  deflation_down->printInfo();

  if(param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS) errorQuda("This function works only with ukqcd gamma basis\n");
  if(param->dirac_order != QUDA_DIRAC_ORDER) errorQuda("This function works only with colors inside the spins\n");

  t1 = MPI_Wtime();

  deflation_up->readEigenValues(filename_eigenValues_up);
  deflation_up->readEigenVectors(filename_eigenVectors_up);
  deflation_up->rotateFromChiralToUKQCD();

  deflation_down->readEigenValues(filename_eigenValues_down);
  deflation_down->readEigenVectors(filename_eigenVectors_down);
  deflation_down->rotateFromChiralToUKQCD();
  if(gauge_param->t_boundary == QUDA_ANTI_PERIODIC_T)deflation_down->multiply_by_phase();
  t2 = MPI_Wtime();

#ifdef TIMING_REPORT
  printfQuda("Timing report for read and transform eigenVectors is %f sec\n",t2-t1);
#endif

  if (!initialized) errorQuda("QUDA not initialized");
  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);

  QKXTM_Gauge_Kepler<double> *K_gauge = new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  K_gauge->packGauge(gaugeToPlaquette);
  K_gauge->loadGauge();
  K_gauge->calculatePlaq();

  bool pc_solution = false;
  bool pc_solve = true;
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) || (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = false;

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;
  
  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;
  // create the dirac operator                                                                                                                                      
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;
  profileInvert.TPSTART(QUDA_PROFILE_H2D);

  //QKXTM: DMH rewite for spinor field memalloc
  ColorSpinorField *b = NULL;
  ColorSpinorField *x = NULL;
  ColorSpinorField *in = NULL;
  ColorSpinorField *out = NULL;
  ColorSpinorField *tmp = NULL;
  const int *X = cudaGauge->X();

  void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  void *output_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  memset(output_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  // wrap CPU host side pointers
  ColorSpinorParam cpuParam(input_vector, *param, X, pc_solution, param->input_location);
  ColorSpinorField *h_b = ColorSpinorField::Create(cpuParam);

  cpuParam.v = output_vector;
  cpuParam.location = param->output_location;
  ColorSpinorField *h_x = ColorSpinorField::Create(cpuParam);

  //Zero out the spinors
  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  x = new cudaColorSpinorField(cudaParam);

  tmp = new cudaColorSpinorField(cudaParam);

  profileInvert.TPSTOP(QUDA_PROFILE_H2D);
  setTuning(param->tune);
  
  blas::zero(*x);
  blas::zero(*b);
  DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
  SolverParam solverParam(*param);
  Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
  QKXTM_Vector_Kepler<double> *K_vector = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<double> *K_guess = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);

  ////////////////////////// Allocate memory for local
  void    *cn_local_up;
  void    *cn_local_down;
  void    *cnTmp;

  if((cudaHostAlloc(&cn_local_up, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("Error allocating memory cn_local_up\n");
  cudaMemset      (cn_local_up, 0, sizeof(double)*2*16*GK_localVolume);

  if((cudaHostAlloc(&cn_local_down, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("Error allocating memory cn_local_down\n");
  cudaMemset      (cn_local_down, 0, sizeof(double)*2*16*GK_localVolume);

  if((cudaHostAlloc(&cnTmp, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("Error allocating memory cnTmp\n");
  cudaMemset      (cnTmp, 0, sizeof(double)*2*16*GK_localVolume);
  ///////////////////////////////////////////////////
  //////////// Allocate memory for one-Der and conserved current
  void    **cnD_up;
  void    **cnC_up;
  void    **cnD_down;
  void    **cnC_down;

  cnD_up   = (void**) malloc(sizeof(double*)*2*4);
  cnC_up   = (void**) malloc(sizeof(double*)*2*4);
  cnD_down   = (void**) malloc(sizeof(double*)*2*4);
  cnC_down   = (void**) malloc(sizeof(double*)*2*4);

  if(cnD_up == NULL)errorQuda("Error allocating memory cnD higher level\n");
  if(cnC_up == NULL)errorQuda("Error allocating memory cnC higher level\n");

  if(cnD_down == NULL)errorQuda("Error allocating memory cnD higher level\n");
  if(cnC_down == NULL)errorQuda("Error allocating memory cnC higher level\n");

  cudaDeviceSynchronize();

  for(int mu = 0; mu < 4 ; mu++){
    if((cudaHostAlloc(&(cnD_up[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("Error allocating memory cnD\n");
    if((cudaHostAlloc(&(cnC_up[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("Error allocating memory cnC\n");

    if((cudaHostAlloc(&(cnD_down[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("Error allocating memory cnD\n");
    if((cudaHostAlloc(&(cnC_down[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("Error allocating memory cnC\n");

  }
  cudaDeviceSynchronize();
  ///////////////////////////////////////////////////
  gsl_rng *rNum = gsl_rng_alloc(gsl_rng_ranlux);
  gsl_rng_set(rNum, seed + comm_rank()*seed);

  if(info.source_type==RANDOM) printfQuda("Will use RANDOM stochastic sources\n");
  else if (info.source_type==UNITY) printfQuda("Will use UNITY stochastic sources\n");

  for(int is = 0 ; is < Nstoch ; is++){

    memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
    getStochasticRandomSource<double>(input_vector,rNum,info.source_type);
    t1 = MPI_Wtime();

#define CROSSCHECK
#ifdef CROSSCHECK
    FILE *ptr_xi;
    ptr_xi=fopen("/users/krikitos/run/test_loop/volumeSource.In","w");
    for(int ii = 0 ; ii < X[0]*X[1]*X[2]*X[3]*spinorSiteSize/2 ; ii++)
      fprintf(ptr_xi,"%+e %+e\n",((double*) input_vector)[ii*2+0], ((double*) input_vector)[ii*2+1]);
#endif

    // for up/////////////////////////////////////
    b->changeTwist(QUDA_TWIST_PLUS);
    x->changeTwist(QUDA_TWIST_PLUS);
    b->Even().changeTwist(QUDA_TWIST_PLUS);
    b->Odd().changeTwist(QUDA_TWIST_PLUS);
    x->Even().changeTwist(QUDA_TWIST_PLUS);
    x->Odd().changeTwist(QUDA_TWIST_PLUS);

    K_vector->packVector((double*) input_vector);
    K_vector->loadVector();
    K_vector->uploadToCuda(b,flag_eo);
    dirac.prepare(in,out,*x,*b,param->solution_type);
    // in is reference to the b but for a parity sinlet
    // out is reference to the x but for a parity sinlet
    cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);
    dirac.Mdag(*in, *tmp_up);
    delete tmp_up;
    // now the the source vector b is ready to perform deflation and find the initial guess
    K_vector->downloadFromCuda(in,flag_eo);
    K_vector->download();
    deflation_up->deflateVector(*K_guess,*K_vector);
    K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
    //  blas::zero(*out); // remove it later , just for test
    (*solve)(*out,*in);
    dirac.reconstruct(*x,*b,param->solution_type);


#ifdef CROSSCHECK
    K_guess->downloadFromCuda(x,flag_eo);
    K_guess->download();
    FILE *ptr_phi_up;
    ptr_phi_up=fopen("/users/krikitos/run/test_loop/volumeSource_up.Out","w");
    for(int ii = 0 ; ii < X[0]*X[1]*X[2]*X[3]*spinorSiteSize/2 ; ii++)
      fprintf(ptr_phi_up,"%+e %+e\n",K_guess->H_elem()[ii*2+0], K_guess->H_elem()[ii*2+1]);
#endif

    K_vector->packVector((double*) input_vector);
    K_vector->loadVector();
    K_vector->uploadToCuda(b,flag_eo);

    volumeSource_w_One_Der<double>(*x,*b,*tmp,param,cn_local_up,cnD_up,cnC_up);
    t2 = MPI_Wtime();
    printfQuda("Stoch %d for up finished in %f sec\n",is,t2-t1);
    if( (is+1)%NdumpStep == 0){
      doCudaFFT_v2<double>(cn_local_up,cnTmp);
      dumpLoop_ultraLocal_v2<double>(cnTmp,filename_out,is+1,info.Q_sq,"ultralocal_up"); 
      for(int mu = 0 ; mu < 4 ; mu++){
	doCudaFFT_v2<double>(cnD_up[mu],cnTmp);
	dumpLoop_oneD_v2<double>(cnTmp,filename_out,is+1,info.Q_sq,mu,"oneD_up"); 

	doCudaFFT_v2<double>(cnC_up[mu],cnTmp);
	dumpLoop_oneD_v2<double>(cnTmp,filename_out,is+1,info.Q_sq,mu,"noe_up"); 
      }
    } // close loop for dump loops

    //////////////////// for down
    t1 = MPI_Wtime();

    b->changeTwist(QUDA_TWIST_MINUS);
    x->changeTwist(QUDA_TWIST_MINUS);
    b->Even().changeTwist(QUDA_TWIST_MINUS);
    b->Odd().changeTwist(QUDA_TWIST_MINUS);
    x->Even().changeTwist(QUDA_TWIST_MINUS);
    x->Odd().changeTwist(QUDA_TWIST_MINUS);

    K_vector->packVector((double*) input_vector);
    K_vector->loadVector();
    K_vector->uploadToCuda(b,flag_eo);
    dirac.prepare(in,out,*x,*b,param->solution_type);
    // in is reference to the b but for a parity sinlet
    // out is reference to the x but for a parity sinlet
    cudaColorSpinorField *tmp_down = new cudaColorSpinorField(*in);
    dirac.Mdag(*in, *tmp_down);
    delete tmp_down;
    // now the the source vector b is ready to perform deflation and find the initial guess
    K_vector->downloadFromCuda(in,flag_eo);
    K_vector->download();
    deflation_down->deflateVector(*K_guess,*K_vector);
    K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
    //  blas::zero(*out); // remove it later , just for test
    (*solve)(*out,*in);
    dirac.reconstruct(*x,*b,param->solution_type);

#ifdef CROSSCHECK
    K_guess->downloadFromCuda(x,flag_eo);
    K_guess->download();
    FILE *ptr_phi_down;
    ptr_phi_down=fopen("/users/krikitos/run/test_loop/volumeSource_down.Out","w");
    for(int ii = 0 ; ii < X[0]*X[1]*X[2]*X[3]*spinorSiteSize/2 ; ii++)
      fprintf(ptr_phi_down,"%+e %+e\n",K_guess->H_elem()[ii*2+0], K_guess->H_elem()[ii*2+1]);
#endif


    K_vector->packVector((double*) input_vector);
    K_vector->loadVector();
    K_vector->uploadToCuda(b,flag_eo);

    volumeSource_w_One_Der<double>(*x,*b,*tmp,param,cn_local_down,cnD_down,cnC_down);
    t2 = MPI_Wtime();
    printfQuda("Stoch %d for down finished in %f sec\n",is,t2-t1);
    if( (is+1)%NdumpStep == 0){
      doCudaFFT_v2<double>(cn_local_down,cnTmp);
      dumpLoop_ultraLocal_v2<double>(cnTmp,filename_out,is+1,info.Q_sq,"ultralocal_down"); // Scalar
      for(int mu = 0 ; mu < 4 ; mu++){
	doCudaFFT_v2<double>(cnD_down[mu],cnTmp);
	dumpLoop_oneD_v2<double>(cnTmp,filename_out,is+1,info.Q_sq,mu,"oneD_down"); // Loops

	doCudaFFT_v2<double>(cnC_down[mu],cnTmp);
	dumpLoop_oneD_v2<double>(cnTmp,filename_out,is+1,info.Q_sq,mu,"noe_down"); // Loops noether
      }
    } // close loop for dump loops

  } // close loop over source positions

  cudaFreeHost(cn_local_up);
  cudaFreeHost(cn_local_down);
  cudaFreeHost(cnTmp);

  for(int mu = 0 ; mu < 4 ; mu++){
    cudaFreeHost(cnD_up[mu]);
    cudaFreeHost(cnD_down[mu]);
    cudaFreeHost(cnC_up[mu]);
    cudaFreeHost(cnC_down[mu]);
  }
  
  free(cnD_up);
  free(cnD_down);
  free(cnC_up);
  free(cnC_down);

  free(input_vector);
  free(output_vector);
  gsl_rng_free(rNum);
  delete solve;
  delete d;
  delete dSloppy;
  delete dPre;
  delete K_guess;
  delete K_vector;
  delete K_gauge;
  delete deflation_down;
  delete h_x;
  delete h_b;
  delete x;
  delete b;
  delete tmp;
  popVerbosity();
  saveTuneCache();
  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);
}

/*
#ifdef TEST
    FILE *ptr_file;
    ptr_file = fopen("/users/krikitos/run/test_loop/source.dat","w");
    if(ptr_file == NULL)errorQuda("cannot open file for writting\n");
    for(int iv = 0 ; iv < GK_localVolume ; iv++)
      for(int mu = 0 ; mu < 4 ; mu++)
	for(int c1 = 0 ; c1 < 3 ; c1++){
	  fprintf(ptr_file,"%f %f\n",((double*)input_vector)[iv*4*3*2+mu*3*2+c1*2+0],((double*)input_vector)[iv*4*3*2+mu*3*2+c1*2+1]);
	}
#endif

 */

void DeflateAndInvert_threepTwop(void **gaugeSmeared, void **gauge, QudaInvertParam *param ,QudaGaugeParam *gauge_param, char *filename_eigenValues_up, char *filename_eigenVectors_up, char *filename_eigenValues_down, char *filename_eigenVectors_down, char *filename_twop, char *filename_threep,int NeV, qudaQKXTMinfo_Kepler info, WHICHPARTICLE NUCLEON, WHICHPROJECTOR PID ){
  bool flag_eo;
  double t1,t2;

  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);
  if(param->solve_type != QUDA_NORMOP_PC_SOLVE) errorQuda("This function works only with even odd preconditioning");
  if(param->inv_type != QUDA_CG_INVERTER) errorQuda("This function works only with CG method");
  if( (param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) && (param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) ) errorQuda("Only asymmetric operators are supported in deflation\n");
  if( param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC )
    flag_eo = true;
  else if(param->matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
    flag_eo = false;

  QKXTM_Deflation_Kepler<double> *deflation_up = new QKXTM_Deflation_Kepler<double>(NeV,flag_eo);
  QKXTM_Deflation_Kepler<double> *deflation_down = new QKXTM_Deflation_Kepler<double>(NeV,flag_eo);
  void *input_vector = malloc(GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]*spinorSiteSize*sizeof(double));
  void *output_vector = malloc(GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]*spinorSiteSize*sizeof(double));
  QKXTM_Gauge_Kepler<double> *K_gaugeSmeared = new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  QKXTM_Gauge_Kepler<float> *K_gaugeContractions = new QKXTM_Gauge_Kepler<float>(BOTH,GAUGE);

  QKXTM_Vector_Kepler<double> *K_vector = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<double> *K_guess = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<float> *K_temp = new QKXTM_Vector_Kepler<float>(BOTH,VECTOR);


  QKXTM_Propagator_Kepler<float> *K_prop_up = new QKXTM_Propagator_Kepler<float>(BOTH,PROPAGATOR);
  QKXTM_Propagator_Kepler<float> *K_prop_down = new QKXTM_Propagator_Kepler<float>(BOTH,PROPAGATOR);  
  QKXTM_Propagator_Kepler<float> *K_seqProp = new QKXTM_Propagator_Kepler<float>(BOTH,PROPAGATOR);

  QKXTM_Propagator3D_Kepler<float> *K_prop3D_up = new QKXTM_Propagator3D_Kepler<float>(BOTH,PROPAGATOR3D);
  QKXTM_Propagator3D_Kepler<float> *K_prop3D_down = new QKXTM_Propagator3D_Kepler<float>(BOTH,PROPAGATOR3D);

  QKXTM_Contraction_Kepler<float> *K_contract = new QKXTM_Contraction_Kepler<float>();
  printfQuda("Memory allocation was successfull\n");

  if(param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS) errorQuda("This function works only with ukqcd gamma basis\n");
  if(param->dirac_order != QUDA_DIRAC_ORDER) errorQuda("This function works only with colors inside the spins\n");

  deflation_up->printInfo();
  t1 = MPI_Wtime();
  deflation_up->readEigenValues(filename_eigenValues_up);
  deflation_up->readEigenVectors(filename_eigenVectors_up);
  deflation_up->rotateFromChiralToUKQCD();
  if(gauge_param->t_boundary == QUDA_ANTI_PERIODIC_T)deflation_up->multiply_by_phase();

  deflation_down->readEigenValues(filename_eigenValues_down);
  deflation_down->readEigenVectors(filename_eigenVectors_down);
  deflation_down->rotateFromChiralToUKQCD();
  if(gauge_param->t_boundary == QUDA_ANTI_PERIODIC_T)deflation_down->multiply_by_phase();
  t2 = MPI_Wtime();

#ifdef TIMING_REPORT
  printfQuda("Timing report for read and transform eigenVectors is %f sec\n",t2-t1);
#endif

  if (!initialized) errorQuda("QUDA not initialized");
  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);


  K_gaugeContractions->packGauge(gauge);
  K_gaugeContractions->loadGauge();
  //  K_gaugeContractions->calculate(); do not do it because I changed the sign due to antiperiodic boundary conditions


  K_gaugeSmeared->packGauge(gaugeSmeared);
  K_gaugeSmeared->loadGauge();
  K_gaugeSmeared->calculatePlaq();

  bool pc_solution = false;
  bool pc_solve = true;
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) || (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = false;

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;
  
  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;
  // create the dirac operator                                                                                                                                      
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;
  profileInvert.TPSTART(QUDA_PROFILE_H2D);

  //QKXTM: DMH rewite for spinor field memalloc
  ColorSpinorField *b = NULL;
  ColorSpinorField *x = NULL;
  ColorSpinorField *in = NULL;
  ColorSpinorField *out = NULL;
  const int *X = cudaGauge->X();

  memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  memset(output_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  // wrap CPU host side pointers
  ColorSpinorParam cpuParam(input_vector, *param, X, pc_solution, param->input_location);
  ColorSpinorField *h_b = ColorSpinorField::Create(cpuParam);

  cpuParam.v = output_vector;
  cpuParam.location = param->output_location;
  ColorSpinorField *h_x = ColorSpinorField::Create(cpuParam);

  //Zero the fields
  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  x = new cudaColorSpinorField(cudaParam);

  profileInvert.TPSTOP(QUDA_PROFILE_H2D);
  setTuning(param->tune);
  
  blas::zero(*x);
  blas::zero(*b);
  DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
  SolverParam solverParam(*param);
  Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);

  int my_src[4];
  char filename_mesons[257];
  char filename_baryons[257];

  for(int isource = 0 ; isource < info.Nsources ; isource++){

     sprintf(filename_mesons,"%s.mesons.SS.%02d.%02d.%02d.%02d.dat",filename_twop,info.sourcePosition[isource][0],info.sourcePosition[isource][1],info.sourcePosition[isource][2],info.sourcePosition[isource][3]);
     sprintf(filename_baryons,"%s.baryons.SS.%02d.%02d.%02d.%02d.dat",filename_twop,info.sourcePosition[isource][0],info.sourcePosition[isource][1],info.sourcePosition[isource][2],info.sourcePosition[isource][3]);

//      bool checkMesons, checkBaryons;
//      checkMesons = exists_file(filename_mesons);
//      checkBaryons = exists_file(filename_baryons);
//      if( (checkMesons == true) && (checkBaryons == true) ) continue; // because threep are written before twop if I checked twop I know that threep are fine

    for(int isc = 0 ; isc < 12 ; isc++){
      ///////////////////////////////////////////////////////////////////////////////// forward prop for up quark ///////////////////////////
      t1 = MPI_Wtime();
      memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
      b->changeTwist(QUDA_TWIST_PLUS);
      x->changeTwist(QUDA_TWIST_PLUS);
      b->Even().changeTwist(QUDA_TWIST_PLUS);
      b->Odd().changeTwist(QUDA_TWIST_PLUS);
      x->Even().changeTwist(QUDA_TWIST_PLUS);
      x->Odd().changeTwist(QUDA_TWIST_PLUS);
      for(int i = 0 ; i < 4 ; i++)
	my_src[i] = info.sourcePosition[isource][i] - comm_coords(default_topo)[i] * X[i];

      if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
	*( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + isc*2 ) = 1.;

      K_vector->packVector((double*) input_vector);
      K_vector->loadVector();
      K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
      K_guess->uploadToCuda(b,flag_eo);
      dirac.prepare(in,out,*x,*b,param->solution_type);
      // in is reference to the b but for a parity sinlet
      // out is reference to the x but for a parity sinlet
      cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);
      dirac.Mdag(*in, *tmp_up);
      delete tmp_up;
      // now the the source vector b is ready to perform deflation and find the initial guess
      K_vector->downloadFromCuda(in,flag_eo);
      K_vector->download();
      deflation_up->deflateVector(*K_guess,*K_vector);
      K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
      //  blas::zero(*out); // remove it later , just for test
      (*solve)(*out,*in);
      dirac.reconstruct(*x,*b,param->solution_type);
      K_vector->downloadFromCuda(x,flag_eo);
      if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	K_vector->scaleVector(2*param->kappa);
      }

      K_temp->castDoubleToFloat(*K_vector);
      K_prop_up->absorbVectorToDevice(*K_temp,isc/3,isc%3);
      t2 = MPI_Wtime();
      printfQuda("Inversion up = %d,  for source = %d finished in time %f sec\n",isc,isource,t2-t1);
      //////////////////////////////////////////////////////////
      ////////////////////////////////////////////////////////////////////////////// Forward prop for down quark ///////////////////////////////////
      /////////////////////////////////////////////////////////
      t1 = MPI_Wtime();
      memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
      b->changeTwist(QUDA_TWIST_MINUS);
      x->changeTwist(QUDA_TWIST_MINUS);
      b->Even().changeTwist(QUDA_TWIST_MINUS);
      b->Odd().changeTwist(QUDA_TWIST_MINUS);
      x->Even().changeTwist(QUDA_TWIST_MINUS);
      x->Odd().changeTwist(QUDA_TWIST_MINUS);
      for(int i = 0 ; i < 4 ; i++)
	my_src[i] = info.sourcePosition[isource][i] - comm_coords(default_topo)[i] * X[i];

      if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
	*( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + isc*2 ) = 1.;

      K_vector->packVector((double*) input_vector);
      K_vector->loadVector();
      K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
      K_guess->uploadToCuda(b,flag_eo);
      dirac.prepare(in,out,*x,*b,param->solution_type);
      // in is reference to the b but for a parity sinlet
      // out is reference to the x but for a parity sinlet
      cudaColorSpinorField *tmp_down = new cudaColorSpinorField(*in);
      dirac.Mdag(*in, *tmp_down);
      delete tmp_down;
      // now the the source vector b is ready to perform deflation and find the initial guess
      K_vector->downloadFromCuda(in,flag_eo);
      K_vector->download();
      deflation_down->deflateVector(*K_guess,*K_vector);
      K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
      //  blas::zero(*out); // remove it later , just for test
      (*solve)(*out,*in);
      dirac.reconstruct(*x,*b,param->solution_type);
      K_vector->downloadFromCuda(x,flag_eo);
      if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	K_vector->scaleVector(2*param->kappa);
      }

      K_temp->castDoubleToFloat(*K_vector);
      K_prop_down->absorbVectorToDevice(*K_temp,isc/3,isc%3);
      t2 = MPI_Wtime();
      printfQuda("Inversion down = %d,  for source = %d finished in time %f sec\n",isc,isource,t2-t1);
    } // close loop over 12 spin-color


    if(info.run3pt_src[isource]){

      /////////////////////////////////// Smearing on the 3D propagators

      //-C.Kallidonis: Loop over the number of sink-source separations
      int my_fixSinkTime;
      char filename_threep_tsink[257];
      for(int its=0;its<info.Ntsink;its++){
	my_fixSinkTime = (info.tsinkSource[its] + info.sourcePosition[isource][3])%GK_totalL[3] - comm_coords(default_topo)[3] * X[3];
	sprintf(filename_threep_tsink,"%s_tsink%d",filename_threep,info.tsinkSource[its]);
	printfQuda("The three-point function base name is: %s\n",filename_threep_tsink);
      
	t1 = MPI_Wtime();
	K_temp->zero_device();
	checkCudaError();
	if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ){
	  K_prop3D_up->absorbTimeSlice(*K_prop_up,my_fixSinkTime);
	  K_prop3D_down->absorbTimeSlice(*K_prop_down,my_fixSinkTime);
	}
	comm_barrier();

	for(int nu = 0 ; nu < 4 ; nu++)
	  for(int c2 = 0 ; c2 < 3 ; c2++){
	    // up //
	    K_temp->zero_device();
	    if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_temp->copyPropagator3D(*K_prop3D_up,my_fixSinkTime,nu,c2);
	    comm_barrier();
	    K_vector->castFloatToDouble(*K_temp);
	    K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	    K_temp->castDoubleToFloat(*K_guess);
	    if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_prop3D_up->absorbVectorTimeSlice(*K_temp,my_fixSinkTime,nu,c2);
	    comm_barrier();
	    K_temp->zero_device();

	    // down //
	    if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_temp->copyPropagator3D(*K_prop3D_down,my_fixSinkTime,nu,c2);
	    comm_barrier();
	    K_vector->castFloatToDouble(*K_temp);
	    K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	    K_temp->castDoubleToFloat(*K_guess);
	    if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_prop3D_down->absorbVectorTimeSlice(*K_temp,my_fixSinkTime,nu,c2);
	    comm_barrier();
	    K_temp->zero_device();	
	  }
	t2 = MPI_Wtime();
	printfQuda("Time needed to prepare the 3D props for sink-source[%d]=%d is %f sec\n",its,info.tsinkSource[its],t2-t1);

	/////////////////////////////////////////sequential propagator for the part 1
	for(int nu = 0 ; nu < 4 ; nu++)
	  for(int c2 = 0 ; c2 < 3 ; c2++){
	    t1 = MPI_Wtime();
	    K_temp->zero_device();
	    if(NUCLEON == PROTON){
	      if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_contract->seqSourceFixSinkPart1(*K_temp,*K_prop3D_up, *K_prop3D_down, my_fixSinkTime, nu, c2, PID, NUCLEON);}
	    else if(NUCLEON == NEUTRON){
	      if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_contract->seqSourceFixSinkPart1(*K_temp,*K_prop3D_down, *K_prop3D_up, my_fixSinkTime, nu, c2, PID, NUCLEON);}
	    comm_barrier();
	    K_temp->conjugate();
	    K_temp->apply_gamma5();
	    K_vector->castFloatToDouble(*K_temp);
	    //
	    K_vector->scaleVector(1e+10);
	    //
	    K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	    if(NUCLEON == PROTON){
	      b->changeTwist(QUDA_TWIST_MINUS); x->changeTwist(QUDA_TWIST_MINUS); b->Even().changeTwist(QUDA_TWIST_MINUS);
	      b->Odd().changeTwist(QUDA_TWIST_MINUS); x->Even().changeTwist(QUDA_TWIST_MINUS); x->Odd().changeTwist(QUDA_TWIST_MINUS);
	    }
	    else{
	      b->changeTwist(QUDA_TWIST_PLUS); x->changeTwist(QUDA_TWIST_PLUS); b->Even().changeTwist(QUDA_TWIST_PLUS);
	      b->Odd().changeTwist(QUDA_TWIST_PLUS); x->Even().changeTwist(QUDA_TWIST_PLUS); x->Odd().changeTwist(QUDA_TWIST_PLUS);
	    }
	    K_guess->uploadToCuda(b,flag_eo);
	    dirac.prepare(in,out,*x,*b,param->solution_type);
	  
	    cudaColorSpinorField *tmp = new cudaColorSpinorField(*in);
	    dirac.Mdag(*in, *tmp);
	    delete tmp;
	    K_vector->downloadFromCuda(in,flag_eo);
	    K_vector->download();
	    if(NUCLEON == PROTON)
	      deflation_down->deflateVector(*K_guess,*K_vector);
	    else if(NUCLEON == NEUTRON)
	      deflation_up->deflateVector(*K_guess,*K_vector);
	    K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
	    (*solve)(*out,*in);
	    dirac.reconstruct(*x,*b,param->solution_type);
	    K_vector->downloadFromCuda(x,flag_eo);
	    if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	      K_vector->scaleVector(2*param->kappa);
	    }
	    //
	    K_vector->scaleVector(1e-10);
	    //
	    K_temp->castDoubleToFloat(*K_vector);
	    K_seqProp->absorbVectorToDevice(*K_temp,nu,c2);
	    t2 = MPI_Wtime();
	    printfQuda("Inversion for seq prop part 1 = %d,  for source = %d and sink-source = %d finished in time %f sec\n",nu*3+c2,isource,info.tsinkSource[its],t2-t1);
	  }

	////////////////// Contractions for part 1 ////////////////
	t1 = MPI_Wtime();
	if(NUCLEON == PROTON){
	  K_contract->contractFixSink(*K_seqProp, *K_prop_up, *K_gaugeContractions, PID, NUCLEON, 1, filename_threep_tsink, isource, info.tsinkSource[its]);
	}
	if(NUCLEON == NEUTRON){
	  K_contract->contractFixSink(*K_seqProp, *K_prop_down, *K_gaugeContractions, PID, NUCLEON, 1, filename_threep_tsink, isource, info.tsinkSource[its]);
	}                
	t2 = MPI_Wtime();
	printfQuda("Time for fix sink contractions for part 1 at sink-source = %d is %f sec\n",info.tsinkSource[its],t2-t1);
	/////////////////////////////////////////sequential propagator for the part 2
	for(int nu = 0 ; nu < 4 ; nu++)
	  for(int c2 = 0 ; c2 < 3 ; c2++){
	    t1 = MPI_Wtime();
	    K_temp->zero_device();
	    if(NUCLEON == PROTON){
	      if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_contract->seqSourceFixSinkPart2(*K_temp,*K_prop3D_up, my_fixSinkTime, nu, c2, PID, NUCLEON);}
	    else if(NUCLEON == NEUTRON){
	      if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_contract->seqSourceFixSinkPart2(*K_temp,*K_prop3D_down, my_fixSinkTime, nu, c2, PID, NUCLEON);}
	    comm_barrier();
	    K_temp->conjugate();
	    K_temp->apply_gamma5();
	    K_vector->castFloatToDouble(*K_temp);
	    //
	    K_vector->scaleVector(1e+10);
	    //
	    K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	    if(NUCLEON == PROTON){
	      b->changeTwist(QUDA_TWIST_PLUS); x->changeTwist(QUDA_TWIST_PLUS); b->Even().changeTwist(QUDA_TWIST_PLUS);
	      b->Odd().changeTwist(QUDA_TWIST_PLUS); x->Even().changeTwist(QUDA_TWIST_PLUS); x->Odd().changeTwist(QUDA_TWIST_PLUS);
	    }
	    else{
	      b->changeTwist(QUDA_TWIST_MINUS); x->changeTwist(QUDA_TWIST_MINUS); b->Even().changeTwist(QUDA_TWIST_MINUS);
	      b->Odd().changeTwist(QUDA_TWIST_MINUS); x->Even().changeTwist(QUDA_TWIST_MINUS); x->Odd().changeTwist(QUDA_TWIST_MINUS);
	    }
	    K_guess->uploadToCuda(b,flag_eo);
	    dirac.prepare(in,out,*x,*b,param->solution_type);
	  
	    cudaColorSpinorField *tmp = new cudaColorSpinorField(*in);
	    dirac.Mdag(*in, *tmp);
	    delete tmp;
	    K_vector->downloadFromCuda(in,flag_eo);
	    K_vector->download();
	    if(NUCLEON == PROTON)
	      deflation_up->deflateVector(*K_guess,*K_vector);
	    else if(NUCLEON == NEUTRON)
	      deflation_down->deflateVector(*K_guess,*K_vector);
	    K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
	    (*solve)(*out,*in);
	    dirac.reconstruct(*x,*b,param->solution_type);
	    K_vector->downloadFromCuda(x,flag_eo);
	    if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	      K_vector->scaleVector(2*param->kappa);
	    }
	    //
	    K_vector->scaleVector(1e-10);
	    //
	    K_temp->castDoubleToFloat(*K_vector);
	    K_seqProp->absorbVectorToDevice(*K_temp,nu,c2);
	    t2 = MPI_Wtime();
	    printfQuda("Inversion for seq prop part 2 = %d,  for source = %d and sink-source = %d finished in time %f sec\n",nu*3+c2,isource,info.tsinkSource[its],t2-t1);
	  }

	////////////////// Contractions for part 2 ////////////////
	t1 = MPI_Wtime();
	if(NUCLEON == PROTON)
	  K_contract->contractFixSink(*K_seqProp, *K_prop_down, *K_gaugeContractions, PID, NUCLEON, 2, filename_threep_tsink, isource, info.tsinkSource[its]);
	if(NUCLEON == NEUTRON)
	  K_contract->contractFixSink(*K_seqProp, *K_prop_up, *K_gaugeContractions, PID, NUCLEON, 2, filename_threep_tsink, isource, info.tsinkSource[its]);
	t2 = MPI_Wtime();

	printfQuda("Time for fix sink contractions for part 2 at sink-source = %d is %f sec\n",info.tsinkSource[its],t2-t1);

      }//-loop over sink-source separations      

    }//-if run for 3pt for specific source

    ////////// At the very end ///////////////////////


    // smear the forward propagators
    for(int nu = 0 ; nu < 4 ; nu++)
      for(int c2 = 0 ; c2 < 3 ; c2++){
	K_temp->copyPropagator(*K_prop_up,nu,c2);
	K_vector->castFloatToDouble(*K_temp);
	K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	K_temp->castDoubleToFloat(*K_guess);
	K_prop_up->absorbVectorToDevice(*K_temp,nu,c2);
	
	K_temp->copyPropagator(*K_prop_down,nu,c2);
	K_vector->castFloatToDouble(*K_temp);
	K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	K_temp->castDoubleToFloat(*K_guess);
	K_prop_down->absorbVectorToDevice(*K_temp,nu,c2);
      }
    /////
    K_prop_up->rotateToPhysicalBase_device(+1);
    K_prop_down->rotateToPhysicalBase_device(-1);
    t1 = MPI_Wtime();
    K_contract->contractMesons(*K_prop_up,*K_prop_down,filename_mesons,isource);
    K_contract->contractBaryons(*K_prop_up,*K_prop_down,filename_baryons,isource);
    t2 = MPI_Wtime();
    printfQuda("Contractions for source = %d finished in time %f sec\n",isource,t2-t1);
  } // close loop over source positions


  free(input_vector);
  free(output_vector);
  delete K_temp;
  delete K_contract;
  delete K_prop_down;
  delete K_prop_up;
  delete solve;
  delete d;
  delete dSloppy;
  delete dPre;
  delete K_guess;
  delete K_vector;
  delete K_gaugeSmeared;
  delete deflation_up;
  delete deflation_down;
  delete h_x;
  delete h_b;
  delete x;
  delete b;
  delete K_gaugeContractions;
  delete K_seqProp;
  delete K_prop3D_up;
  delete K_prop3D_down;

  popVerbosity();
  saveTuneCache();
  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);

}


//=============================================================================================//
//========================= E I G E N S O L V E R   F U N C T I O N S =========================//
//=============================================================================================//


void calcEigenVectors(QudaInvertParam *param , qudaQKXTM_arpackInfo arpackInfo){

  printfQuda("Running eigensolver...\n");

  QKXTM_Deflation_Kepler<double> *deflation = new QKXTM_Deflation_Kepler<double>(param,arpackInfo);

  deflation->eigenSolver();
  
  delete deflation;
}


void calcEigenVectors_Check(QudaInvertParam *param , qudaQKXTM_arpackInfo arpackInfo){

  printfQuda("Running eigensolver and checking the eigenvectors...\n");

  QKXTM_Deflation_Kepler<double> *deflation = new QKXTM_Deflation_Kepler<double>(param,arpackInfo);

  long int length_per_NeV = deflation->Length_Per_NeV();
  size_t bytes_length_per_NeV = deflation->Bytes_Per_NeV();

  double *vec_in  = (double*) malloc(bytes_length_per_NeV);
  double *vec_out = (double*) malloc(bytes_length_per_NeV);

  printfQuda("The pointer to vec in is %p\n",vec_in);
  printfQuda("The pointer to vec out is %p\n",vec_out);

  FILE *ptr_evecs;
  char filename[257];
  char filename2[257];

  int n_elem_write = length_per_NeV/2;

  //-Calculate the eigenvectors
  deflation->printInfo();
  deflation->eigenSolver();

//   for(int i=0;i<arpackInfo.nEv;i++){
//     for(int k=0;k<240;k++){
//       printfQuda("BEFORE ANYTHING: i = %04d, k = %04d: %+e  %+e\n",i,k,deflation->H_elem()[i*length_per_NeV+2*k],deflation->H_elem()[i*length_per_NeV+2*k+1]);
//     }
//     printfQuda("\n\n");
//   }

  sprintf(filename2,"h_elem_evenodd");
  deflation->writeEigenVectors_ASCII(filename2);

  if(arpackInfo.isFullOp) deflation->MapEvenOddToFull();

  for(int i=0;i<arpackInfo.nEv;i++){
    sprintf(filename,"eigenvecs_applyOp.%04d.txt",i);
    if( (ptr_evecs=fopen(filename,"w"))==NULL ) errorQuda("Cannot open filename for test\n");

    memset(vec_in ,0,bytes_length_per_NeV);
    memset(vec_out,0,bytes_length_per_NeV);
    deflation->copyEigenVectorToQKXTM_Vector_Kepler(i,vec_in);

    deflation->ApplyMdagM(vec_out,vec_in,param);

    for(int k=0;k<n_elem_write;k++){
      fprintf(ptr_evecs,"i = %04d , k = %10d: %+e  %+e     %+e  %+e     %7.5f  %7.5f\n",i,k,vec_out[2*k],vec_out[2*k+1],vec_in[2*k],vec_in[2*k+1],
	      vec_out[2*k]/(deflation->EigenValues()[2*i]*vec_in[2*k]),vec_out[2*k+1]/(deflation->EigenValues()[2*i]*vec_in[2*k+1]));
    }
    fclose(ptr_evecs);
  }

  sprintf(filename2,"h_elem_full");
  deflation->writeEigenVectors_ASCII(filename2);

  free(vec_in);
  free(vec_out);
  delete deflation;

  printfQuda("Calculation and checking of EigenVectors completed succesfully\n");
}

void calcEigenVectors_loop_wOneD_EvenOdd(void **gaugeToPlaquette, QudaInvertParam *param ,QudaGaugeParam *gauge_param,  qudaQKXTM_arpackInfo arpackInfo, qudaQKXTM_loopInfo loopInfo, qudaQKXTMinfo_Kepler info){

  bool flag_eo;
  double t1,t2;

  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);
  if( (arpackInfo.isFullOp) || (param->solve_type != QUDA_NORMOP_PC_SOLVE) ) errorQuda("calcEigenVectors_loop_wOneD: This function works only with even-odd preconditioning\n");
  
  if(param->inv_type != QUDA_CG_INVERTER) errorQuda("calcEigenVectors_loop_wOneD_EvenOdd: This function works only with CG method");
  
  if(param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS) errorQuda("calcEigenVectors_loop_wOneD_EvenOdd: This function works only with ukqcd gamma basis\n");
  if(param->dirac_order != QUDA_DIRAC_ORDER) errorQuda("calcEigenVectors_loop_wOneD_EvenOdd: This function works only with color-inside-spin\n");
  
  if( (param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) && (param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) ) errorQuda("Only asymmetric operators are supported in deflation\n");

  if( arpackInfo.isEven && (param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) ){
    errorQuda("calcEigenVectors_loop_wOneD_EvenOdd: Inconsistency between operator types!");
  }
  if( (!arpackInfo.isEven) && (param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) ){
    errorQuda("calcEigenVectors_loop_wOneD_EvenOdd: Inconsistency between operator types!");
  }

  if(arpackInfo.isEven){
    printfQuda("calcEigenVectors_loop_wOneD_EvenOdd: Solving for the Even-Even operator\n");
    flag_eo = true;
  }
  else{
    printfQuda("calcEigenVectors_loop_wOneD_EvenOdd: Solving for the Odd-Odd operator\n");
    flag_eo = false;
  }

  if (!initialized) errorQuda("calcEigenVectors_loop_wOneD_EvenOdd: QUDA not initialized");
  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);
  
  int Nstoch = loopInfo.Nstoch;
  unsigned long int seed = loopInfo.seed;
  int NdumpStep = loopInfo.Ndump;
  int Nprint = loopInfo.Nprint;
  loopInfo.Nmoms = GK_Nmoms;
  int Nmoms = GK_Nmoms;
  char filename_out[512];
  strcpy(filename_out,loopInfo.loop_fname);

  FILE_WRITE_FORMAT LoopFileFormat = loopInfo.FileFormat;

  loopInfo.loop_type[0] = "Scalar";      loopInfo.loop_oneD[0] = false;   // std-ultra_local
  loopInfo.loop_type[1] = "dOp";         loopInfo.loop_oneD[1] = false;   // gen-ultra_local
  loopInfo.loop_type[2] = "Loops";       loopInfo.loop_oneD[2] = true;    // std-one_derivative
  loopInfo.loop_type[3] = "LoopsCv";     loopInfo.loop_oneD[3] = true;    // std-conserved current
  loopInfo.loop_type[4] = "LpsDw";       loopInfo.loop_oneD[4] = true;    // gen-one_derivative
  loopInfo.loop_type[5] = "LpsDwCv";     loopInfo.loop_oneD[5] = true;    // gen-conserved current

  printfQuda("Loop Calculation Info\n");
  printfQuda("=====================\n");
  printfQuda("No. of noise vectors: %d\n",Nstoch);
  printfQuda("The seed is: %ld\n",seed);
  printfQuda("The conf trajectory is: %04d\n",loopInfo.traj);
  printfQuda("Will produce the loop for %d Momentum Combinations\n",Nmoms);
  printfQuda("Will dump every %d noise vectors, thus %d times\n",NdumpStep,Nprint);
  printfQuda("The loop file format is %s\n", (LoopFileFormat == ASCII_FORM) ? "ASCII" : "HDF5");
  printfQuda("The loop base name is %s\n",filename_out);
  printfQuda("=====================\n");

  //- Calculate the eigenVectors
  QKXTM_Deflation_Kepler<double> *deflation = new QKXTM_Deflation_Kepler<double>(param,arpackInfo);
  deflation->printInfo();

  t1 = MPI_Wtime();
  deflation->eigenSolver();
  t2 = MPI_Wtime();
  printfQuda("calcEigenVectors_loop_wOneD_EvenOdd TIME REPORT: EigenVector Calculation: %f sec\n",t2-t1);
  //--------------------------------------------------------------

  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);

  QKXTM_Gauge_Kepler<double> *K_gauge = new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  K_gauge->packGauge(gaugeToPlaquette);
  K_gauge->loadGauge();
  K_gauge->calculatePlaq();

  //  bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) ||
  //  (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

  bool pc_solution = false;
  bool pc_solve = true;
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) || (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = false;

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;
  
  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;
  // create the dirac operator                                                                                                                                      
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;
  profileInvert.TPSTART(QUDA_PROFILE_H2D);

  ColorSpinorField *b = NULL;
  ColorSpinorField *x = NULL;
  ColorSpinorField *in = NULL;
  ColorSpinorField *out = NULL;
  ColorSpinorField *tmp3 = NULL;
  ColorSpinorField *tmp4 = NULL;
  const int *X = cudaGauge->X();

  void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  void *output_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  memset(output_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  // wrap CPU host side pointers
  ColorSpinorParam cpuParam(input_vector, *param, X, pc_solution, param->input_location);
  ColorSpinorField *h_b = ColorSpinorField::Create(cpuParam);

  cpuParam.v = output_vector;
  cpuParam.location = param->output_location;
  ColorSpinorField *h_x = ColorSpinorField::Create(cpuParam);

  //Zero out the solution
  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  x = new cudaColorSpinorField(cudaParam);

  tmp3 = new cudaColorSpinorField(cudaParam);
  tmp4 = new cudaColorSpinorField(cudaParam);

  profileInvert.TPSTOP(QUDA_PROFILE_H2D);
  setTuning(param->tune);
  
  blas::zero(*x);
  blas::zero(*b);
  DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
  SolverParam solverParam(*param);
  Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
  QKXTM_Vector_Kepler<double> *K_vector = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<double> *K_guess = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);

  ////////////////////////// Allocate memory for local
  void    *cnRes_vv;
  void    *cnRes_gv;
  void    *cnTmp;

  if((cudaHostAlloc(&cnRes_vv, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("Error allocating memory cnRes_vv\n");
  if((cudaHostAlloc(&cnRes_gv, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("Error allocating memory cnRes_gv\n");

  cudaMemset      (cnRes_vv, 0, sizeof(double)*2*16*GK_localVolume);
  cudaMemset      (cnRes_gv, 0, sizeof(double)*2*16*GK_localVolume);

  if((cudaHostAlloc(&cnTmp, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("Error allocating memory cnTmp\n");

  cudaMemset      (cnTmp, 0, sizeof(double)*2*16*GK_localVolume);
  ///////////////////////////////////////////////////
  //////////// Allocate memory for one-Der and conserved current
  void    **cnD_vv;
  void    **cnD_gv;
  void    **cnC_vv;
  void    **cnC_gv;

  cnD_vv   = (void**) malloc(sizeof(double*)*2*4);
  cnD_gv   = (void**) malloc(sizeof(double*)*2*4);
  cnC_vv   = (void**) malloc(sizeof(double*)*2*4);
  cnC_gv   = (void**) malloc(sizeof(double*)*2*4);

  if(cnD_gv == NULL)errorQuda("Error allocating memory cnD_gv higher level\n");
  if(cnD_vv == NULL)errorQuda("Error allocating memory cnD_vv higher level\n");
  if(cnC_gv == NULL)errorQuda("Error allocating memory cnC_gv higher level\n");
  if(cnC_vv == NULL)errorQuda("Error allocating memory cnC_vv higher level\n");
  cudaDeviceSynchronize();

  for(int mu = 0; mu < 4 ; mu++){
    if((cudaHostAlloc(&(cnD_vv[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("Error allocating memory cnD_vv\n");
    if((cudaHostAlloc(&(cnD_gv[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("Error allocating memory cnD_gv\n");
    if((cudaHostAlloc(&(cnC_vv[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("Error allocating memory cnC_vv\n");
    if((cudaHostAlloc(&(cnC_gv[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("Error allocating memory cnC_gv\n");

    cudaMemset(cnD_vv[mu], 0, sizeof(double)*2*16*GK_localVolume);
    cudaMemset(cnD_gv[mu], 0, sizeof(double)*2*16*GK_localVolume);
    cudaMemset(cnC_vv[mu], 0, sizeof(double)*2*16*GK_localVolume);
    cudaMemset(cnC_gv[mu], 0, sizeof(double)*2*16*GK_localVolume);
  }
  cudaDeviceSynchronize();
  //--------------------------------------------------------------------------------------------

  //-Allocate memory for the write buffers
  double *buf_std_uloc,*buf_gen_uloc;
  double **buf_std_oneD;
  double **buf_gen_oneD;
  double **buf_std_csvC;
  double **buf_gen_csvC;

  if( (buf_std_uloc = (double*) malloc(Nprint*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_std_uloc failed.\n");
  if( (buf_gen_uloc = (double*) malloc(Nprint*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_gen_uloc failed.\n");

  if( (buf_std_oneD = (double**) malloc(4*sizeof(double*)))==NULL ) errorQuda("Allocation of buffer buf_std_oneD failed.\n");
  if( (buf_gen_oneD = (double**) malloc(4*sizeof(double*)))==NULL ) errorQuda("Allocation of buffer buf_gen_oneD failed.\n");
  if( (buf_std_csvC = (double**) malloc(4*sizeof(double*)))==NULL ) errorQuda("Allocation of buffer buf_std_csvC failed.\n");
  if( (buf_gen_csvC = (double**) malloc(4*sizeof(double*)))==NULL ) errorQuda("Allocation of buffer buf_gen_csvC failed.\n");
  
  for(int mu = 0; mu < 4 ; mu++){
    if( (buf_std_oneD[mu] = (double*) malloc(Nprint*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_std_oneD[%d] failed.\n",mu);
    if( (buf_gen_oneD[mu] = (double*) malloc(Nprint*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_gen_oneD[%d] failed.\n",mu);
    if( (buf_std_csvC[mu] = (double*) malloc(Nprint*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_std_csvC[%d] failed.\n",mu);
    if( (buf_gen_csvC[mu] = (double*) malloc(Nprint*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_gen_csvC[%d] failed.\n",mu);
  }
  //---------------------------------------------------------------

  gsl_rng *rNum = gsl_rng_alloc(gsl_rng_ranlux);
  gsl_rng_set(rNum, seed + comm_rank()*seed);

  if(info.source_type==RANDOM) printfQuda("Will use RANDOM stochastic sources\n");
  else if (info.source_type==UNITY) printfQuda("Will use UNITY stochastic sources\n");


  //-Allocate the momenta
  int **mom,**momQsq;
  long int SplV = GK_totalL[0]*GK_totalL[1]*GK_totalL[2];
  if((mom =    (int**) malloc(sizeof(int*)*SplV )) == NULL) errorQuda("Error in allocating mom\n");
  if((momQsq = (int**) malloc(sizeof(int*)*Nmoms)) == NULL) errorQuda("Error in allocating momQsq\n");
  for(int ip=0; ip<SplV; ip++)
    if((mom[ip] = (int*) malloc(sizeof(int)*3)) == NULL) errorQuda("Error in allocating mom[%d]\n",ip);

  for(int ip=0; ip<Nmoms; ip++)
    if((momQsq[ip] = (int *) malloc(sizeof(int)*3)) == NULL) errorQuda("Error in allocating momQsq[%d]\n",ip);

  createLoopMomenta(mom,momQsq,info.Q_sq,Nmoms);
  printfQuda("Momenta created\n");
  //----------------------------------------------------

  int iPrint = 0;
  for(int is = 0 ; is < Nstoch ; is++){
    t1 = MPI_Wtime();
    memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
    getStochasticRandomSource<double>(input_vector,rNum,info.source_type);
    K_vector->packVector((double*) input_vector);
    K_vector->loadVector();
    K_vector->uploadToCuda(b,flag_eo);
    dirac.prepare(in,out,*x,*b,param->solution_type);
    // in is reference to the b but for a parity sinlet
    // out is reference to the x but for a parity sinlet
    cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);
    dirac.Mdag(*in, *tmp_up);
    delete tmp_up;
    // now the the source vector b is ready to perform deflation and find the initial guess
    K_vector->downloadFromCuda(in,flag_eo);
    K_vector->download();
    deflation->deflateVector(*K_guess,*K_vector);
    K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
    //  blas::zero(*out); // remove it later , just for test
    (*solve)(*out,*in);
    dirac.reconstruct(*x,*b,param->solution_type);
    t2 = MPI_Wtime();
    printfQuda("TIME_REPORT: Inversion for Stoch %04d is %f sec\n",is+1,t2-t1);

    t1 = MPI_Wtime();
    oneEndTrick_w_One_Der<double>(*x,*tmp3,*tmp4,param,cnRes_gv,cnRes_vv,cnD_gv,cnD_vv,cnC_gv,cnC_vv);
    t2 = MPI_Wtime();
    printfQuda("TIME_REPORT: One-end trick for Stoch %04d is %f sec\n",is+1,t2-t1);

    if( (is+1)%NdumpStep == 0){
      if(GK_nProc[2]==1){      
	doCudaFFT_v2<double>(cnRes_vv,cnTmp); // Scalar
	copyLoopToWriteBuf(buf_std_uloc,cnTmp,iPrint,info.Q_sq,Nmoms,mom);
	doCudaFFT_v2<double>(cnRes_gv,cnTmp); // dOp
	copyLoopToWriteBuf(buf_gen_uloc,cnTmp,iPrint,info.Q_sq,Nmoms,mom);
	
	for(int mu = 0 ; mu < 4 ; mu++){
	  doCudaFFT_v2<double>(cnD_vv[mu],cnTmp); // Loops
	  copyLoopToWriteBuf(buf_std_oneD[mu],cnTmp,iPrint,info.Q_sq,Nmoms,mom);
	  doCudaFFT_v2<double>(cnC_vv[mu],cnTmp); // LoopsCv
	  copyLoopToWriteBuf(buf_std_csvC[mu],cnTmp,iPrint,info.Q_sq,Nmoms,mom);
	  
	  doCudaFFT_v2<double>(cnD_gv[mu],cnTmp); // LpsDw
	  copyLoopToWriteBuf(buf_gen_oneD[mu],cnTmp,iPrint,info.Q_sq,Nmoms,mom);
	  doCudaFFT_v2<double>(cnC_gv[mu],cnTmp); // LpsDwCv
	  copyLoopToWriteBuf(buf_gen_csvC[mu],cnTmp,iPrint,info.Q_sq,Nmoms,mom);
	}
      }
      else if(GK_nProc[2]>1){
	t1 = MPI_Wtime();
	performFFT<double>(buf_std_uloc, cnRes_vv, iPrint, Nmoms, momQsq);
	performFFT<double>(buf_gen_uloc, cnRes_gv, iPrint, Nmoms, momQsq);

	for(int mu=0;mu<4;mu++){
	  performFFT<double>(buf_std_oneD[mu], cnD_vv[mu], iPrint, Nmoms, momQsq);
	  performFFT<double>(buf_std_csvC[mu], cnC_vv[mu], iPrint, Nmoms, momQsq);
	  performFFT<double>(buf_gen_oneD[mu], cnD_gv[mu], iPrint, Nmoms, momQsq);
	  performFFT<double>(buf_gen_csvC[mu], cnC_gv[mu], iPrint, Nmoms, momQsq);
	}
	t2 = MPI_Wtime();
	printfQuda("TIME_REPORT: FFT and copying to Write Buffers is %f sec\n",t2-t1);
      }

      printfQuda("Loops for Nstoch = %d copied to write buffers\n",is+1);
      iPrint++;
    }//-if (is+1)
  }//-loop over stochastic noise vectors

  bool stoch_part = false;

  printfQuda("Will write the loops in %s format\n", (LoopFileFormat == ASCII_FORM) ? "ASCII" : "HDF5");
  t1 = MPI_Wtime();
  if(LoopFileFormat==ASCII_FORM){ // Write the loops in ASCII format
    writeLoops_ASCII(buf_std_uloc, filename_out, loopInfo, momQsq, 0, 0, stoch_part, false, false); // Scalar
    writeLoops_ASCII(buf_gen_uloc, filename_out, loopInfo, momQsq, 1, 0, stoch_part, false, false); // dOp
    for(int mu = 0 ; mu < 4 ; mu++){
      writeLoops_ASCII(buf_std_oneD[mu], filename_out, loopInfo, momQsq, 2, mu, stoch_part, false, false); // Loops
      writeLoops_ASCII(buf_std_csvC[mu], filename_out, loopInfo, momQsq, 3, mu, stoch_part, false, false); // LoopsCv
      writeLoops_ASCII(buf_gen_oneD[mu], filename_out, loopInfo, momQsq, 4, mu, stoch_part, false, false); // LpsDw
      writeLoops_ASCII(buf_gen_csvC[mu], filename_out, loopInfo, momQsq, 5, mu, stoch_part, false, false); // LpsDwCv
    }
  }
  else if(LoopFileFormat==HDF5_FORM){ // Write the loops in HDF5 format
    writeLoops_HDF5(buf_std_uloc, buf_gen_uloc, buf_std_oneD, buf_std_csvC, buf_gen_oneD, buf_gen_csvC, filename_out, loopInfo, momQsq, stoch_part, false, false);
  }
  t2 = MPI_Wtime();
  printfQuda("Writing the loops completed in %f sec.\n",t2-t1);


  //- Free loop cuda buffers
  cudaFreeHost(cnRes_gv);
  cudaFreeHost(cnRes_vv);
  cudaFreeHost(cnTmp);
  for(int mu = 0 ; mu < 4 ; mu++){
    cudaFreeHost(cnD_vv[mu]);
    cudaFreeHost(cnD_gv[mu]);
    cudaFreeHost(cnC_vv[mu]);
    cudaFreeHost(cnC_gv[mu]);
  }  
  free(cnD_vv);
  free(cnD_gv);
  free(cnC_vv);
  free(cnC_gv);
  //---------------------------

  //-Free loop write buffers
  free(buf_std_uloc);
  free(buf_gen_uloc);
  for(int mu = 0 ; mu < 4 ; mu++){
    free(buf_std_oneD[mu]);
    free(buf_std_csvC[mu]);
    free(buf_gen_oneD[mu]);
    free(buf_gen_csvC[mu]);
  }
  free(buf_std_oneD);
  free(buf_std_csvC);
  free(buf_gen_oneD);
  free(buf_gen_csvC);
  //---------------------------

  //-Free the momentum matrices
  for(int ip=0; ip<SplV; ip++) free(mom[ip]);
  free(mom);
  for(int ip=0;ip<Nmoms;ip++) free(momQsq[ip]);
  free(momQsq);
  //---------------------------

  free(input_vector);
  free(output_vector);
  gsl_rng_free(rNum);
  delete solve;
  delete d;
  delete dSloppy;
  delete dPre;
  delete K_guess;
  delete K_vector;
  delete K_gauge;
  delete deflation;
  delete h_x;
  delete h_b;
  delete x;
  delete b;
  delete tmp3;
  delete tmp4;
  popVerbosity();
  saveTuneCache();
  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);
}


void calcEigenVectors_loop_wOneD_FullOp(void **gaugeToPlaquette, QudaInvertParam *param ,QudaGaugeParam *gauge_param,
					qudaQKXTM_arpackInfo arpackInfo, qudaQKXTM_arpackInfo arpackInfoEO, qudaQKXTM_loopInfo loopInfo, qudaQKXTMinfo_Kepler info){

  double t1,t2,t3,t4;

  if (!initialized) errorQuda("calcEigenVectors_loop_wOneD_FullOp: QUDA not initialized");
  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);
  
  printfQuda("\n### calcEigenVectors_loop_wOneD_FullOp: Loop calculation begins now\n\n");

  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);
  if( (param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) && (param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) ) errorQuda("Only asymmetric operators are supported in deflation\n");
  if(param->inv_type != QUDA_CG_INVERTER) errorQuda("calcEigenVectors_loop_wOneD_FullOp: This function works only with CG method");  
  if(param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS) errorQuda("calcEigenVectors_loop_wOneD_FullOp: This function works only with ukqcd gamma basis\n");
  if(param->dirac_order != QUDA_DIRAC_ORDER) errorQuda("calcEigenVectors_loop_wOneD_FullOp: This function works only with color-inside-spin\n");
  if( arpackInfo.isEven    && (param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) ) errorQuda("calcEigenVectors_loop_wOneD_FullOp: Inconsistency between operator types!");
  if( (!arpackInfo.isEven) && (param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) )   errorQuda("calcEigenVectors_loop_wOneD_FullOp: Inconsistency between operator types!");
  
  bool stochEO = loopInfo.fullOp_stochEO;
  bool pc_solution = false; // Construct the full solution spinor
  bool pc_solve;
  bool flag_eo;
  if(stochEO){
    pc_solve = true;
    if(arpackInfo.isEven){
      flag_eo = true;
      printfQuda("calcEigenVectors_loop_wOneD_FullOp: Will solve the stochastic part using the Even-Odd, Even-Even Asymmetric operator\n");
    }
    else{
      flag_eo = false;
      printfQuda("calcEigenVectors_loop_wOneD_FullOp: Will solve the stochastic part using the Even-Odd, Odd-Odd Asymmetric operator\n");
    }
  }
  else{
    flag_eo = false;
    pc_solve = false;
    printfQuda("calcEigenVectors_loop_wOneD_FullOp: Will solve the stochastic part using the Full operator\n");
  }
  //------------------------------------------------------------------------------------------------------------------

  int Nstoch = loopInfo.Nstoch;
  unsigned long int seed = loopInfo.seed;
  int Ndump = loopInfo.Ndump;
  int Nprint = loopInfo.Nprint;
  loopInfo.Nmoms = GK_Nmoms;
  int Nmoms = GK_Nmoms;
  char filename_out[512];
  int smethod = loopInfo.smethod;

  FILE_WRITE_FORMAT LoopFileFormat = loopInfo.FileFormat;

  char loop_exact_fname[512];
  char loop_stoch_fname[512];

  //-C.K. Truncated solver method params
  bool useTSM = loopInfo.useTSM;
  int TSM_NHP = loopInfo.TSM_NHP;
  int TSM_NLP = loopInfo.TSM_NLP;
  int TSM_NdumpHP = loopInfo.TSM_NdumpHP;
  int TSM_NdumpLP = loopInfo.TSM_NdumpLP;
  int TSM_NprintHP = loopInfo.TSM_NprintHP;
  int TSM_NprintLP = loopInfo.TSM_NprintLP;
  long int TSM_maxiter = 0;
  double TSM_tol = 0.0;
  if( (loopInfo.TSM_tol == 0) && (loopInfo.TSM_maxiter !=0 ) ) TSM_maxiter = loopInfo.TSM_maxiter; // LP criterion fixed by iteration number
  else if( (loopInfo.TSM_tol != 0) && (loopInfo.TSM_maxiter == 0) ) TSM_tol = loopInfo.TSM_tol; // LP criterion fixed by tolerance
  else if( useTSM && (loopInfo.TSM_tol != 0) && (loopInfo.TSM_maxiter != 0) ){
    warningQuda("Both max-iter = %ld and tolerance = %lf defined as criterions for the TSM. Proceeding with max-iter = %ld criterion.\n",loopInfo.TSM_maxiter,loopInfo.TSM_tol,loopInfo.TSM_maxiter);
    TSM_maxiter = loopInfo.TSM_maxiter; // LP criterion fixed by iteration number
  }
  //---------------------------------------------------------------------------

  loopInfo.loop_type[0] = "Scalar";      loopInfo.loop_oneD[0] = false;   // std-ultra_local
  loopInfo.loop_type[1] = "dOp";         loopInfo.loop_oneD[1] = false;   // gen-ultra_local
  loopInfo.loop_type[2] = "Loops";       loopInfo.loop_oneD[2] = true;    // std-one_derivative
  loopInfo.loop_type[3] = "LoopsCv";     loopInfo.loop_oneD[3] = true;    // std-conserved current
  loopInfo.loop_type[4] = "LpsDw";       loopInfo.loop_oneD[4] = true;    // gen-one_derivative
  loopInfo.loop_type[5] = "LpsDwCv";     loopInfo.loop_oneD[5] = true;    // gen-conserved current

  printfQuda("\nLoop Calculation Info\n");
  printfQuda("=====================\n");
  if(useTSM){
    printfQuda(" Will perform the Truncated Solver method using the following parameters:\n");
    printfQuda("  -N_HP = %d\n",TSM_NHP);
    printfQuda("  -N_LP = %d\n",TSM_NLP);
    if (TSM_maxiter == 0) printfQuda("  -CG stopping criterion is: tol = %e\n",TSM_tol);
    else printfQuda("  -CG stopping criterion is: max-iter = %ld\n",TSM_maxiter);
    printfQuda("  -Will dump every %d high-precision noise vectors, thus %d times\n",TSM_NdumpHP,TSM_NprintHP);
    printfQuda("  -Will dump every %d low-precision noise vectors , thus %d times\n",TSM_NdumpLP,TSM_NprintLP);
  }
  else{
    printfQuda(" Will not perform the Truncated Solver method\n");
    printfQuda(" No. of noise vectors: %d\n",Nstoch);
    printfQuda(" Will dump every %d noise vectors, thus %d times\n",Ndump,Nprint);
  }
  printfQuda(" The seed is: %ld\n",seed);
  printfQuda(" The conf trajectory is: %04d\n",loopInfo.traj);
  printfQuda(" Will produce the loop for %d Momentum Combinations\n",loopInfo.Nmoms);
  printfQuda(" The loop file format is %s\n", (LoopFileFormat == ASCII_FORM) ? "ASCII" : "HDF5");
  printfQuda(" The loop base name is %s\n",loopInfo.loop_fname);
  printfQuda(" Will perform the loop for the following %d numbers of eigenvalues:",loopInfo.nSteps_defl);
  for(int s=0;s<loopInfo.nSteps_defl;s++){
    printfQuda("  %d",loopInfo.deflStep[s]);
  }
  if(smethod==1) printfQuda("\n Stochastic part according to: MdagM psi = (1-P)Mdag xi (smethod=1)\n");
  else printfQuda("\n Stochastic part according to: MdagM phi = Mdag xi (smethod=0)\n");
  if(info.source_type==RANDOM) printfQuda(" Will use RANDOM stochastic sources\n");
  else if (info.source_type==UNITY) printfQuda(" Will use UNITY stochastic sources\n");
  printfQuda("=====================\n\n");
  
  bool exact_part = true;
  bool stoch_part = false;

  bool LowPrecSum = true;
  bool HighPrecSum = false;
  //----------------------------------------------------------------------------------------------------------------

  //- Allocate memory for local loops
  void *std_uloc;
  void *gen_uloc;
  void *tmp_loop;
  
  if((cudaHostAlloc(&std_uloc, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("calcEigenVectors_loop_wOneD_FullOp: Error allocating memory std_uloc\n");
  if((cudaHostAlloc(&gen_uloc, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("calcEigenVectors_loop_wOneD_FullOp: Error allocating memory gen_uloc\n");
  if((cudaHostAlloc(&tmp_loop, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("calcEigenVectors_loop_wOneD_FullOp: Error allocating memory tmp_loop\n");
  
  cudaMemset(std_uloc, 0, sizeof(double)*2*16*GK_localVolume);
  cudaMemset(gen_uloc, 0, sizeof(double)*2*16*GK_localVolume);
  cudaMemset(tmp_loop, 0, sizeof(double)*2*16*GK_localVolume);
  cudaDeviceSynchronize();

  //- Allocate memory for one-Derivative and conserved current loops
  void **std_oneD;
  void **gen_oneD;
  void **std_csvC;
  void **gen_csvC;
  
  std_oneD = (void**) malloc(4*sizeof(double*));
  gen_oneD = (void**) malloc(4*sizeof(double*));
  std_csvC = (void**) malloc(4*sizeof(double*));
  gen_csvC = (void**) malloc(4*sizeof(double*));

  if(gen_oneD == NULL) errorQuda("calcEigenVectors_loop_wOneD_FullOp: Error allocating memory gen_oneD higher level\n");
  if(std_oneD == NULL) errorQuda("calcEigenVectors_loop_wOneD_FullOp: Error allocating memory std_oneD higher level\n");
  if(gen_csvC == NULL) errorQuda("calcEigenVectors_loop_wOneD_FullOp: Error allocating memory gen_csvC higher level\n");
  if(std_csvC == NULL) errorQuda("calcEigenVectors_loop_wOneD_FullOp: Error allocating memory std_csvC higher level\n");
  cudaDeviceSynchronize();
  
  for(int mu = 0; mu < 4 ; mu++){
    if((cudaHostAlloc(&(std_oneD[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("calcEigenVectors_loop_wOneD_FullOp: Error allocating memory std_oneD\n");
    if((cudaHostAlloc(&(gen_oneD[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("calcEigenVectors_loop_wOneD_FullOp: Error allocating memory gen_oneD\n");
    if((cudaHostAlloc(&(std_csvC[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("calcEigenVectors_loop_wOneD_FullOp: Error allocating memory std_csvC\n");
    if((cudaHostAlloc(&(gen_csvC[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("calcEigenVectors_loop_wOneD_FullOp: Error allocating memory gen_csvC\n");
    
    cudaMemset(std_oneD[mu], 0, sizeof(double)*2*16*GK_localVolume);
    cudaMemset(gen_oneD[mu], 0, sizeof(double)*2*16*GK_localVolume);
    cudaMemset(std_csvC[mu], 0, sizeof(double)*2*16*GK_localVolume);
    cudaMemset(gen_csvC[mu], 0, sizeof(double)*2*16*GK_localVolume);
  }
  cudaDeviceSynchronize();
  //--------------------------------------

  //-Allocate memory for the write buffers
  double *buf_std_uloc,*buf_gen_uloc;
  double **buf_std_oneD;
  double **buf_gen_oneD;
  double **buf_std_csvC;
  double **buf_gen_csvC;

  int Nprt = ( useTSM ? TSM_NprintLP : Nprint );

  if( (buf_std_uloc = (double*) malloc(Nprt*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_std_uloc failed.\n");
  if( (buf_gen_uloc = (double*) malloc(Nprt*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_gen_uloc failed.\n");

  if( (buf_std_oneD = (double**) malloc(4*sizeof(double*)))==NULL ) errorQuda("Allocation of buffer buf_std_oneD failed.\n");
  if( (buf_gen_oneD = (double**) malloc(4*sizeof(double*)))==NULL ) errorQuda("Allocation of buffer buf_gen_oneD failed.\n");
  if( (buf_std_csvC = (double**) malloc(4*sizeof(double*)))==NULL ) errorQuda("Allocation of buffer buf_std_csvC failed.\n");
  if( (buf_gen_csvC = (double**) malloc(4*sizeof(double*)))==NULL ) errorQuda("Allocation of buffer buf_gen_csvC failed.\n");

  for(int mu = 0; mu < 4 ; mu++){
    if( (buf_std_oneD[mu] = (double*) malloc(Nprt*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_std_oneD[%d] failed.\n",mu);
    if( (buf_gen_oneD[mu] = (double*) malloc(Nprt*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_gen_oneD[%d] failed.\n",mu);
    if( (buf_std_csvC[mu] = (double*) malloc(Nprt*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_std_csvC[%d] failed.\n",mu);
    if( (buf_gen_csvC[mu] = (double*) malloc(Nprt*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_gen_csvC[%d] failed.\n",mu);
  }
  //------------------------------------------------------------------------------------------------

  //- Allocate extra memory if using TSM
  void *std_uloc_LP, *gen_uloc_LP;
  void **std_oneD_LP, **gen_oneD_LP, **std_csvC_LP, **gen_csvC_LP;

  double *buf_std_uloc_LP,*buf_gen_uloc_LP;
  double *buf_std_uloc_HP,*buf_gen_uloc_HP;
  double **buf_std_oneD_LP, **buf_gen_oneD_LP, **buf_std_csvC_LP, **buf_gen_csvC_LP;
  double **buf_std_oneD_HP, **buf_gen_oneD_HP, **buf_std_csvC_HP, **buf_gen_csvC_HP;
  
  if(useTSM){
    //- local 
    if((cudaHostAlloc(&std_uloc_LP, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("calcEigenVectors_loop_wOneD_FullOp: Error allocating memory std_uloc_LP\n");
    if((cudaHostAlloc(&gen_uloc_LP, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("calcEigenVectors_loop_wOneD_FullOp: Error allocating memory gen_uloc_LP\n");
  
    cudaMemset(std_uloc_LP, 0, sizeof(double)*2*16*GK_localVolume);
    cudaMemset(gen_uloc_LP, 0, sizeof(double)*2*16*GK_localVolume);
    cudaDeviceSynchronize();

    //- one-Derivative and conserved current loops
    std_oneD_LP = (void**) malloc(4*sizeof(double*));
    gen_oneD_LP = (void**) malloc(4*sizeof(double*));
    std_csvC_LP = (void**) malloc(4*sizeof(double*));
    gen_csvC_LP = (void**) malloc(4*sizeof(double*));

    if(gen_oneD_LP == NULL) errorQuda("calcEigenVectors_loop_wOneD_FullOp: Error allocating memory gen_oneD_LP higher level\n");
    if(std_oneD_LP == NULL) errorQuda("calcEigenVectors_loop_wOneD_FullOp: Error allocating memory std_oneD_LP higher level\n");
    if(gen_csvC_LP == NULL) errorQuda("calcEigenVectors_loop_wOneD_FullOp: Error allocating memory gen_csvC_LP higher level\n");
    if(std_csvC_LP == NULL) errorQuda("calcEigenVectors_loop_wOneD_FullOp: Error allocating memory std_csvC_LP higher level\n");
    cudaDeviceSynchronize();
  
    for(int mu = 0; mu < 4 ; mu++){
      if((cudaHostAlloc(&(std_oneD_LP[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
	errorQuda("calcEigenVectors_loop_wOneD_FullOp: Error allocating memory std_oneD_LP\n");
      if((cudaHostAlloc(&(gen_oneD_LP[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
	errorQuda("calcEigenVectors_loop_wOneD_FullOp: Error allocating memory gen_oneD_LP\n");
      if((cudaHostAlloc(&(std_csvC_LP[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
	errorQuda("calcEigenVectors_loop_wOneD_FullOp: Error allocating memory std_csvC_LP\n");
      if((cudaHostAlloc(&(gen_csvC_LP[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
	errorQuda("calcEigenVectors_loop_wOneD_FullOp: Error allocating memory gen_csvC_LP\n");    
    
      cudaMemset(std_oneD_LP[mu], 0, sizeof(double)*2*16*GK_localVolume);
      cudaMemset(gen_oneD_LP[mu], 0, sizeof(double)*2*16*GK_localVolume);
      cudaMemset(std_csvC_LP[mu], 0, sizeof(double)*2*16*GK_localVolume);
      cudaMemset(gen_csvC_LP[mu], 0, sizeof(double)*2*16*GK_localVolume);
    }
    cudaDeviceSynchronize();
    //------------------------------

    //-write buffers for Low-precision loops
    if( (buf_std_uloc_LP = (double*) malloc(TSM_NprintHP*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_std_uloc_LP failed.\n");
    if( (buf_gen_uloc_LP = (double*) malloc(TSM_NprintHP*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_gen_uloc_LP failed.\n");
    
    if( (buf_std_oneD_LP = (double**) malloc(4*sizeof(double*)))==NULL ) errorQuda("Allocation of buffer buf_std_oneD_LP failed.\n");
    if( (buf_gen_oneD_LP = (double**) malloc(4*sizeof(double*)))==NULL ) errorQuda("Allocation of buffer buf_gen_oneD_LP failed.\n");
    if( (buf_std_csvC_LP = (double**) malloc(4*sizeof(double*)))==NULL ) errorQuda("Allocation of buffer buf_std_csvC_LP failed.\n");
    if( (buf_gen_csvC_LP = (double**) malloc(4*sizeof(double*)))==NULL ) errorQuda("Allocation of buffer buf_gen_csvC_LP failed.\n");
    
    for(int mu = 0; mu < 4 ; mu++){
      if( (buf_std_oneD_LP[mu] = (double*) malloc(TSM_NprintHP*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_std_oneD_LP[%d] failed.\n",mu);
      if( (buf_gen_oneD_LP[mu] = (double*) malloc(TSM_NprintHP*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_gen_oneD_LP[%d] failed.\n",mu);
      if( (buf_std_csvC_LP[mu] = (double*) malloc(TSM_NprintHP*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_std_csvC_LP[%d] failed.\n",mu);
      if( (buf_gen_csvC_LP[mu] = (double*) malloc(TSM_NprintHP*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_gen_csvC_LP[%d] failed.\n",mu);
    }

    //-write buffers for High-precision loops
    if( (buf_std_uloc_HP = (double*) malloc(TSM_NprintHP*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_std_uloc_HP failed.\n");
    if( (buf_gen_uloc_HP = (double*) malloc(TSM_NprintHP*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_gen_uloc_HP failed.\n");
    
    if( (buf_std_oneD_HP = (double**) malloc(4*sizeof(double*)))==NULL ) errorQuda("Allocation of buffer buf_std_oneD_HP failed.\n");
    if( (buf_gen_oneD_HP = (double**) malloc(4*sizeof(double*)))==NULL ) errorQuda("Allocation of buffer buf_gen_oneD_HP failed.\n");
    if( (buf_std_csvC_HP = (double**) malloc(4*sizeof(double*)))==NULL ) errorQuda("Allocation of buffer buf_std_csvC_HP failed.\n");
    if( (buf_gen_csvC_HP = (double**) malloc(4*sizeof(double*)))==NULL ) errorQuda("Allocation of buffer buf_gen_csvC_HP failed.\n");
    
    for(int mu = 0; mu < 4 ; mu++){
      if( (buf_std_oneD_HP[mu] = (double*) malloc(TSM_NprintHP*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_std_oneD_HP[%d] failed.\n",mu);
      if( (buf_gen_oneD_HP[mu] = (double*) malloc(TSM_NprintHP*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_gen_oneD_HP[%d] failed.\n",mu);
      if( (buf_std_csvC_HP[mu] = (double*) malloc(TSM_NprintHP*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_std_csvC_HP[%d] failed.\n",mu);
      if( (buf_gen_csvC_HP[mu] = (double*) malloc(TSM_NprintHP*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_gen_csvC_HP[%d] failed.\n",mu);
    }

  }//-if useTSM
  //------------------------------------------------------------------------------------------------

  //-Allocate the momenta
  int **mom,**momQsq;
  long int SplV = GK_totalL[0]*GK_totalL[1]*GK_totalL[2];
  if((mom =    (int**) malloc(sizeof(int*)*SplV )) == NULL) errorQuda("Error in allocating mom\n");
  if((momQsq = (int**) malloc(sizeof(int*)*Nmoms)) == NULL) errorQuda("Error in allocating momQsq\n");
  for(int ip=0; ip<SplV; ip++)
    if((mom[ip] = (int*) malloc(sizeof(int)*3)) == NULL) errorQuda("Error in allocating mom[%d]\n",ip);

  for(int ip=0; ip<Nmoms; ip++)
    if((momQsq[ip] = (int *) malloc(sizeof(int)*3)) == NULL) errorQuda("Error in allocating momQsq[%d]\n",ip);

  createLoopMomenta(mom,momQsq,info.Q_sq,Nmoms);
  printfQuda("Momenta created\n");
  //----------------------------------------------------


  //=========================================================================================//
  //=================================  E X A C T   P A R T  =================================//
  //=========================================================================================//

  printfQuda("\n ### Exact part calculation ###\n");

  int NeV_Full = arpackInfo.nEv;  
  
  QKXTM_Deflation_Kepler<double> *deflation = new QKXTM_Deflation_Kepler<double>(param,arpackInfo);
  deflation->printInfo();
  
  //- Calculate the eigenVectors
  t1 = MPI_Wtime(); 
  deflation->eigenSolver();
  t2 = MPI_Wtime();
  printfQuda("calcEigenVectors_loop_wOneD_FullOp TIME REPORT: Full Operator EigenVector Calculation: %f sec\n",t2-t1);

  deflation->MapEvenOddToFull();

  //- Calculate the exact part of the loop
  int iPrint = 0;
  int s = 0;
  for(int n=0;n<NeV_Full;n++){
    t1 = MPI_Wtime();
    deflation->Loop_w_One_Der_FullOp_Exact(n, param, gen_uloc, std_uloc, gen_oneD, std_oneD, gen_csvC, std_csvC);
    t2 = MPI_Wtime();
    printfQuda("TIME_REPORT: Exact part for eigenvector %d done in: %f sec\n",n+1,t2-t1);
    
    if( (n+1)==loopInfo.deflStep[s] ){     
      if(GK_nProc[2]==1){      
	doCudaFFT_v2<double>(std_uloc,tmp_loop); // Scalar
	copyLoopToWriteBuf(buf_std_uloc,tmp_loop,iPrint,info.Q_sq,Nmoms,mom);
	doCudaFFT_v2<double>(gen_uloc,tmp_loop); // dOp
	copyLoopToWriteBuf(buf_gen_uloc,tmp_loop,iPrint,info.Q_sq,Nmoms,mom);
	
	for(int mu = 0 ; mu < 4 ; mu++){
	  doCudaFFT_v2<double>(std_oneD[mu],tmp_loop); // Loops
	  copyLoopToWriteBuf(buf_std_oneD[mu],tmp_loop,iPrint,info.Q_sq,Nmoms,mom);
	  doCudaFFT_v2<double>(std_csvC[mu],tmp_loop); // LoopsCv
	  copyLoopToWriteBuf(buf_std_csvC[mu],tmp_loop,iPrint,info.Q_sq,Nmoms,mom);
	  
	  doCudaFFT_v2<double>(gen_oneD[mu],tmp_loop); // LpsDw
	  copyLoopToWriteBuf(buf_gen_oneD[mu],tmp_loop,iPrint,info.Q_sq,Nmoms,mom);
	  doCudaFFT_v2<double>(gen_csvC[mu],tmp_loop); // LpsDwCv
	  copyLoopToWriteBuf(buf_gen_csvC[mu],tmp_loop,iPrint,info.Q_sq,Nmoms,mom);
	}
	printfQuda("Exact part of Loops for NeV = %d copied to write buffers\n",n+1);
      }
      else if(GK_nProc[2]>1){
	t1 = MPI_Wtime();
	performFFT<double>(buf_std_uloc, std_uloc, iPrint, Nmoms, momQsq);
	performFFT<double>(buf_gen_uloc, gen_uloc, iPrint, Nmoms, momQsq);
	
	for(int mu=0;mu<4;mu++){
	  performFFT<double>(buf_std_oneD[mu], std_oneD[mu], iPrint, Nmoms, momQsq);
	  performFFT<double>(buf_std_csvC[mu], std_csvC[mu], iPrint, Nmoms, momQsq);
	  performFFT<double>(buf_gen_oneD[mu], gen_oneD[mu], iPrint, Nmoms, momQsq);
	  performFFT<double>(buf_gen_csvC[mu], gen_csvC[mu], iPrint, Nmoms, momQsq);
	}
	t2 = MPI_Wtime();
	printfQuda("TIME_REPORT: FFT and copying to Write Buffers is %f sec\n",t2-t1);
      }

      //-Write the exact part of the loop
      sprintf(loop_exact_fname,"%s_exact_NeV%d",loopInfo.loop_fname,n+1);
      if(LoopFileFormat==ASCII_FORM){ // Write the loops in ASCII format
	writeLoops_ASCII(buf_std_uloc, loop_exact_fname, loopInfo, momQsq, 0, 0, exact_part, false, false); // Scalar
	writeLoops_ASCII(buf_gen_uloc, loop_exact_fname, loopInfo, momQsq, 1, 0, exact_part, false ,false); // dOp
	for(int mu = 0 ; mu < 4 ; mu++){
	  writeLoops_ASCII(buf_std_oneD[mu], loop_exact_fname, loopInfo, momQsq, 2, mu, exact_part, false, false); // Loops
	  writeLoops_ASCII(buf_std_csvC[mu], loop_exact_fname, loopInfo, momQsq, 3, mu, exact_part, false, false); // LoopsCv
	  writeLoops_ASCII(buf_gen_oneD[mu], loop_exact_fname, loopInfo, momQsq, 4, mu, exact_part, false, false); // LpsDw
	  writeLoops_ASCII(buf_gen_csvC[mu], loop_exact_fname, loopInfo, momQsq, 5, mu, exact_part, false, false); // LpsDwCv
	}
      }
      else if(LoopFileFormat==HDF5_FORM){ // Write the loops in HDF5 format
	writeLoops_HDF5(buf_std_uloc, buf_gen_uloc, buf_std_oneD, buf_std_csvC, buf_gen_oneD, buf_gen_csvC, loop_exact_fname, loopInfo, momQsq, exact_part, false, false);
      }

      printfQuda("Writing the Exact part of the loops for NeV = %d completed.\n",n+1);
      s++;
    }//-if
  }//-for NeV_Full

  printfQuda("\n ### Exact part calculation Done ###\n");

  //=========================================================================================//
  //============================  S T O C H A S T I C   P A R T  ============================//
  //=========================================================================================//

  printfQuda("\n ### Stochastic part calculation ###\n\n");

  QKXTM_Deflation_Kepler<double> *deflationEO;
  if(stochEO){    
    deflationEO = new QKXTM_Deflation_Kepler<double>(param, arpackInfoEO);
    deflationEO->printInfo();

    //- Calculate the eigenVectors of the Even-Odd operator
    t1 = MPI_Wtime(); 
    deflationEO->eigenSolver();
    t2 = MPI_Wtime();
    printfQuda("calcEigenVectors_loop_wOneD_FullOp TIME REPORT: Even-Odd Operator EigenVector Calculation: %f sec\n",t2-t1);
  }

  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);

  QKXTM_Gauge_Kepler<double> *K_gauge = new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  K_gauge->packGauge(gaugeToPlaquette);
  K_gauge->loadGauge();
  K_gauge->calculatePlaq();

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;

  // Create the dirac operator                                                                                                                                      
  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;
  profileInvert.TPSTART(QUDA_PROFILE_H2D);
  //----------------------------------------------

  // Create the solvers
  DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
  SolverParam solverParam(*param);
  Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);

  // Create the LP solver for TSM                                                                                                                                      
  Solver *solve_LP;
  if(useTSM){
    double orig_tol = param->tol;
    long int orig_maxiter = param->maxiter;
    if(TSM_maxiter==0) param->tol = TSM_tol;            // Set the
    else if(TSM_tol==0) param->maxiter = TSM_maxiter;   // low-precision criterion

    SolverParam solverParam_LP(*param);                                                  //
    solve_LP = Solver::create(solverParam_LP, m, mSloppy, mPre, profileInvert); // Create the low-precision solver

    if(TSM_maxiter==0) param->tol = orig_tol;            // Set the
    else if(TSM_tol==0) param->maxiter = orig_maxiter;   // original, high-precision values
  }
  //---------------------------------------------------------------------------------------------------

  void *input_vector  = malloc(GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]*spinorSiteSize*sizeof(double));
  memset(input_vector ,0,GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]*spinorSiteSize*sizeof(double));

  ColorSpinorField *b      = NULL;
  ColorSpinorField *x      = NULL;
  ColorSpinorField *in     = NULL;
  ColorSpinorField *out    = NULL;
  ColorSpinorField *tmp3   = NULL;
  ColorSpinorField *tmp4   = NULL;
  ColorSpinorField *x_LP   = NULL;
  ColorSpinorField *out_LP = NULL;

  ColorSpinorParam cpuParam(input_vector,*param,GK_localL,pc_solution);
  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  b    = new cudaColorSpinorField(cudaParam);
  x    = new cudaColorSpinorField(cudaParam);
  tmp3 = new cudaColorSpinorField(cudaParam);
  tmp4 = new cudaColorSpinorField(cudaParam);

  if(useTSM) x_LP = new cudaColorSpinorField(cudaParam);

  profileInvert.TPSTOP(QUDA_PROFILE_H2D);
  setTuning(param->tune);
  //---------------------------------------------------------------------------------------------------

  QKXTM_Vector_Kepler<double> *K_vector = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<double> *K_vecdef = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);

  for(int dstep=0;dstep<loopInfo.nSteps_defl;dstep++){
    int NeV_defl = loopInfo.deflStep[dstep];
    printfQuda("\n# Performing the stochastic inversions for NeV = %d\n\n",NeV_defl);
    
    //- Prepare the loops for the stochastic part
    cudaMemset(std_uloc, 0, sizeof(double)*2*16*GK_localVolume);
    cudaMemset(gen_uloc, 0, sizeof(double)*2*16*GK_localVolume);
    cudaMemset(tmp_loop, 0, sizeof(double)*2*16*GK_localVolume);
    
    for(int mu = 0; mu < 4 ; mu++){
      cudaMemset(std_oneD[mu], 0, sizeof(double)*2*16*GK_localVolume);
      cudaMemset(gen_oneD[mu], 0, sizeof(double)*2*16*GK_localVolume);
      cudaMemset(std_csvC[mu], 0, sizeof(double)*2*16*GK_localVolume);
      cudaMemset(gen_csvC[mu], 0, sizeof(double)*2*16*GK_localVolume);
    }
    cudaDeviceSynchronize();
    //----------------

    gsl_rng *rNum = gsl_rng_alloc(gsl_rng_ranlux);
    gsl_rng_set(rNum, seed + comm_rank()*seed);

    int Nrun;
    int Nd;
    char *msg_str;
    if(useTSM){
      Nrun = TSM_NLP;
      Nd = TSM_NdumpLP;
      asprintf(&msg_str,"NLP");
    }
    else{
      Nrun = Nstoch;
      Nd = Ndump;
      asprintf(&msg_str,"Stoch.");
    }
   
    int iPrint = 0;
    for(int is = 0 ; is < Nrun ; is++){
      t3 = MPI_Wtime();
      t1 = MPI_Wtime();
      memset(input_vector,0,GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]*spinorSiteSize*sizeof(double));
      getStochasticRandomSource<double>(input_vector,rNum,info.source_type);
      t2 = MPI_Wtime();
      printfQuda("TIME_REPORT: %s %04d - Source creation: %f sec\n",msg_str,is+1,t2-t1);
      K_vector->packVector((double*) input_vector);
      K_vector->loadVector();
      K_vector->uploadToCuda(b,flag_eo);
      dirac.prepare(in,out,*x,*b,param->solution_type); // in -> b, out -> x, for parity singlets
      cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);
      dirac.Mdag(*in,*tmp_up); // in <- M^dag * b
      delete tmp_up;
      K_vector->downloadFromCuda(in,flag_eo);
      K_vector->download();
      
      t1 = MPI_Wtime();
      if(smethod==1){  // Deflate the source vector (Christos)
	if(stochEO) errorQuda("Stochastic part with Even-Odd operator not applicable with smethod = 1.\n");

	deflation->projectVector(*K_vecdef,*K_vector,is+1,NeV_defl);  
	t2 = MPI_Wtime();
	printfQuda("TIME_REPORT: %s %04d - Source projection: %f sec\n",msg_str,is+1,t2-t1);
	K_vecdef->uploadToCuda(in,flag_eo);   // Source Vector is projected and put into "in", in <- (1-UU^dag) M^dag b
	blas::zero(*out);                       // Set the initial guess to zero!

	if(useTSM) (*solve_LP)(*out,*in);
	else       (*solve)   (*out,*in);
	dirac.reconstruct(*x,*b,param->solution_type);
      }
      else{  // Deflate the initial guess and solution (Abdou's procedure)
	if(loopInfo.nSteps_defl>1) errorQuda("Cannot performed stepped-deflation with smethod different than 1. (yet)\n"); 

	if(stochEO) deflationEO->deflateVector(*K_vecdef,*K_vector);
	else deflation->deflateVector(*K_vecdef,*K_vector);
	t2 = MPI_Wtime();
	printfQuda("TIME_REPORT %s %04d - Init.guess deflation: %f sec\n",msg_str,is+1,t2-t1);
	K_vecdef->uploadToCuda(out,flag_eo);             // Initial guess is deflated and put into "out", out = U(\Lambda^-1)U^dag M^dag b

	if(useTSM) (*solve_LP)(*out,*in);
	else       (*solve)   (*out,*in);
	dirac.reconstruct(*x,*b,param->solution_type);

	t1 = MPI_Wtime();	
        K_vector->downloadFromCuda(x,flag_eo);
        K_vector->download();
        deflation->projectVector(*K_vecdef,*K_vector,is+1,NeV_defl);
        K_vecdef->uploadToCuda(x,flag_eo);              // Solution is projected and put into x, x <- (1-UU^dag) x
	t2 = MPI_Wtime();
	printfQuda("TIME_REPORT: %s %04d - Solution projection: %f sec\n",msg_str,is+1,t2-t1);
      }
      
      t1 = MPI_Wtime();
      oneEndTrick_w_One_Der<double>(*x,*tmp3,*tmp4,param, gen_uloc, std_uloc, gen_oneD, std_oneD, gen_csvC, std_csvC);
      t2 = MPI_Wtime();
      printfQuda("TIME_REPORT: %s %04d - Contractions: %f sec\n",msg_str,is+1,t2-t1);

      t4 = MPI_Wtime();
      printfQuda("### TIME_REPORT: %s %04d - Finished in %f sec\n",msg_str,is+1,t4-t3);      

      if( (is+1)%Nd == 0){
	t1 = MPI_Wtime();
	if(GK_nProc[2]==1){      
	  doCudaFFT_v2<double>(std_uloc,tmp_loop); // Scalar
	  copyLoopToWriteBuf(buf_std_uloc,tmp_loop,iPrint,info.Q_sq,Nmoms,mom);
	  doCudaFFT_v2<double>(gen_uloc,tmp_loop); // dOp
	  copyLoopToWriteBuf(buf_gen_uloc,tmp_loop,iPrint,info.Q_sq,Nmoms,mom);
	  
	  for(int mu = 0 ; mu < 4 ; mu++){
	    doCudaFFT_v2<double>(std_oneD[mu],tmp_loop); // Loops
	    copyLoopToWriteBuf(buf_std_oneD[mu],tmp_loop,iPrint,info.Q_sq,Nmoms,mom);
	    doCudaFFT_v2<double>(std_csvC[mu],tmp_loop); // LoopsCv
	    copyLoopToWriteBuf(buf_std_csvC[mu],tmp_loop,iPrint,info.Q_sq,Nmoms,mom);
	    
	    doCudaFFT_v2<double>(gen_oneD[mu],tmp_loop); // LpsDw
	    copyLoopToWriteBuf(buf_gen_oneD[mu],tmp_loop,iPrint,info.Q_sq,Nmoms,mom);
	    doCudaFFT_v2<double>(gen_csvC[mu],tmp_loop); // LpsDwCv
	    copyLoopToWriteBuf(buf_gen_csvC[mu],tmp_loop,iPrint,info.Q_sq,Nmoms,mom);
	  }
	}
	else if(GK_nProc[2]>1){
	  performFFT<double>(buf_std_uloc, std_uloc, iPrint, Nmoms, momQsq);
	  performFFT<double>(buf_gen_uloc, gen_uloc, iPrint, Nmoms, momQsq);

	  for(int mu=0;mu<4;mu++){
	    performFFT<double>(buf_std_oneD[mu], std_oneD[mu], iPrint, Nmoms, momQsq);
	    performFFT<double>(buf_std_csvC[mu], std_csvC[mu], iPrint, Nmoms, momQsq);
	    performFFT<double>(buf_gen_oneD[mu], gen_oneD[mu], iPrint, Nmoms, momQsq);
	    performFFT<double>(buf_gen_csvC[mu], gen_csvC[mu], iPrint, Nmoms, momQsq);
	  }
	}
	t2 = MPI_Wtime();
	printfQuda("Loops for %s = %04d FFT'ed and copied to write buffers in %f sec\n",msg_str,is+1,t2-t1);
	iPrint++;
      }//-if (is+1)

    }//-is-loop

    //-Write the stochastic part of the loops
    t1 = MPI_Wtime();
    sprintf(loop_stoch_fname,"%s_stoch%sNeV%d",loopInfo.loop_fname, useTSM ? "_TSM_" : "_", NeV_defl);
    if(LoopFileFormat==ASCII_FORM){ // Write the loops in ASCII format
      writeLoops_ASCII(buf_std_uloc, loop_stoch_fname, loopInfo, momQsq, 0, 0, stoch_part, useTSM, LowPrecSum); // Scalar
      writeLoops_ASCII(buf_gen_uloc, loop_stoch_fname, loopInfo, momQsq, 1, 0, stoch_part, useTSM, LowPrecSum); // dOp
      for(int mu = 0 ; mu < 4 ; mu++){
	writeLoops_ASCII(buf_std_oneD[mu], loop_stoch_fname, loopInfo, momQsq, 2, mu, stoch_part, useTSM, LowPrecSum); // Loops
	writeLoops_ASCII(buf_std_csvC[mu], loop_stoch_fname, loopInfo, momQsq, 3, mu, stoch_part, useTSM, LowPrecSum); // LoopsCv
	writeLoops_ASCII(buf_gen_oneD[mu], loop_stoch_fname, loopInfo, momQsq, 4, mu, stoch_part, useTSM, LowPrecSum); // LpsDw
	writeLoops_ASCII(buf_gen_csvC[mu], loop_stoch_fname, loopInfo, momQsq, 5, mu, stoch_part, useTSM, LowPrecSum); // LpsDwCv
      }
    }
    else if(LoopFileFormat==HDF5_FORM){ // Write the loops in HDF5 format
      writeLoops_HDF5(buf_std_uloc, buf_gen_uloc, buf_std_oneD, buf_std_csvC, buf_gen_oneD, buf_gen_csvC, loop_stoch_fname, loopInfo, momQsq, stoch_part, useTSM, LowPrecSum);
    }
    t2 = MPI_Wtime();
    printfQuda("Writing the Stochastic part of the loops for NeV = %d completed in %f sec.\n",NeV_defl,t2-t1);


    //-Perform the NHP loops, for the low-precision and the high-precision inversions
    if(useTSM){
      printfQuda("\nWill Perform the HP and LP inversions\n\n");

      //- These one-end trick buffers are to be re-used for the high-precision vectors
      cudaMemset(std_uloc, 0, sizeof(double)*2*16*GK_localVolume);
      cudaMemset(gen_uloc, 0, sizeof(double)*2*16*GK_localVolume);
      cudaMemset(tmp_loop, 0, sizeof(double)*2*16*GK_localVolume);
    
      for(int mu = 0; mu < 4 ; mu++){
	cudaMemset(std_oneD[mu], 0, sizeof(double)*2*16*GK_localVolume);
	cudaMemset(gen_oneD[mu], 0, sizeof(double)*2*16*GK_localVolume);
	cudaMemset(std_csvC[mu], 0, sizeof(double)*2*16*GK_localVolume);
	cudaMemset(gen_csvC[mu], 0, sizeof(double)*2*16*GK_localVolume);
      }
      cudaDeviceSynchronize();
      //---------------------------

      Nrun = TSM_NHP;
      Nd = TSM_NdumpHP;
      iPrint = 0;
      for(int is = 0 ; is < Nrun ; is++){
	t3 = MPI_Wtime();
	t1 = MPI_Wtime();
	memset(input_vector,0,GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]*spinorSiteSize*sizeof(double));
	getStochasticRandomSource<double>(input_vector,rNum,info.source_type);
	t2 = MPI_Wtime();
	printfQuda("TIME_REPORT: %s %04d - Source creation: %f sec\n",msg_str,is+1,t2-t1);
	K_vector->packVector((double*) input_vector);
	K_vector->loadVector();
	K_vector->uploadToCuda(b,flag_eo);
	dirac.prepare(in,out   ,*x   ,*b,param->solution_type); // in -> b, out -> x, for parity singlets
	dirac.prepare(in,out_LP,*x_LP,*b,param->solution_type); //
	cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);
	dirac.Mdag(*in, *tmp_up);  // in <- M^dag b
	delete tmp_up;
	K_vector->downloadFromCuda(in,flag_eo);
	K_vector->download();
      
	t1 = MPI_Wtime();
	if(smethod==1){  // Deflate the source vector (Christos)
	  deflation->projectVector(*K_vecdef,*K_vector,is+1,NeV_defl);  
	  t2 = MPI_Wtime();
	  printfQuda("TIME_REPORT: NHP %04d - Source projection: %f sec\n",is+1,t2-t1);
	  K_vecdef->uploadToCuda(in,flag_eo);              // Source Vector is projected and put into "in", in = (1-UU^dag) M^dag b
	  blas::zero(*out);    // Set the HP
	  blas::zero(*out_LP); // and LP initial guess to zero!

	  (*solve)(*out,*in);       //-High-precision inversion
	  dirac.reconstruct(*x,*b,param->solution_type);
	  (*solve_LP)(*out_LP,*in); //-Low-recision inversion
	  dirac.reconstruct(*x_LP,*b,param->solution_type);
	}
	else{
	  if(stochEO) deflationEO->deflateVector(*K_vecdef,*K_vector);
	  else deflation->deflateVector(*K_vecdef,*K_vector);
	  t2 = MPI_Wtime();
	  printfQuda("TIME_REPORT: NHP %04d - Init.guess deflation: %f sec\n",is+1,t2-t1);
	  K_vecdef->uploadToCuda(out   ,flag_eo);   // Initial guess is deflated and put into "out", out = U(\Lambda^-1)U^dag M^dag b
	  K_vecdef->uploadToCuda(out_LP,flag_eo);   //

	  (*solve)(*out,*in);       //-High-precision inversion
	  dirac.reconstruct(*x,*b,param->solution_type);
	  (*solve_LP)(*out_LP,*in); //-Low-recision inversion
	  dirac.reconstruct(*x_LP,*b,param->solution_type);

	  t1 = MPI_Wtime();
	  K_vector->downloadFromCuda(x,flag_eo);
	  K_vector->download();
	  deflation->projectVector(*K_vecdef,*K_vector,is+1,NeV_defl);
	  K_vecdef->uploadToCuda(x,flag_eo);              // HP Solution is projected and put into x, x <- (1-UU^dag) x
          t2 = MPI_Wtime();
          printfQuda("TIME_REPORT: NHP %04d - HP solution projection: %f sec\n",is+1,t2-t1);

	  t1 = MPI_Wtime();
	  K_vector->downloadFromCuda(x_LP,flag_eo);
	  K_vector->download();
	  deflation->projectVector(*K_vecdef,*K_vector,is+1,NeV_defl);
	  K_vecdef->uploadToCuda(x_LP,flag_eo);           // LP Solution is projected and put into x_LP, x_LP <- (1-UU^dag) x_LP
          t2 = MPI_Wtime();
          printfQuda("TIME_REPORT: NHP %04d - LP solution projection: %f sec\n",is+1,t2-t1);
	}

	t1 = MPI_Wtime();
	oneEndTrick_w_One_Der<double>(*x,*tmp3,*tmp4,param, gen_uloc, std_uloc, gen_oneD, std_oneD, gen_csvC, std_csvC); //-high-precision
	t2 = MPI_Wtime();
	printfQuda("TIME_REPORT: NHP %04d - HP Contractions: %f sec\n",is+1,t2-t1);
	t1 = MPI_Wtime();
	oneEndTrick_w_One_Der<double>(*x_LP,*tmp3,*tmp4,param, gen_uloc_LP, std_uloc_LP, gen_oneD_LP, std_oneD_LP, gen_csvC_LP, std_csvC_LP); //-low-precision
	t2 = MPI_Wtime();
	printfQuda("TIME_REPORT: NHP %04d - LP Contractions: %f sec\n",is+1,t2-t1);
	
	t4 = MPI_Wtime();
	printfQuda("### TIME_REPORT: NHP %04d - Finished in %f sec\n",is+1,t4-t3);
	
	if( (is+1)%Nd == 0){
	  t1 = MPI_Wtime();
	  if(GK_nProc[2]==1){      
	    doCudaFFT_v2<double>(std_uloc   ,tmp_loop);  copyLoopToWriteBuf(buf_std_uloc_HP,tmp_loop,iPrint,info.Q_sq,Nmoms,mom); // Scalar
	    doCudaFFT_v2<double>(std_uloc_LP,tmp_loop);  copyLoopToWriteBuf(buf_std_uloc_LP,tmp_loop,iPrint,info.Q_sq,Nmoms,mom); //
	    doCudaFFT_v2<double>(gen_uloc   ,tmp_loop);  copyLoopToWriteBuf(buf_gen_uloc_HP,tmp_loop,iPrint,info.Q_sq,Nmoms,mom); // dOp
	    doCudaFFT_v2<double>(gen_uloc_LP,tmp_loop);  copyLoopToWriteBuf(buf_gen_uloc_LP,tmp_loop,iPrint,info.Q_sq,Nmoms,mom); //

	    for(int mu = 0 ; mu < 4 ; mu++){
	      doCudaFFT_v2<double>(std_oneD[mu]   ,tmp_loop);  copyLoopToWriteBuf(buf_std_oneD_HP[mu],tmp_loop,iPrint,info.Q_sq,Nmoms,mom); // Loops
	      doCudaFFT_v2<double>(std_oneD_LP[mu],tmp_loop);  copyLoopToWriteBuf(buf_std_oneD_LP[mu],tmp_loop,iPrint,info.Q_sq,Nmoms,mom); //

	      doCudaFFT_v2<double>(std_csvC[mu]   ,tmp_loop);  copyLoopToWriteBuf(buf_std_csvC_HP[mu],tmp_loop,iPrint,info.Q_sq,Nmoms,mom); // LoopsCv
	      doCudaFFT_v2<double>(std_csvC_LP[mu],tmp_loop);  copyLoopToWriteBuf(buf_std_csvC_LP[mu],tmp_loop,iPrint,info.Q_sq,Nmoms,mom); //

	      doCudaFFT_v2<double>(gen_oneD[mu]   ,tmp_loop);  copyLoopToWriteBuf(buf_gen_oneD_HP[mu],tmp_loop,iPrint,info.Q_sq,Nmoms,mom); // LpsDw
	      doCudaFFT_v2<double>(gen_oneD_LP[mu],tmp_loop);  copyLoopToWriteBuf(buf_gen_oneD_LP[mu],tmp_loop,iPrint,info.Q_sq,Nmoms,mom); //

	      doCudaFFT_v2<double>(gen_csvC[mu]   ,tmp_loop);  copyLoopToWriteBuf(buf_gen_csvC_HP[mu],tmp_loop,iPrint,info.Q_sq,Nmoms,mom); // LpsDwCv
	      doCudaFFT_v2<double>(gen_csvC_LP[mu],tmp_loop);  copyLoopToWriteBuf(buf_gen_csvC_LP[mu],tmp_loop,iPrint,info.Q_sq,Nmoms,mom); //
	    }
	  }
	  else if(GK_nProc[2]>1){
	    performFFT<double>(buf_std_uloc_HP, std_uloc, iPrint, Nmoms, momQsq);  performFFT<double>(buf_std_uloc_LP, std_uloc_LP, iPrint, Nmoms, momQsq);
	    performFFT<double>(buf_gen_uloc_HP, gen_uloc, iPrint, Nmoms, momQsq);  performFFT<double>(buf_gen_uloc_LP, gen_uloc_LP, iPrint, Nmoms, momQsq);
	    
	    for(int mu=0;mu<4;mu++){
	      performFFT<double>(buf_std_oneD_HP[mu], std_oneD[mu], iPrint, Nmoms, momQsq);  performFFT<double>(buf_std_oneD_LP[mu], std_oneD_LP[mu], iPrint, Nmoms, momQsq);
	      performFFT<double>(buf_std_csvC_HP[mu], std_csvC[mu], iPrint, Nmoms, momQsq);  performFFT<double>(buf_std_csvC_LP[mu], std_csvC_LP[mu], iPrint, Nmoms, momQsq);
	      performFFT<double>(buf_gen_oneD_HP[mu], gen_oneD[mu], iPrint, Nmoms, momQsq);  performFFT<double>(buf_gen_oneD_LP[mu], gen_oneD_LP[mu], iPrint, Nmoms, momQsq);
	      performFFT<double>(buf_gen_csvC_HP[mu], gen_csvC[mu], iPrint, Nmoms, momQsq);  performFFT<double>(buf_gen_csvC_LP[mu], gen_csvC_LP[mu], iPrint, Nmoms, momQsq);
	    }
	  }
	  t2 = MPI_Wtime();
	  printfQuda("Loops for NHP = %04d FFT'ed and copied to write buffers in %f sec\n",is+1,t2-t1);
	  iPrint++;
	}//-if (is+1)
      } // close loop over noise vectors
 
      //-Write the high-precision part
      t1 = MPI_Wtime();
      sprintf(loop_stoch_fname,"%s_stoch_TSM_NeV%d_HighPrec",loopInfo.loop_fname, NeV_defl);
      if(LoopFileFormat==ASCII_FORM){ // Write the loops in ASCII format
	writeLoops_ASCII(buf_std_uloc_HP, loop_stoch_fname, loopInfo, momQsq, 0, 0, stoch_part, useTSM, HighPrecSum); // Scalar
	writeLoops_ASCII(buf_gen_uloc_HP, loop_stoch_fname, loopInfo, momQsq, 1, 0, stoch_part, useTSM, HighPrecSum); // dOp
	for(int mu = 0 ; mu < 4 ; mu++){
	  writeLoops_ASCII(buf_std_oneD_HP[mu], loop_stoch_fname, loopInfo, momQsq, 2, mu, stoch_part, useTSM, HighPrecSum); // Loops
	  writeLoops_ASCII(buf_std_csvC_HP[mu], loop_stoch_fname, loopInfo, momQsq, 3, mu, stoch_part, useTSM, HighPrecSum); // LoopsCv
	  writeLoops_ASCII(buf_gen_oneD_HP[mu], loop_stoch_fname, loopInfo, momQsq, 4, mu, stoch_part, useTSM, HighPrecSum); // LpsDw
	  writeLoops_ASCII(buf_gen_csvC_HP[mu], loop_stoch_fname, loopInfo, momQsq, 5, mu, stoch_part, useTSM, HighPrecSum); // LpsDwCv
	}
      }
      else if(LoopFileFormat==HDF5_FORM){ // Write the loops in HDF5 format
	writeLoops_HDF5(buf_std_uloc, buf_gen_uloc, buf_std_oneD, buf_std_csvC, buf_gen_oneD, buf_gen_csvC, loop_stoch_fname, loopInfo, momQsq, stoch_part, useTSM, HighPrecSum);
      }
      t2 = MPI_Wtime();
      printfQuda("Writing the high-precision loops for NeV = %d completed in %f sec.\n",NeV_defl,t2-t1);

      //-Write the low-precision part
      t1 = MPI_Wtime();
      sprintf(loop_stoch_fname,"%s_stoch_TSM_NeV%d_LowPrec",loopInfo.loop_fname, NeV_defl);
      if(LoopFileFormat==ASCII_FORM){ // Write the loops in ASCII format
	writeLoops_ASCII(buf_std_uloc_LP, loop_stoch_fname, loopInfo, momQsq, 0, 0, stoch_part, useTSM, HighPrecSum); // Scalar
	writeLoops_ASCII(buf_gen_uloc_LP, loop_stoch_fname, loopInfo, momQsq, 1, 0, stoch_part, useTSM, HighPrecSum); // dOp
	for(int mu = 0 ; mu < 4 ; mu++){
	  writeLoops_ASCII(buf_std_oneD_LP[mu], loop_stoch_fname, loopInfo, momQsq, 2, mu, stoch_part, useTSM, HighPrecSum); // Loops
	  writeLoops_ASCII(buf_std_csvC_LP[mu], loop_stoch_fname, loopInfo, momQsq, 3, mu, stoch_part, useTSM, HighPrecSum); // LoopsCv
	  writeLoops_ASCII(buf_gen_oneD_LP[mu], loop_stoch_fname, loopInfo, momQsq, 4, mu, stoch_part, useTSM, HighPrecSum); // LpsDw
	  writeLoops_ASCII(buf_gen_csvC_LP[mu], loop_stoch_fname, loopInfo, momQsq, 5, mu, stoch_part, useTSM, HighPrecSum); // LpsDwCv
	}
      }
      else if(LoopFileFormat==HDF5_FORM){ // Write the loops in HDF5 format
	writeLoops_HDF5(buf_std_uloc_LP, buf_gen_uloc_LP, buf_std_oneD_LP, buf_std_csvC_LP, buf_gen_oneD_LP, buf_gen_csvC_LP, loop_stoch_fname, loopInfo, momQsq, stoch_part, useTSM, HighPrecSum);
      }
      t2 = MPI_Wtime();
      printfQuda("Writing the low-precision loops for NeV = %d completed in %f sec.\n",NeV_defl,t2-t1);
    }//-useTSM
    
    gsl_rng_free(rNum);
  }//-dstep
    
  printfQuda("\n ### Stochastic part calculation Done ###\n");
  printfQuda("\nCleaning up...\n");

  //-Free the momentum matrices
  for(int ip=0; ip<SplV; ip++) free(mom[ip]);
  free(mom);
  for(int ip=0;ip<Nmoms;ip++) free(momQsq[ip]);
  free(momQsq);
  //---------------------------

  //-Free loop write buffers
  free(buf_std_uloc);
  free(buf_gen_uloc);
  for(int mu = 0 ; mu < 4 ; mu++){
    free(buf_std_oneD[mu]);
    free(buf_std_csvC[mu]);
    free(buf_gen_oneD[mu]);
    free(buf_gen_csvC[mu]);
  }
  free(buf_std_oneD);
  free(buf_std_csvC);
  free(buf_gen_oneD);
  free(buf_gen_csvC);
  //---------------------------

  //-Free the Cuda loop buffers
  cudaFreeHost(std_uloc);
  cudaFreeHost(gen_uloc);
  cudaFreeHost(tmp_loop);
  for(int mu = 0 ; mu < 4 ; mu++){
    cudaFreeHost(std_oneD[mu]);
    cudaFreeHost(gen_oneD[mu]);
    cudaFreeHost(std_csvC[mu]);
    cudaFreeHost(gen_csvC[mu]);
  }
  free(std_oneD);
  free(gen_oneD);
  free(std_csvC);
  free(gen_csvC);
  //------------------------------------

  //-Free the extra buffers if using TSM
  if(useTSM){
    free(buf_std_uloc_LP); free(buf_std_uloc_HP);
    free(buf_gen_uloc_LP); free(buf_gen_uloc_HP);
    for(int mu = 0 ; mu < 4 ; mu++){
      free(buf_std_oneD_LP[mu]); free(buf_std_oneD_HP[mu]);
      free(buf_std_csvC_LP[mu]); free(buf_std_csvC_HP[mu]);
      free(buf_gen_oneD_LP[mu]); free(buf_gen_oneD_HP[mu]);
      free(buf_gen_csvC_LP[mu]); free(buf_gen_csvC_HP[mu]);
    }
    free(buf_std_oneD_LP); free(buf_std_oneD_HP);
    free(buf_std_csvC_LP); free(buf_std_csvC_HP);
    free(buf_gen_oneD_LP); free(buf_gen_oneD_HP);
    free(buf_gen_csvC_LP); free(buf_gen_csvC_HP);

    cudaFreeHost(std_uloc_LP);
    cudaFreeHost(gen_uloc_LP);
    for(int mu = 0 ; mu < 4 ; mu++){
      cudaFreeHost(std_oneD_LP[mu]);
      cudaFreeHost(gen_oneD_LP[mu]);
      cudaFreeHost(std_csvC_LP[mu]);
      cudaFreeHost(gen_csvC_LP[mu]);
    }
    free(std_oneD_LP);
    free(gen_oneD_LP);
    free(std_csvC_LP);
    free(gen_csvC_LP);
  }
  //------------------------------------


  free(input_vector);
  delete deflation;
  delete solve;
  delete d;
  delete dSloppy;
  delete dPre;
  delete K_vecdef;
  delete K_vector;
  delete K_gauge;
  delete x;
  delete b;
  delete tmp3;
  delete tmp4;

  if(useTSM){
    delete solve_LP;
    delete x_LP;
  }

  if(stochEO) delete deflationEO;

  printfQuda("...Done\n");
  popVerbosity();
  saveTuneCache();
  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);
}

//template<typename Float>
void calcEigenVectors_threepTwop_EvenOdd(void **gaugeSmeared, void **gauge, QudaGaugeParam *gauge_param, QudaInvertParam *param, qudaQKXTM_arpackInfo arpackInfo, qudaQKXTMinfo_Kepler info,
					 char *filename_twop, char *filename_threep, WHICHPARTICLE NUCLEON){

  bool flag_eo;
  double t1,t2,t3,t4;

  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);
  if(param->solve_type != QUDA_NORMOP_PC_SOLVE) errorQuda("This function works only with even odd preconditioning");
  if(param->inv_type != QUDA_CG_INVERTER) errorQuda("This function works only with CG method");
  if( (param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) && (param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) ) errorQuda("Only asymmetric operators are supported in deflation\n");
  if( param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC )
    flag_eo = true;
  else if(param->matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
    flag_eo = false;

  void *input_vector = malloc(GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]*spinorSiteSize*sizeof(double));
  void *output_vector = malloc(GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]*spinorSiteSize*sizeof(double));
  QKXTM_Gauge_Kepler<double> *K_gaugeSmeared = new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  QKXTM_Gauge_Kepler<float> *K_gaugeContractions = new QKXTM_Gauge_Kepler<float>(BOTH,GAUGE);

  QKXTM_Vector_Kepler<double> *K_vector = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<double> *K_guess = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<float> *K_temp = new QKXTM_Vector_Kepler<float>(BOTH,VECTOR);


  QKXTM_Propagator_Kepler<float> *K_prop_up = new QKXTM_Propagator_Kepler<float>(BOTH,PROPAGATOR);
  QKXTM_Propagator_Kepler<float> *K_prop_down = new QKXTM_Propagator_Kepler<float>(BOTH,PROPAGATOR);  
  QKXTM_Propagator_Kepler<float> *K_seqProp = new QKXTM_Propagator_Kepler<float>(BOTH,PROPAGATOR);

  QKXTM_Propagator3D_Kepler<float> *K_prop3D_up = new QKXTM_Propagator3D_Kepler<float>(BOTH,PROPAGATOR3D);
  QKXTM_Propagator3D_Kepler<float> *K_prop3D_down = new QKXTM_Propagator3D_Kepler<float>(BOTH,PROPAGATOR3D);

  QKXTM_Contraction_Kepler<float> *K_contract = new QKXTM_Contraction_Kepler<float>();
  printfQuda("Memory allocation was successfull\n");

  if(param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS) errorQuda("This function works only with ukqcd gamma basis\n");
  if(param->dirac_order != QUDA_DIRAC_ORDER) errorQuda("This function works only with colors inside the spins\n");

  //- Calculate the eigenVectors of the +mu
  QKXTM_Deflation_Kepler<double> *deflation_up = new QKXTM_Deflation_Kepler<double>(param,arpackInfo);
  deflation_up->printInfo();

  t1 = MPI_Wtime();
  deflation_up->eigenSolver();
  t2 = MPI_Wtime();
  printfQuda("calcEigenVectors_threepTwop_EvenOdd TIME REPORT: EigenVector +mu Calculation: %f sec\n",t2-t1);
  //----------------------------------------------------

  //- Calculate the eigenVectors of the -mu
  param->twist_flavor = QUDA_TWIST_MINUS;
  QKXTM_Deflation_Kepler<double> *deflation_down = new QKXTM_Deflation_Kepler<double>(param,arpackInfo);
  deflation_down->printInfo();

  t1 = MPI_Wtime();
  deflation_down->eigenSolver();
  t2 = MPI_Wtime();
  printfQuda("calcEigenVectors_threepTwop_EvenOdd TIME REPORT: EigenVector -mu Calculation: %f sec\n",t2-t1);
  param->twist_flavor = QUDA_TWIST_PLUS;
  //----------------------------------------------------

  if (!initialized) errorQuda("QUDA not initialized");
  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);


  K_gaugeContractions->packGauge(gauge);
  K_gaugeContractions->loadGauge();
  //  K_gaugeContractions->calculate(); do not do it because I changed the sign due to antiperiodic boundary conditions


  K_gaugeSmeared->packGauge(gaugeSmeared);
  K_gaugeSmeared->loadGauge();
  K_gaugeSmeared->calculatePlaq();

  bool pc_solution = false;
  bool pc_solve = true;
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) || (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = false;

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;
  
  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;
  // create the dirac operator                                                                                                                                      
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;
  profileInvert.TPSTART(QUDA_PROFILE_H2D);

  //QKXTM: DMH rewite for spinor field memalloc
  ColorSpinorField *b = NULL;
  ColorSpinorField *x = NULL;
  ColorSpinorField *in = NULL;
  ColorSpinorField *out = NULL;
  const int *X = cudaGauge->X();

  memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  memset(output_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  // wrap CPU host side pointers
  ColorSpinorParam cpuParam(input_vector, *param, X, pc_solution, param->input_location);
  ColorSpinorField *h_b = ColorSpinorField::Create(cpuParam);

  cpuParam.v = output_vector;
  cpuParam.location = param->output_location;
  ColorSpinorField *h_x = ColorSpinorField::Create(cpuParam);

  //Zero out spinors
  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  x = new cudaColorSpinorField(cudaParam);

  profileInvert.TPSTOP(QUDA_PROFILE_H2D);
  setTuning(param->tune);
  
  blas::zero(*x);
  blas::zero(*b);
  DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
  SolverParam solverParam(*param);
  Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);

  int my_src[4];
  char filename_mesons[257];
  char filename_baryons[257];

  info.thrp_type[0] = "ultra_local";
  info.thrp_type[1] = "noether";
  info.thrp_type[2] = "oneD";

  info.thrp_proj_type[0] = "G4";
  info.thrp_proj_type[1] = "G5G123";
  info.thrp_proj_type[2] = "G5G1";
  info.thrp_proj_type[3] = "G5G2";
  info.thrp_proj_type[4] = "G5G3";

  info.baryon_type[0] = "nucl_nucl";
  info.baryon_type[1] = "nucl_roper";
  info.baryon_type[2] = "roper_nucl";
  info.baryon_type[3] = "roper_roper";
  info.baryon_type[4] = "deltapp_deltamm_11";
  info.baryon_type[5] = "deltapp_deltamm_22";
  info.baryon_type[6] = "deltapp_deltamm_33";
  info.baryon_type[7] = "deltap_deltaz_11";
  info.baryon_type[8] = "deltap_deltaz_22";
  info.baryon_type[9] = "deltap_deltaz_33";

  info.meson_type[0] = "pseudoscalar";
  info.meson_type[1] = "scalar";
  info.meson_type[2] = "g5g1";
  info.meson_type[3] = "g5g2";
  info.meson_type[4] = "g5g3";
  info.meson_type[5] = "g5g4";
  info.meson_type[6] = "g1";
  info.meson_type[7] = "g2";
  info.meson_type[8] = "g3";
  info.meson_type[9] = "g4";

  printfQuda("\nThe total number of source-positions is %d\n",info.Nsources);

  int nRun3pt = 0;
  for(int i=0;i<info.Nsources;i++) nRun3pt += info.run3pt_src[i];

  int NprojMax = 0;
  if(nRun3pt==0) printfQuda("Will NOT perform the three-point function for any of the source positions\n");
  else if (nRun3pt>0){
    printfQuda("Will perform the three-point function for %d source positions, for the following source-sink separations and projectors:\n",nRun3pt);
    for(int its=0;its<info.Ntsink;its++){
      if(info.Nproj[its] >= NprojMax) NprojMax = info.Nproj[its];
      
      printfQuda(" sink-source = %d:\n",info.tsinkSource[its]);
      for(int p=0;p<info.Nproj[its];p++) printfQuda("  %s\n",info.thrp_proj_type[info.proj_list[its][p]]);
    }
  }
  else errorQuda("Check your option for running the three-point function! Exiting.\n");


  CORR_SPACE CorrSpace = info.CorrSpace; // Flag to determine whether to write the correlation functions in position/momentum space
  printfQuda("Will write the correlation functions in %s-space!\n", (CorrSpace == POSITION_SPACE) ? "position" : "momentum");

  if(CorrSpace==POSITION_SPACE && info.CorrFileFormat==ASCII_FORM){
    warningQuda("ASCII format not supported for writing the correlation functions in position-space! Switching to HDF5 format...\n");
    info.CorrFileFormat = HDF5_FORM;
  }
  FILE_WRITE_FORMAT CorrFileFormat = info.CorrFileFormat;
  printfQuda("Will write the correlation functions in %s format\n", (CorrFileFormat == ASCII_FORM) ? "ASCII" : "HDF5");


  //-Allocate the Two-point and Three-point function data buffers
  long int alloc_size;
  if(CorrSpace==MOMENTUM_SPACE) alloc_size = GK_localL[3]*GK_Nmoms;
  else if(CorrSpace==POSITION_SPACE) alloc_size = GK_localVolume;

  //-Three-Point function
  double *corrThp_local   = (double*) calloc(alloc_size  *16*2,sizeof(double));
  double *corrThp_noether = (double*) calloc(alloc_size*4   *2,sizeof(double));
  double *corrThp_oneD    = (double*) calloc(alloc_size*4*16*2,sizeof(double));
  if(corrThp_local == NULL || corrThp_noether == NULL || corrThp_oneD == NULL) errorQuda("Cannot allocate memory for Three-point function write Buffers.");

  //-Two-point function
  double (*corrMesons)[2][N_MESONS] = (double(*)[2][N_MESONS]) calloc(alloc_size*2*N_MESONS*2,sizeof(double));
  double (*corrBaryons)[2][N_BARYONS][4][4] = (double(*)[2][N_BARYONS][4][4]) calloc(alloc_size*2*N_BARYONS*4*4*2,sizeof(double));
  if(corrMesons == NULL || corrBaryons == NULL) errorQuda("Cannot allocate memory for Two-point function write Buffers.");


  //-HDF5 buffers for the three-point and two-point function
  double *Thrp_local_HDF5   = NULL;
  double *Thrp_noether_HDF5 = NULL;
  double **Thrp_oneD_HDF5   = NULL;
  
  double *Twop_baryons_HDF5 = NULL;
  double *Twop_mesons_HDF5  = NULL;

  if( CorrFileFormat==HDF5_FORM ){
    if( (Thrp_local_HDF5   = (double*) malloc(2*16*alloc_size*2*info.Ntsink*NprojMax*sizeof(double)))==NULL ) errorQuda("Cannot allocate memory for Thrp_local_HDF5.\n");
    if( (Thrp_noether_HDF5 = (double*) malloc(2* 4*alloc_size*2*info.Ntsink*NprojMax*sizeof(double)))==NULL ) errorQuda("Cannot allocate memory for Thrp_noether_HDF5.\n");

    memset(Thrp_local_HDF5  , 0, 2*16*alloc_size*2*info.Ntsink*NprojMax*sizeof(double));
    memset(Thrp_noether_HDF5, 0, 2* 4*alloc_size*2*info.Ntsink*NprojMax*sizeof(double));

    if( (Thrp_oneD_HDF5 = (double**) malloc(4*sizeof(double*))) == NULL ) errorQuda("Cannot allocate memory for Thrp_oneD_HDF5.\n");
    for(int mu=0;mu<4;mu++){
      if( (Thrp_oneD_HDF5[mu] = (double*) malloc(2*16*alloc_size*2*info.Ntsink*NprojMax*sizeof(double)))==NULL ) errorQuda("Cannot allocate memory for Thrp_oned_HDF5[%d].\n",mu);

      memset(Thrp_oneD_HDF5[mu], 0, 2*16*alloc_size*2*info.Ntsink*NprojMax*sizeof(double));
    }

    if( (Twop_baryons_HDF5 = (double*) malloc(2*16*alloc_size*2*N_BARYONS*sizeof(double)))==NULL ) errorQuda("Cannot allocate memory for Twop_baryons_HDF5.\n");
    if( (Twop_mesons_HDF5  = (double*) malloc(2   *alloc_size*2*N_MESONS *sizeof(double)))==NULL ) errorQuda("Cannot allocate memory for Twop_mesons_HDF5.\n");

    memset(Twop_baryons_HDF5, 0, 2*16*alloc_size*2*N_BARYONS*sizeof(double));
    memset(Twop_mesons_HDF5 , 0, 2   *alloc_size*2*N_MESONS *sizeof(double));
  }
  //------------------------------------------------------------------------


  for(int isource = 0 ; isource < info.Nsources ; isource++){
    t3 = MPI_Wtime();
    printfQuda("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",isource,info.sourcePosition[isource][0],info.sourcePosition[isource][1],info.sourcePosition[isource][2],info.sourcePosition[isource][3]);

    if( CorrFileFormat==ASCII_FORM ){
      sprintf(filename_mesons,"%s.mesons.SS.%02d.%02d.%02d.%02d.dat",filename_twop,info.sourcePosition[isource][0],info.sourcePosition[isource][1],info.sourcePosition[isource][2],info.sourcePosition[isource][3]);
      sprintf(filename_baryons,"%s.baryons.SS.%02d.%02d.%02d.%02d.dat",filename_twop,info.sourcePosition[isource][0],info.sourcePosition[isource][1],info.sourcePosition[isource][2],info.sourcePosition[isource][3]);
    }
    else if( CorrFileFormat==HDF5_FORM ){
      char *str;
      if(CorrSpace==MOMENTUM_SPACE) asprintf(&str,"Qsq%d",info.Q_sq);
      else if (CorrSpace==POSITION_SPACE) asprintf(&str,"PosSpace");
      sprintf(filename_mesons ,"%s_mesons_%s_SS.%02d.%02d.%02d.%02d.h5" ,filename_twop,str,info.sourcePosition[isource][0],info.sourcePosition[isource][1],info.sourcePosition[isource][2],info.sourcePosition[isource][3]);
      sprintf(filename_baryons,"%s_baryons_%s_SS.%02d.%02d.%02d.%02d.h5",filename_twop,str,info.sourcePosition[isource][0],info.sourcePosition[isource][1],info.sourcePosition[isource][2],info.sourcePosition[isource][3]);
    }

    if(info.check_files){
      bool checkMesons, checkBaryons;
      checkMesons = exists_file(filename_mesons);
      checkBaryons = exists_file(filename_baryons);
      if( (checkMesons == true) && (checkBaryons == true) ) continue;
    }

    printfQuda("Forward Inversions:\n");
    t1 = MPI_Wtime();
    for(int isc = 0 ; isc < 12 ; isc++){
      ///////////////////////////////////////////////////////////////////////////////// forward prop for up quark ///////////////////////////
      memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
      b->changeTwist(QUDA_TWIST_PLUS);
      x->changeTwist(QUDA_TWIST_PLUS);
      b->Even().changeTwist(QUDA_TWIST_PLUS);
      b->Odd().changeTwist(QUDA_TWIST_PLUS);
      x->Even().changeTwist(QUDA_TWIST_PLUS);
      x->Odd().changeTwist(QUDA_TWIST_PLUS);
      for(int i = 0 ; i < 4 ; i++)
	my_src[i] = info.sourcePosition[isource][i] - comm_coords(default_topo)[i] * X[i];

      if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
	*( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + isc*2 ) = 1.;

      K_vector->packVector((double*) input_vector);
      K_vector->loadVector();
      K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
      K_guess->uploadToCuda(b,flag_eo);
      dirac.prepare(in,out,*x,*b,param->solution_type);
      // in is reference to the b but for a parity sinlet
      // out is reference to the x but for a parity sinlet
      cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);
      dirac.Mdag(*in, *tmp_up);
      delete tmp_up;
      // now the the source vector b is ready to perform deflation and find the initial guess
      K_vector->downloadFromCuda(in,flag_eo);
      K_vector->download();
      deflation_up->deflateVector(*K_guess,*K_vector);
      K_guess->uploadToCuda(out,flag_eo); // initial guess is ready

      printfQuda(" up - %02d: ",isc);
      (*solve)(*out,*in);
      dirac.reconstruct(*x,*b,param->solution_type);
      K_vector->downloadFromCuda(x,flag_eo);
      if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	K_vector->scaleVector(2*param->kappa);
      }

      K_temp->castDoubleToFloat(*K_vector);
      K_prop_up->absorbVectorToDevice(*K_temp,isc/3,isc%3);
      //      printfQuda("Inversion up = %d,  for source = %d finished in time %f sec\n",isc,isource,t2-t1);
      //////////////////////////////////////////////////////////
      ////////////////////////////////////////////////////////////////////////////// Forward prop for down quark ///////////////////////////////////
      /////////////////////////////////////////////////////////
      memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
      b->changeTwist(QUDA_TWIST_MINUS);
      x->changeTwist(QUDA_TWIST_MINUS);
      b->Even().changeTwist(QUDA_TWIST_MINUS);
      b->Odd().changeTwist(QUDA_TWIST_MINUS);
      x->Even().changeTwist(QUDA_TWIST_MINUS);
      x->Odd().changeTwist(QUDA_TWIST_MINUS);
      for(int i = 0 ; i < 4 ; i++)
	my_src[i] = info.sourcePosition[isource][i] - comm_coords(default_topo)[i] * X[i];

      if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
	*( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + isc*2 ) = 1.;

      K_vector->packVector((double*) input_vector);
      K_vector->loadVector();
      K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
      K_guess->uploadToCuda(b,flag_eo);
      dirac.prepare(in,out,*x,*b,param->solution_type);
      // in is reference to the b but for a parity sinlet
      // out is reference to the x but for a parity sinlet
      cudaColorSpinorField *tmp_down = new cudaColorSpinorField(*in);
      dirac.Mdag(*in, *tmp_down);
      delete tmp_down;
      // now the the source vector b is ready to perform deflation and find the initial guess
      K_vector->downloadFromCuda(in,flag_eo);
      K_vector->download();
      deflation_down->deflateVector(*K_guess,*K_vector);
      K_guess->uploadToCuda(out,flag_eo); // initial guess is ready

      printfQuda(" dn - %02d: ",isc);
      (*solve)(*out,*in);
      dirac.reconstruct(*x,*b,param->solution_type);
      K_vector->downloadFromCuda(x,flag_eo);
      if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	K_vector->scaleVector(2*param->kappa);
      }

      K_temp->castDoubleToFloat(*K_vector);
      K_prop_down->absorbVectorToDevice(*K_temp,isc/3,isc%3);
      //      printfQuda("Inversion down = %d,  for source = %d finished in time %f sec\n",isc,isource,t2-t1);
    } // close loop over 12 spin-color
    t2 = MPI_Wtime();
    printfQuda("TIME_REPORT - Forward Inversions: %f sec.\n\n",t2-t1);

    if(info.run3pt_src[isource]){

      /////////////////////////////////// Smearing on the 3D propagators

      //-C.K: Loop over the number of sink-source separations
      int my_fixSinkTime;
      char *filename_threep_base;
      for(int its=0;its<info.Ntsink;its++){
	my_fixSinkTime = (info.tsinkSource[its] + info.sourcePosition[isource][3])%GK_totalL[3] - comm_coords(default_topo)[3] * X[3];

	t1 = MPI_Wtime();
	K_temp->zero_device();
	checkCudaError();
	if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ){
	  K_prop3D_up->absorbTimeSlice(*K_prop_up,my_fixSinkTime);
	  K_prop3D_down->absorbTimeSlice(*K_prop_down,my_fixSinkTime);
	}
	comm_barrier();

	for(int nu = 0 ; nu < 4 ; nu++)
	  for(int c2 = 0 ; c2 < 3 ; c2++){
	    // up //
	    K_temp->zero_device();
	    if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_temp->copyPropagator3D(*K_prop3D_up,my_fixSinkTime,nu,c2);
	    comm_barrier();
	    K_vector->castFloatToDouble(*K_temp);
	    K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	    K_temp->castDoubleToFloat(*K_guess);
	    if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_prop3D_up->absorbVectorTimeSlice(*K_temp,my_fixSinkTime,nu,c2);
	    comm_barrier();
	    K_temp->zero_device();

	    // down //
	    if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_temp->copyPropagator3D(*K_prop3D_down,my_fixSinkTime,nu,c2);
	    comm_barrier();
	    K_vector->castFloatToDouble(*K_temp);
	    K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	    K_temp->castDoubleToFloat(*K_guess);
	    if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_prop3D_down->absorbVectorTimeSlice(*K_temp,my_fixSinkTime,nu,c2);
	    comm_barrier();
	    K_temp->zero_device();	
	  }
	t2 = MPI_Wtime();
	//printfQuda("TIME_REPORT - 3d Props preparation for sink-source[%d]=%d: %f sec\n",its,info.tsinkSource[its],t2-t1);

	for(int proj=0;proj<info.Nproj[its];proj++){
	  WHICHPROJECTOR PID = (WHICHPROJECTOR) info.proj_list[its][proj];
	  char *proj_str;
	  asprintf(&proj_str,"%s",info.thrp_proj_type[info.proj_list[its][proj]]);

	  printfQuda("\n# Three-point function calculation for source-position = %d, sink-source = %d, projector %s begins now\n",isource,info.tsinkSource[its],proj_str);

	  if( CorrFileFormat==ASCII_FORM ){
	    asprintf(&filename_threep_base,"%s_tsink%d_proj%s",filename_threep,info.tsinkSource[its],proj_str);
	    printfQuda("The three-point function ASCII base name is: %s\n",filename_threep_base);
	  }

	  /////////////////////////////////////////sequential propagator for the part 1
	  printfQuda("Sequential Inversions, flavor %s:\n",NUCLEON == NEUTRON ? "dn" : "up");
	  t1 = MPI_Wtime();
	  for(int nu = 0 ; nu < 4 ; nu++)
	    for(int c2 = 0 ; c2 < 3 ; c2++){
	      K_temp->zero_device();
	      if(NUCLEON == PROTON){
		if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_contract->seqSourceFixSinkPart1(*K_temp,*K_prop3D_up, *K_prop3D_down, my_fixSinkTime, nu, c2, PID, NUCLEON);}
	      else if(NUCLEON == NEUTRON){
		if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_contract->seqSourceFixSinkPart1(*K_temp,*K_prop3D_down, *K_prop3D_up, my_fixSinkTime, nu, c2, PID, NUCLEON);}
	      comm_barrier();
	      K_temp->conjugate();
	      K_temp->apply_gamma5();
	      K_vector->castFloatToDouble(*K_temp);
	      //
	      K_vector->scaleVector(1e+10);
	      //
	      K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	      if(NUCLEON == PROTON){
		b->changeTwist(QUDA_TWIST_MINUS); x->changeTwist(QUDA_TWIST_MINUS); b->Even().changeTwist(QUDA_TWIST_MINUS);
		b->Odd().changeTwist(QUDA_TWIST_MINUS); x->Even().changeTwist(QUDA_TWIST_MINUS); x->Odd().changeTwist(QUDA_TWIST_MINUS);
	      }
	      else{
		b->changeTwist(QUDA_TWIST_PLUS); x->changeTwist(QUDA_TWIST_PLUS); b->Even().changeTwist(QUDA_TWIST_PLUS);
		b->Odd().changeTwist(QUDA_TWIST_PLUS); x->Even().changeTwist(QUDA_TWIST_PLUS); x->Odd().changeTwist(QUDA_TWIST_PLUS);
	      }
	      K_guess->uploadToCuda(b,flag_eo);
	      dirac.prepare(in,out,*x,*b,param->solution_type);
	  
	      cudaColorSpinorField *tmp = new cudaColorSpinorField(*in);
	      dirac.Mdag(*in, *tmp);
	      delete tmp;
	      K_vector->downloadFromCuda(in,flag_eo);
	      K_vector->download();
	      if(NUCLEON == PROTON)
		deflation_down->deflateVector(*K_guess,*K_vector);
	      else if(NUCLEON == NEUTRON)
		deflation_up->deflateVector(*K_guess,*K_vector);
	      K_guess->uploadToCuda(out,flag_eo); // initial guess is ready

	      printfQuda("%02d - ",nu*3+c2);
	      (*solve)(*out,*in);
	      dirac.reconstruct(*x,*b,param->solution_type);
	      K_vector->downloadFromCuda(x,flag_eo);
	      if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
		K_vector->scaleVector(2*param->kappa);
	      }
	      //
	      K_vector->scaleVector(1e-10);
	      //
	      K_temp->castDoubleToFloat(*K_vector);
	      K_seqProp->absorbVectorToDevice(*K_temp,nu,c2);
	      //	      printfQuda("Inversion time for seq prop part 1 = %d, source = %d at sink-source = %d, projector %s is: %f sec\n",nu*3+c2,isource,info.tsinkSource[its],proj_str,t2-t1);
	    }
	  t2 = MPI_Wtime();
	  printfQuda("TIME_REPORT - Sequential Inversions, flavor %s: %f sec\n",NUCLEON == NEUTRON ? "dn" : "up",t2-t1);

	  ////////////////// Contractions for part 1 ////////////////
	  t1 = MPI_Wtime();
	  if(NUCLEON == PROTON)  K_contract->contractFixSink(*K_seqProp, *K_prop_up  , *K_gaugeContractions, corrThp_local, corrThp_noether, corrThp_oneD, PID, NUCLEON, 1, isource, CorrSpace);
	  if(NUCLEON == NEUTRON) K_contract->contractFixSink(*K_seqProp, *K_prop_down, *K_gaugeContractions, corrThp_local, corrThp_noether, corrThp_oneD, PID, NUCLEON, 1, isource, CorrSpace);
	  t2 = MPI_Wtime();
	  printfQuda("TIME_REPORT - Three-point Contractions, flavor %s: %f sec\n",NUCLEON == NEUTRON ? "dn" : "up",t2-t1);

	  t1 = MPI_Wtime();
	  if( CorrFileFormat==ASCII_FORM ){
	    K_contract->writeThrp_ASCII(corrThp_local, corrThp_noether, corrThp_oneD, NUCLEON, 1, filename_threep_base, isource, info.tsinkSource[its], CorrSpace);
	    t2 = MPI_Wtime();
	    printfQuda("TIME_REPORT - Done: Three-point function for source-position = %d, sink-source = %d, projector %s, flavor %s written in ASCII format in %f sec.\n",
		       isource,info.tsinkSource[its],proj_str,NUCLEON == NEUTRON ? "dn" : "up",t2-t1);
	  }
	  else if( CorrFileFormat==HDF5_FORM ){
	    int uOrd;
	    if(NUCLEON == PROTON ) uOrd = 0;
	    if(NUCLEON == NEUTRON) uOrd = 1;

	    int thrp_sign = ( info.tsinkSource[its] + GK_sourcePosition[isource][3] ) >= GK_totalL[3] ? -1 : +1;

	    K_contract->copyThrpToHDF5_Buf((void*)Thrp_local_HDF5  , (void*)corrThp_local  , 0, uOrd, its, info.Ntsink, proj, thrp_sign, THRP_LOCAL  , CorrSpace);
	    K_contract->copyThrpToHDF5_Buf((void*)Thrp_noether_HDF5, (void*)corrThp_noether, 0, uOrd, its, info.Ntsink, proj, thrp_sign, THRP_NOETHER, CorrSpace);
	    for(int mu = 0;mu<4;mu++)
	      K_contract->copyThrpToHDF5_Buf((void*)Thrp_oneD_HDF5[mu],(void*)corrThp_oneD ,mu, uOrd, its, info.Ntsink, proj, thrp_sign, THRP_ONED   , CorrSpace);

	    t2 = MPI_Wtime();
	    printfQuda("TIME_REPORT - Three-point function for flavor %s copied to HDF5 write buffers in %f sec.\n",NUCLEON == NEUTRON ? "dn" : "up",t2-t1);
	  }

	  /////////////////////////////////////////sequential propagator for the part 2
	  printfQuda("Sequential Inversions, flavor %s:\n",NUCLEON == NEUTRON ? "up" : "dn");
	  t1 = MPI_Wtime();
	  for(int nu = 0 ; nu < 4 ; nu++)
	    for(int c2 = 0 ; c2 < 3 ; c2++){
	      K_temp->zero_device();
	      if(NUCLEON == PROTON){
		if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_contract->seqSourceFixSinkPart2(*K_temp,*K_prop3D_up, my_fixSinkTime, nu, c2, PID, NUCLEON);}
	      else if(NUCLEON == NEUTRON){
		if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_contract->seqSourceFixSinkPart2(*K_temp,*K_prop3D_down, my_fixSinkTime, nu, c2, PID, NUCLEON);}
	      comm_barrier();
	      K_temp->conjugate();
	      K_temp->apply_gamma5();
	      K_vector->castFloatToDouble(*K_temp);
	      //
	      K_vector->scaleVector(1e+10);
	      //
	      K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	      if(NUCLEON == PROTON){
		b->changeTwist(QUDA_TWIST_PLUS); x->changeTwist(QUDA_TWIST_PLUS); b->Even().changeTwist(QUDA_TWIST_PLUS);
		b->Odd().changeTwist(QUDA_TWIST_PLUS); x->Even().changeTwist(QUDA_TWIST_PLUS); x->Odd().changeTwist(QUDA_TWIST_PLUS);
	      }
	      else{
		b->changeTwist(QUDA_TWIST_MINUS); x->changeTwist(QUDA_TWIST_MINUS); b->Even().changeTwist(QUDA_TWIST_MINUS);
		b->Odd().changeTwist(QUDA_TWIST_MINUS); x->Even().changeTwist(QUDA_TWIST_MINUS); x->Odd().changeTwist(QUDA_TWIST_MINUS);
	      }
	      K_guess->uploadToCuda(b,flag_eo);
	      dirac.prepare(in,out,*x,*b,param->solution_type);
	  
	      cudaColorSpinorField *tmp = new cudaColorSpinorField(*in);
	      dirac.Mdag(*in, *tmp);
	      delete tmp;
	      K_vector->downloadFromCuda(in,flag_eo);
	      K_vector->download();
	      if(NUCLEON == PROTON)
		deflation_up->deflateVector(*K_guess,*K_vector);
	      else if(NUCLEON == NEUTRON)
		deflation_down->deflateVector(*K_guess,*K_vector);
	      K_guess->uploadToCuda(out,flag_eo); // initial guess is ready

	      printfQuda("%02d - ",nu*3+c2);
	      (*solve)(*out,*in);
	      dirac.reconstruct(*x,*b,param->solution_type);
	      K_vector->downloadFromCuda(x,flag_eo);
	      if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
		K_vector->scaleVector(2*param->kappa);
	      }
	      //
	      K_vector->scaleVector(1e-10);
	      //
	      K_temp->castDoubleToFloat(*K_vector);
	      K_seqProp->absorbVectorToDevice(*K_temp,nu,c2);
	      //	      printfQuda("Inversion time for seq prop part 2 = %d, source = %d at sink-source = %d, projector %s is: %f sec\n",nu*3+c2,isource,info.tsinkSource[its],proj_str,t2-t1);
	    }
	  t2 = MPI_Wtime();
	  printfQuda("TIME_REPORT - Sequential Inversions, flavor %s: %f sec\n",NUCLEON == NEUTRON ? "up" : "dn",t2-t1);

	  ////////////////// Contractions for part 2 ////////////////
	  t1 = MPI_Wtime();
	  if(NUCLEON == PROTON)  K_contract->contractFixSink(*K_seqProp, *K_prop_down, *K_gaugeContractions, corrThp_local, corrThp_noether, corrThp_oneD, PID, NUCLEON, 2, isource, CorrSpace);
	  if(NUCLEON == NEUTRON) K_contract->contractFixSink(*K_seqProp, *K_prop_up  , *K_gaugeContractions, corrThp_local, corrThp_noether, corrThp_oneD, PID, NUCLEON, 2, isource, CorrSpace);
	  t2 = MPI_Wtime();
	  printfQuda("TIME_REPORT - Three-point Contractions, flavor %s: %f sec\n",NUCLEON == NEUTRON ? "up" : "dn",t2-t1);

	  t1 = MPI_Wtime();
	  if( CorrFileFormat==ASCII_FORM ){
	    K_contract->writeThrp_ASCII(corrThp_local, corrThp_noether, corrThp_oneD, NUCLEON, 2, filename_threep_base, isource, info.tsinkSource[its], CorrSpace);
	    t2 = MPI_Wtime();
	    printfQuda("TIME_REPORT - Done: Three-point function for source-position = %d, sink-source = %d, projector %s, flavor %s written in ASCII format in %f sec.\n",
		       isource,info.tsinkSource[its],proj_str,NUCLEON == NEUTRON ? "up" : "dn",t2-t1);
	  }
	  else if( CorrFileFormat==HDF5_FORM ){
	    int uOrd;
	    if(NUCLEON == PROTON ) uOrd = 1;
	    if(NUCLEON == NEUTRON) uOrd = 0;

	    int thrp_sign = ( info.tsinkSource[its] + GK_sourcePosition[isource][3] ) >= GK_totalL[3] ? -1 : +1;

	    K_contract->copyThrpToHDF5_Buf((void*)Thrp_local_HDF5  , (void*)corrThp_local  , 0, uOrd, its, info.Ntsink, proj, thrp_sign, THRP_LOCAL  , CorrSpace);
	    K_contract->copyThrpToHDF5_Buf((void*)Thrp_noether_HDF5, (void*)corrThp_noether, 0, uOrd, its, info.Ntsink, proj, thrp_sign, THRP_NOETHER, CorrSpace);
	    for(int mu = 0;mu<4;mu++)
	      K_contract->copyThrpToHDF5_Buf((void*)Thrp_oneD_HDF5[mu],(void*)corrThp_oneD ,mu, uOrd, its, info.Ntsink, proj, thrp_sign, THRP_ONED   , CorrSpace);
	  
	    t2 = MPI_Wtime();
	    printfQuda("TIME_REPORT - Three-point function for flavor %s copied to HDF5 write buffers in %f sec.\n",NUCLEON == NEUTRON ? "up" : "dn",t2-t1);
	  }

	}//-loop over projectors
      }//-loop over sink-source separations      

      //-C.K. Write the three-point function in HDF5 format
      if( CorrFileFormat==HDF5_FORM ){
	char *str;
	if(CorrSpace==MOMENTUM_SPACE) asprintf(&str,"Qsq%d",info.Q_sq);
	else if (CorrSpace==POSITION_SPACE) asprintf(&str,"PosSpace");

	t1 = MPI_Wtime();
	asprintf(&filename_threep_base,"%s_%s_%s_SS.%02d.%02d.%02d.%02d.h5",filename_threep, (NUCLEON == PROTON) ? "proton" : "neutron",str,
		 GK_sourcePosition[isource][0],GK_sourcePosition[isource][1],GK_sourcePosition[isource][2],GK_sourcePosition[isource][3]);
	printfQuda("\nThe three-point function HDF5 filename is: %s\n",filename_threep_base);
	
	K_contract->writeThrpHDF5((void*) Thrp_local_HDF5, (void*) Thrp_noether_HDF5, (void**)Thrp_oneD_HDF5, filename_threep_base, info, isource, NUCLEON);
	
	t2 = MPI_Wtime();
	printfQuda("TIME_REPORT - Done: Three-point function for source-position = %d written in HDF5 format in %f sec.\n",isource,t2-t1);
      }

      printfQuda("\n");
    }//-if running for the specific isource


    ////////// At the very end ///////////////////////


    // smear the forward propagators
    for(int nu = 0 ; nu < 4 ; nu++)
      for(int c2 = 0 ; c2 < 3 ; c2++){
	K_temp->copyPropagator(*K_prop_up,nu,c2);
	K_vector->castFloatToDouble(*K_temp);
	K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	K_temp->castDoubleToFloat(*K_guess);
	K_prop_up->absorbVectorToDevice(*K_temp,nu,c2);
	
	K_temp->copyPropagator(*K_prop_down,nu,c2);
	K_vector->castFloatToDouble(*K_temp);
	K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	K_temp->castDoubleToFloat(*K_guess);
	K_prop_down->absorbVectorToDevice(*K_temp,nu,c2);
      }
    /////
    K_prop_up->rotateToPhysicalBase_device(+1);
    K_prop_down->rotateToPhysicalBase_device(-1);
    t1 = MPI_Wtime();
    K_contract->contractBaryons(*K_prop_up,*K_prop_down, corrBaryons, isource, CorrSpace);
    K_contract->contractMesons (*K_prop_up,*K_prop_down, corrMesons , isource, CorrSpace);
    t2 = MPI_Wtime();
    printfQuda("TIME_REPORT - Two-point Contractions: %f sec\n",t2-t1);

    printfQuda("\nThe baryons two-point function %s filename is: %s\n",(CorrFileFormat==ASCII_FORM) ? "ASCII" : "HDF5",filename_baryons);
    printfQuda("The mesons two-point function %s filename is: %s\n" ,(CorrFileFormat==ASCII_FORM) ? "ASCII" : "HDF5",filename_mesons);
    if( CorrFileFormat==ASCII_FORM ){
      t1 = MPI_Wtime();
      K_contract->writeTwopBaryons_ASCII(corrBaryons, filename_baryons, isource, CorrSpace);
      K_contract->writeTwopMesons_ASCII (corrMesons , filename_mesons , isource, CorrSpace);
      t2 = MPI_Wtime();
      printfQuda("TIME_REPORT - Done: Two-point function for Mesons and Baryons for source-position = %d written in ASCII format in %f sec.\n",isource,t2-t1);
    }
    else if( CorrFileFormat==HDF5_FORM ){
      t1 = MPI_Wtime();
      K_contract->copyTwopBaryonsToHDF5_Buf((void*)Twop_baryons_HDF5, (void*)corrBaryons, isource, CorrSpace);
      K_contract->copyTwopMesonsToHDF5_Buf ((void*)Twop_mesons_HDF5 , (void*)corrMesons, CorrSpace);
      t2 = MPI_Wtime();
      printfQuda("TIME_REPORT - Two-point function for baryons and mesons copied to HDF5 write buffers in %f sec.\n",t2-t1);
      
      t1 = MPI_Wtime();
      K_contract->writeTwopBaryonsHDF5((void*) Twop_baryons_HDF5, filename_baryons, info, isource);
      K_contract->writeTwopMesonsHDF5 ((void*) Twop_mesons_HDF5 , filename_mesons , info, isource);
      t2 = MPI_Wtime();
      printfQuda("TIME_REPORT - Done: Two-point function for Baryons and Mesons for source-position = %d written in HDF5 format in %f sec.\n",isource,t2-t1);
    }

    t4 = MPI_Wtime();
    printfQuda("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d Completed in %f sec. ###\n",isource,
	       info.sourcePosition[isource][0],info.sourcePosition[isource][1],info.sourcePosition[isource][2],info.sourcePosition[isource][3],t4-t3);
  } // close loop over source positions

  printfQuda("\nCleaning up...\n");
  free(corrThp_local);
  free(corrThp_noether);
  free(corrThp_oneD);
  free(corrMesons);
  free(corrBaryons);

  if( CorrFileFormat==HDF5_FORM ){
    free(Thrp_local_HDF5);
    free(Thrp_noether_HDF5);
    for(int mu=0;mu<4;mu++) free(Thrp_oneD_HDF5[mu]);
    free(Thrp_oneD_HDF5);
    free(Twop_baryons_HDF5);
    free(Twop_mesons_HDF5);
  }

  free(input_vector);
  free(output_vector);
  delete K_temp;
  delete K_contract;
  delete K_prop_down;
  delete K_prop_up;
  delete solve;
  delete d;
  delete dSloppy;
  delete dPre;
  delete K_guess;
  delete K_vector;
  delete K_gaugeSmeared;
  delete deflation_up;
  delete deflation_down;
  delete h_x;
  delete h_b;
  delete x;
  delete b;
  delete K_gaugeContractions;
  delete K_seqProp;
  delete K_prop3D_up;
  delete K_prop3D_down;

  printfQuda("...Done\n");

  popVerbosity();
  saveTuneCache();
  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);

}

void calcEigenVectors_loop_wOneD_2pt3pt_EvenOdd(void **gaugeSmeared, void **gauge, QudaGaugeParam *gauge_param, QudaInvertParam *param, void **gauge_loop, QudaGaugeParam *gauge_param_loop, QudaInvertParam *param_loop,
						qudaQKXTM_arpackInfo arpackInfo, qudaQKXTM_loopInfo loopInfo, qudaQKXTMinfo_Kepler info, char *filename_twop, char *filename_threep, WHICHPARTICLE NUCLEON, WHICHPROJECTOR PID ){

  bool flag_eo;
  double t1,t2;

  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);
  if( (arpackInfo.isFullOp) || (param->solve_type != QUDA_NORMOP_PC_SOLVE) ) errorQuda("This function works only with even-odd preconditioning\n");
  if(param->inv_type != QUDA_CG_INVERTER) errorQuda("This function works only with CG method");
  if( (param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) && (param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) ) errorQuda("Only asymmetric operators are supported in deflation\n");

  if( arpackInfo.isEven && (param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) ){
    errorQuda("Inconsistency between operator types!");
  }
  if( (!arpackInfo.isEven) && (param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) ){
    errorQuda("Inconsistency between operator types!");
  }

  if( param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC )
    flag_eo = true;
  else if(param->matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
    flag_eo = false;


  void *input_vector = malloc(GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]*spinorSiteSize*sizeof(double));
  void *output_vector = malloc(GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]*spinorSiteSize*sizeof(double));
  QKXTM_Gauge_Kepler<double> *K_gaugeSmeared = new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  QKXTM_Gauge_Kepler<float> *K_gaugeContractions = new QKXTM_Gauge_Kepler<float>(BOTH,GAUGE);

  QKXTM_Vector_Kepler<double> *K_vector = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<double> *K_guess = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<float> *K_temp = new QKXTM_Vector_Kepler<float>(BOTH,VECTOR);


  QKXTM_Propagator_Kepler<float> *K_prop_up = new QKXTM_Propagator_Kepler<float>(BOTH,PROPAGATOR);
  QKXTM_Propagator_Kepler<float> *K_prop_down = new QKXTM_Propagator_Kepler<float>(BOTH,PROPAGATOR);  
  QKXTM_Propagator_Kepler<float> *K_seqProp = new QKXTM_Propagator_Kepler<float>(BOTH,PROPAGATOR);

  QKXTM_Propagator3D_Kepler<float> *K_prop3D_up = new QKXTM_Propagator3D_Kepler<float>(BOTH,PROPAGATOR3D);
  QKXTM_Propagator3D_Kepler<float> *K_prop3D_down = new QKXTM_Propagator3D_Kepler<float>(BOTH,PROPAGATOR3D);

  QKXTM_Contraction_Kepler<float> *K_contract = new QKXTM_Contraction_Kepler<float>();
  printfQuda("Memory allocation was successfull\n");

  if(param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS) errorQuda("This function works only with ukqcd gamma basis\n");
  if(param->dirac_order != QUDA_DIRAC_ORDER) errorQuda("This function works only with colors inside the spins\n");

  if (!initialized) errorQuda("QUDA not initialized");
  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  //- Calculate the eigenVectors of the +mu
  QKXTM_Deflation_Kepler<double> *deflation_up = new QKXTM_Deflation_Kepler<double>(param,arpackInfo);
  deflation_up->printInfo();

  t1 = MPI_Wtime();
  deflation_up->eigenSolver();
  t2 = MPI_Wtime();
  printfQuda("TIME REPORT: EigenVector +mu Calculation: %f sec\n",t2-t1);
  //----------------------------------------------------

  //- Calculate the eigenVectors of the -mu
  param->twist_flavor = QUDA_TWIST_MINUS;
  QKXTM_Deflation_Kepler<double> *deflation_down = new QKXTM_Deflation_Kepler<double>(param,arpackInfo);
  deflation_down->printInfo();

  t1 = MPI_Wtime();
  deflation_down->eigenSolver();
  t2 = MPI_Wtime();
  printfQuda("TIME REPORT: EigenVector -mu Calculation: %f sec\n",t2-t1);
  param->twist_flavor = QUDA_TWIST_PLUS;
  //---------------------------------------------------- 


  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);

  K_gaugeContractions->packGauge(gauge);
  K_gaugeContractions->loadGauge();
  //  K_gaugeContractions->calculate(); do not do it because I changed the sign due to antiperiodic boundary conditions

  K_gaugeSmeared->packGauge(gaugeSmeared);
  K_gaugeSmeared->loadGauge();
  K_gaugeSmeared->calculatePlaq();

  bool pc_solution = false;
  bool pc_solve = true;
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) || (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = false;

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;
  
  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;
  // create the dirac operator                                                                                                                                      
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;
  profileInvert.TPSTART(QUDA_PROFILE_H2D);

  //QKXTM: DMH rewite for spinor field memalloc
  ColorSpinorField *b = NULL;
  ColorSpinorField *x = NULL;
  ColorSpinorField *in = NULL;
  ColorSpinorField *out = NULL;
  const int *X = cudaGauge->X();

  memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  memset(output_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  // wrap CPU host side pointers
  ColorSpinorParam cpuParam(input_vector, *param, X, pc_solution, param->input_location);
  ColorSpinorField *h_b = ColorSpinorField::Create(cpuParam);

  cpuParam.v = output_vector;
  cpuParam.location = param->output_location;
  ColorSpinorField *h_x = ColorSpinorField::Create(cpuParam);

  //Zero out the solution
  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  x = new cudaColorSpinorField(cudaParam);

  profileInvert.TPSTOP(QUDA_PROFILE_H2D);
  setTuning(param->tune);
  
  blas::zero(*x);
  blas::zero(*b);
  DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
  SolverParam solverParam(*param);
  Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);

  int my_src[4];
  char filename_mesons[257];
  char filename_baryons[257];

  for(int isource = 0 ; isource < info.Nsources ; isource++){

     sprintf(filename_mesons,"%s.mesons.SS.%02d.%02d.%02d.%02d.dat",filename_twop,info.sourcePosition[isource][0],info.sourcePosition[isource][1],info.sourcePosition[isource][2],info.sourcePosition[isource][3]);
     sprintf(filename_baryons,"%s.baryons.SS.%02d.%02d.%02d.%02d.dat",filename_twop,info.sourcePosition[isource][0],info.sourcePosition[isource][1],info.sourcePosition[isource][2],info.sourcePosition[isource][3]);

//      bool checkMesons, checkBaryons;
//      checkMesons = exists_file(filename_mesons);
//      checkBaryons = exists_file(filename_baryons);
//      if( (checkMesons == true) && (checkBaryons == true) ) continue; // because threep are written before twop if I checked twop I know that threep are fine

    for(int isc = 0 ; isc < 12 ; isc++){
      ///////////////////////////////////////////////////////////////////////////////// forward prop for up quark ///////////////////////////
      t1 = MPI_Wtime();
      memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
      b->changeTwist(QUDA_TWIST_PLUS);
      x->changeTwist(QUDA_TWIST_PLUS);
      b->Even().changeTwist(QUDA_TWIST_PLUS);
      b->Odd().changeTwist(QUDA_TWIST_PLUS);
      x->Even().changeTwist(QUDA_TWIST_PLUS);
      x->Odd().changeTwist(QUDA_TWIST_PLUS);
      for(int i = 0 ; i < 4 ; i++)
	my_src[i] = info.sourcePosition[isource][i] - comm_coords(default_topo)[i] * X[i];

      if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
	*( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + isc*2 ) = 1.;

      K_vector->packVector((double*) input_vector);
      K_vector->loadVector();
      K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
      K_guess->uploadToCuda(b,flag_eo);
      dirac.prepare(in,out,*x,*b,param->solution_type);
      // in is reference to the b but for a parity sinlet
      // out is reference to the x but for a parity sinlet
      cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);
      dirac.Mdag(*in, *tmp_up);
      delete tmp_up;
      // now the the source vector b is ready to perform deflation and find the initial guess
      K_vector->downloadFromCuda(in,flag_eo);
      K_vector->download();
      deflation_up->deflateVector(*K_guess,*K_vector);
      K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
      //  blas::zero(*out); // remove it later , just for test
      (*solve)(*out,*in);
      dirac.reconstruct(*x,*b,param->solution_type);
      K_vector->downloadFromCuda(x,flag_eo);
      if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	K_vector->scaleVector(2*param->kappa);
      }

      K_temp->castDoubleToFloat(*K_vector);
      K_prop_up->absorbVectorToDevice(*K_temp,isc/3,isc%3);
      t2 = MPI_Wtime();
      printfQuda("Inversion up = %d,  for source = %d finished in time %f sec\n",isc,isource,t2-t1);
      //////////////////////////////////////////////////////////
      ////////////////////////////////////////////////////////////////////////////// Forward prop for down quark ///////////////////////////////////
      /////////////////////////////////////////////////////////
      t1 = MPI_Wtime();
      memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
      b->changeTwist(QUDA_TWIST_MINUS);
      x->changeTwist(QUDA_TWIST_MINUS);
      b->Even().changeTwist(QUDA_TWIST_MINUS);
      b->Odd().changeTwist(QUDA_TWIST_MINUS);
      x->Even().changeTwist(QUDA_TWIST_MINUS);
      x->Odd().changeTwist(QUDA_TWIST_MINUS);
      for(int i = 0 ; i < 4 ; i++)
	my_src[i] = info.sourcePosition[isource][i] - comm_coords(default_topo)[i] * X[i];

      if( (my_src[0]>=0) && (my_src[0]<X[0]) && (my_src[1]>=0) && (my_src[1]<X[1]) && (my_src[2]>=0) && (my_src[2]<X[2]) && (my_src[3]>=0) && (my_src[3]<X[3]))
	*( (double*)input_vector + my_src[3]*X[2]*X[1]*X[0]*24 + my_src[2]*X[1]*X[0]*24 + my_src[1]*X[0]*24 + my_src[0]*24 + isc*2 ) = 1.;

      K_vector->packVector((double*) input_vector);
      K_vector->loadVector();
      K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
      K_guess->uploadToCuda(b,flag_eo);
      dirac.prepare(in,out,*x,*b,param->solution_type);
      // in is reference to the b but for a parity sinlet
      // out is reference to the x but for a parity sinlet
      cudaColorSpinorField *tmp_down = new cudaColorSpinorField(*in);
      dirac.Mdag(*in, *tmp_down);
      delete tmp_down;
      // now the the source vector b is ready to perform deflation and find the initial guess
      K_vector->downloadFromCuda(in,flag_eo);
      K_vector->download();
      deflation_down->deflateVector(*K_guess,*K_vector);
      K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
      //  blas::zero(*out); // remove it later , just for test
      (*solve)(*out,*in);
      dirac.reconstruct(*x,*b,param->solution_type);
      K_vector->downloadFromCuda(x,flag_eo);
      if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	K_vector->scaleVector(2*param->kappa);
      }

      K_temp->castDoubleToFloat(*K_vector);
      K_prop_down->absorbVectorToDevice(*K_temp,isc/3,isc%3);
      t2 = MPI_Wtime();
      printfQuda("Inversion down = %d,  for source = %d finished in time %f sec\n",isc,isource,t2-t1);
    } // close loop over 12 spin-color


    if(info.run3pt_src[isource]){

      /////////////////////////////////// Smearing on the 3D propagators

      //-C.Kallidonis: Loop over the number of sink-source separations
      int my_fixSinkTime;
      char filename_threep_tsink[257];
      for(int its=0;its<info.Ntsink;its++){
	my_fixSinkTime = (info.tsinkSource[its] + info.sourcePosition[isource][3])%GK_totalL[3] - comm_coords(default_topo)[3] * X[3];
	sprintf(filename_threep_tsink,"%s_tsink%d",filename_threep,info.tsinkSource[its]);
	printfQuda("The three-point function base name is: %s\n",filename_threep_tsink);
      
	t1 = MPI_Wtime();
	K_temp->zero_device();
	checkCudaError();
	if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ){
	  K_prop3D_up->absorbTimeSlice(*K_prop_up,my_fixSinkTime);
	  K_prop3D_down->absorbTimeSlice(*K_prop_down,my_fixSinkTime);
	}
	comm_barrier();

	for(int nu = 0 ; nu < 4 ; nu++)
	  for(int c2 = 0 ; c2 < 3 ; c2++){
	    // up //
	    K_temp->zero_device();
	    if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_temp->copyPropagator3D(*K_prop3D_up,my_fixSinkTime,nu,c2);
	    comm_barrier();
	    K_vector->castFloatToDouble(*K_temp);
	    K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	    K_temp->castDoubleToFloat(*K_guess);
	    if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_prop3D_up->absorbVectorTimeSlice(*K_temp,my_fixSinkTime,nu,c2);
	    comm_barrier();
	    K_temp->zero_device();

	    // down //
	    if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_temp->copyPropagator3D(*K_prop3D_down,my_fixSinkTime,nu,c2);
	    comm_barrier();
	    K_vector->castFloatToDouble(*K_temp);
	    K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	    K_temp->castDoubleToFloat(*K_guess);
	    if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_prop3D_down->absorbVectorTimeSlice(*K_temp,my_fixSinkTime,nu,c2);
	    comm_barrier();
	    K_temp->zero_device();	
	  }
	t2 = MPI_Wtime();
	printfQuda("Time needed to prepare the 3D props for sink-source[%d]=%d is %f sec\n",its,info.tsinkSource[its],t2-t1);

	/////////////////////////////////////////sequential propagator for the part 1
	for(int nu = 0 ; nu < 4 ; nu++)
	  for(int c2 = 0 ; c2 < 3 ; c2++){
	    t1 = MPI_Wtime();
	    K_temp->zero_device();
	    if(NUCLEON == PROTON){
	      if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_contract->seqSourceFixSinkPart1(*K_temp,*K_prop3D_up, *K_prop3D_down, my_fixSinkTime, nu, c2, PID, NUCLEON);}
	    else if(NUCLEON == NEUTRON){
	      if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_contract->seqSourceFixSinkPart1(*K_temp,*K_prop3D_down, *K_prop3D_up, my_fixSinkTime, nu, c2, PID, NUCLEON);}
	    comm_barrier();
	    K_temp->conjugate();
	    K_temp->apply_gamma5();
	    K_vector->castFloatToDouble(*K_temp);
	    //
	    K_vector->scaleVector(1e+10);
	    //
	    K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	    if(NUCLEON == PROTON){
	      b->changeTwist(QUDA_TWIST_MINUS); x->changeTwist(QUDA_TWIST_MINUS); b->Even().changeTwist(QUDA_TWIST_MINUS);
	      b->Odd().changeTwist(QUDA_TWIST_MINUS); x->Even().changeTwist(QUDA_TWIST_MINUS); x->Odd().changeTwist(QUDA_TWIST_MINUS);
	    }
	    else{
	      b->changeTwist(QUDA_TWIST_PLUS); x->changeTwist(QUDA_TWIST_PLUS); b->Even().changeTwist(QUDA_TWIST_PLUS);
	      b->Odd().changeTwist(QUDA_TWIST_PLUS); x->Even().changeTwist(QUDA_TWIST_PLUS); x->Odd().changeTwist(QUDA_TWIST_PLUS);
	    }
	    K_guess->uploadToCuda(b,flag_eo);
	    dirac.prepare(in,out,*x,*b,param->solution_type);
	  
	    cudaColorSpinorField *tmp = new cudaColorSpinorField(*in);
	    dirac.Mdag(*in, *tmp);
	    delete tmp;
	    K_vector->downloadFromCuda(in,flag_eo);
	    K_vector->download();
	    if(NUCLEON == PROTON)
	      deflation_down->deflateVector(*K_guess,*K_vector);
	    else if(NUCLEON == NEUTRON)
	      deflation_up->deflateVector(*K_guess,*K_vector);
	    K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
	    (*solve)(*out,*in);
	    dirac.reconstruct(*x,*b,param->solution_type);
	    K_vector->downloadFromCuda(x,flag_eo);
	    if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	      K_vector->scaleVector(2*param->kappa);
	    }
	    //
	    K_vector->scaleVector(1e-10);
	    //
	    K_temp->castDoubleToFloat(*K_vector);
	    K_seqProp->absorbVectorToDevice(*K_temp,nu,c2);
	    t2 = MPI_Wtime();
	    printfQuda("Inversion for seq prop part 1 = %d,  for source = %d and sink-source = %d finished in time %f sec\n",nu*3+c2,isource,info.tsinkSource[its],t2-t1);
	  }

	////////////////// Contractions for part 1 ////////////////
	t1 = MPI_Wtime();
	if(NUCLEON == PROTON){
	  K_contract->contractFixSink(*K_seqProp, *K_prop_up, *K_gaugeContractions, PID, NUCLEON, 1, filename_threep_tsink, isource, info.tsinkSource[its]);
	}
	if(NUCLEON == NEUTRON){
	  K_contract->contractFixSink(*K_seqProp, *K_prop_down, *K_gaugeContractions, PID, NUCLEON, 1, filename_threep_tsink, isource, info.tsinkSource[its]);
	}                
	t2 = MPI_Wtime();
	printfQuda("Time for fix sink contractions for part 1 at sink-source = %d is %f sec\n",info.tsinkSource[its],t2-t1);
	/////////////////////////////////////////sequential propagator for the part 2
	for(int nu = 0 ; nu < 4 ; nu++)
	  for(int c2 = 0 ; c2 < 3 ; c2++){
	    t1 = MPI_Wtime();
	    K_temp->zero_device();
	    if(NUCLEON == PROTON){
	      if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_contract->seqSourceFixSinkPart2(*K_temp,*K_prop3D_up, my_fixSinkTime, nu, c2, PID, NUCLEON);}
	    else if(NUCLEON == NEUTRON){
	      if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) K_contract->seqSourceFixSinkPart2(*K_temp,*K_prop3D_down, my_fixSinkTime, nu, c2, PID, NUCLEON);}
	    comm_barrier();
	    K_temp->conjugate();
	    K_temp->apply_gamma5();
	    K_vector->castFloatToDouble(*K_temp);
	    //
	    K_vector->scaleVector(1e+10);
	    //
	    K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	    if(NUCLEON == PROTON){
	      b->changeTwist(QUDA_TWIST_PLUS); x->changeTwist(QUDA_TWIST_PLUS); b->Even().changeTwist(QUDA_TWIST_PLUS);
	      b->Odd().changeTwist(QUDA_TWIST_PLUS); x->Even().changeTwist(QUDA_TWIST_PLUS); x->Odd().changeTwist(QUDA_TWIST_PLUS);
	    }
	    else{
	      b->changeTwist(QUDA_TWIST_MINUS); x->changeTwist(QUDA_TWIST_MINUS); b->Even().changeTwist(QUDA_TWIST_MINUS);
	      b->Odd().changeTwist(QUDA_TWIST_MINUS); x->Even().changeTwist(QUDA_TWIST_MINUS); x->Odd().changeTwist(QUDA_TWIST_MINUS);
	    }
	    K_guess->uploadToCuda(b,flag_eo);
	    dirac.prepare(in,out,*x,*b,param->solution_type);
	  
	    cudaColorSpinorField *tmp = new cudaColorSpinorField(*in);
	    dirac.Mdag(*in, *tmp);
	    delete tmp;
	    K_vector->downloadFromCuda(in,flag_eo);
	    K_vector->download();
	    if(NUCLEON == PROTON)
	      deflation_up->deflateVector(*K_guess,*K_vector);
	    else if(NUCLEON == NEUTRON)
	      deflation_down->deflateVector(*K_guess,*K_vector);
	    K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
	    (*solve)(*out,*in);
	    dirac.reconstruct(*x,*b,param->solution_type);
	    K_vector->downloadFromCuda(x,flag_eo);
	    if (param->mass_normalization == QUDA_MASS_NORMALIZATION || param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	      K_vector->scaleVector(2*param->kappa);
	    }
	    //
	    K_vector->scaleVector(1e-10);
	    //
	    K_temp->castDoubleToFloat(*K_vector);
	    K_seqProp->absorbVectorToDevice(*K_temp,nu,c2);
	    t2 = MPI_Wtime();
	    printfQuda("Inversion for seq prop part 2 = %d,  for source = %d and sink-source = %d finished in time %f sec\n",nu*3+c2,isource,info.tsinkSource[its],t2-t1);
	  }

	////////////////// Contractions for part 2 ////////////////
	t1 = MPI_Wtime();
	if(NUCLEON == PROTON)
	  K_contract->contractFixSink(*K_seqProp, *K_prop_down, *K_gaugeContractions, PID, NUCLEON, 2, filename_threep_tsink, isource, info.tsinkSource[its]);
	if(NUCLEON == NEUTRON)
	  K_contract->contractFixSink(*K_seqProp, *K_prop_up, *K_gaugeContractions, PID, NUCLEON, 2, filename_threep_tsink, isource, info.tsinkSource[its]);
	t2 = MPI_Wtime();

	printfQuda("Time for fix sink contractions for part 2 at sink-source = %d is %f sec\n",info.tsinkSource[its],t2-t1);

      }//-loop over sink-source separations      

    }
    ////////// At the very end ///////////////////////


    // smear the forward propagators
    for(int nu = 0 ; nu < 4 ; nu++)
      for(int c2 = 0 ; c2 < 3 ; c2++){
	K_temp->copyPropagator(*K_prop_up,nu,c2);
	K_vector->castFloatToDouble(*K_temp);
	K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	K_temp->castDoubleToFloat(*K_guess);
	K_prop_up->absorbVectorToDevice(*K_temp,nu,c2);
	
	K_temp->copyPropagator(*K_prop_down,nu,c2);
	K_vector->castFloatToDouble(*K_temp);
	K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	K_temp->castDoubleToFloat(*K_guess);
	K_prop_down->absorbVectorToDevice(*K_temp,nu,c2);
      }
    /////
    K_prop_up->rotateToPhysicalBase_device(+1);
    K_prop_down->rotateToPhysicalBase_device(-1);
    t1 = MPI_Wtime();
    K_contract->contractMesons(*K_prop_up,*K_prop_down,filename_mesons,isource);
    K_contract->contractBaryons(*K_prop_up,*K_prop_down,filename_baryons,isource);
    t2 = MPI_Wtime();
    printfQuda("Contractions for source = %d finished in time %f sec\n",isource,t2-t1);
  } // close loop over source positions


  free(input_vector);
  free(output_vector);
  delete deflation_up;
  delete K_temp;
  delete K_contract;
  delete K_prop_down;
  delete K_prop_up;
  delete solve;
  delete d;
  delete dSloppy;
  delete dPre;
  delete K_guess;
  delete K_vector;
  delete K_gaugeSmeared;
  delete h_x;
  delete h_b;
  delete x;
  delete b;
  delete K_gaugeContractions;
  delete K_seqProp;
  delete K_prop3D_up;
  delete K_prop3D_down;

  printfQuda("\n\n###################\n");
  printfQuda("### TWO-POINT AND THREE-POINT FUNCTION CALCULATION COMPLETED SUCCESSFULLY ###\n");
  printfQuda("###################\n\n");

  //=================== L O O P ===================//
  printfQuda("\n###################\n");
  printfQuda("### LOOP CALCULATION FOLLOWS ###\n");

//   char filename2[512];
//   sprintf(filename2,"h_elem_before");
//   deflation_down->writeEigenVectors_ASCII(filename2);

  calcEigenVectors_loop_wOneD_EvenOdd_noDefl(deflation_down->H_elem(), deflation_down->EigenValues(), gauge_loop, param_loop, gauge_param_loop, arpackInfo, loopInfo, info);

  printfQuda("### LOOP CALCULATION COMPLETED SUCCESSFULLY ###\n\n");

  delete deflation_down;

  popVerbosity();
  saveTuneCache();
  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);

}


void calcEigenVectors_loop_wOneD_EvenOdd_noDefl(double *eigVecs_d, double *eigVals_d, void **gaugeToPlaquette, QudaInvertParam *param ,QudaGaugeParam *gauge_param,  qudaQKXTM_arpackInfo arpackInfo, qudaQKXTM_loopInfo loopInfo, qudaQKXTMinfo_Kepler info){

  bool flag_eo;
  double t1,t2;

  if( (arpackInfo.isFullOp) || (param->solve_type != QUDA_NORMOP_PC_SOLVE) ) errorQuda("calcEigenVectors_loop_wOneD: This function works only with even-odd preconditioning\n");
  
  if(param->inv_type != QUDA_CG_INVERTER) errorQuda("calcEigenVectors_loop_wOneD_EvenOdd: This function works only with CG method");
  
  if(param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS) errorQuda("calcEigenVectors_loop_wOneD_EvenOdd: This function works only with ukqcd gamma basis\n");
  if(param->dirac_order != QUDA_DIRAC_ORDER) errorQuda("calcEigenVectors_loop_wOneD_EvenOdd: This function works only with color-inside-spin\n");
  
  if( (param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) && (param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) ) errorQuda("Only asymmetric operators are supported in deflation\n");

  if( arpackInfo.isEven && (param->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) ){
    errorQuda("calcEigenVectors_loop_wOneD_EvenOdd: Inconsistency between operator types!");
  }
  if( (!arpackInfo.isEven) && (param->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) ){
    errorQuda("calcEigenVectors_loop_wOneD_EvenOdd: Inconsistency between operator types!");
  }

  if(arpackInfo.isEven){
    printfQuda("calcEigenVectors_loop_wOneD_EvenOdd: Solving for the Even-Even operator\n");
    flag_eo = true;
  }
  else{
    printfQuda("calcEigenVectors_loop_wOneD_EvenOdd: Solving for the Odd-Odd operator\n");
    flag_eo = false;
  }

  if (!initialized) errorQuda("calcEigenVectors_loop_wOneD_EvenOdd: QUDA not initialized");
  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);
  
  int Nstoch = loopInfo.Nstoch;
  unsigned long int seed = loopInfo.seed;
  int NdumpStep = loopInfo.Ndump;

  char filename_out[512];
  strcpy(filename_out,loopInfo.loop_fname);

  printfQuda("Loop Calculation Info\n");
  printfQuda("=====================\n");
  printfQuda("No. of noise vectors: %d\n",Nstoch);
  printfQuda("The seed is: %ld\n",seed);
  printfQuda("Will dump every %d noise vectors\n",NdumpStep);
  printfQuda("The loop base name is %s\n",filename_out);
  printfQuda("=====================\n");

  //- Calculate the eigenVectors
  param->twist_flavor = QUDA_TWIST_MINUS;  
  QKXTM_Deflation_Kepler<double> *deflation = new QKXTM_Deflation_Kepler<double>(param,arpackInfo);
  deflation->printInfo();

  deflation->copyToEigenVector(eigVecs_d,eigVals_d);

//   char filename2[512];
//   sprintf(filename2,"h_elem_after");
//   deflation->writeEigenVectors_ASCII(filename2);

  //  t1 = MPI_Wtime();
  //  deflation->eigenSolver();
  //  t2 = MPI_Wtime();
  //  printfQuda("calcEigenVectors_loop_wOneD_EvenOdd TIME REPORT: EigenVector Calculation: %f sec\n",t2-t1);
  //--------------------------------------------------------------

  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);

  QKXTM_Gauge_Kepler<double> *K_gauge = new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  K_gauge->packGauge(gaugeToPlaquette);
  K_gauge->loadGauge();
  K_gauge->calculatePlaq();

  //  bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) ||
  //  (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

  bool pc_solution = false;
  bool pc_solve = true;
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) || (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = false;

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;
  
  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;
  // create the dirac operator                                                                                                                                      
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;
  profileInvert.TPSTART(QUDA_PROFILE_H2D);

  ColorSpinorField *b = NULL;
  ColorSpinorField *x = NULL;
  ColorSpinorField *in = NULL;
  ColorSpinorField *out = NULL;
  ColorSpinorField *tmp3 = NULL;
  ColorSpinorField *tmp4 = NULL;
  const int *X = cudaGauge->X();

  void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  void *output_vector = malloc(X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  memset(output_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  // wrap CPU host side pointers
  ColorSpinorParam cpuParam(input_vector, *param, X, pc_solution, param->input_location);
  ColorSpinorField *h_b = ColorSpinorField::Create(cpuParam);

  cpuParam.v = output_vector;
  cpuParam.location = param->output_location;
  ColorSpinorField *h_x = ColorSpinorField::Create(cpuParam);

  //Zero out the solution
  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  x = new cudaColorSpinorField(cudaParam);


  tmp3 = new cudaColorSpinorField(cudaParam);
  tmp4 = new cudaColorSpinorField(cudaParam);

  profileInvert.TPSTOP(QUDA_PROFILE_H2D);
  setTuning(param->tune);
  
  blas::zero(*x);
  blas::zero(*b);
  DiracMdagM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
  SolverParam solverParam(*param);
  Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, profileInvert);
  QKXTM_Vector_Kepler<double> *K_vector = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<double> *K_guess = new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);

  ////////////////////////// Allocate memory for local
  void    *cnRes_vv;
  void    *cnRes_gv;
  void    *cnTmp;

  if((cudaHostAlloc(&cnRes_vv, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("Error allocating memory cnRes_vv\n");
  if((cudaHostAlloc(&cnRes_gv, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("Error allocating memory cnRes_gv\n");

  cudaMemset      (cnRes_vv, 0, sizeof(double)*2*16*GK_localVolume);
  cudaMemset      (cnRes_gv, 0, sizeof(double)*2*16*GK_localVolume);

  if((cudaHostAlloc(&cnTmp, sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("Error allocating memory cnTmp\n");

  cudaMemset      (cnTmp, 0, sizeof(double)*2*16*GK_localVolume);
  ///////////////////////////////////////////////////
  //////////// Allocate memory for one-Der and conserved current
  void    **cnD_vv;
  void    **cnD_gv;
  void    **cnC_vv;
  void    **cnC_gv;

  cnD_vv   = (void**) malloc(sizeof(double*)*2*4);
  cnD_gv   = (void**) malloc(sizeof(double*)*2*4);
  cnC_vv   = (void**) malloc(sizeof(double*)*2*4);
  cnC_gv   = (void**) malloc(sizeof(double*)*2*4);

  if(cnD_gv == NULL)errorQuda("Error allocating memory cnD_gv higher level\n");
  if(cnD_vv == NULL)errorQuda("Error allocating memory cnD_vv higher level\n");
  if(cnC_gv == NULL)errorQuda("Error allocating memory cnC_gv higher level\n");
  if(cnC_vv == NULL)errorQuda("Error allocating memory cnC_vv higher level\n");
  cudaDeviceSynchronize();

  for(int mu = 0; mu < 4 ; mu++){
    if((cudaHostAlloc(&(cnD_vv[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("Error allocating memory cnD_vv\n");
    if((cudaHostAlloc(&(cnD_gv[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("Error allocating memory cnD_gv\n");
    if((cudaHostAlloc(&(cnC_vv[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("Error allocating memory cnC_vv\n");
    if((cudaHostAlloc(&(cnC_gv[mu]), sizeof(double)*2*16*GK_localVolume, cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("Error allocating memory cnC_gv\n");
  }
  cudaDeviceSynchronize();

  printfQuda("Memory for loops allocated properly\n");
  ///////////////////////////////////////////////////
  gsl_rng *rNum = gsl_rng_alloc(gsl_rng_ranlux);
  gsl_rng_set(rNum, seed + comm_rank()*seed);

  if(info.source_type==RANDOM) printfQuda("Will use RANDOM stochastic sources\n");
  else if (info.source_type==UNITY) printfQuda("Will use UNITY stochastic sources\n");

  for(int is = 0 ; is < Nstoch ; is++){
    t1 = MPI_Wtime();

    memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
    getStochasticRandomSource<double>(input_vector,rNum,info.source_type);
    K_vector->packVector((double*) input_vector);
    K_vector->loadVector();
    K_vector->uploadToCuda(b,flag_eo);
    dirac.prepare(in,out,*x,*b,param->solution_type);
    // in is reference to the b but for a parity sinlet
    // out is reference to the x but for a parity sinlet
    cudaColorSpinorField *tmp_up = new cudaColorSpinorField(*in);
    dirac.Mdag(*in, *tmp_up);
    delete tmp_up;
    // now the the source vector b is ready to perform deflation and find the initial guess
    K_vector->downloadFromCuda(in,flag_eo);
    K_vector->download();
    deflation->deflateVector(*K_guess,*K_vector);
    K_guess->uploadToCuda(out,flag_eo); // initial guess is ready
    //  blas::zero(*out); // remove it later , just for test
    (*solve)(*out,*in);
    dirac.reconstruct(*x,*b,param->solution_type);
    oneEndTrick_w_One_Der<double>(*x,*tmp3,*tmp4,param,cnRes_gv,cnRes_vv,cnD_gv,cnD_vv,cnC_gv,cnC_vv);

    t2 = MPI_Wtime();
    printfQuda("TIME_REPORT: One-end trick for Stoch %04d is %f sec\n",is+1,t2-t1);

    if( (is+1)%NdumpStep == 0){
      doCudaFFT_v2<double>(cnRes_vv,cnTmp);
      dumpLoop_ultraLocal<double>(cnTmp,filename_out,is+1,info.Q_sq_loop,0); // Scalar
      doCudaFFT_v2<double>(cnRes_gv,cnTmp);
      dumpLoop_ultraLocal<double>(cnTmp,filename_out,is+1,info.Q_sq_loop,1); // dOp
      for(int mu = 0 ; mu < 4 ; mu++){
	doCudaFFT_v2<double>(cnD_vv[mu],cnTmp);
	dumpLoop_oneD<double>(cnTmp,filename_out,is+1,info.Q_sq_loop,mu,0); // Loops
	doCudaFFT_v2<double>(cnD_gv[mu],cnTmp);
	dumpLoop_oneD<double>(cnTmp,filename_out,is+1,info.Q_sq_loop,mu,1); // LpsDw

	doCudaFFT_v2<double>(cnC_vv[mu],cnTmp);
	dumpLoop_oneD<double>(cnTmp,filename_out,is+1,info.Q_sq_loop,mu,2); // LpsDw noether
	doCudaFFT_v2<double>(cnC_gv[mu],cnTmp);
	dumpLoop_oneD<double>(cnTmp,filename_out,is+1,info.Q_sq_loop,mu,3); // LpsDw noether
      }
    } // close loop for dump loops
  } // close loop over stochastic noise vectors

  cudaFreeHost(cnRes_gv);
  cudaFreeHost(cnRes_vv);

  cudaFreeHost(cnTmp);

  for(int mu = 0 ; mu < 4 ; mu++){
    cudaFreeHost(cnD_vv[mu]);
    cudaFreeHost(cnD_gv[mu]);
    cudaFreeHost(cnC_vv[mu]);
    cudaFreeHost(cnC_gv[mu]);
  }
  
  free(cnD_vv);
  free(cnD_gv);
  free(cnC_vv);
  free(cnC_gv);

  free(input_vector);
  free(output_vector);
  gsl_rng_free(rNum);
  delete solve;
  delete d;
  delete dSloppy;
  delete dPre;
  delete K_guess;
  delete K_vector;
  delete K_gauge;
  delete deflation;
  delete h_x;
  delete h_b;
  delete x;
  delete b;
  delete tmp3;
  delete tmp4;

}



void calcEigenVectors_threepTwop_FullOp(void **gaugeSmeared, void **gauge, 
					QudaGaugeParam *gauge_param, 
					QudaInvertParam *param, 
					qudaQKXTM_arpackInfo arpackInfo, 
					qudaQKXTMinfo_Kepler info,
					char *filename_twop, 
					char *filename_threep, 
					WHICHPARTICLE NUCLEON, 
					WHICHPROJECTOR PID ){
  double t1,t2;

  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);
  //if(param->solve_type != QUDA_NORMOP_PC_SOLVE) errorQuda("This function works only with even odd preconditioning");
  if(param->inv_type != QUDA_CG_INVERTER) errorQuda("This function works only with CG method");
  if( !arpackInfo.isFullOp ) errorQuda("This function works only with the Full Operator\n");
  printfQuda("Solving for the FULL operator\n");

  if( param->solve_type != QUDA_NORMOP_SOLVE ) errorQuda("This function is intended to solve for the Full Operator\n");
  else printfQuda("Solving for the FULL operator\n");

  bool pc_solve = !arpackInfo.isFullOp;
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) || (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = false;

  QKXTM_Deflation_Kepler<double> *deflation_up = 
    new QKXTM_Deflation_Kepler<double>(param,arpackInfo);

  //  param.mu *= -1.0;
  //  QKXTM_Deflation_Kepler<double> *deflation_down = new QKXTM_Deflation_Kepler<double>(param,arpackInfo);
  //  param.mu *= -1.0;

  QKXTM_Gauge_Kepler<double> *K_gaugeSmeared = 
    new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  QKXTM_Gauge_Kepler<float> *K_gaugeContractions = 
    new QKXTM_Gauge_Kepler<float>(BOTH,GAUGE);

  QKXTM_Vector_Kepler<double> *K_vector = 
    new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<double> *K_guess = 
    new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<float> *K_temp = 
    new QKXTM_Vector_Kepler<float>(BOTH,VECTOR);

  QKXTM_Propagator_Kepler<float> *K_prop_up = 
    new QKXTM_Propagator_Kepler<float>(BOTH,PROPAGATOR);
  QKXTM_Propagator_Kepler<float> *K_prop_down = 
    new QKXTM_Propagator_Kepler<float>(BOTH,PROPAGATOR);  
  QKXTM_Propagator_Kepler<float> *K_seqProp = 
    new QKXTM_Propagator_Kepler<float>(BOTH,PROPAGATOR);

  QKXTM_Propagator3D_Kepler<float> *K_prop3D_up = 
    new QKXTM_Propagator3D_Kepler<float>(BOTH,PROPAGATOR3D);
  QKXTM_Propagator3D_Kepler<float> *K_prop3D_down = 
    new QKXTM_Propagator3D_Kepler<float>(BOTH,PROPAGATOR3D);

  QKXTM_Contraction_Kepler<float> *K_contract = 
    new QKXTM_Contraction_Kepler<float>();

  printfQuda("Memory allocation was successfull\n");

  if(param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS) 
    errorQuda("This function works only with ukqcd gamma basis\n");
  if(param->dirac_order != QUDA_DIRAC_ORDER) 
    errorQuda("This function works only with colors inside the spins\n");

  //-Calculate the eigenvectors for the +mu
  deflation_up->printInfo();
  t1 = MPI_Wtime();
  deflation_up->eigenSolver();
  t2 = MPI_Wtime();
  printfQuda("TIME_REPORT: ARPACK for +mu:  %f sec\n",t2-t1);
  //-----------------------------------------------------------------

  //-Calculate the eigenvectors for the -mu
  //   deflation_down->printInfo();
  //   t1 = MPI_Wtime();
  //   deflation_down->eigenSolver();
  //   t2 = MPI_Wtime();
  //   printfQuda("TIME_REPORT: ARPACK for -mu:  %f sec\n",t2-t1);


  if (!initialized) errorQuda("QUDA not initialized");
  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);

  K_gaugeContractions->packGauge(gauge);
  K_gaugeContractions->loadGauge();

  K_gaugeSmeared->packGauge(gaugeSmeared);
  K_gaugeSmeared->loadGauge();
  K_gaugeSmeared->calculatePlaq();

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? 
		       sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 
			 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 
			 8 : 9)/(double)(1<<30);
  }
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;
  
  // create the dirac operator
  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;

  profileInvert.TPSTART(QUDA_PROFILE_H2D);

  ColorSpinorField *b = NULL;
  ColorSpinorField *x = NULL;
  ColorSpinorField *in = NULL;
  ColorSpinorField *out = NULL;

  void *input_vector  = malloc(GK_localL[0]*
			       GK_localL[1]*
			       GK_localL[2]*
			       GK_localL[3]*spinorSiteSize*sizeof(double));

  void *output_vector = malloc(GK_localL[0]*
			       GK_localL[1]*
			       GK_localL[2]*
			       GK_localL[3]*spinorSiteSize*sizeof(double));

  memset(input_vector,0,
	 GK_localL[0]*
	 GK_localL[1]*
	 GK_localL[2]*
	 GK_localL[3]*spinorSiteSize*sizeof(double));

  memset(output_vector,0,
	 GK_localL[0]*
	 GK_localL[1]*
	 GK_localL[2]*
	 GK_localL[3]*spinorSiteSize*sizeof(double));


  // wrap CPU host side pointers
  ColorSpinorParam cpuParam(input_vector, *param, GK_localL, pc_solve, param->input_location);
  ColorSpinorField *h_b = ColorSpinorField::Create(cpuParam);

  cpuParam.v = output_vector;
  cpuParam.location = param->output_location;
  ColorSpinorField *h_x = ColorSpinorField::Create(cpuParam);

  //Zero out the solution
  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_COPY_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  x = new cudaColorSpinorField(cudaParam);

  profileInvert.TPSTOP(QUDA_PROFILE_H2D);
  setTuning(param->tune);
  
  blas::zero(*x);
  blas::zero(*b);

  DiracMMdag m(dirac), mSloppy(diracSloppy), mPre(diracPre);
  SolverParam solverParam(*param);
  Solver *solve = Solver::create(solverParam,m,mSloppy,mPre,profileInvert);

  int my_src[4];
  char filename_mesons[257];
  char filename_baryons[257];

  for(int isource = 0 ; isource < info.Nsources ; isource++){

     sprintf(filename_mesons,"%s.mesons.SS.%02d.%02d.%02d.%02d.dat",
	     filename_twop,
	     info.sourcePosition[isource][0],
	     info.sourcePosition[isource][1],
	     info.sourcePosition[isource][2],
	     info.sourcePosition[isource][3]);

     sprintf(filename_baryons,"%s.baryons.SS.%02d.%02d.%02d.%02d.dat",
	     filename_twop,
	     info.sourcePosition[isource][0],
	     info.sourcePosition[isource][1],
	     info.sourcePosition[isource][2],
	     info.sourcePosition[isource][3]);
     
    for(int isc = 0 ; isc < 12 ; isc++){
      //// forward propagators for up AND down quarks ////
      t1 = MPI_Wtime();
      memset(input_vector,0,
	     GK_localL[0]*
	     GK_localL[1]*
	     GK_localL[2]*
	     GK_localL[3]*spinorSiteSize*sizeof(double));
      
      b->changeTwist(QUDA_TWIST_PLUS);
      x->changeTwist(QUDA_TWIST_PLUS);
      b->Even().changeTwist(QUDA_TWIST_PLUS);
      b->Odd().changeTwist(QUDA_TWIST_PLUS);
      x->Even().changeTwist(QUDA_TWIST_PLUS);
      x->Odd().changeTwist(QUDA_TWIST_PLUS);

      for(int i = 0 ; i < 4 ; i++)
	my_src[i] = info.sourcePosition[isource][i] - 
	  comm_coords(default_topo)[i] * GK_localL[i];

      if( (my_src[0]>=0) && 
	  (my_src[0]<GK_localL[0]) && 
	  (my_src[1]>=0) && 
	  (my_src[1]<GK_localL[1]) && 
	  (my_src[2]>=0) && 
	  (my_src[2]<GK_localL[2]) && 
	  (my_src[3]>=0) && 
	  (my_src[3]<GK_localL[3])) {
	*( (double*)input_vector + 
	   my_src[3]*GK_localL[2]*GK_localL[1]*GK_localL[0]*24 + 
	   my_src[2]*GK_localL[1]*GK_localL[0]*24 + 
	   my_src[1]*GK_localL[0]*24 + 
	   my_src[0]*24 + isc*2 ) = 1.0;
      }
      K_vector->packVector((double*) input_vector);
      K_vector->loadVector();
      K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
      K_guess->uploadToCuda(b,pc_solve);
      dirac.prepare(in,out,*x,*b,param->solution_type);
      // in is reference to the b but for a parity singlet
      // out is reference to the x but for a parity singlet
      // now the source vector b is ready to perform deflation and find 
      // the initial guess
      K_vector->downloadFromCuda(in,pc_solve);
      K_vector->download();
      deflation_up->deflateVector(*K_guess,*K_vector);
      K_guess->uploadToCuda(out,pc_solve); // initial guess is ready
      (*solve)(*out,*in);
      dirac.reconstruct(*x,*b,param->solution_type);

      cudaColorSpinorField *y = new cudaColorSpinorField(*x);

      //-Get the solution for the up flavor
      dirac.Mdag(*x, *y);

      K_vector->downloadFromCuda(x,pc_solve);
      if (param->mass_normalization == QUDA_MASS_NORMALIZATION || 
	  param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	K_vector->scaleVector(2*param->kappa);
      }

      K_temp->castDoubleToFloat(*K_vector);
      K_prop_up->absorbVectorToDevice(*K_temp,isc/3,isc%3);
      //----------------------------------

      //-Get the solution for the down flavor
      y->changeTwist(QUDA_TWIST_MINUS);
      x->changeTwist(QUDA_TWIST_MINUS);
      dirac.Mdag(*x, *y);

      K_vector->downloadFromCuda(x,pc_solve);
      if (param->mass_normalization == QUDA_MASS_NORMALIZATION || 
	  param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	K_vector->scaleVector(2*param->kappa);
      }
      K_temp->castDoubleToFloat(*K_vector);
      K_prop_down->absorbVectorToDevice(*K_temp,isc/3,isc%3);
      //----------------------------------

      t2 = MPI_Wtime();
      printfQuda("Forward Inversion up and down = %d,  for source = %d finished in time %f sec\n",isc,isource,t2-t1);
    } // close loop over 12 spin-color

    // smear the forward propagators
    for(int nu = 0 ; nu < 4 ; nu++)
      for(int c2 = 0 ; c2 < 3 ; c2++){
	K_temp->copyPropagator(*K_prop_up,nu,c2);
	K_vector->castFloatToDouble(*K_temp);
	K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	K_temp->castDoubleToFloat(*K_guess);
	K_prop_up->absorbVectorToDevice(*K_temp,nu,c2);
	
	K_temp->copyPropagator(*K_prop_down,nu,c2);
	K_vector->castFloatToDouble(*K_temp);
	K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	K_temp->castDoubleToFloat(*K_guess);
	K_prop_down->absorbVectorToDevice(*K_temp,nu,c2);
      }
    
    K_prop_up->rotateToPhysicalBase_device(+1);
    K_prop_down->rotateToPhysicalBase_device(-1);

    t1 = MPI_Wtime();
    K_contract->contractMesons(*K_prop_up,*K_prop_down,
			       filename_mesons,isource);
    K_contract->contractBaryons(*K_prop_up,*K_prop_down,
				filename_baryons,isource);
    t2 = MPI_Wtime();

    printfQuda("Contractions for source = %d finished in time %f sec\n",
	       isource,t2-t1);
  } // close loop over source positions
  

  free(input_vector);
  free(output_vector);
  delete K_temp;
  delete K_contract;
  delete K_prop_down;
  delete K_prop_up;
  delete solve;
  delete d;
  delete dSloppy;
  delete dPre;
  delete K_guess;
  delete K_vector;
  delete K_gaugeSmeared;
  delete deflation_up;
  //  delete deflation_down;
  delete h_x;
  delete h_b;
  delete x;
  delete b;
  delete K_gaugeContractions;
  delete K_seqProp;
  delete K_prop3D_up;
  delete K_prop3D_down;

  popVerbosity();
  saveTuneCache();
  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);

}


//-C.K. old way to dump the Loops in ASCII format

//       doCudaFFT_v2<double>(cnRes_vv,cnTmp);
//       dumpLoop_ultraLocal<double>(cnTmp,filename_out,is+1,info.Q_sq,0); // Scalar
//       doCudaFFT_v2<double>(cnRes_gv,cnTmp);
//       dumpLoop_ultraLocal<double>(cnTmp,filename_out,is+1,info.Q_sq,1); // dOp
//       for(int mu = 0 ; mu < 4 ; mu++){
// 	doCudaFFT_v2<double>(cnD_vv[mu],cnTmp);
// 	dumpLoop_oneD<double>(cnTmp,filename_out,is+1,info.Q_sq,mu,0); // Loops
// 	doCudaFFT_v2<double>(cnD_gv[mu],cnTmp);
// 	dumpLoop_oneD<double>(cnTmp,filename_out,is+1,info.Q_sq,mu,1); // LpsDw

// 	doCudaFFT_v2<double>(cnC_vv[mu],cnTmp);
// 	dumpLoop_oneD<double>(cnTmp,filename_out,is+1,info.Q_sq,mu,2); // Loops noether
// 	doCudaFFT_v2<double>(cnC_gv[mu],cnTmp);
// 	dumpLoop_oneD<double>(cnTmp,filename_out,is+1,info.Q_sq,mu,3); // LpsDw noether
//       }



//-C.K. old way to dump the exact part of the Loops in ASCII format

//       doCudaFFT_v2<double>(std_uloc,tmp_loop);
//       dumpLoop_ultraLocal_Exact<double>(tmp_loop,loop_exact_fname,info.Q_sq,0); // Std. Ultra-local - Scalar  
//       doCudaFFT_v2<double>(gen_uloc,tmp_loop);
//       dumpLoop_ultraLocal_Exact<double>(tmp_loop,loop_exact_fname,info.Q_sq,1); // Gen. Ultra-local - dOp
      
//       for(int mu = 0 ; mu < 4 ; mu++){
// 	doCudaFFT_v2<double>(std_oneD[mu],tmp_loop);
// 	dumpLoop_oneD_Exact<double>(tmp_loop,loop_exact_fname,info.Q_sq,mu,0); // Std. oneD - Loops    
// 	doCudaFFT_v2<double>(gen_oneD[mu],tmp_loop);
// 	dumpLoop_oneD_Exact<double>(tmp_loop,loop_exact_fname,info.Q_sq,mu,1); // Gen. oneD - LpsDw    
// 	doCudaFFT_v2<double>(std_csvC[mu],tmp_loop);
// 	dumpLoop_oneD_Exact<double>(tmp_loop,loop_exact_fname,info.Q_sq,mu,2); // Std. Conserved current - LoopsCv    
// 	doCudaFFT_v2<double>(gen_csvC[mu],tmp_loop);
// 	dumpLoop_oneD_Exact<double>(tmp_loop,loop_exact_fname,info.Q_sq,mu,3); // Gen. Conserved current - LpsDwCv
//       }



//-C.K. old way to dump the stochastic part of the Loops in ASCII format

// 	doCudaFFT_v2<double>(std_uloc,tmp_loop);
// 	dumpLoop_ultraLocal<double>(tmp_loop,loop_stoch_fname,is+1,info.Q_sq,0); // Std. Ultra-local - Scalar
// 	doCudaFFT_v2<double>(gen_uloc,tmp_loop);
// 	dumpLoop_ultraLocal<double>(tmp_loop,loop_stoch_fname,is+1,info.Q_sq,1); // Gen. Ultra-local - dOp
	
// 	for(int mu = 0 ; mu < 4 ; mu++){
// 	  doCudaFFT_v2<double>(std_oneD[mu],tmp_loop);
// 	  dumpLoop_oneD<double>(tmp_loop,loop_stoch_fname,is+1,info.Q_sq,mu,0); // Std. oneD - Loops
// 	  doCudaFFT_v2<double>(gen_oneD[mu],tmp_loop);
// 	  dumpLoop_oneD<double>(tmp_loop,loop_stoch_fname,is+1,info.Q_sq,mu,1); // Gen. oneD - LpsDw
// 	  doCudaFFT_v2<double>(std_csvC[mu],tmp_loop);
// 	  dumpLoop_oneD<double>(tmp_loop,loop_stoch_fname,is+1,info.Q_sq,mu,2); // Std. Conserved current - LoopsCv
// 	  doCudaFFT_v2<double>(gen_csvC[mu],tmp_loop);
// 	  dumpLoop_oneD<double>(tmp_loop,loop_stoch_fname,is+1,info.Q_sq,mu,3); // Gen. Conserved current - LpsDwCv
// 	}

