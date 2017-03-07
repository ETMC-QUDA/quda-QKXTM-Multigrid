#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <QKXTM_util.h>
#include <dslash_util.h>
#include <blas_reference.h>
#include <wilson_dslash_reference.h>
#include <domain_wall_dslash_reference.h>
#include "misc.h"

#include "face_quda.h"

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

#include <gauge_qio.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>
#include <contractQuda.h>
#include <qudaQKXTM_Kepler.h>

//========================================================================//
//====== P A R A M E T E R   S E T T I N G S   A N D   C H E C K S =======//
//========================================================================//

// Wilson, clover-improved Wilson, twisted mass, and domain wall are supported.
extern QudaDslashType dslash_type;
extern bool tune;
extern int device;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern int Lsdim;
extern int gridsize_from_cmdline[];
extern QudaReconstructType link_recon;
extern QudaPrecision prec;
extern QudaPrecision  prec_sloppy;
extern QudaPrecision  prec_precondition;
extern QudaReconstructType link_recon_sloppy;
extern QudaReconstructType link_recon_precondition;
extern double mass; // mass of Dirac operator
extern double mu;
//QKXTM: DMH Experimental MG additions
extern double delta_muPR;
extern double delta_kappaPR;
extern double delta_cswPR;
extern double delta_muCG;
extern double delta_kappaCG;
extern double delta_cswCG;

extern double anisotropy;
extern double tol; // tolerance for inverter
extern double tol_hq; // heavy-quark tolerance for inverter
extern QudaMassNormalization normalization; // mass normalization of Dirac operators

extern int niter;
extern int nvec[];
extern int mg_levels;

extern bool generate_nullspace;
extern bool generate_all_levels;
extern int nu_pre;
extern int nu_post;
extern int geo_block_size[QUDA_MAX_MG_LEVEL][QUDA_MAX_DIM];

extern QudaInverterType smoother_type;

extern QudaMatPCType matpc_type;
extern QudaSolveType solve_type;

extern char vec_infile[];
extern char vec_outfile[];

//Twisted mass flavor type
extern QudaTwistFlavorType twist_flavor;

extern void usage(char** );

extern double clover_coeff;
extern bool compute_clover;

//QKXTM: DMH extra inputs
//If using MG, these are fixed to be GCR and MG respectively.
//extern QudaInverterType  inv_type;
//extern QudaInverterType  precon_type;
extern int multishift; // whether to test multi-shift or standard solver


extern char latfile[];
extern char latfile_smeared[];
extern char verbosity_level[];
extern int traj;
extern bool isEven;
extern int Q_sq;
extern double kappa;
extern char prop_path[];
extern double csw;

//-C.K. Loop parameters
extern int Nstoch;
extern unsigned long int seed;
extern char loop_fname[];
extern char *loop_file_format;
extern int Ndump;
extern char source_type[];
extern char filename_dSteps[];
extern bool useTSM;
extern int TSM_NHP;
extern int TSM_NLP;
extern int TSM_NdumpHP;
extern int TSM_NdumpLP;
extern int TSM_maxiter;
extern double TSM_tol;

//-C.K. ARPACK Parameters
extern int PolyDeg;
extern int nEv;
extern int nKv;
extern char *spectrumPart;
extern bool isACC;
extern double tolArpack;
extern int maxIterArpack;
extern char arpack_logfile[];
extern double amin;
extern double amax;
extern bool isEven;
extern bool isFullOp;


namespace quda {
  extern void setTransferGPU(bool);
}

void display_test_info() {
  printfQuda("running the following test:\n");
    
  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension Ls_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d         %d\n",
	     get_prec_str(prec),get_prec_str(prec_sloppy),
	     get_recon_str(link_recon), 
	     get_recon_str(link_recon_sloppy),  xdim, ydim, zdim, tdim, Lsdim);     

  printfQuda("MG parameters\n");
  printfQuda(" - number of levels %d\n", mg_levels);
  for (int i=0; i<mg_levels-1; i++) printfQuda(" - level %d number of null-space vectors %d\n", i+1, nvec[i]);
  printfQuda(" - number of pre-smoother applications %d\n", nu_pre);
  printfQuda(" - number of post-smoother applications %d\n", nu_post);

  printfQuda("Grid partition info:     X  Y  Z  T\n"); 
  printfQuda("                         %d  %d  %d  %d\n", 
	     dimPartitioned(0),
	     dimPartitioned(1),
	     dimPartitioned(2),
	     dimPartitioned(3)); 
  
  return ;
  
}

QudaPrecision &cpu_prec = prec;
QudaPrecision &cuda_prec = prec;
QudaPrecision &cuda_prec_sloppy = prec_sloppy;
QudaPrecision &cuda_prec_precondition = prec_precondition;

void setGaugeParam(QudaGaugeParam &gauge_param) {

  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;

  gauge_param.anisotropy = anisotropy;
  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;
  
  gauge_param.cpu_prec = cpu_prec;
  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = link_recon;

  gauge_param.cuda_prec_sloppy = cuda_prec_sloppy;
  gauge_param.reconstruct_sloppy = link_recon_sloppy;

  gauge_param.cuda_prec_precondition = cuda_prec_precondition;
  gauge_param.reconstruct_precondition = link_recon_precondition;

  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  gauge_param.ga_pad = 0;
  // For multi-GPU, ga_pad must be large enough to store a time-slice
#ifdef MULTI_GPU
  int x_face_size = gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
  int y_face_size = gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
  int z_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
  int t_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
  int pad_size =MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;    
#endif
}

void setInvertParam(QudaInvertParam &inv_param) {

  inv_param.kappa = kappa;
  inv_param.mass = 0.5/kappa - 4.0;
  inv_param.epsilon = 0.0;

  inv_param.Ls = 1;

  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;

  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.gamma_basis = QUDA_UKQCD_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || 
      dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = cuda_prec_precondition;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    inv_param.clover_coeff = csw*inv_param.kappa;
  }

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.tune = tune ? QUDA_TUNE_YES : QUDA_TUNE_NO;

  inv_param.dslash_type = dslash_type;

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || 
      dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.mu = mu;
    inv_param.twist_flavor = twist_flavor;
    inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) ? 
      2 : 1;

    if (twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
      printfQuda("Twisted-mass doublet non supported (yet)\n");
      exit(0);
    }
  }

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = normalization;
  inv_param.solver_normalization = QUDA_DEFAULT_NORMALIZATION;

  inv_param.pipeline = 0;

  // offsets used only by multi-shift solver
  inv_param.num_offset = 4;
  double offset[4] = {0.01, 0.02, 0.03, 0.04};
  for (int i=0; i<inv_param.num_offset; i++) inv_param.offset[i] = offset[i];

  // do we want to use an even-odd preconditioned solve or not
  if(isEven) inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  else inv_param.matpc_type = QUDA_MATPC_ODD_ODD;
  
  // do we want full solution or single-parity solution
  inv_param.solution_type = QUDA_MAT_SOLUTION;

  // Using Even-Odd operator
  inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;  
  
  inv_param.inv_type = QUDA_GCR_INVERTER;
  inv_param.verbosity = QUDA_VERBOSE;
  inv_param.verbosity_precondition = QUDA_SILENT;

  inv_param.inv_type_precondition = QUDA_MG_INVERTER;

  inv_param.Nsteps = 20;
  inv_param.gcrNkrylov = 10;
  inv_param.tol = tol;
  inv_param.tol_restart = 1e-3;

  // require both L2 relative and heavy quark residual to determine 
  // convergence
  inv_param.residual_type = 
    static_cast<QudaResidualType>(QUDA_L2_RELATIVE_RESIDUAL);
  // specify a tolerance for the residual for heavy quark residual
  inv_param.tol_hq = tol_hq; 
  
  // these can be set individually
  for (int i=0; i<inv_param.num_offset; i++) {
    inv_param.tol_offset[i] = inv_param.tol;
    inv_param.tol_hq_offset[i] = inv_param.tol_hq;
  }
  inv_param.maxiter = niter;
  inv_param.reliable_delta = 1e-2;
  inv_param.use_sloppy_partial_accumulator = 0;
  inv_param.max_res_increase = 1;

  // domain decomposition preconditioner parameters
  inv_param.schwarz_type = QUDA_ADDITIVE_SCHWARZ;
  inv_param.precondition_cycle = 1;
  inv_param.tol_precondition = 1e-1;
  inv_param.maxiter_precondition = 10;
  inv_param.omega = 1.0;


  //if(strcmp(verbosity_level,"verbose")==0) 
  //inv_param.verbosity = QUDA_VERBOSE;
  //else if(strcmp(verbosity_level,"summarize")==0) 
  //inv_param.verbosity = QUDA_SUMMARIZE;
  //else if(strcmp(verbosity_level,"silent")==0) 
  //inv_param.verbosity = QUDA_SILENT;
  //else{
  //warningQuda("Unknown verbosity level %s. Proceeding with QUDA_SUMMARIZE verbosity level\n",verbosity_level);
  //inv_param.verbosity = QUDA_SUMMARIZE;
  //}
}

void setMultigridParam(QudaMultigridParam &mg_param) {
  QudaInvertParam &inv_param = *mg_param.invert_param;

  inv_param.kappa = kappa;
  inv_param.mass = 0.5/kappa - 4.0;

  inv_param.Ls = 1;

  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || 
      dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = cuda_prec_precondition;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    inv_param.clover_coeff = csw*inv_param.kappa;
  }

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.tune = tune ? QUDA_TUNE_YES : QUDA_TUNE_NO;

  inv_param.dslash_type = dslash_type;

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || 
      dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.mu = mu;
    mg_param.delta_muPR = delta_muPR;
    mg_param.delta_muCG = delta_muCG;
    mg_param.delta_kappaPR = delta_kappaPR;
    mg_param.delta_kappaCG = delta_kappaCG;

    //inv_param.twist_flavor = twist_flavor;
    inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) ? 
      2 : 1;
    
    if (twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
      printfQuda("Twisted-mass doublet non supported (yet)\n");
      exit(0);
    }
  }
  
  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = normalization;
  inv_param.solver_normalization = QUDA_DEFAULT_NORMALIZATION;

  // do we want to use an even-odd preconditioned solve or not
  if(isEven) inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  else inv_param.matpc_type = QUDA_MATPC_ODD_ODD;

  inv_param.solution_type = QUDA_MAT_SOLUTION;

  inv_param.solve_type = QUDA_DIRECT_SOLVE;

  mg_param.invert_param = &inv_param;
  mg_param.n_level = mg_levels;
  for (int i=0; i<mg_param.n_level; i++) {
    for (int j=0; j<QUDA_MAX_DIM; j++) {
	// if not defined use 4
      mg_param.geo_block_size[i][j] = geo_block_size[i][j] ? 
	geo_block_size[i][j] : 4;      
    }
    mg_param.spin_block_size[i] = 1;
    
    //QKXTM: DMH Develop branch code
    // default to 24 vectors if not set
    mg_param.n_vec[i] = nvec[i] == 0 ? 24 : nvec[i]; 
    mg_param.nu_pre[i] = nu_pre;
    mg_param.nu_post[i] = nu_post;
    
    mg_param.cycle_type[i] = QUDA_MG_CYCLE_RECURSIVE;
    
    mg_param.smoother[i] = smoother_type;

    // set the smoother / bottom solver tolerance 
    // (for MR smoothing this will be ignored)
    // repurpose heavy-quark tolerance for now

    mg_param.smoother_tol[i] = tol_hq; 
    mg_param.global_reduction[i] = QUDA_BOOLEAN_YES;

    // set to QUDA_DIRECT_SOLVE for no even/odd 
    // preconditioning on the smoother
    // set to QUDA_DIRECT_PC_SOLVE for to enable even/odd 
    // preconditioning on the smoother
    mg_param.smoother_solve_type[i] = QUDA_DIRECT_PC_SOLVE; // EVEN-ODD

    // set to QUDA_MAT_SOLUTION to inject a full field into coarse grid
    // set to QUDA_MATPC_SOLUTION to inject single parity field into 
    // coarse grid

    // if we are using an outer even-odd preconditioned solve, then we
    // use single parity injection into the coarse grid
    mg_param.coarse_grid_solution_type[i] = solve_type == QUDA_DIRECT_PC_SOLVE ? QUDA_MATPC_SOLUTION : QUDA_MAT_SOLUTION;

    mg_param.omega[i] = 0.85; // over/under relaxation factor

    mg_param.location[i] = QUDA_CUDA_FIELD_LOCATION;
  }

  // only coarsen the spin on the first restriction
  mg_param.spin_block_size[0] = 2;

  // coarse grid solver is GCR
  mg_param.smoother[mg_levels-1] = QUDA_GCR_INVERTER;

  //QKXTM: DMH tmLQCD code
  //mg_param.compute_null_vector = QUDA_COMPUTE_NULL_VECTOR_YES;;
  //mg_param.generate_all_levels = QUDA_BOOLEAN_YES;
  
  //QKXTM: DMH develop code
  mg_param.compute_null_vector = generate_nullspace ? 
    QUDA_COMPUTE_NULL_VECTOR_YES : QUDA_COMPUTE_NULL_VECTOR_NO;
  mg_param.generate_all_levels = generate_all_levels ? 
    QUDA_BOOLEAN_YES :  QUDA_BOOLEAN_NO;

  mg_param.run_verify = QUDA_BOOLEAN_NO;

  // set file i/o parameters
  strcpy(mg_param.vec_infile, vec_infile);
  strcpy(mg_param.vec_outfile, vec_outfile);

  // these need to be set for now but are actually ignored by the MG setup
  // needed to make it pass the initialization test
  inv_param.inv_type = QUDA_GCR_INVERTER;
  inv_param.tol = tol;
  inv_param.maxiter = niter;
  inv_param.reliable_delta = 1e-10;
  inv_param.gcrNkrylov = 10;
  //inv_param.max_res_increase = 4;

  inv_param.verbosity = QUDA_SUMMARIZE;
  inv_param.verbosity_precondition = QUDA_SUMMARIZE;
}


//=======================================================================//
//== C O N T A I N E R   A N D   Q U D A   I N I T I A L I S A T I O N  =//
//=======================================================================//

int main(int argc, char **argv)
{
  using namespace quda;

  for (int i = 1; i < argc; i++){
    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    } 
    printfQuda("ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (prec_precondition == QUDA_INVALID_PRECISION) prec_precondition = prec_sloppy;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;
  if (link_recon_precondition == QUDA_RECONSTRUCT_INVALID) link_recon_precondition = link_recon_sloppy;

  // initialize QMP or MPI
#if defined(QMP_COMMS)
  QMP_thread_level_t tl;
  QMP_init_msg_passing(&argc, &argv, QMP_THREAD_SINGLE, &tl);
#elif defined(MPI_COMMS)
  MPI_Init(&argc, &argv);
#endif

  // call srand() with a rank-dependent seed
  initRand();

  display_test_info();

  //QKXTM: qkxtm specfic inputs
  //--------------------------------------------------------------------
  //-C.K. Pass ARPACK parameters to arpackInfo  
  qudaQKXTM_arpackInfo arpackInfo;
  arpackInfo.PolyDeg = PolyDeg;
  arpackInfo.nEv = nEv;
  arpackInfo.nKv = nKv;
  arpackInfo.isACC = isACC;
  arpackInfo.tolArpack = tolArpack;
  arpackInfo.maxIterArpack = maxIterArpack;
  strcpy(arpackInfo.arpack_logfile,arpack_logfile);
  arpackInfo.amin = amin;
  arpackInfo.amax = amax;
  arpackInfo.isEven = isEven;
  arpackInfo.isFullOp = isFullOp;

  if(strcmp(spectrumPart,"SR")==0) arpackInfo.spectrumPart = SR;
  else if(strcmp(spectrumPart,"LR")==0) arpackInfo.spectrumPart = LR;
  else if(strcmp(spectrumPart,"SM")==0) arpackInfo.spectrumPart = SM;
  else if(strcmp(spectrumPart,"LM")==0) arpackInfo.spectrumPart = LM;
  else if(strcmp(spectrumPart,"SI")==0) arpackInfo.spectrumPart = SI;
  else if(strcmp(spectrumPart,"LI")==0) arpackInfo.spectrumPart = LI;
  else{
    printf("Error: Your spectrumPart option is suspicious\n");
    exit(-1);
  }

  //-C.K. General QKXTM information
  qudaQKXTMinfo_Kepler info;
  info.lL[0] = xdim;
  info.lL[1] = ydim;
  info.lL[2] = zdim;
  info.lL[3] = tdim;
  info.Q_sq = Q_sq;
  info.isEven = isEven;
  if( strcmp(source_type,"random")==0 ) info.source_type = RANDOM;
  else if( strcmp(source_type,"unity")==0 ) info.source_type = UNITY;
  else{
    printf("Wrong type for stochastic source type. Must be either random/unity. Exiting.\n");
    exit(1);
  }

  //-C.K. Pass loop parameters to loopInfo
  qudaQKXTM_loopInfo loopInfo;
  loopInfo.Nstoch = Nstoch;
  loopInfo.seed = seed;
  loopInfo.Ndump = Ndump;
  loopInfo.traj = traj;
  loopInfo.Qsq = Q_sq;
  strcpy(loopInfo.loop_fname,loop_fname);

  if( strcmp(loop_file_format,"ASCII")==0 || 
      strcmp(loop_file_format,"ascii")==0 ) {
    // Determine whether to write the loops in ASCII
    loopInfo.FileFormat = ASCII_FORM;
  }
  else if( strcmp(loop_file_format,"HDF5")==0 || 
	   strcmp(loop_file_format,"hdf5")==0 ) {
    // Determine whether to write the loops in HDF5
    loopInfo.FileFormat = HDF5_FORM; 
  }
  else fprintf(stderr,"Undefined option for --loop-file-format. Options are ASCII(ascii)/HDF5(hdf5)\n");

  if(loopInfo.Nstoch%loopInfo.Ndump==0) loopInfo.Nprint = loopInfo.Nstoch/loopInfo.Ndump;
  else errorQuda("NdumpStep MUST divide Nstoch exactly! Exiting.\n");

  //- TSM parameters
  loopInfo.useTSM = useTSM;
  if(useTSM){
    loopInfo.TSM_NHP = TSM_NHP;
    loopInfo.TSM_NLP = TSM_NLP;
    loopInfo.TSM_NdumpHP = TSM_NdumpHP;
    loopInfo.TSM_NdumpLP = TSM_NdumpLP;
    
    if(loopInfo.TSM_NHP%loopInfo.TSM_NdumpHP==0) {
      loopInfo.TSM_NprintHP = loopInfo.TSM_NHP/loopInfo.TSM_NdumpHP;
    } else errorQuda("TSM_NdumpHP MUST divide TSM_NHP exactly! Exiting.\n");
    
    if(loopInfo.TSM_NLP%loopInfo.TSM_NdumpLP==0) {
      loopInfo.TSM_NprintLP = loopInfo.TSM_NLP/loopInfo.TSM_NdumpLP;
    } else errorQuda("TSM_NdumpLP MUST divide TSM_NLP exactly! Exiting.\n");
    
    loopInfo.TSM_tol = TSM_tol;
    loopInfo.TSM_maxiter = TSM_maxiter;
    if( (TSM_maxiter==0) && (TSM_tol==0) ) {
      errorQuda("Criterion for low-precision sources not set!\n");
    }
    if(TSM_tol!=0) errorQuda("Setting the tolerance as low-precision criterion for Truncated Solver method not supported! Re-run using --TSM_maxiter <iter> as criterion.\n");
  }

  // QUDA parameters begin here.
  //-----------------------------------------------------------------
  if ( dslash_type != QUDA_TWISTED_MASS_DSLASH && 
       dslash_type != QUDA_TWISTED_CLOVER_DSLASH && 
       dslash_type != QUDA_CLOVER_WILSON_DSLASH){
    printfQuda("This test is only for twisted mass or twisted clover operator\n");
    exit(-1);
  }
  
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);

  QudaInvertParam mg_inv_param = newQudaInvertParam();
  QudaMultigridParam mg_param = newQudaMultigridParam();
  mg_param.invert_param = &mg_inv_param;
  setMultigridParam(mg_param);

  QudaInvertParam inv_param = newQudaInvertParam();
  setInvertParam(inv_param);

  QudaInvertParam EVinv_param = newQudaInvertParam();
  EVinv_param = inv_param;
  if(isEven) EVinv_param.matpc_type = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
  else EVinv_param.matpc_type = QUDA_MATPC_ODD_ODD_ASYMMETRIC;


  // declare the dimensions of the communication grid
  initCommsGridQuda(4, gridsize_from_cmdline, NULL, NULL);

  setDims(gauge_param.X);
  setSpinorSiteSize(24);

  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? 
    sizeof(double) : sizeof(float);
  size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? 
    sizeof(double) : sizeof(float);

  void *gauge[4], *clover_inv=0, *clover=0;
  void *gauge_Plaq[4];

  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V*gaugeSiteSize*gSize);
    gauge_Plaq[dir] = malloc(V*gaugeSiteSize*gSize);
  }

  // load in the command line supplied gauge field
  readLimeGauge(gauge, latfile, &gauge_param, &inv_param, 
		gridsize_from_cmdline);
  for(int mu = 0 ; mu < 4 ; mu++)
    memcpy(gauge_Plaq[mu],gauge[mu],V*9*2*sizeof(double));
  mapEvenOddToNormalGauge(gauge_Plaq,gauge_param,xdim,ydim,zdim,tdim);

  // start the timer
  double time0 = -((double)clock());

  // initialize the QUDA library
  initQuda(device);
  //Print remaining info to stdout
  init_qudaQKXTM_Kepler(&info);
  printf_qudaQKXTM_Kepler();

  // load the gauge field
  loadGaugeQuda((void*)gauge, &gauge_param);

  for(int i = 0 ; i < 4 ; i++){
    free(gauge[i]);
  } 

  printfQuda("Before clover term\n");
  // This line ensure that if we need to construct the clover inverse 
  // (in either the smoother or the solver) we do so
  if (mg_param.smoother_solve_type[0] == QUDA_DIRECT_PC_SOLVE || 
      solve_type == QUDA_DIRECT_PC_SOLVE) {
    inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
  }

  if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) 
    loadCloverQuda(NULL, NULL, &inv_param);
  printfQuda("After clover term\n");
  
  //QKXTM: DMH EXP
  // setup the multigrid solver.
  mg_param.invert_param->twist_flavor = twist_flavor;
  void *mg_preconditioner = newMultigridQuda(&mg_param);
  inv_param.preconditioner = mg_preconditioner;  

  //Launch calculation.
  calcMG_loop_wOneD_TSM_wExact(gauge_Plaq, &EVinv_param, &inv_param, 
			       &gauge_param, arpackInfo, loopInfo, info);
  
  // free the multigrid solver
  destroyMultigridQuda(mg_preconditioner);
  
  freeGaugeQuda();
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || 
      dslash_type == QUDA_TWISTED_CLOVER_DSLASH) freeCloverQuda();

  for(int i = 0 ; i < 4 ; i++){
    free(gauge_Plaq[i]);
  }
  
  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
#if defined(QMP_COMMS)
  QMP_finalize_msg_passing();
#elif defined(MPI_COMMS)
  MPI_Finalize();
#endif
  
  return 0;
}
