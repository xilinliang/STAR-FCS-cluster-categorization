// trainTMVA.C - train the FCS ECal cluster category classifier with TMVA.
//
// Run it with the SAME ROOT that root4star uses, i.e. inside the SL7 container
// after starver. Compile it with ACLiC - note the trailing '+':
//
//     root4star -b -q 'trainTMVA.C+("fcsEcalClusterFeatures.root","FcsCat",13)'
//
// The '+' matters. Interpreted, this macro goes through CINT, which parses
// namespaces and inline functions in StFcsClusterFeatures.h only
// approximately; ACLiC hands it to the real compiler instead - the same one
// cons uses - so the feature code that runs here is the code that runs in the
// makers. Without the '+' you may get a bare
//     Error: Too many '}' tmpfile:NN
// which is CINT giving up on the parse, not a problem with your input file.
// TMVA weight XML is not guaranteed to be readable across ROOT major versions,
// and a weight file trained in some other ROOT is the classic way to lose a week.
//
// Nothing to install: TMVA ships inside ROOT, which ships inside the STAR
// library stack.
//
// Input : the `clusters` tree written by StFcsClusterFeatureMaker (.fzd or
//         MuDst) or StFcsPicoFeatureMaker (picoDst) - one schema, either way,
//         run on simulation so the truth branches are filled
// Output: weights/<jobname><set>_BDTG.weights.xml , ..._MLP.weights.xml
//         and <jobname><set>.root for the TMVA GUI
//
// The input variables come from StFcsClusterFeatures.h - the same header the
// inference maker uses - so training and application cannot drift apart.
//
// Feature sets: 3, 6, 13 or 34. Set 3 is the 3x3 one - the id names the
// window, not the variable count, and it has 13 variables. It is the good
// starting point: raw tower energies plus E, sigmaMax, sigmaMin and E1/E, with
// none of the correlated derived triplets of set 13.
//
// Set 6 is the tower-free one: it is computable from any input without
// reconstructing anything, which used to be the only way to work on picoDst.
// StPicoFcsCluster still stores no tower list, but StFcsTowerAssoc.h now
// recovers it geometrically (92 % exact tower count, median energy difference
// zero - see BUILD.md), so sets 3, 13 and 34 work on picoDst too. Set 6 costs
// you the tower-level shape detail, so expect it to separate 1-photon from
// 2-photon clusters less sharply; train both on the same sample and compare.
//
// The input file can come from a .fzd, a MuDst or a picoDst - all three write
// this same tree, and all three carry the generator-level label mcLabel.
//
// author: generated for Xilin Liang

#include "TFile.h"
#include "TString.h"
#include "TSystem.h"
#include "TTree.h"
#include "TMVA/Factory.h"
#include "TMVA/Tools.h"

#include "StRoot/StFcsMLCategoryMaker/StFcsClusterFeatures.h"

void trainTMVA(const char* infile = "fcsEcalClusterFeatures.root",
               const char* jobname = "FcsCat",
               int featureSet = 13,          // 3, 6, 13 or 34 - see the note above
               const char* treename = "clusters",
               float purityCut = 0.8,
               float eMin = 0.5) {
   using namespace StFcsClusterFeatures;
   gSystem->Load("libTMVA");
   TMVA::Tools::Instance();

   const int NVAR = nVar(featureSet);
   const char** varname = varNames(featureSet);
   const TString job = Form("%s%d", jobname, featureSet);
   printf("training feature set %d (%d variables), job %s\n", featureSet, NVAR, job.Data());

   // ---------------------------------------------------------------- input
   TFile* fin = TFile::Open(infile);
   if (!fin || fin->IsZombie()) { printf("cannot open %s\n", infile); return; }
   TTree* in = (TTree*)fin->Get(treename);
   if (!in) { printf("no tree %s in %s\n", treename, infile); return; }

   const int NW = 11;  // StFcsClusterFeatureMaker::kNW
   Float_t e, x, y, sigmaMin, sigmaMax, theta, xw, yw, truthPurity;
   Float_t img[NW * NW], mask[NW * NW];
   Int_t nTowers, nNeighbor, seedRow, seedCol, catStar, truthNPhoton, mcLabel;
   in->SetBranchAddress("e", &e);
   in->SetBranchAddress("x", &x);
   in->SetBranchAddress("y", &y);
   in->SetBranchAddress("sigmaMin", &sigmaMin);
   in->SetBranchAddress("sigmaMax", &sigmaMax);
   in->SetBranchAddress("theta", &theta);
   in->SetBranchAddress("xw", &xw);
   in->SetBranchAddress("yw", &yw);
   in->SetBranchAddress("img", img);
   in->SetBranchAddress("mask", mask);
   in->SetBranchAddress("nTowers", &nTowers);
   in->SetBranchAddress("nNeighbor", &nNeighbor);
   in->SetBranchAddress("seedRow", &seedRow);
   in->SetBranchAddress("seedCol", &seedCol);
   in->SetBranchAddress("catStar", &catStar);
   in->SetBranchAddress("truthNPhoton", &truthNPhoton);
   // generator-level label; -1 on feature files written before it existed
   mcLabel = -1;
   if (in->GetBranch("mcLabel")) in->SetBranchAddress("mcLabel", &mcLabel);
   in->SetBranchAddress("truthPurity", &truthPurity);

   // ------------------------------------------------- flat training trees
   TFile* ftmp = new TFile(job + "_train.root", "RECREATE");
   Float_t v[kNVarMax];
   TTree* t[3];
   const char* clsname[3] = {"other", "onePhoton", "twoPhoton"};
   for (int c = 0; c < 3; c++) {
      t[c] = new TTree(clsname[c], clsname[c]);
      for (int i = 0; i < NVAR; i++)
         t[c]->Branch(varname[i], &v[i], Form("%s/F", varname[i]));
   }

   const int half = NW / 2;
   float te[NW * NW];
   int trow[NW * NW], tcol[NW * NW];
   Long64_t n = in->GetEntries();
   Long64_t kept[3] = {0, 0, 0};

   // accounting, so that an empty training sample explains itself
   const bool haveMcBranch = (in->GetBranch("mcLabel") != 0);
   Long64_t nBelowE = 0, nNoTruth = 0, nImpure = 0, nNoTowers = 0;

   // range of every input variable over the kept clusters. TMVA aborts the
   // whole job on a constant one - see the check after the loop.
   Float_t vmin[kNVarMax], vmax[kNVarMax];
   for (int i = 0; i < NVAR; i++) {
      vmin[i] = 1e30;
      vmax[i] = -1e30;
   }

   for (Long64_t i = 0; i < n; i++) {
      in->GetEntry(i);
      if (e < eMin) {
         nBelowE++;
         continue;
      }

      // LABEL SOURCE. mcLabel is generator level: it counts the GENERATED
      // photons projecting onto this cluster, so it does not care how GEANT
      // shared the energy deposits between primary and shower tracks. Prefer it
      // whenever it is present, and note that it needs no purity cut - which
      // matters for hadronic clusters, whose energy is spread over many tracks
      // so their leading-track purity is naturally low and the old rule threw
      // most of them away, leaving class 0 nearly empty.
      //
      // truthNPhoton (hit level) is the fallback for feature files made before
      // the generator-level branches existed.
      int cls = -1;
      if (mcLabel >= 0) {
         cls = mcLabel;  // 0, 1 or 2 already
      } else if (truthNPhoton >= 0) {
         if (truthNPhoton >= 2) {
            cls = 2;
         } else if (truthNPhoton == 1 && truthPurity >= purityCut) {
            cls = 1;
         } else if (truthNPhoton == 0 && truthPurity >= purityCut) {
            cls = 0;
         }
      }
      if (cls < 0) {
         if (mcLabel < 0 && truthNPhoton < 0)
            nNoTruth++;  // no truth at all on this cluster
         else
            nImpure++;   // truth present but the purity cut rejected it
         continue;
      }

      // rebuild the cluster's towers from the stored image and mask
      int nTow = 0;
      for (int dr = -half; dr <= half; dr++) {
         for (int dc = -half; dc <= half; dc++) {
            const int pix = (dr + half) * NW + (dc + half);
            if (mask[pix] <= 0 || img[pix] <= 0) continue;
            te[nTow] = img[pix];
            trow[nTow] = seedRow + dr;
            tcol[nTow] = seedCol + dc;
            nTow++;
         }
      }

      ClusterInput c;
      c.e = e;
      c.x = x;
      c.y = y;
      c.sigmaMin = sigmaMin;
      c.sigmaMax = sigmaMax;
      c.theta = theta;
      c.nTowers = nTowers;
      c.nNeighbor = nNeighbor;
      c.xw = xw;
      c.yw = yw;
      c.nTow = nTow;
      c.towerE = te;
      c.towerRow = trow;
      c.towerCol = tcol;

      if (compute(featureSet, c, v) != NVAR) {
         nNoTowers++;
         continue;
      }
      for (int i = 0; i < NVAR; i++) {
         if (v[i] < vmin[i]) vmin[i] = v[i];
         if (v[i] > vmax[i]) vmax[i] = v[i];
      }
      t[cls]->Fill();
      kept[cls]++;
   }
   // ------------------------------------------------------------ accounting
   printf("\n--- where the %lld clusters in %s went ---\n", n, infile);
   printf("  below eMin = %.2f GeV        : %lld\n", eMin, nBelowE);
   printf("  no truth on the cluster      : %lld\n", nNoTruth);
   printf("  truth present, cut by purity : %lld\n", nImpure);
   printf("  no usable tower list         : %lld\n", nNoTowers);
   printf("  KEPT  other=%lld  1photon=%lld  2photon=%lld\n", kept[0], kept[1], kept[2]);
   printf("  label source: %s\n", haveMcBranch ? "mcLabel (generator level)"
                                               : "truthNPhoton (hit level); no mcLabel branch");

   if (kept[0] + kept[1] + kept[2] == 0) {
      printf("\nNothing to train on. Reading the numbers above:\n");
      if (n == 0) {
         printf("  The tree is empty. The feature dumper ran but wrote no clusters -\n"
                "  check the ECal cluster count in the job that produced it.\n");
      } else if (nNoTruth == n - nBelowE) {
         printf("  Every cluster has no truth, so the input carried no MC arrays.\n"
                "  All three inputs CAN be labelled - the generator-level truth survives\n"
                "  into every tier - so check, in order:\n"
                "    .fzd     : is fcsSim in the chain? no fast simulator, no hits\n"
                "    MuDst    : does it have StMuMcTrack/StMuMcVertex branches, and was\n"
                "               StFcsMuMcTruthMaker constructed BEFORE the dumper?\n"
                "    picoDst  : SetStatus(\"McTrack*\",1) and (\"McVertex*\",1), and was the\n"
                "               picoDst produced from a MuDst that had the MC arrays?\n"
                "  Only the HIT-level branches (trkPid, truthNPhoton) are exclusive to the\n"
                "  .fzd; mcLabel, which is what this macro trains on, is not.\n");
      } else if (nBelowE == n) {
         printf("  Every cluster is below eMin = %.2f GeV. Lower it, or check the energy\n"
                "  scale in the dumper.\n", eMin);
      } else if (nImpure > 0) {
         printf("  Truth is present but the purity cut rejected everything. Lower\n"
                "  purityCut, or re-dump with the generator-level branches so mcLabel\n"
                "  is used instead - it needs no purity cut.\n");
      }
      printf("\nStopping before TMVA, which would abort on empty trees.\n");
      ftmp->Close();
      return;
   }
   if (kept[1] < 100 || kept[2] < 100)
      printf("  WARNING: thin classes - the model will not be worth much yet\n");
   printf("\n");

   // ------------------------------------------------- constant-variable check
   //
   // TMVA does not skip a variable that never changes, it kills the job:
   //   <FATAL> DataSetFactory : Variable nNeighbor is constant. Please remove
   //                            the variable.
   //   ***> abort program execution
   // and by then it has already printed two screens of setup, so the cause is
   // easy to miss. Catch it here, where the reason can be explained.
   //
   // Dropping the offending variable automatically would be worse than
   // stopping: the weight file would then hold NVAR-1 variables while
   // StFcsClusterFeatures::compute() still produces NVAR in a fixed order, and
   // TMVA::Reader would refuse the model at application time - or, worse,
   // silently pair up the wrong ones. Training and inference share one
   // definition of the inputs in this package, and that is worth keeping.
   {
      int nConst = 0;
      for (int i = 0; i < NVAR; i++) {
         if (vmax[i] > vmin[i]) continue;
         if (nConst == 0)
            printf("--- constant input variables (TMVA would abort on these) ---\n");
         printf("  %-12s is always %g\n", varname[i], vmin[i]);
         nConst++;
      }
      if (nConst > 0) {
         printf("\nA variable with no spread carries no information, and TMVA refuses it.\n");
         printf("Usual cause: the input tier does not store that quantity, so the dumper\n");
         printf("wrote a placeholder on every cluster. nNeighbor and nPoints are the two\n");
         printf("that picoDst does not carry directly.\n\n");
         printf("Options, in order of preference:\n");
         printf("  - use a feature set that does not need it. Only set 13 uses nNeighbor;\n");
         printf("    sets 3, 6 and 34 do not:\n");
         printf("      trainTMVA.C+(\"%s\",\"%s\",3)\n", infile, jobname);
         printf("  - re-dump with a maker that fills it. StFcsPicoFeatureMaker computes\n");
         printf("    nNeighbor from the recovered tower adjacency; a feature file made\n");
         printf("    before that existed has it at -1 everywhere.\n");
         printf("  - if the sample itself is the reason (a single-particle gun where no\n");
         printf("    cluster ever has a neighbour), train on a mixed sample, or use set 3.\n");
         printf("\nStopping before TMVA.\n");
         ftmp->Close();
         return;
      }
   }

   // --------------------------------------------------------------- TMVA
   TFile* fout = TFile::Open(job + ".root", "RECREATE");
   TMVA::Factory* factory = new TMVA::Factory(
       job, fout,
       "!V:!Silent:Color:DrawProgressBar:Transformations=I;N:AnalysisType=multiclass");

   // ROOT 5 TMVA API: the Factory owns the variables and the trees directly.
   // There is deliberately no #if ROOT_VERSION here - CINT mishandles
   // preprocessor branches inside a function body and reports
   //   Error: Too many '}' tmpfile:NN
   // which is a parse failure, not a problem with your input file. STAR DEV is
   // ROOT 5.34.38. On ROOT 6.08+ the same calls move to a TMVA::DataLoader:
   //   TMVA::DataLoader* dl = new TMVA::DataLoader("dataset");
   //   dl->AddVariable(...); dl->AddTree(...); dl->PrepareTrainingAndTestTree(...);
   //   factory->BookMethod(dl, TMVA::Types::kBDT, "BDTG", "...");
   for (int i = 0; i < NVAR; i++) factory->AddVariable(varname[i], 'F');
   for (int c = 0; c < 3; c++) factory->AddTree(t[c], clsname[c]);
   factory->PrepareTrainingAndTestTree("", "SplitMode=Random:NormMode=NumEvents:!V");
   factory->BookMethod(TMVA::Types::kBDT, "BDTG",
                       "!H:!V:NTrees=600:MaxDepth=4:BoostType=Grad:Shrinkage=0.10:"
                       "UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=40");
   factory->BookMethod(TMVA::Types::kMLP, "MLP",
                       "!H:!V:NeuronType=tanh:NCycles=600:HiddenLayers=N+5,N:"
                       "TestRate=5:EstimatorType=CE:UseRegulator:VarTransform=Norm");

   factory->TrainAllMethods();
   factory->TestAllMethods();
   factory->EvaluateAllMethods();

   fout->Close();
   ftmp->Close();
   printf("\nweights are in weights/%s_BDTG.weights.xml (and _MLP)\n", job.Data());
   printf("point StFcsMLCategoryMaker at one of them:\n");
   printf("  mlcat->setFeatureSet(%d);\n", featureSet);
   printf("  mlcat->setTMVAMethod(\"BDTG\");\n");
   printf("  mlcat->setWeightFile(\"weights/%s_BDTG.weights.xml\");\n", job.Data());
   delete factory;
}
