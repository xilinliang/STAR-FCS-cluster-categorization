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
// Input : the tree written by StFcsClusterFeatureMaker (run on simulation, so
//         the truth branches are filled)
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
// Use 6 when the model will be applied to picoDst
// input: StPicoFcsCluster stores no tower list, so those six variables are the
// only ones that exist on both sides. The training sample comes from a MuDst
// chain either way, because that is where the GEANT truth links live - see
// StFcsClusterFeatureMaker. Set 6 costs you the tower-level shape detail, so
// expect it to separate 1-photon from 2-photon clusters less sharply than 13;
// train both on the same sample and compare before deciding it is good enough.
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

   for (Long64_t i = 0; i < n; i++) {
      in->GetEntry(i);
      if (e < eMin) continue;

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
      if (cls < 0) continue;  // no usable truth: data, or an impure cluster

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

      if (compute(featureSet, c, v) != NVAR) continue;
      t[cls]->Fill();
      kept[cls]++;
   }
   printf("training sample: other=%lld  1photon=%lld  2photon=%lld\n", kept[0], kept[1], kept[2]);
   if (kept[1] < 100 || kept[2] < 100)
      printf("not enough labelled clusters - check that you ran on simulation with truth\n");

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
