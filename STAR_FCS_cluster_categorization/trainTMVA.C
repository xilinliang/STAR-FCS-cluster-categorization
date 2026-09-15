// trainTMVA.C - train the FCS ECal cluster category classifier with TMVA.
//
// Run it with the SAME ROOT that root4star uses, i.e. inside the SL7 container
// after starver:
//     root4star -b -q 'trainTMVA.C("fcsEcalClusterFeatures.root","FcsCat",13)'
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
// Feature sets: 6, 13 or 34. Use 6 when the model will be applied to picoDst
// input: StPicoFcsCluster stores no tower list, so those six variables are the
// only ones that exist on both sides. The training sample comes from a MuDst
// chain either way, because that is where the GEANT truth links live - see
// StFcsClusterFeatureMaker. Set 6 costs you the tower-level shape detail, so
// expect it to separate 1-photon from 2-photon clusters less sharply than 13;
// train both on the same sample and compare before deciding it is good enough.
//
// author: generated for Xilin Liang

#include "RVersion.h"
#include "TFile.h"
#include "TSystem.h"
#include "TTree.h"
#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 8, 0)
#include "TMVA/DataLoader.h"
#endif

#include "StRoot/StFcsMLCategoryMaker/StFcsClusterFeatures.h"

void trainTMVA(const char* infile = "fcsEcalClusterFeatures.root",
               const char* jobname = "FcsCat",
               int featureSet = 13,          // 6, 13 or 34 - see the note above
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
   Int_t nTowers, nNeighbor, seedRow, seedCol, catStar, truthNPhoton;
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
      if (truthNPhoton < 0) continue;  // no truth: data, or truth not stored

      int cls;
      if (truthNPhoton >= 2) {
         cls = 2;
      } else if (truthNPhoton == 1 && truthPurity >= purityCut) {
         cls = 1;
      } else if (truthNPhoton == 0 && truthPurity >= purityCut) {
         cls = 0;  // a clean hadronic / non-photon cluster
      } else {
         continue;  // impure, unlabelled
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

#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 8, 0)
   TMVA::DataLoader* dl = new TMVA::DataLoader("dataset");
   for (int i = 0; i < NVAR; i++) dl->AddVariable(varname[i], 'F');
   for (int c = 0; c < 3; c++) dl->AddTree(t[c], clsname[c]);
   dl->PrepareTrainingAndTestTree("", "SplitMode=Random:NormMode=NumEvents:!V");
   factory->BookMethod(dl, TMVA::Types::kBDT, "BDTG",
                       "!H:!V:NTrees=600:MaxDepth=4:BoostType=Grad:Shrinkage=0.10:"
                       "UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=40");
   factory->BookMethod(dl, TMVA::Types::kMLP, "MLP",
                       "!H:!V:NeuronType=tanh:NCycles=600:HiddenLayers=N+5,N:"
                       "TestRate=5:EstimatorType=CE:UseRegulator:VarTransform=Norm");
#else
   for (int i = 0; i < NVAR; i++) factory->AddVariable(varname[i], 'F');
   for (int c = 0; c < 3; c++) factory->AddTree(t[c], clsname[c]);
   factory->PrepareTrainingAndTestTree("", "SplitMode=Random:NormMode=NumEvents:!V");
   factory->BookMethod(TMVA::Types::kBDT, "BDTG",
                       "!H:!V:NTrees=600:MaxDepth=4:BoostType=Grad:Shrinkage=0.10:"
                       "UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=40");
   factory->BookMethod(TMVA::Types::kMLP, "MLP",
                       "!H:!V:NeuronType=tanh:NCycles=600:HiddenLayers=N+5,N:"
                       "TestRate=5:EstimatorType=CE:UseRegulator:VarTransform=Norm");
#endif

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
