// trainTMVA.C - train the FCS ECal cluster category classifier with TMVA.
//
// Run it with the SAME ROOT that root4star uses, i.e.
//     starver dev            # or whichever version you run the chain with
//     root4star -b -q 'trainTMVA.C("fcsEcalClusterFeatures.root","FcsCat")'
// TMVA weight XML is not guaranteed to be readable across ROOT major versions,
// and a weight file trained in a conda ROOT is the classic way to lose a week.
//
// Input : the tree written by StFcsClusterFeatureMaker (simulation, so that the
//         truth branches are filled)
// Output: weights/<jobname>_BDTG.weights.xml , weights/<jobname>_MLP.weights.xml
//         and <jobname>.root for the TMVA GUI
//
// The 13 variables are built here exactly as StFcsMLCategoryMaker::features()
// builds them, into a flat intermediate tree, so that training and application
// cannot drift apart through a TTreeFormula subtlety.
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

void trainTMVA(const char* infile = "fcsEcalClusterFeatures.root",
               const char* jobname = "FcsCat",
               const char* treename = "clusters",
               float purityCut = 0.8,
               float eMin = 0.5) {
   gSystem->Load("libTMVA");
   TMVA::Tools::Instance();

   // ---------------------------------------------------------------- input
   TFile* fin = TFile::Open(infile);
   if (!fin || fin->IsZombie()) { printf("cannot open %s\n", infile); return; }
   TTree* in = (TTree*)fin->Get(treename);
   if (!in) { printf("no tree %s in %s\n", treename, infile); return; }

   Float_t e, sigmaMax, sigmaMin, theta, seedFrac, e2Frac, e1e2Frac, sigX, sigY, sigXY;
   Int_t nTowers, nNeighbor, catStar, truthNPhoton;
   Float_t truthPurity;
   in->SetBranchAddress("e", &e);
   in->SetBranchAddress("sigmaMax", &sigmaMax);
   in->SetBranchAddress("sigmaMin", &sigmaMin);
   in->SetBranchAddress("theta", &theta);
   in->SetBranchAddress("seedFrac", &seedFrac);
   in->SetBranchAddress("e2Frac", &e2Frac);
   in->SetBranchAddress("e1e2Frac", &e1e2Frac);
   in->SetBranchAddress("sigX", &sigX);
   in->SetBranchAddress("sigY", &sigY);
   in->SetBranchAddress("sigXY", &sigXY);
   in->SetBranchAddress("nTowers", &nTowers);
   in->SetBranchAddress("nNeighbor", &nNeighbor);
   in->SetBranchAddress("catStar", &catStar);
   in->SetBranchAddress("truthNPhoton", &truthNPhoton);
   in->SetBranchAddress("truthPurity", &truthPurity);

   // ------------------------------------------------- flat training tree
   TFile* ftmp = new TFile(Form("%s_train.root", jobname), "RECREATE");
   Float_t v[13];
   TTree* t[3];
   const char* clsname[3] = {"other", "onePhoton", "twoPhoton"};
   const char* varname[13] = {"logE", "nTowers", "sigmaMax", "sigmaMin", "sigmaRatio",
                              "theta", "seedFrac", "e2Frac", "e1e2Asym", "sigX",
                              "sigY", "sigXY", "nNeighbor"};
   for (int c = 0; c < 3; c++) {
      t[c] = new TTree(clsname[c], clsname[c]);
      for (int i = 0; i < 13; i++) t[c]->Branch(varname[i], &v[i], Form("%s/F", varname[i]));
   }

   Long64_t n = in->GetEntries();
   Long64_t kept[3] = {0, 0, 0};
   for (Long64_t i = 0; i < n; i++) {
      in->GetEntry(i);
      if (e < eMin) continue;
      if (truthNPhoton < 0) continue;  // no truth: data, or fast simulator truth not stored

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

      v[0] = (e > 0) ? log(e) : -10.0;
      v[1] = nTowers;
      v[2] = sigmaMax;
      v[3] = sigmaMin;
      v[4] = (sigmaMax > 0) ? sigmaMin / sigmaMax : 0.0;
      v[5] = theta;
      v[6] = seedFrac;
      v[7] = e2Frac;
      v[8] = e1e2Frac;
      v[9] = sigX;
      v[10] = sigY;
      v[11] = sigXY;
      v[12] = nNeighbor;
      t[cls]->Fill();
      kept[cls]++;
   }
   printf("training sample: other=%lld  1photon=%lld  2photon=%lld\n", kept[0], kept[1], kept[2]);
   if (kept[1] < 100 || kept[2] < 100) {
      printf("not enough labelled clusters - check that you ran on simulation with truth\n");
   }

   // --------------------------------------------------------------- TMVA
   TFile* fout = TFile::Open(Form("%s.root", jobname), "RECREATE");
   TMVA::Factory* factory = new TMVA::Factory(
       jobname, fout,
       "!V:!Silent:Color:DrawProgressBar:Transformations=I;N:AnalysisType=multiclass");

#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 8, 0)
   TMVA::DataLoader* dl = new TMVA::DataLoader("dataset");
   for (int i = 0; i < 13; i++) dl->AddVariable(varname[i], 'F');
   for (int c = 0; c < 3; c++) dl->AddTree(t[c], clsname[c]);
   dl->PrepareTrainingAndTestTree("", "SplitMode=Random:NormMode=NumEvents:!V");
   factory->BookMethod(dl, TMVA::Types::kBDT, "BDTG",
                       "!H:!V:NTrees=600:MaxDepth=4:BoostType=Grad:Shrinkage=0.10:"
                       "UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=40");
   factory->BookMethod(dl, TMVA::Types::kMLP, "MLP",
                       "!H:!V:NeuronType=tanh:NCycles=600:HiddenLayers=N+5,N:"
                       "TestRate=5:EstimatorType=CE:UseRegulator");
#else
   for (int i = 0; i < 13; i++) factory->AddVariable(varname[i], 'F');
   for (int c = 0; c < 3; c++) factory->AddTree(t[c], clsname[c]);
   factory->PrepareTrainingAndTestTree("", "SplitMode=Random:NormMode=NumEvents:!V");
   factory->BookMethod(TMVA::Types::kBDT, "BDTG",
                       "!H:!V:NTrees=600:MaxDepth=4:BoostType=Grad:Shrinkage=0.10:"
                       "UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=40");
   factory->BookMethod(TMVA::Types::kMLP, "MLP",
                       "!H:!V:NeuronType=tanh:NCycles=600:HiddenLayers=N+5,N:"
                       "TestRate=5:EstimatorType=CE:UseRegulator");
#endif

   factory->TrainAllMethods();
   factory->TestAllMethods();
   factory->EvaluateAllMethods();

   fout->Close();
   ftmp->Close();
   printf("\nweights are in weights/%s_BDTG.weights.xml (and _MLP)\n", jobname);
   printf("point StFcsMLCategoryMaker at one of them:\n");
   printf("  mlcat->setBackend(0); mlcat->setTMVAMethod(\"BDTG\");\n");
   printf("  mlcat->setWeightFile(\"weights/%s_BDTG.weights.xml\");\n", jobname);
   delete factory;
}
