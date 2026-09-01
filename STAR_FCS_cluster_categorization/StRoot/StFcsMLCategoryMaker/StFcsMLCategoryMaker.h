// class StFcsMLCategoryMaker
//
// Replaces the hard-coded sigma-vs-energy category cut of StFcsClusterMaker
// with a trained model, so that StFcsPointMaker fits 1 or 2 photons based on
// the model's decision.
//
// Chain position - this maker MUST sit between the two:
//   StFcsClusterMaker -> StFcsMLCategoryMaker -> StFcsPointMaker
//
// Category convention (unchanged from STAR):
//   0 = ambiguous -> StFcsPointMaker tries both 1- and 2-photon fits
//   1 = single photon
//   2 = two photons
//
// The input variables are defined once, in StFcsClusterFeatures.h, which
// trainTMVA.C includes as well - training and inference literally run the same
// code. Two sets, selected with setFeatureSet():
//   13 (default) - shape summary variables. Start here.
//   34           - the ePIC-style set with the 5x5 tower energies.
// The set is part of the contract with the weight file: train and apply with
// the same number, and keep it in the weight file name.
//
// Two inference backends:
//   kTMVA    - TMVA::Reader, multiclass. ROOT ships TMVA, so nothing extra has
//              to be installed at RCF. Train the weight file with the SAME ROOT
//              that root4star uses - TMVA weight XML is not guaranteed to read
//              across ROOT major versions.
//   kTextMLP - StFcsMLP.h, a plain-text dense network, only needed if a model
//              is trained outside ROOT. Ignore it if you are using TMVA.
//
// author: generated for Xilin Liang

#ifndef STAR_StFcsMLCategoryMaker_HH
#define STAR_StFcsMLCategoryMaker_HH

#include <string>

#include "StFcsClusterFeatures.h"
#include "StFcsMLP.h"
#include "StMaker.h"

namespace TMVA {
class Reader;
}
class TFile;
class TH1F;
class TH2F;
class StFcsDb;
class StFcsCollection;
class StFcsCluster;

class StFcsMLCategoryMaker : public StMaker {
  public:
   enum Mode { kQaOnly = 0, kOverride = 1, kOverrideIfConfident = 2 };
   enum Backend { kTMVA = 0, kTextMLP = 1 };

   StFcsMLCategoryMaker(const Char_t* name = "FcsMLCat");
   ~StFcsMLCategoryMaker();

   Int_t Init();
   Int_t Make();
   Int_t Finish();

   void setFeatureSet(int n) { mFeatureSet = (n == 34) ? 34 : 13; }
   void setBackend(int b) { mBackend = b; }
   void setWeightFile(const char* f) { mWeightFile = f; }     // TMVA XML or StFcsMLP text
   void setTMVAMethod(const char* m) { mTMVAMethod = m; }     // e.g. "BDTG", "MLP"
   void setQaFile(const char* f) { mQaFile = f; }
   void setMode(int m) { mMode = m; }
   void setConfidence(float c) { mConfidence = c; }  // used by kOverrideIfConfident
   void setEnergyThreshold(float e) { mEmin = e; }   // below this, keep the STAR category

   // Pull an StFcsCluster apart into the plain arrays StFcsClusterFeatures
   // wants, then compute. Returns the number of variables filled, 0 on failure.
   static int features(StFcsCluster* clu, StFcsDb* db, int set, float* out);

  private:
   StFcsDb* mFcsDb = 0;
   StFcsCollection* mFcsColl = 0;

   int mFeatureSet = 13;
   int mBackend = kTMVA;
   std::string mWeightFile = "weights/FcsCat13_BDTG.weights.xml";
   std::string mTMVAMethod = "BDTG";
   TMVA::Reader* mReader = 0;
   Float_t mVar[StFcsClusterFeatures::kNVarMax];  // TMVA::Reader needs stable addresses
   StFcsMLP mNet;

   std::string mQaFile = "";
   int mMode = kOverride;
   float mConfidence = 0.7;
   float mEmin = 0.5;

   Long64_t mNCluster = 0;
   Long64_t mNChanged = 0;

   TH2F* h2_catStar_vs_catML = 0;
   TH1F* h1_prob[3] = {0, 0, 0};

#ifndef SKIPDefImp
   ClassDef(StFcsMLCategoryMaker, 0)
#endif
};

#endif
