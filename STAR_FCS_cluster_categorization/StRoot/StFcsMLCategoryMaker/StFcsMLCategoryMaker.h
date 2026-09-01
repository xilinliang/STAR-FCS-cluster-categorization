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
// Two inference backends:
//   kTMVA    - TMVA::Reader, multiclass. ROOT ships TMVA, so nothing new has to
//              be built at RCF. Train the weight file with the SAME ROOT that
//              root4star uses (run trainTMVA.C under root4star) - TMVA weight
//              XML is not guaranteed to read across ROOT major versions.
//   kTextMLP - StFcsMLP.h, a plain-text dense network. Use it if you would
//              rather train in PyTorch/sklearn; no ROOT/TMVA involvement.
//
// author: generated for Xilin Liang

#ifndef STAR_StFcsMLCategoryMaker_HH
#define STAR_StFcsMLCategoryMaker_HH

#include <string>
#include <vector>

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

   static const int kNVar = 13;
   static const char* kVarNames[kNVar];  // must match the names used in training

   StFcsMLCategoryMaker(const Char_t* name = "FcsMLCat");
   ~StFcsMLCategoryMaker();

   Int_t Init();
   Int_t Make();
   Int_t Finish();

   void setBackend(int b) { mBackend = b; }
   void setWeightFile(const char* f) { mWeightFile = f; }     // TMVA XML or StFcsMLP text
   void setTMVAMethod(const char* m) { mTMVAMethod = m; }     // e.g. "BDTG", "MLP"
   void setQaFile(const char* f) { mQaFile = f; }
   void setMode(int m) { mMode = m; }
   void setConfidence(float c) { mConfidence = c; }  // used by kOverrideIfConfident
   void setEnergyThreshold(float e) { mEmin = e; }   // below this, keep the STAR category

   // Feature vector fed to the model. Keep this in lockstep with
   // python/star_features.py::build_features and with trainTMVA.C - same order,
   // same definitions, same names.
   static std::vector<float> features(StFcsCluster* clu, StFcsDb* db);

  private:
   StFcsDb* mFcsDb = 0;
   StFcsCollection* mFcsColl = 0;

   int mBackend = kTMVA;
   std::string mWeightFile = "weights/FcsCat_BDTG.weights.xml";
   std::string mTMVAMethod = "BDTG";
   TMVA::Reader* mReader = 0;
   Float_t mVar[kNVar];  // TMVA::Reader needs stable addresses
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
