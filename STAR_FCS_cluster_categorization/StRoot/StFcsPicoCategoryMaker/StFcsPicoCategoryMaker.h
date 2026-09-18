// class StFcsPicoCategoryMaker
//
// The picoDst counterpart of StFcsMLCategoryMaker: reads FCS ECal clusters out
// of StPicoDst, computes the feature vector, evaluates the trained model, and
// records the result. Runs in a StPicoDstMaker chain:
//
//   StPicoDstMaker(IoRead) -> StFcsPicoCategoryMaker
//
// WHAT PICODST DOES AND DOES NOT CARRY
//
// StPicoFcsCluster stores the cluster summary only - id, detectorId, category,
// nTowers, x, y, sigmaMin, sigmaMax, theta, chi2Ndf1/2Photon and the four
// momentum. It does NOT store which towers went into the cluster:
// StPicoDstMaker::fillFcsClusters() copies the scalars and drops
// StMuFcsCluster::hits(). FcsHits is written as an independent flat collection
// with no back-pointer to a cluster.
//
// FEATURE SETS. Set 6 (logE, nTowers, sigmaMax, sigmaMin, sigmaRatio, theta) is
// the subset that needs no tower list at all, so it always works and is the
// safe default. The tower-level sets - 3, 13 and 34 - are available too, by
// re-associating the FcsHits to the clusters geometrically: see
// StFcsTowerAssoc.h, which documents the method and how closely it reproduces
// the stored cluster (92 % exact tower count, median energy difference zero).
// That is a reconstruction of the clustering rather than a reading of it, so it
// is opt-in: ask for set 3/13/34 and you get it, with the association running;
// ask for nothing and you get set 6.
//
// Whichever you use, train and apply on the SAME feature set - the weight file
// records the variable names and TMVA will refuse a mismatch.
//
// Two further things picoDst does not give you, both worth knowing before you
// read the output:
//   - no StFcsPoint collection, so the pi0 analysis here pairs CLUSTERS, not
//     the 1-/2-photon fitted points;
//   - chi2Ndf1Photon / chi2Ndf2Photon are only meaningful if StFcsPointMaker
//     ran in the chain that produced the picoDst. In the sample that came with
//     this code they are identically zero.
//
// Unlike StFcsMLCategoryMaker this maker CANNOT feed the point fitter - there
// is no reconstruction downstream of a picoDst. It is for evaluating a trained
// model on pico input and for the physics comparison, not for production.
//
// author: generated for Xilin Liang

#ifndef STAR_StFcsPicoCategoryMaker_HH
#define STAR_StFcsPicoCategoryMaker_HH

#include <string>
#ifndef __CINT__
#include <vector>
#endif

#include "StMaker.h"

class TFile;
class TTree;
class TH1F;
class TH2F;
class StPicoDstMaker;
class StPicoDst;
class StPicoFcsCluster;
class StFcsDb;

#ifndef __CINT__
namespace TMVA {
class Reader;
}
#endif

class StFcsPicoCategoryMaker : public StMaker {
  public:
   enum Mode { kQaOnly = 0, kOverride = 1, kOverrideIfConfident = 2 };

   // must equal StFcsClusterFeatures::kNVarMax; checked in the .cxx
   static const int kNVarMax = 34;

   StFcsPicoCategoryMaker(StPicoDstMaker* picoMaker, const Char_t* name = "FcsPicoCat");
   ~StFcsPicoCategoryMaker();

   Int_t Init();
   Int_t Make();
   Int_t Finish();

   void setFeatureSet(int n) { mFeatureSet = n; }         // 6 for picoDst input
   void setWeightFile(const char* f) { mWeightFile = f; }
   void setTMVAMethod(const char* m) { mTMVAMethod = m; }
   void setOutputFile(const char* f) { mOutFile = f; }
   void setMode(int m) { mMode = m; }
   void setConfidence(float c) { mConfidence = c; }
   void setEnergyThreshold(float e) { mEmin = e; }        // per cluster, GeV
   void setPairEnergyThreshold(float e) { mPairEmin = e; }// per cluster in a pi0 pair
   void setZggMax(float z) { mZggMax = z; }
   // Run without a model: dump features and the STAR category only. Use this
   // first, to see the distributions before there is anything to evaluate.
   void setNoModel(int v) { mNoModel = v; }

   // Tower-association tuning, used only for feature sets 3 / 13 / 34.
   // See StFcsTowerAssoc.h; the defaults are the measured plateau.
   void setMaxTowerDistance(float cells) { mMaxDist = cells; }
   void setUseStoredNTowers(int v) { mUseNTow = v; }

  private:
   void bookHistograms();
   void fillPi0(int useMlCategory);
   // Evaluates one cluster: fills feat[] and, unless mNoModel, prob[] too.
   // Returns the ML category, or -1 when there is no model or no response.
   // towE/towRow/towCol carry the recovered tower list (nTow entries, row and
   // column 1-based); pass nTow = 0 for set 6, which does not need one.
   int evaluate(StPicoFcsCluster* clu, int nTow, const float* towE, const int* towRow,
                const int* towCol, float* feat, float* prob);

   StPicoDstMaker* mPicoDstMaker;
   StPicoDst* mPicoDst;
   StFcsDb* mFcsDb;

   int mFeatureSet;
   std::string mWeightFile;
   std::string mTMVAMethod;
   std::string mOutFile;
   int mMode;
   int mNoModel;
   float mConfidence;
   float mEmin;
   float mPairEmin;
   float mZggMax;
   float mMaxDist;
   int mUseNTow;

   Float_t mVar[kNVarMax];

   TFile* mFile;
   TTree* mTree;

   // one entry per ECal cluster
   Int_t bRun, bEvent, bDet, bClId, bNCluDet, bNTowers, bNTowRec, bCatStar, bCatML;
   Float_t bE, bX, bY, bStarX, bStarY, bStarZ, bPt, bEta, bPhi;
   Float_t bSigmaMin, bSigmaMax, bTheta, bChi2Ndf1, bChi2Ndf2, bVz;
   Float_t bFeat[kNVarMax];
   Float_t bProb[3];

   Long64_t mNEvents;
   Long64_t mNCluster;
   Long64_t mNChanged;

   TH2F* h2_catStar_vs_catML;
   TH1F* h1_prob[3];
   TH1F* h1_invmass_star;
   TH1F* h1_invmass_ml;
   TH1F* h1_zgg;
   TH1F* h1_dgg;
   TH1F* h1_nCluEcal;
   TH2F* h2_invmass_vs_zgg;

#ifndef __CINT__
   TMVA::Reader* mReader;
   // ML category per picoDst cluster index, rebuilt each event so the pi0
   // pairing can look it up instead of re-evaluating the model per pair
   std::vector<int> mCatML;
#endif

#ifndef SKIPDefImp
   ClassDef(StFcsPicoCategoryMaker, 0)
#endif
};

#endif
