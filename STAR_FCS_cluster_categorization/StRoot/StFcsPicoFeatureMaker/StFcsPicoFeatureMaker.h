// class StFcsPicoFeatureMaker
//
// Dumps the SAME `clusters` tree as StFcsClusterFeatureMaker, but from a
// picoDst instead of a MuDst/StEvent chain. trainTMVA.C reads either one
// without changing a line.
//
// Chain position (see runPicoDst_ml.C):
//   StPicoDstMaker -> StFcsDbMaker(setDbAccess(0)) -> StFcsPicoFeatureMaker
//
// WHAT PICODST GIVES AND WHAT THIS MAKER HAS TO REBUILD
//
//   exact, read straight from StPicoFcsCluster:
//     energy, x, y, nTowers, sigmaMin, sigmaMax, theta, chi2Ndf1/2Photon,
//     category, four-momentum
//   rebuilt here from StPicoFcsHit by StFcsTowerAssoc (see that header for how
//   well, with numbers):
//     the cluster's tower list, and everything derived from it - seed tower,
//     seedFrac / e2Frac / e1e2Frac, sigX / sigY / sigXY, the 11x11 img and mask
//   generator-level truth, from StPicoMcTrack + StPicoMcVertex:
//     mcLabel, nMcPhoton, mcE, mcDr, mcSep, mcSepCell, mcZgg
//     nNeighbor - not stored either, but the tower adjacency it is built from
//     survives, so it is recomputed as the number of DISTINCT neighbouring
//     clusters; StFcsClusterFeatureMaker de-duplicates StFcsCluster::neighbor()
//     to the same definition, so the two tiers agree. See StFcsTowerAssoc.h.
//   NOT available at all:
//     hit-level GEANT truth. StPicoFcsHit has no track links, so nTrk stays 0
//     and truthNPhoton stays -1, exactly as on a MuDst. Train on mcLabel.
//     nPoints, which needs an StFcsPoint collection: written as -1.
//
// So a picoDst is now a full training input, not just an application input:
// mcLabel comes from the MC arrays and the tower list is recovered, which makes
// feature sets 3, 13 and 34 computable here as well as 6.
//
// THE ONE THING TO WATCH. The recovered tower list is a reconstruction, not the
// original. It agrees with the stored cluster to 92 % on tower count and to a
// median of exactly zero on energy, but if you are going to train a model on
// picoDst and apply it on MuDst (or the other way round), dump both from the
// same events once and compare - the QA branches nTowRec, eRec, dxRec, dyRec are
// there for precisely that. On the MuDst side the association is exact, so any
// disagreement you find is this maker's.
//
// author: generated for Xilin Liang

#ifndef STAR_StFcsPicoFeatureMaker_HH
#define STAR_StFcsPicoFeatureMaker_HH

#include <string>
#ifndef __CINT__
#include <vector>
#endif

#include "StMaker.h"

class TFile;
class TTree;
class StFcsDb;
class StPicoDst;
class StPicoDstMaker;
class StPicoFcsCluster;

class StFcsPicoFeatureMaker : public StMaker {
  public:
   // must match StFcsClusterFeatureMaker::kNW / kNPix / kMaxMc, or the two trees
   // are not the same schema any more and trainTMVA.C reads garbage
   static const int kNW = 11;
   static const int kNPix = kNW * kNW;
   static const int kMaxMc = 8;

   StFcsPicoFeatureMaker(StPicoDstMaker* picoMaker = 0, const Char_t* name = "FcsPicoFeat");
   ~StFcsPicoFeatureMaker();

   Int_t Init();
   Int_t Make();
   Int_t Finish();

   void setPicoDstMaker(StPicoDstMaker* m) { mPicoDstMaker = m; }
   void setOutputFile(const char* f) { mFileName = f; }
   void setEnergyThreshold(float e) { mEmin = e; }      // min cluster energy to store
   void setTowerThreshold(float e) { mTowerEmin = e; }  // min tower energy to consider

   // Association tuning - see StFcsTowerAssoc.h. The defaults are the measured
   // plateau; there is no reason to change them unless you are studying the
   // association itself.
   void setMaxTowerDistance(float cells) { mMaxDist = cells; }
   void setUseStoredNTowers(int v) { mUseNTow = v; }  // 1 = truncate to the stored nTowers
   // Tower-adjacency distance for counting neighbour clusters, in cells. 1.01
   // is StFcsClusterMaker's ECal setting (mNeighborDistance_Ecal).
   void setNeighborDistance(float cells) { mNeighborDist = cells; }

   void setSaveMcTruth(int v) { mSaveMcTruth = v; }
   void setMcMatchRadius(float cm) { mMcMatchR = cm; }
   void setPhotonEnergyThreshold(float e) { mMcEmin = e; }

  private:
   void resetBranches();
   void collectMcPhotons();
   void matchMcPhotons(float cluX, float cluY, int det);

   StPicoDstMaker* mPicoDstMaker;
   StPicoDst* mPicoDst;
   StFcsDb* mFcsDb;

   std::string mFileName;
   TFile* mFile;
   TTree* mTree;

   float mEmin;
   float mTowerEmin;
   float mMaxDist;
   int mUseNTow;
   float mNeighborDist;
   int mSaveMcTruth;
   float mMcMatchR;
   float mMcEmin;
   int mHaveMc;  // 1 when this event carries an MC record, photons or not

   Long64_t mNEvents, mNCluster, mNNoMcArray;

   // ---- branch buffers: identical names and types to StFcsClusterFeatureMaker ----
   Int_t bRun, bEvent, bDet, bClId, bNCluDet;
   Float_t bE, bX, bY;
   Float_t bStarX, bStarY, bStarZ;
   Float_t bEta, bPhi, bPt;
   Float_t bSigmaMax, bSigmaMin, bTheta;
   Int_t bNTowers, bNNeighbor, bNPoints;
   Int_t bCatStar;
   Float_t bChi2Ndf1, bChi2Ndf2;
   Int_t bSeedId, bSeedRow, bSeedCol;
   Float_t bSeedE, bSeedFrac, bE2Frac, bE1e2Frac;
   Float_t bXW, bYW;
   Float_t bSigX, bSigY, bSigXY;
   Float_t bImg[kNPix];
   Float_t bMask[kNPix];

   Int_t bNTrk;  // always 0 here: picoDst has no hit-level GEANT truth
   Int_t bTruthNPhoton, bTruthSameParent;
   Float_t bTruthPurity;

   Int_t bMcLabel, bNMcPhoton;
   Int_t bMcTrkId[kMaxMc], bMcParent[kMaxMc];
   Int_t bMcParentPid;
   Float_t bMcE[kMaxMc], bMcDr[kMaxMc], bMcX[kMaxMc], bMcY[kMaxMc];
   Float_t bMcSep, bMcSepCell, bMcZgg;
   Int_t bNMcPhotonEvent;

   // ---- picoDst-only extras, for judging the association ----
   Float_t bVz;
   Int_t bNTowRec;    // towers the association gave this cluster
   Float_t bERec;     // their energy sum; compare with e
   Float_t bDxRec;    // recovered centroid - stored x [cells]
   Float_t bDyRec;    // recovered centroid - stored y [cells]

#ifndef __CINT__
   struct McPhoton {
      int id;
      int parent;
      int parentPid;
      float e;
      float p[3];
      float v[3];
      float proj[2][2];
      int projOk[2];
   };
   std::vector<McPhoton> mMcPhotons;
   void projectPhoton(McPhoton& ph);
#endif

#ifndef SKIPDefImp
   ClassDef(StFcsPicoFeatureMaker, 0)
#endif
};

#endif
