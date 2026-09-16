// class StFcsMLCategoryMaker
// see header for chain position, feature sets, backends and category convention

#include "StFcsMLCategoryMaker.h"

#include <cmath>
#include <vector>

// The implementation headers live here, not in StFcsMLCategoryMaker.h - CINT
// cannot parse them and rootcint would fail while building the dictionary.
#include "StFcsClusterFeatures.h"
#include "StFcsMLP.h"

#include "StEvent/StEnumerations.h"
#include "StEvent/StEvent.h"
#include "StEvent/StFcsCluster.h"
#include "StEvent/StFcsCollection.h"
#include "StEvent/StFcsHit.h"
#include "StEventTypes.h"
#include "StFcsDbMaker/StFcsDb.h"
#include "StMessMgr.h"
#include "Stypes.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TMVA/Reader.h"

using namespace StFcsClusterFeatures;

// keep the CINT-safe copy of the array size honest
#if __cplusplus >= 201103L || defined(__GXX_EXPERIMENTAL_CXX0X__)
static_assert(StFcsMLCategoryMaker::kNVarMax == StFcsClusterFeatures::kNVarMax,
              "StFcsMLCategoryMaker::kNVarMax must match StFcsClusterFeatures::kNVarMax");
#endif

#ifndef SKIPDefImp
ClassImp(StFcsMLCategoryMaker)
#endif

    // definition for the in-class-initialised static constant, so that any
    // odr-use of it resolves at dlopen time rather than failing with
    //   undefined symbol: _ZN20StFcsMLCategoryMaker8kNVarMaxE
    const int StFcsMLCategoryMaker::kNVarMax;

    StFcsMLCategoryMaker::StFcsMLCategoryMaker(const Char_t* name)
    : StMaker(name),
      mFcsDb(0),
      mFcsColl(0),
      mFeatureSet(13),
      mBackend(kTMVA),
      mWeightFile("weights/FcsCat13_BDTG.weights.xml"),
      mTMVAMethod("BDTG"),
      mQaFile(""),
      mMode(kOverride),
      mConfidence(0.7),
      mEmin(0.5),
      mNCluster(0),
      mNChanged(0),
      h2_catStar_vs_catML(0),
      mReader(0),
      mNet(0) {
   for (int i = 0; i < kNVarMax; i++) mVar[i] = 0.0;
   for (int i = 0; i < 3; i++) h1_prob[i] = 0;
}

StFcsMLCategoryMaker::~StFcsMLCategoryMaker() {
   if (mReader) delete mReader;
   if (mNet) delete mNet;
}

//-----------------------------------------------------------------------------
// Adapter: StFcsCluster -> StFcsClusterFeatures::ClusterInput -> compute().
// No variable is defined here; the definitions live in StFcsClusterFeatures.h
// so that trainTMVA.C computes exactly the same numbers from the tree.
int StFcsMLCategoryMaker::features(StFcsCluster* clu, StFcsDb* db, int set, float* out) {
   StPtrVecFcsHit& hits = clu->hits();
   const int n = (int)hits.size();
   if (n <= 0) return 0;
   const int det = clu->detectorId();

   std::vector<float> te(n);
   std::vector<int> trow(n);
   std::vector<int> tcol(n);
   for (int k = 0; k < n; k++) {
      te[k] = hits[k]->energy();
      trow[k] = db->getRowNumber(det, hits[k]->id());
      tcol[k] = db->getColumnNumber(det, hits[k]->id());
   }

   ClusterInput c;
   c.e = clu->energy();
   c.x = clu->x();
   c.y = clu->y();
   c.sigmaMin = clu->sigmaMin();
   c.sigmaMax = clu->sigmaMax();
   c.theta = clu->theta();
   c.nTowers = clu->nTowers();
   c.nNeighbor = clu->nNeighbor();
   c.xw = db->getXWidth(det);
   c.yw = db->getYWidth(det);
   c.nTow = n;
   c.towerE = &te[0];
   c.towerRow = &trow[0];
   c.towerCol = &tcol[0];

   return compute(set, c, out);
}

//-----------------------------------------------------------------------------
Int_t StFcsMLCategoryMaker::Init() {
   mFcsDb = static_cast<StFcsDb*>(GetDataSet("fcsDb"));
   if (!mFcsDb) {
      LOG_ERROR << "StFcsMLCategoryMaker::Init failed to get StFcsDb" << endm;
      return kStFatal;
   }

   const int nv = nVar(mFeatureSet);
   const char** names = varNames(mFeatureSet);
   LOG_INFO << "StFcsMLCategoryMaker: feature set " << mFeatureSet << " (" << nv << " variables)" << endm;

   if (mBackend == kTMVA) {
      mReader = new TMVA::Reader("!Color:!Silent");
      for (int i = 0; i < nv; i++) mReader->AddVariable(names[i], &mVar[i]);
      // BookMVA does not throw on a bad path, it complains and returns 0. It
      // also fails when the weight file was trained with different variable
      // names - which is what a feature-set mismatch looks like.
      if (!mReader->BookMVA(mTMVAMethod.c_str(), mWeightFile.c_str())) {
         LOG_ERROR << "StFcsMLCategoryMaker::Init TMVA could not book " << mTMVAMethod << " from "
                   << mWeightFile << " - check it was trained with feature set " << mFeatureSet << endm;
         return kStFatal;
      }
      LOG_INFO << "StFcsMLCategoryMaker: TMVA method " << mTMVAMethod << " from " << mWeightFile << endm;
   } else {
      mNet = new StFcsMLP();
      if (!mNet->load(mWeightFile.c_str())) {
         LOG_ERROR << "StFcsMLCategoryMaker::Init failed to load " << mWeightFile << endm;
         return kStFatal;
      }
      if (mNet->nOutput() != 3 || mNet->nInput() != nv) {
         LOG_ERROR << "StFcsMLCategoryMaker::Init model shape mismatch: nin=" << mNet->nInput()
                   << " nout=" << mNet->nOutput() << " expected " << nv << " and 3" << endm;
         return kStFatal;
      }
   }

   h2_catStar_vs_catML = new TH2F("h2_catStar_vs_catML", "STAR category vs ML category;STAR;ML", 3, 0, 3, 3, 0, 3);
   h1_prob[0] = new TH1F("h1_prob0", "response, class 0 (ambiguous)", 100, 0, 1);
   h1_prob[1] = new TH1F("h1_prob1", "response, class 1 (1 photon)", 100, 0, 1);
   h1_prob[2] = new TH1F("h1_prob2", "response, class 2 (2 photons)", 100, 0, 1);
   return kStOK;
}

//-----------------------------------------------------------------------------
Int_t StFcsMLCategoryMaker::Make() {
   StEvent* event = (StEvent*)GetInputDS("StEvent");
   if (!event) {
      LOG_ERROR << "StFcsMLCategoryMaker::Make did not find StEvent" << endm;
      return kStErr;
   }
   mFcsColl = event->fcsCollection();
   if (!mFcsColl) return kStErr;

   const int nv = nVar(mFeatureSet);
   float f[kNVarMax];

   for (int det = 0; det < 2; det++) {
      if (mFcsDb->ecalHcalPres(det) != 0) continue;
      StSPtrVecFcsCluster& clusters = mFcsColl->clusters(det);
      const int nc = mFcsColl->numberOfClusters(det);
      for (int ic = 0; ic < nc; ic++) {
         StFcsCluster* clu = clusters[ic];
         if (clu->energy() < mEmin) continue;
         if (features(clu, mFcsDb, mFeatureSet, f) != nv) continue;

         float p[3] = {0, 0, 0};
         if (mBackend == kTMVA) {
            for (int i = 0; i < nv; i++) mVar[i] = f[i];
            const std::vector<Float_t>& r = mReader->EvaluateMulticlass(mTMVAMethod.c_str());
            if (r.size() != 3) {
               LOG_WARN << "StFcsMLCategoryMaker: multiclass response has " << r.size()
                        << " entries, expected 3 - was the method trained in multiclass mode?" << endm;
               continue;
            }
            for (int i = 0; i < 3; i++) p[i] = r[i];
         } else {
            std::vector<float> in(f, f + nv);
            const std::vector<float> r = mNet->eval(in);
            if (r.size() != 3) continue;
            for (int i = 0; i < 3; i++) p[i] = r[i];
         }

         int best = 0;
         for (int i = 1; i < 3; i++)
            if (p[i] > p[best]) best = i;

         mNCluster++;
         const int old = clu->category();
         h2_catStar_vs_catML->Fill(old, best);
         for (int i = 0; i < 3; i++) h1_prob[i]->Fill(p[i]);

         if (mMode == kQaOnly) continue;
         if (mMode == kOverrideIfConfident && p[best] < mConfidence) continue;

         if (best != old) mNChanged++;
         clu->setCategory(best);
      }
   }
   return kStOK;
}

//-----------------------------------------------------------------------------
Int_t StFcsMLCategoryMaker::Finish() {
   LOG_INFO << Form("StFcsMLCategoryMaker: %lld clusters, %lld categories changed (%.1f%%)", mNCluster,
                    mNChanged, mNCluster ? 100.0 * mNChanged / mNCluster : 0.0)
            << endm;
   if (mQaFile.length() > 0) {
      TFile f(mQaFile.c_str(), "RECREATE");
      h2_catStar_vs_catML->Write();
      for (int i = 0; i < 3; i++) h1_prob[i]->Write();
      f.Close();
   }
   return kStOK;
}
