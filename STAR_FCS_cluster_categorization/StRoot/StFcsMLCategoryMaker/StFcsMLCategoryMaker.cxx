// class StFcsMLCategoryMaker
// see header for chain position, backends and category convention

#include "StFcsMLCategoryMaker.h"

#include <cmath>

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

#ifndef SKIPDefImp
ClassImp(StFcsMLCategoryMaker)
#endif

    // Variable names. TMVA matches training and application by NAME, so these
    // strings have to be identical to the ones trainTMVA.C uses.
    const char* StFcsMLCategoryMaker::kVarNames[StFcsMLCategoryMaker::kNVar] = {
        "logE", "nTowers", "sigmaMax", "sigmaMin", "sigmaRatio", "theta", "seedFrac",
        "e2Frac", "e1e2Asym", "sigX", "sigY", "sigXY", "nNeighbor"};

StFcsMLCategoryMaker::StFcsMLCategoryMaker(const Char_t* name) : StMaker(name) {}

StFcsMLCategoryMaker::~StFcsMLCategoryMaker() {
   if (mReader) delete mReader;
}

//-----------------------------------------------------------------------------
// Feature vector. THIS ORDER IS THE CONTRACT with python/star_features.py,
// trainTMVA.C and kVarNames above.
//   0  logE        log(E)
//   1  nTowers
//   2  sigmaMax
//   3  sigmaMin
//   4  sigmaRatio  sigmaMin / sigmaMax
//   5  theta
//   6  seedFrac    e1 / E
//   7  e2Frac      e2 / E
//   8  e1e2Asym    (e1-e2)/(e1+e2)
//   9  sigX        second moments in cell units
//   10 sigY
//   11 sigXY
//   12 nNeighbor
std::vector<float> StFcsMLCategoryMaker::features(StFcsCluster* clu, StFcsDb* db) {
   std::vector<float> f(kNVar, 0.0);
   const float E = clu->energy();
   const int det = clu->detectorId();

   float e1 = -1, e2 = -1;
   double wtot = 0, sx = 0, sy = 0, sxy = 0, mx = 0, my = 0;
   StPtrVecFcsHit& hits = clu->hits();
   for (size_t k = 0; k < hits.size(); k++) {
      const float he = hits[k]->energy();
      const int row = db->getRowNumber(det, hits[k]->id());
      const int col = db->getColumnNumber(det, hits[k]->id());
      if (he > e1) {
         e2 = e1;
         e1 = he;
      } else if (he > e2) {
         e2 = he;
      }
      wtot += he;
      mx += he * col;
      my += he * row;
      sx += he * col * col;
      sy += he * row * row;
      sxy += he * col * row;
   }
   if (wtot <= 0) return f;
   mx /= wtot;
   my /= wtot;
   if (e2 < 0) e2 = 0;

   const float smax = clu->sigmaMax();
   f[0] = (E > 0) ? log(E) : -10.0;
   f[1] = clu->nTowers();
   f[2] = smax;
   f[3] = clu->sigmaMin();
   f[4] = (smax > 0) ? clu->sigmaMin() / smax : 0.0;
   f[5] = clu->theta();
   f[6] = (E > 0) ? e1 / E : 0.0;
   f[7] = (E > 0) ? e2 / E : 0.0;
   f[8] = (e1 + e2 > 0) ? (e1 - e2) / (e1 + e2) : 1.0;
   f[9] = sqrt(fabs(sx / wtot - mx * mx));
   f[10] = sqrt(fabs(sy / wtot - my * my));
   f[11] = sxy / wtot - mx * my;
   f[12] = clu->nNeighbor();
   return f;
}

//-----------------------------------------------------------------------------
Int_t StFcsMLCategoryMaker::Init() {
   mFcsDb = static_cast<StFcsDb*>(GetDataSet("fcsDb"));
   if (!mFcsDb) {
      LOG_ERROR << "StFcsMLCategoryMaker::Init failed to get StFcsDb" << endm;
      return kStFatal;
   }

   if (mBackend == kTMVA) {
      mReader = new TMVA::Reader("!Color:!Silent");
      for (int i = 0; i < kNVar; i++) mReader->AddVariable(kVarNames[i], &mVar[i]);
      // BookMVA does not throw on a bad path, it complains and returns 0
      if (!mReader->BookMVA(mTMVAMethod.c_str(), mWeightFile.c_str())) {
         LOG_ERROR << "StFcsMLCategoryMaker::Init TMVA could not book " << mTMVAMethod << " from "
                   << mWeightFile << endm;
         return kStFatal;
      }
      LOG_INFO << "StFcsMLCategoryMaker: TMVA method " << mTMVAMethod << " from " << mWeightFile << endm;
   } else {
      if (!mNet.load(mWeightFile.c_str())) {
         LOG_ERROR << "StFcsMLCategoryMaker::Init failed to load " << mWeightFile << endm;
         return kStFatal;
      }
      if (mNet.nOutput() != 3 || mNet.nInput() != kNVar) {
         LOG_ERROR << "StFcsMLCategoryMaker::Init model shape mismatch: nin=" << mNet.nInput()
                   << " nout=" << mNet.nOutput() << " expected " << kNVar << " and 3" << endm;
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

   for (int det = 0; det < 2; det++) {
      if (mFcsDb->ecalHcalPres(det) != 0) continue;
      StSPtrVecFcsCluster& clusters = mFcsColl->clusters(det);
      const int nc = mFcsColl->numberOfClusters(det);
      for (int ic = 0; ic < nc; ic++) {
         StFcsCluster* clu = clusters[ic];
         if (clu->energy() < mEmin) continue;

         const std::vector<float> f = features(clu, mFcsDb);

         float p[3] = {0, 0, 0};
         if (mBackend == kTMVA) {
            for (int i = 0; i < kNVar; i++) mVar[i] = f[i];
            const std::vector<Float_t>& r = mReader->EvaluateMulticlass(mTMVAMethod.c_str());
            if (r.size() != 3) {
               LOG_WARN << "StFcsMLCategoryMaker: multiclass response has " << r.size()
                        << " entries, expected 3 - was the method trained in multiclass mode?" << endm;
               continue;
            }
            for (int i = 0; i < 3; i++) p[i] = r[i];
         } else {
            const std::vector<float> r = mNet.eval(f);
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
