// class StFcsPicoCategoryMaker
// see the header for what picoDst carries and what it does not

#include "StFcsPicoCategoryMaker.h"

#include <cmath>
#include <vector>

#include "StMessMgr.h"
#include "Stypes.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TLorentzVector.h"
#include "TMVA/Reader.h"
#include "TTree.h"
#include "TVector3.h"

#include "StFcsDbMaker/StFcsDb.h"
#include "StPicoDstMaker/StPicoDstMaker.h"
#include "StPicoEvent/StPicoDst.h"
#include "StPicoEvent/StPicoEvent.h"
#include "StPicoEvent/StPicoFcsCluster.h"
#include "StPicoEvent/StPicoFcsHit.h"

// the feature definitions, shared with StFcsMLCategoryMaker and trainTMVA.C
#include "StFcsMLCategoryMaker/StFcsClusterFeatures.h"
// recovers which towers belong to which cluster - needed by sets 3, 13 and 34
#include "StFcsMLCategoryMaker/StFcsTowerAssoc.h"

using namespace StFcsClusterFeatures;

#if __cplusplus >= 201103L || defined(__GXX_EXPERIMENTAL_CXX0X__)
static_assert(StFcsPicoCategoryMaker::kNVarMax == StFcsClusterFeatures::kNVarMax,
              "kNVarMax must match StFcsClusterFeatures::kNVarMax");
#endif

#ifndef SKIPDefImp
ClassImp(StFcsPicoCategoryMaker)
#endif

    // definition for the in-class-initialised static constant - see the note in
    // StFcsClusterFeatureMaker.cxx
    const int StFcsPicoCategoryMaker::kNVarMax;

    StFcsPicoCategoryMaker::StFcsPicoCategoryMaker(StPicoDstMaker* picoMaker, const Char_t* name)
    : StMaker(name),
      mPicoDstMaker(picoMaker),
      mPicoDst(0),
      mFcsDb(0),
      mFeatureSet(6),
      mWeightFile("weights/FcsCat6_BDTG.weights.xml"),
      mTMVAMethod("BDTG"),
      mOutFile("fcsPicoCategory.root"),
      mMode(kOverride),
      mNoModel(0),
      mConfidence(0.7),
      mEmin(0.5),
      mPairEmin(1.0),
      mZggMax(0.7),
      mMaxDist(5.0),
      mUseNTow(1),
      mNeighborDist(1.01),
      mFile(0),
      mTree(0),
      mNEvents(0),
      mNCluster(0),
      mNChanged(0),
      h2_catStar_vs_catML(0),
      h1_invmass_star(0),
      h1_invmass_ml(0),
      h1_zgg(0),
      h1_dgg(0),
      h1_nCluEcal(0),
      h2_invmass_vs_zgg(0),
      mReader(0) {
   for (int i = 0; i < kNVarMax; i++) mVar[i] = 0.0;
   for (int i = 0; i < 3; i++) h1_prob[i] = 0;
}

StFcsPicoCategoryMaker::~StFcsPicoCategoryMaker() {
   if (mReader) delete mReader;
}

//-----------------------------------------------------------------------------
Int_t StFcsPicoCategoryMaker::Init() {
   if (!mPicoDstMaker) {
      LOG_ERROR << "StFcsPicoCategoryMaker::Init no StPicoDstMaker given" << endm;
      return kStFatal;
   }

   const int nv = nVar(mFeatureSet);
   const char** names = varNames(mFeatureSet);
   LOG_INFO << "StFcsPicoCategoryMaker: feature set " << mFeatureSet << " (" << nv << " variables)" << endm;

   // Sets 3, 13 and 34 need the cluster's tower list, which picoDst does not
   // store. StFcsTowerAssoc recovers it geometrically, and that needs the tower
   // map and cell size from StFcsDb - geometry only, no database: run the
   // StFcsDbMaker with setDbAccess(0).
   if (mFeatureSet != 6) {
      mFcsDb = static_cast<StFcsDb*>(GetDataSet("fcsDb"));
      if (!mFcsDb) {
         LOG_ERROR << "StFcsPicoCategoryMaker::Init feature set " << mFeatureSet
                   << " needs the tower list, which is recovered with StFcsDb geometry, but there "
                      "is no StFcsDbMaker in the chain. Add one (setDbAccess(0) is enough), or use "
                      "feature set 6, which needs no tower list."
                   << endm;
         return kStFatal;
      }
      LOG_INFO << Form("StFcsPicoCategoryMaker: recovering the tower list, maxDist=%.1f cells, "
                       "truncate to stored nTowers=%d",
                       mMaxDist, mUseNTow)
               << endm;
   }

   if (!mNoModel) {
      mReader = new TMVA::Reader("!Color:!Silent");
      for (int i = 0; i < nv; i++) mReader->AddVariable(names[i], &mVar[i]);
      if (!mReader->BookMVA(mTMVAMethod.c_str(), mWeightFile.c_str())) {
         LOG_ERROR << "StFcsPicoCategoryMaker::Init TMVA could not book " << mTMVAMethod << " from "
                   << mWeightFile << " - was it trained with feature set " << mFeatureSet << "?" << endm;
         return kStFatal;
      }
      LOG_INFO << "StFcsPicoCategoryMaker: TMVA " << mTMVAMethod << " from " << mWeightFile << endm;
   } else {
      LOG_INFO << "StFcsPicoCategoryMaker: no model, dumping features and the STAR category only" << endm;
   }

   mFile = new TFile(mOutFile.c_str(), "RECREATE");
   mTree = new TTree("clusters", "FCS ECal clusters from picoDst");
   mTree->Branch("run", &bRun, "run/I");
   mTree->Branch("event", &bEvent, "event/I");
   mTree->Branch("vz", &bVz, "vz/F");
   mTree->Branch("det", &bDet, "det/I");
   mTree->Branch("clid", &bClId, "clid/I");
   mTree->Branch("ncluDet", &bNCluDet, "ncluDet/I");
   mTree->Branch("e", &bE, "e/F");
   mTree->Branch("x", &bX, "x/F");
   mTree->Branch("y", &bY, "y/F");
   mTree->Branch("pt", &bPt, "pt/F");
   mTree->Branch("eta", &bEta, "eta/F");
   mTree->Branch("phi", &bPhi, "phi/F");
   mTree->Branch("nTowers", &bNTowers, "nTowers/I");
   mTree->Branch("nTowRec", &bNTowRec, "nTowRec/I");  // towers the association gave it
   mTree->Branch("nNeighbor", &bNNeighbor, "nNeighbor/I");
   mTree->Branch("sigmaMin", &bSigmaMin, "sigmaMin/F");
   mTree->Branch("sigmaMax", &bSigmaMax, "sigmaMax/F");
   mTree->Branch("theta", &bTheta, "theta/F");
   mTree->Branch("chi2ndf1", &bChi2Ndf1, "chi2ndf1/F");
   mTree->Branch("chi2ndf2", &bChi2Ndf2, "chi2ndf2/F");
   mTree->Branch("catStar", &bCatStar, "catStar/I");
   mTree->Branch("catML", &bCatML, "catML/I");
   mTree->Branch("feat", bFeat, Form("feat[%d]/F", nv));
   mTree->Branch("prob", bProb, "prob[3]/F");

   bookHistograms();
   return kStOK;
}

//-----------------------------------------------------------------------------
void StFcsPicoCategoryMaker::bookHistograms() {
   h2_catStar_vs_catML = new TH2F("h2_catStar_vs_catML", "STAR category vs ML category;STAR;ML", 3, 0, 3, 3, 0, 3);
   h1_prob[0] = new TH1F("h1_prob0", "response, class 0 (ambiguous)", 100, 0, 1);
   h1_prob[1] = new TH1F("h1_prob1", "response, class 1 (1 photon)", 100, 0, 1);
   h1_prob[2] = new TH1F("h1_prob2", "response, class 2 (2 photons)", 100, 0, 1);
   h1_nCluEcal = new TH1F("h1_nCluEcal", "ECal clusters per event", 40, 0, 40);
   h1_invmass_star = new TH1F("h1_invmass_star", "cluster pair mass, STAR category;m_{#gamma#gamma} [GeV]", 150, 0, 0.4);
   h1_invmass_ml = new TH1F("h1_invmass_ml", "cluster pair mass, ML category;m_{#gamma#gamma} [GeV]", 150, 0, 0.4);
   h1_zgg = new TH1F("h1_zgg", "energy asymmetry;Z_{#gamma#gamma}", 150, 0, 1);
   h1_dgg = new TH1F("h1_dgg", "opening angle;#theta_{#gamma#gamma} [rad]", 150, 0, 0.1);
   h2_invmass_vs_zgg = new TH2F("h2_invmass_vs_zgg", "mass vs Z_{#gamma#gamma};Z_{#gamma#gamma};m [GeV]", 35, 0, 0.7, 60, 0, 0.4);
}

//-----------------------------------------------------------------------------
// Build the feature vector for one picoDst cluster and, if a model is loaded,
// evaluate it. feat[] must hold nVar(mFeatureSet) floats, prob[] three.
int StFcsPicoCategoryMaker::evaluate(StPicoFcsCluster* clu, int nTow, const float* towE,
                                     const int* towRow, const int* towCol, int nNeighbor,
                                     float* feat, float* prob) {
   const int nv = nVar(mFeatureSet);
   for (int k = 0; k < 3; k++) prob[k] = 0.0;

   const int det = clu->detectorId();

   ClusterInput c;
   c.e = clu->energy();
   c.x = clu->x();
   c.y = clu->y();
   c.sigmaMin = clu->sigmaMin();
   c.sigmaMax = clu->sigmaMax();
   c.theta = clu->theta();
   c.nTowers = clu->nTowers();
   // Not stored in picoDst - recomputed from the tower adjacency by
   // StFcsTowerAssoc::neighborCounts, so that set 13 sees the same variable here
   // as it did in training. Only set 13 uses it.
   c.nNeighbor = nNeighbor;
   c.xw = mFcsDb ? mFcsDb->getXWidth(det) : 0;
   c.yw = mFcsDb ? mFcsDb->getYWidth(det) : 0;
   // Empty for set 6, which needs no tower list; recovered by StFcsTowerAssoc
   // for sets 3, 13 and 34.
   c.nTow = nTow;
   c.towerE = towE;
   c.towerRow = towRow;
   c.towerCol = towCol;

   if (compute(mFeatureSet, c, feat) != nv) return -1;
   if (mNoModel) return -1;

   for (int k = 0; k < nv; k++) mVar[k] = feat[k];
   const std::vector<Float_t>& r = mReader->EvaluateMulticlass(mTMVAMethod.c_str());
   if (r.size() != 3) {
      LOG_WARN << "StFcsPicoCategoryMaker: multiclass response has " << r.size()
               << " entries, expected 3 - was the method trained in multiclass mode?" << endm;
      return -1;
   }
   int best = 0;
   for (int k = 0; k < 3; k++) {
      prob[k] = r[k];
      if (r[k] > r[best]) best = k;
   }
   if (mMode == kOverrideIfConfident && r[best] < mConfidence) return clu->category();
   return best;
}

//-----------------------------------------------------------------------------
Int_t StFcsPicoCategoryMaker::Make() {
   mPicoDst = mPicoDstMaker->picoDst();
   if (!mPicoDst) {
      LOG_ERROR << "StFcsPicoCategoryMaker::Make no StPicoDst" << endm;
      return kStErr;
   }
   StPicoEvent* ev = mPicoDst->event();
   if (!ev) return kStErr;

   mNEvents++;
   bRun = ev->runId();
   bEvent = ev->eventId();
   bVz = ev->primaryVertex().z();

   const int nv = nVar(mFeatureSet);
   const unsigned int nclu = mPicoDst->numberOfFcsClusters();

   // count ECal clusters first, for the per-event branch and the QA
   int nEcal = 0;
   for (unsigned int i = 0; i < nclu; i++) {
      StPicoFcsCluster* c = mPicoDst->fcsCluster(i);
      if (c && c->detectorId() < 2) nEcal++;
   }
   h1_nCluEcal->Fill(nEcal);
   bNCluDet = nEcal;

   // evaluate every ECal cluster once; the pi0 pairing reads the result back
   mCatML.assign(nclu, -1);

   // One detector half at a time, because the tower association is per half.
   for (int det = 0; det < 2; det++) {
      // ---- towers of this half, only if the feature set needs them ----
      std::vector<StFcsTowerAssoc::Tower> tow;
      if (mFcsDb) {
         const int nCol = mFcsDb->nColumn(det);
         const int nRow = mFcsDb->nRow(det);
         const unsigned int nhit = mPicoDst->numberOfFcsHits();
         for (unsigned int i = 0; i < nhit; i++) {
            StPicoFcsHit* h = mPicoDst->fcsHit(i);
            if (!h || (int)h->detectorId() != det) continue;
            const int row = mFcsDb->getRowNumber(det, h->id());
            const int col = mFcsDb->getColumnNumber(det, h->id());
            if (row < 1 || row > nRow || col < 1 || col > nCol) continue;
            StFcsTowerAssoc::Tower t;
            t.id = h->id();
            t.row = row;
            t.col = col;
            t.e = h->energy();
            tow.push_back(t);
         }
      }

      // ---- clusters of this half ----
      std::vector<int> cluIdx;
      std::vector<float> cx, cy;
      std::vector<int> cn;
      for (unsigned int i = 0; i < nclu; i++) {
         StPicoFcsCluster* c = mPicoDst->fcsCluster(i);
         if (!c || (int)c->detectorId() != det) continue;
         cluIdx.push_back((int)i);
         cx.push_back(c->x());
         cy.push_back(c->y());
         cn.push_back(mUseNTow ? c->nTowers() : 0);
      }
      const int nc = (int)cluIdx.size();
      if (nc == 0) continue;

      std::vector<int> owner;
      std::vector<int> nNbr;
      if (!tow.empty()) {
         StFcsTowerAssoc::assign(tow, nc, &cx[0], &cy[0], &cn[0], mMaxDist, owner);
         StFcsTowerAssoc::neighborCounts(tow, owner, nc, mFcsDb->nRow(det), mFcsDb->nColumn(det),
                                         mNeighborDist, nNbr);
      }

      std::vector<StFcsTowerAssoc::Tower> mine;
      std::vector<float> te;
      std::vector<int> trow, tcol;

      for (int ic = 0; ic < nc; ic++) {
         const int i = cluIdx[ic];
         StPicoFcsCluster* clu = mPicoDst->fcsCluster(i);
         if (!clu) continue;

         mine.clear();
         if (!owner.empty()) StFcsTowerAssoc::gather(tow, owner, ic, mine);
         te.clear();
         trow.clear();
         tcol.clear();
         for (size_t k = 0; k < mine.size(); k++) {
            te.push_back(mine[k].e);
            trow.push_back(mine[k].row);
            tcol.push_back(mine[k].col);
         }
         const int nTow = (int)te.size();

         float f[kNVarMax];
         float p[3];
         const int nNb = (ic < (int)nNbr.size()) ? nNbr[ic] : 0;
         const int catML = evaluate(clu, nTow, nTow ? &te[0] : 0, nTow ? &trow[0] : 0,
                                    nTow ? &tcol[0] : 0, nNb, f, p);
         mCatML[i] = catML;

         if (clu->energy() < mEmin) continue;  // tree and QA threshold

         bDet = det;
         bClId = clu->id();
         bE = clu->energy();
         bX = clu->x();
         bY = clu->y();
         bNTowers = clu->nTowers();
         bNTowRec = nTow;
         bNNeighbor = nNb;
         bSigmaMin = clu->sigmaMin();
         bSigmaMax = clu->sigmaMax();
         bTheta = clu->theta();
         bChi2Ndf1 = clu->chi2Ndf1Photon();
         bChi2Ndf2 = clu->chi2Ndf2Photon();
         bCatStar = clu->category();
         const TLorentzVector p4 = clu->fourMomentum();
         bPt = p4.Pt();
         bEta = p4.Eta();
         bPhi = p4.Phi();
         for (int k = 0; k < nv; k++) bFeat[k] = f[k];
         for (int k = 0; k < 3; k++) bProb[k] = p[k];
         bCatML = catML;

         if (catML >= 0) {
            for (int k = 0; k < 3; k++) h1_prob[k]->Fill(p[k]);
            h2_catStar_vs_catML->Fill(bCatStar, bCatML);
            if (bCatML != bCatStar) mNChanged++;
         }

         mNCluster++;
         mTree->Fill();
      }
   }

   // pi0 QA: pair ECal clusters, once selected on the STAR category and once on
   // the ML category. NOTE this is a SELECTION, not a refit - on picoDst there
   // is no StFcsPointMaker downstream, so the category can only decide which
   // clusters enter the pair, never split one cluster into two photons. The
   // full benefit of a better categorizer needs it to run before the point
   // fitter, which means the MuDst/StEvent chain.
   fillPi0(0);
   if (!mNoModel) fillPi0(1);

   return kStOK;
}

//-----------------------------------------------------------------------------
// Pair ECal clusters within the same detector half. useMlCategory=0 selects on
// the STAR category, 1 on the model's. Only single-photon-like clusters
// (category 1) are paired.
void StFcsPicoCategoryMaker::fillPi0(int useMlCategory) {
   const unsigned int nclu = mPicoDst->numberOfFcsClusters();

   for (unsigned int i = 0; i < nclu; i++) {
      StPicoFcsCluster* ci = mPicoDst->fcsCluster(i);
      if (!ci || ci->detectorId() > 1 || ci->energy() < mPairEmin) continue;

      const int cati = useMlCategory ? mCatML[i] : ci->category();
      if (cati != 1) continue;  // keep single-photon-like clusters

      for (unsigned int j = i + 1; j < nclu; j++) {
         StPicoFcsCluster* cj = mPicoDst->fcsCluster(j);
         if (!cj || cj->detectorId() != ci->detectorId() || cj->energy() < mPairEmin) continue;

         const int catj = useMlCategory ? mCatML[j] : cj->category();
         if (catj != 1) continue;

         const float ei = ci->energy(), ej = cj->energy();
         const float zgg = fabs(ei - ej) / (ei + ej);
         if (zgg > mZggMax) continue;

         const TLorentzVector pi = ci->fourMomentum();
         const TLorentzVector pj = cj->fourMomentum();
         const float m = (pi + pj).M();
         const float ang = pi.Vect().Angle(pj.Vect());

         if (useMlCategory) {
            h1_invmass_ml->Fill(m);
         } else {
            h1_invmass_star->Fill(m);
            h1_zgg->Fill(zgg);
            h1_dgg->Fill(ang);
            h2_invmass_vs_zgg->Fill(zgg, m);
         }
      }
   }
}

//-----------------------------------------------------------------------------
Int_t StFcsPicoCategoryMaker::Finish() {
   LOG_INFO << Form("StFcsPicoCategoryMaker: %lld events, %lld ECal clusters, %lld categories changed",
                    mNEvents, mNCluster, mNChanged)
            << endm;
   if (!mFile) return kStOK;
   mFile->cd();
   mTree->Write();
   h2_catStar_vs_catML->Write();
   for (int i = 0; i < 3; i++) h1_prob[i]->Write();
   h1_nCluEcal->Write();
   h1_invmass_star->Write();
   h1_invmass_ml->Write();
   h1_zgg->Write();
   h1_dgg->Write();
   h2_invmass_vs_zgg->Write();
   mFile->Close();
   LOG_INFO << "StFcsPicoCategoryMaker wrote " << mOutFile << endm;
   return kStOK;
}
