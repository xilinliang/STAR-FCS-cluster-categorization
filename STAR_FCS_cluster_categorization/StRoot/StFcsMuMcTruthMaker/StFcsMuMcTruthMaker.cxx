// class StFcsMuMcTruthMaker
// see the header for why this is a separate package

#include "StFcsMuMcTruthMaker.h"

#include "StChain.h"
#include "StMessMgr.h"
#include "Stypes.h"
#include "TClonesArray.h"
#include "TList.h"

#include "StMuDSTMaker/COMMON/StMuDst.h"
#include "StMuDSTMaker/COMMON/StMuMcTrack.h"
#include "StMuDSTMaker/COMMON/StMuMcVertex.h"

#include "StFcsClusterFeatureMaker/StFcsClusterFeatureMaker.h"

// StMuArrays: mcArrayNames[] = { "StMuMcVertex", "StMuMcTrack" }
static const int kMcVertexArray = 0;
static const int kMcTrackArray = 1;

#ifndef SKIPDefImp
ClassImp(StFcsMuMcTruthMaker)
#endif

    StFcsMuMcTruthMaker::StFcsMuMcTruthMaker(const Char_t* name, const Char_t* featureMakerName)
    : StMaker(name),
      mFeat(0),
      mFeatName(featureMakerName),
      mEmin(0.0),
      mNEvents(0),
      mNPhotons(0),
      mNNoMcArray(0) {}

StFcsMuMcTruthMaker::StFcsMuMcTruthMaker(StFcsClusterFeatureMaker* feat, const Char_t* name)
    : StMaker(name), mFeat(feat), mFeatName(""), mEmin(0.0), mNEvents(0), mNPhotons(0), mNNoMcArray(0) {}

StFcsMuMcTruthMaker::~StFcsMuMcTruthMaker() {}

//-----------------------------------------------------------------------------
Int_t StFcsMuMcTruthMaker::Init() {
   // Resolve the dumper by name if it was not handed over directly. Init() runs
   // after every maker in the chain has been constructed, so the dumper exists
   // by now even though it was built after this maker.
   if (!mFeat && mFeatName.Length() > 0)
      mFeat = (StFcsClusterFeatureMaker*)GetMaker(mFeatName.Data());

   if (!mFeat) {
      LOG_ERROR << "StFcsMuMcTruthMaker::Init could not find StFcsClusterFeatureMaker \""
                << mFeatName << "\" in the chain" << endm;
      return kStFatal;
   }

   // Warn about the ordering mistake that would otherwise be invisible: if this
   // maker runs AFTER the dumper, every event's photons arrive one event late.
   StMaker* first = 0;
   TList* makers = GetParentChain() ? GetParentChain()->GetMakeList() : 0;
   if (makers) {
      TIter next(makers);
      StMaker* m = 0;
      while ((m = (StMaker*)next())) {
         if (m == this || m == (StMaker*)mFeat) {
            first = m;
            break;
         }
      }
      if (first == (StMaker*)mFeat)
         LOG_WARN << "StFcsMuMcTruthMaker runs AFTER StFcsClusterFeatureMaker: the generated "
                  << "photons will be one event late. Construct this maker first." << endm;
   }
   return kStOK;
}

//-----------------------------------------------------------------------------
Int_t StFcsMuMcTruthMaker::Make() {
   mNEvents++;

   TClonesArray* mcTracks = StMuDst::mcArray(kMcTrackArray);
   TClonesArray* mcVertices = StMuDst::mcArray(kMcVertexArray);
   if (!mcTracks || !mcVertices) {
      // A MuDst written without the MC arrays. Leave the feature maker alone so
      // that it falls back to the g2t tables if they happen to be there.
      mNNoMcArray++;
      return kStOK;
   }

   mFeat->clearMcPhotons();

   const int ntrk = mcTracks->GetEntriesFast();
   const int nvtx = mcVertices->GetEntriesFast();

   // The generated (gun) particle, by the same rule as the other two tiers:
   // track id 1 is one of the generator's particles, the primaries are the
   // tracks sharing its start vertex, and the most energetic one is recorded.
   {
      StMuMcTrack* first = 0;
      for (int i = 0; i < ntrk; i++) {
         StMuMcTrack* t = (StMuMcTrack*)mcTracks->UncheckedAt(i);
         if (t && t->Id() == 1) {
            first = t;
            break;
         }
      }
      if (!first && ntrk > 0) first = (StMuMcTrack*)mcTracks->UncheckedAt(0);
      if (first) {
         const int v1 = first->IdVx();
         int nPrim = 0, pid = 0;
         float eMax = 0;
         for (int i = 0; i < ntrk; i++) {
            StMuMcTrack* t = (StMuMcTrack*)mcTracks->UncheckedAt(i);
            if (!t || t->IdVx() != v1) continue;
            nPrim++;
            if (t->E() > eMax) {
               eMax = t->E();
               pid = t->GePid();
            }
         }
         mFeat->setGenerated(pid, eMax, nPrim);
      }
   }

   for (int i = 0; i < ntrk; i++) {
      StMuMcTrack* t = (StMuMcTrack*)mcTracks->UncheckedAt(i);
      if (!t) continue;
      if (t->GePid() != 1) continue;    // GEANT3 pid 1 = gamma
      if (t->IsShower()) continue;      // drop electromagnetic cascade products
      if (t->E() <= mEmin) continue;

      // start vertex, by id
      const int idVx = t->IdVx();
      float vx = 0, vy = 0, vz = 0;
      int found = 0;
      for (int j = 0; j < nvtx; j++) {
         StMuMcVertex* v = (StMuMcVertex*)mcVertices->UncheckedAt(j);
         if (!v || v->Id() != idVx) continue;
         vx = v->XyzV().x();
         vy = v->XyzV().y();
         vz = v->XyzV().z();
         found = 1;
         break;
      }
      if (!found) continue;

      // parent: StMuMcTrack has no parent-track pointer, only the start vertex,
      // so the parent is the track that ENDS at this vertex. For a pi0 gun that
      // is the pi0 itself.
      int parentId = 0, parentPid = 0;
      for (int j = 0; j < ntrk; j++) {
         StMuMcTrack* p = (StMuMcTrack*)mcTracks->UncheckedAt(j);
         if (!p || p->Id() == t->Id()) continue;
         if (p->IdVxEnd() != idVx) continue;
         parentId = p->Id();
         parentPid = p->GePid();
         break;
      }

      mFeat->addMcPhoton(t->Id(), parentId, parentPid, t->E(),
                         t->Pxyz().x(), t->Pxyz().y(), t->Pxyz().z(), vx, vy, vz);
      mNPhotons++;
   }

   return kStOK;
}

//-----------------------------------------------------------------------------
Int_t StFcsMuMcTruthMaker::Finish() {
   LOG_INFO << Form("StFcsMuMcTruthMaker: %lld events, %lld generated photons (%.2f/event)", mNEvents,
                    mNPhotons, mNEvents ? double(mNPhotons) / mNEvents : 0.0)
            << endm;
   if (mNNoMcArray > 0) {
      LOG_WARN << Form(
                      "StFcsMuMcTruthMaker: %lld events had no StMuMcTrack/StMuMcVertex array. "
                      "That MuDst was produced without the MC arrays, so there is no "
                      "generator-level truth in it and mcLabel will stay at -1.",
                      mNNoMcArray)
               << endm;
   }
   return kStOK;
}
