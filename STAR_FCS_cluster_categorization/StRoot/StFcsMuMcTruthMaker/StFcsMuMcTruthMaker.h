// class StFcsMuMcTruthMaker
//
// Feeds generator-level truth to StFcsClusterFeatureMaker when the input is a
// MuDst rather than a GEANT .fzd.
//
// WHY THIS EXISTS
//
// Hit-level truth - which GEANT track deposited energy in which tower - does
// not survive into a MuDst: StMuFcsHit stores detectorId, id, adc and energy
// and nothing else. But GENERATOR-level truth does. StMuMcTrack is built
// directly from g2t_track_st and keeps GePid(), Id(), IsShower(), IdVx(), E()
// and Pxyz(), with StMuMcVertex::XyzV() for the start position - exactly the
// inputs the photon projection needs.
//
// So a MuDst produced with the MC arrays gives labels after all. Put this maker
// anywhere before StFcsClusterFeatureMaker in a MuDst chain and mcLabel, mcSep
// and mcZgg get filled the same way they do on the .fzd path. What you do NOT
// get from a MuDst is the hit-level set - trkPid, trkE, truthNPhoton - because
// those links are gone.
//
// Chain position:
//   StMuDstMaker -> ... -> StFcsClusterMaker -> StFcsPointMaker
//                       -> StFcsMuMcTruthMaker -> StFcsClusterFeatureMaker
//
// This lives in its own package on purpose. It links against StMuDSTMaker, and
// StFcsClusterFeatureMaker must not: that library also loads in the .fzd chain,
// where the MuDst libraries are absent, and an unresolved StMuDst symbol would
// make it fail to dlopen.
//
// author: generated for Xilin Liang

#ifndef STAR_StFcsMuMcTruthMaker_HH
#define STAR_StFcsMuMcTruthMaker_HH

#include "TString.h"

#include "StMaker.h"

class StFcsClusterFeatureMaker;

class StFcsMuMcTruthMaker : public StMaker {
  public:
   // ORDER MATTERS. StChain runs makers in the order they were constructed, so
   // this one must be constructed BEFORE StFcsClusterFeatureMaker or the
   // photons arrive an event too late. That is awkward with a pointer argument,
   // since the dumper does not exist yet - so the default constructor takes the
   // dumper's NAME and resolves it in Init(), which runs after every maker has
   // been built. Construct this one first, the dumper second.
   StFcsMuMcTruthMaker(const Char_t* name = "FcsMuMcTruth",
                       const Char_t* featureMakerName = "FcsClusFeat");
   // Explicit-pointer form, for a chain where the dumper already exists.
   StFcsMuMcTruthMaker(StFcsClusterFeatureMaker* feat, const Char_t* name);
   ~StFcsMuMcTruthMaker();

   Int_t Init();
   Int_t Make();
   Int_t Finish();

   // Keep only photons above this energy. 0 keeps them all.
   void setPhotonEnergyThreshold(float e) { mEmin = e; }

  private:
   StFcsClusterFeatureMaker* mFeat;
   TString mFeatName;
   float mEmin;
   Long64_t mNEvents;
   Long64_t mNPhotons;
   Long64_t mNNoMcArray;

#ifndef SKIPDefImp
   ClassDef(StFcsMuMcTruthMaker, 0)
#endif
};

#endif
