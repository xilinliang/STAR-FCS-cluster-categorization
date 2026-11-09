// StFcsSampleLabel.h
//
// THE SAMPLE-BASED truth definition - "which input sample is this cluster
// from" - shared by trainTMVA.C (labelDef=2) and evalCategory.C (truthDef=3)
// so the two cannot drift. Same reason StFcsTrainTestSplit.h holds the other
// label rules and StFcsEnergyWeights.h holds the weight.
//
// WHY IT EXISTS. The other definitions decide a cluster's class from what is
// inside that ONE cluster, and the cleaned one (sampleLabel, labelDef=1 /
// truthDef=1) DROPS every cluster it cannot place - above all the two clusters
// of a resolved pi0, which hold one photon each. A dropped cluster leaves both
// the efficiency numerator and the PURITY DENOMINATOR, so the most natural
// fake at low energy - one photon of a resolved pi0 tagged as a merged pi0 -
// costs nothing there. This definition has no such exit: every cluster of a
// recognised gun event lands in exactly one of three samples, so a purity
// computed with it is over every cluster the model was shown.
//
// THE SAMPLES, decided per EVENT:
//
//   gamma event (GEANT3 pid 1)  every cluster -> 1EM           (class 1)
//   pi- event   (pid 9)         every cluster -> Hadron        (class 0)
//   pi0 event   (pid 7)         the highest-energy cluster with mcLabel >= 2,
//                               if the event has one           -> MergedPi0 (2)
//                               every other cluster of it      -> 1EM       (1)
//   any other gun               no class, dropped
//
// So a RESOLVED pi0 contributes both its clusters to the 1EM sample, which is
// what they physically are - one photon each - and a MERGED pi0 contributes
// its one cluster to the MergedPi0 sample. A pi0 event never contributes to
// the Hadron sample, and a pi- event never leaves it, whatever the shower did:
// a cluster in a pi- event IS hadronic background, even when a shower pi0 put
// a photon in it. That is the difference from mcLabel (truthDef=0), which
// would call such a cluster single-EM.
//
// Then, for each class k, over the three samples together:
//   efficiency(k) = (model says k, among the k sample) / (the k sample)
//   purity(k)     = (model says k, among the k sample) / (model says k, in ALL
//                                                         three samples)
//
// WHAT IT IS NOT. This asks "which gun fired the event, one cluster at a
// time", with the pi0 events split by whether the two photons landed in one
// cluster. It is a cross-check of the other definitions, not a replacement:
// the efficiency it reports for 1EM mixes prompt photons with the photons of
// resolved pi0s, which are the same object to a shower-shape classifier but
// not to an analysis. Read it beside truthDef=1, not instead of it.
//
// THE PI0 SPLIT uses mcLabel >= 2 - two or more GENERATED photons projecting
// within the feature maker's match radius (setMcMatchRadius, 11 cm, about two
// ECal towers). That radius puts a ceiling on the separation a pair can have
// and still count as merged: roughly four tower widths for a symmetric pair,
// nearer two for a very asymmetric one. Beyond it the event is called resolved
// and both of its clusters go to 1EM.
//
// MECHANICS. The class of a cluster depends on the OTHER clusters of its
// event, so it cannot be decided in the single pass that reads the features.
// build() makes one extra pass over the tree, keyed on (run, event) exactly as
// StFcsTrainTestSplit::EventSplitter is, and returns one label per entry for
// the caller to index by entry number.
//
// It opens the file AGAIN on its own handle rather than borrowing the caller's
// TTree. That costs one more open and is worth it: hooking up branch addresses
// on a tree the caller is already reading, and then resetting them, silently
// detaches every feature branch the caller set up, and nothing about the
// resulting garbage looks like an error. With a private handle the call is
// safe wherever it is placed.
//
// author: generated for Xilin Liang

#ifndef STAR_StFcsSampleLabel_HH
#define STAR_StFcsSampleLabel_HH

#include <cstdio>
#include <vector>
#include "Rtypes.h"
#include "TFile.h"
#include "TTree.h"

namespace StFcsSampleLabel {

// GEANT3 particle ids of the three guns
const int kGamma = 1;
const int kPi0 = 7;
const int kPiMinus = 9;

// What the pass found, for the printed summary.
struct Report {
   Long64_t nClus[3];  // clusters per sample: 0 Hadron, 1 1EM, 2 MergedPi0
   Long64_t nEvt[3];   // events per gun: 0 gamma, 1 pi0, 2 pi-
   Long64_t nMerged;   // pi0 events that produced a merged cluster
   Long64_t nNoGun;    // clusters of an unrecognised gun, left without a class
   Report() : nMerged(0), nNoGun(0) {
      for (int i = 0; i < 3; i++) nClus[i] = nEvt[i] = 0;
   }
};

// Close out one event: label every cluster it collected. A free function and
// not a lambda, because ROOT 5 builds this as C++98.
inline void finishEvent(int gun, const std::vector<Long64_t>& idx, const std::vector<float>& ee,
                        const std::vector<int>& ml, float eMin,
                        std::vector<signed char>& label, Report& rep) {
   const size_t n = idx.size();
   if (n == 0) return;

   if (gun == kGamma || gun == kPiMinus) {
      const signed char k = (gun == kGamma) ? 1 : 0;
      for (size_t i = 0; i < n; i++) label[(size_t)idx[i]] = k;
      rep.nClus[(int)k] += (Long64_t)n;
      rep.nEvt[(gun == kGamma) ? 0 : 2] += 1;
      return;
   }

   if (gun == kPi0) {
      // the merged cluster, if the event made one: the highest energy among
      // the clusters holding two or more generated photons. A sub-eMin cluster
      // cannot be it, or an event could spend its merged slot on a cluster
      // that is then cut away.
      int best = -1;
      for (size_t i = 0; i < n; i++) {
         if (ml[i] < 2 || ee[i] < eMin) continue;
         if (best < 0 || ee[i] > ee[(size_t)best]) best = (int)i;
      }
      for (size_t i = 0; i < n; i++) {
         const bool merged = ((int)i == best);
         label[(size_t)idx[i]] = merged ? 2 : 1;
         rep.nClus[merged ? 2 : 1] += 1;
      }
      rep.nEvt[1] += 1;
      if (best >= 0) rep.nMerged += 1;
      return;
   }

   rep.nNoGun += (Long64_t)n;  // label stays -1
}

// One extra pass over the tree, on a private file handle. label[i] is the
// class of entry i, or -1. Returns the number of entries labelled, or -1 when
// the file cannot support the definition (unreadable, no tree, or no genPid /
// mcLabel branch).
inline Long64_t build(const char* infile, const char* treename,
                      std::vector<signed char>& label, float eMin, Report* out = 0) {
   label.clear();
   TFile* f = TFile::Open(infile, "READ");
   if (!f || f->IsZombie()) {
      if (f) delete f;
      return -1;
   }
   TTree* t = (TTree*)f->Get(treename);
   if (!t || !t->GetBranch("genPid") || !t->GetBranch("mcLabel")) {
      delete f;
      return -1;
   }

   Int_t run = 0, event = 0, genPid = 0, mcLabel = -1;
   Float_t e = 0;
   // only the five branches the definition needs, so the pass is cheap
   t->SetBranchStatus("*", false);
   const char* want[5] = {"run", "event", "genPid", "mcLabel", "e"};
   for (int b = 0; b < 5; b++) t->SetBranchStatus(want[b], true);
   t->SetBranchAddress("run", &run);
   t->SetBranchAddress("event", &event);
   t->SetBranchAddress("genPid", &genPid);
   t->SetBranchAddress("mcLabel", &mcLabel);
   t->SetBranchAddress("e", &e);

   const Long64_t n = t->GetEntries();
   label.assign((size_t)n, (signed char)-1);

   Report rep;
   std::vector<Long64_t> idx;
   std::vector<float> ee;
   std::vector<int> ml;
   int gun = 0;
   Int_t curRun = 0, curEvt = 0;
   bool have = false;

   for (Long64_t i = 0; i < n; i++) {
      t->GetEntry(i);
      if (!have || run != curRun || event != curEvt) {
         finishEvent(gun, idx, ee, ml, eMin, label, rep);
         idx.clear();
         ee.clear();
         ml.clear();
         curRun = run;
         curEvt = event;
         have = true;
         // the gun of an event is the gun of its first cluster; a single-
         // particle sample has exactly one per event by construction
         gun = (genPid == kGamma || genPid == kPi0 || genPid == kPiMinus) ? genPid : 0;
      }
      idx.push_back(i);
      ee.push_back(e);
      ml.push_back(mcLabel);
   }
   finishEvent(gun, idx, ee, ml, eMin, label, rep);

   delete f;  // takes the tree with it; the labels are already copied out
   if (out) *out = rep;
   return n;
}

inline void printReport(const Report& rep) {
   const Long64_t tot = rep.nClus[0] + rep.nClus[1] + rep.nClus[2];
   const char* nm[3] = {"Hadron", "1EM", "MergedPi0"};
   printf("\n  sample-based truth: every cluster of a recognised gun event has a class\n");
   printf("    %-12s %10s %9s\n", "sample", "clusters", "share");
   for (int k = 0; k < 3; k++)
      printf("    %-12s %10lld %8.1f%%\n", nm[k], rep.nClus[k],
             tot > 0 ? 100.0 * rep.nClus[k] / tot : 0.0);
   printf("    events: %lld gamma, %lld pi0, %lld pi-\n", rep.nEvt[0], rep.nEvt[1], rep.nEvt[2]);
   printf("    pi0 events with a merged cluster: %lld of %lld (%.1f%%); the rest resolved,\n"
          "      with both of their clusters counted as 1EM\n",
          rep.nMerged, rep.nEvt[1],
          rep.nEvt[1] > 0 ? 100.0 * rep.nMerged / rep.nEvt[1] : 0.0);
   if (rep.nNoGun > 0)
      printf("    %lld cluster(s) of an unrecognised gun, left without a class\n", rep.nNoGun);
}

}  // namespace StFcsSampleLabel

#endif
