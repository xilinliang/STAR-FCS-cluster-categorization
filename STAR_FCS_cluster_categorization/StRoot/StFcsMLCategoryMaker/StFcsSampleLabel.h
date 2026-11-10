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
//   gamma event (GEANT3 pid 1)  the HIGHEST-ENERGY cluster -> 1EM  (class 1)
//                               every other cluster        -> dropped
//   pi- event   (pid 9)         every cluster              -> Hadron (class 0)
//   pi0 event   (pid 7)         if the event has a cluster with mcLabel >= 2,
//                                 the highest-energy such  -> MergedPi0  (2)
//                                 and every OTHER cluster  -> dropped
//                               otherwise (resolved):
//                                 the TWO highest-energy   -> 1EM        (1)
//                                 any further cluster      -> dropped
//   any other gun               no class, dropped
//
// Only clusters at or above eMin are considered; a cluster below it is never
// labelled, never "dropped", and never a candidate, so what the labels
// describe is exactly what the caller will use.
//
// WHY THE 1EM SAMPLE IS TRIMMED. A photon gun makes ONE photon and a resolved
// pi0 makes TWO, so one and two clusters respectively are all the single-EM
// objects an event can contain. Anything further is a fragment, a secondary or
// a split-off, and feeding those to the 1EM class teaches it that the junk
// around a shower is a photon. The energy ranking is the selection: no truth
// matching, so it stays close to what an analysis could do for itself.
//
// AND WHY A MERGED pi0 EVENT CONTRIBUTES NOTHING TO 1EM. The event's two
// photons are both inside the merged cluster, so there is no single-EM object
// left in it. Any other cluster in such an event is, by construction, not one
// of the pi0's photons.
//
// A pi- event never leaves the Hadron sample, whatever the shower did: a
// cluster in a pi- event IS hadronic background, even when a shower pi0 put a
// photon in it. That is the difference from mcLabel (truthDef=0), which would
// call such a cluster single-EM. The hadronic sample is the one place every
// cluster is kept, because a hadron genuinely does spread over several and no
// subset of them is more "the hadron" than the rest.
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
   Long64_t nKept[3];  // clusters given a class, per gun (pi0 feeds two samples)
   Long64_t nDrop[3];  // clusters above eMin left out by the ranking, per gun
   Long64_t nMerged;   // pi0 events that produced a merged cluster
   Long64_t nBelowE;   // clusters below eMin, never candidates
   Long64_t nNoGun;    // clusters of an unrecognised gun, left without a class
   Report() : nMerged(0), nBelowE(0), nNoGun(0) {
      for (int i = 0; i < 3; i++) nClus[i] = nEvt[i] = nKept[i] = nDrop[i] = 0;
   }
};

// Close out one event: label every cluster it collected. A free function and
// not a lambda, because ROOT 5 builds this as C++98.
inline void finishEvent(int gun, const std::vector<Long64_t>& idx, const std::vector<float>& ee,
                        const std::vector<int>& ml, float eMin,
                        std::vector<signed char>& label, Report& rep) {
   const size_t n = idx.size();
   if (n == 0) return;
   if (gun != kGamma && gun != kPi0 && gun != kPiMinus) {
      rep.nNoGun += (Long64_t)n;  // labels stay -1
      return;
   }

   // only the clusters the caller will actually use can be candidates
   std::vector<size_t> ok;
   for (size_t i = 0; i < n; i++) {
      if (ee[i] < eMin)
         rep.nBelowE += 1;
      else
         ok.push_back(i);
   }
   const int gs = (gun == kGamma) ? 0 : (gun == kPi0) ? 1 : 2;
   rep.nEvt[gs] += 1;
   if (ok.empty()) return;

   // pi-: every cluster, because a hadron genuinely spreads over several and
   // no subset of them is more "the hadron" than the rest
   if (gun == kPiMinus) {
      for (size_t j = 0; j < ok.size(); j++) {
         label[(size_t)idx[ok[j]]] = 0;
         rep.nClus[0] += 1;
         rep.nKept[2] += 1;
      }
      return;
   }

   // gamma: one photon, so one single-EM object - the leading cluster
   if (gun == kGamma) {
      size_t best = ok[0];
      for (size_t j = 1; j < ok.size(); j++)
         if (ee[ok[j]] > ee[best]) best = ok[j];
      label[(size_t)idx[best]] = 1;
      rep.nClus[1] += 1;
      rep.nKept[0] += 1;
      rep.nDrop[0] += (Long64_t)(ok.size() - 1);
      return;
   }

   // pi0: merged, or resolved
   int merged = -1;
   for (size_t j = 0; j < ok.size(); j++) {
      const size_t i = ok[j];
      if (ml[i] < 2) continue;
      if (merged < 0 || ee[i] > ee[(size_t)merged]) merged = (int)i;
   }
   if (merged >= 0) {
      // both photons are inside this one cluster, so the event holds no
      // single-EM object at all: nothing else from it is kept
      label[(size_t)idx[(size_t)merged]] = 2;
      rep.nClus[2] += 1;
      rep.nKept[1] += 1;
      rep.nDrop[1] += (Long64_t)(ok.size() - 1);
      rep.nMerged += 1;
      return;
   }
   // resolved: the two leading clusters are the two photons
   size_t b1 = ok[0], b2 = 0;
   bool haveB2 = false;
   for (size_t j = 1; j < ok.size(); j++) {
      const size_t i = ok[j];
      if (ee[i] > ee[b1]) {
         b2 = b1;
         haveB2 = true;
         b1 = i;
      } else if (!haveB2 || ee[i] > ee[b2]) {
         b2 = i;
         haveB2 = true;
      }
   }
   label[(size_t)idx[b1]] = 1;
   rep.nClus[1] += 1;
   rep.nKept[1] += 1;
   if (haveB2) {
      label[(size_t)idx[b2]] = 1;
      rep.nClus[1] += 1;
      rep.nKept[1] += 1;
   }
   rep.nDrop[1] += (Long64_t)(ok.size() - (haveB2 ? 2 : 1));
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
   const char* gn[3] = {"gamma", "pi0", "pi-"};
   printf("\n  sample-based truth (StFcsSampleLabel.h)\n");
   printf("    %-12s %10s %8s\n", "sample", "clusters", "share");
   for (int k = 0; k < 3; k++)
      printf("    %-12s %10lld %7.1f%%\n", nm[k], rep.nClus[k],
             tot > 0 ? 100.0 * rep.nClus[k] / tot : 0.0);
   printf("    %-12s %10lld\n", "TOTAL", tot);
   printf("\n    %-8s %9s %9s %9s %8s   %s\n", "gun", "events", "kept", "dropped", "kept/evt",
          "selection rule");
   const char* how[3] = {"leading cluster only",
                         "merged: that one alone; resolved: top 2", "every cluster"};
   for (int g = 0; g < 3; g++)
      printf("    %-8s %9lld %9lld %9lld %8.2f   %s\n", gn[g], rep.nEvt[g], rep.nKept[g],
             rep.nDrop[g], rep.nEvt[g] > 0 ? (double)rep.nKept[g] / rep.nEvt[g] : 0.0, how[g]);
   printf("    (a pi0 event feeds MergedPi0 or 1EM, never both; gamma feeds 1EM, pi- Hadron)\n");
   printf("\n    pi0 events with a merged cluster: %lld of %lld (%.1f%%). The rest are\n"
          "      resolved and give their two leading clusters to 1EM.\n",
          rep.nMerged, rep.nEvt[1],
          rep.nEvt[1] > 0 ? 100.0 * rep.nMerged / rep.nEvt[1] : 0.0);
   printf("    dropped = above eMin but not selected by the ranking: %lld in all\n",
          rep.nDrop[0] + rep.nDrop[1] + rep.nDrop[2]);
   if (rep.nBelowE > 0) printf("    below eMin, never candidates: %lld\n", rep.nBelowE);
   if (rep.nNoGun > 0)
      printf("    %lld cluster(s) of an unrecognised gun, left without a class\n", rep.nNoGun);
}

}  // namespace StFcsSampleLabel

#endif
