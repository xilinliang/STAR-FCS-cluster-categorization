// StFcsTrainTestSplit.h - the training bookkeeping shared by trainTMVA.C and
// evalCategory.C, so the two can never disagree about it:
//   - which clusters are TRAINING and which are TEST   (EventSplitter)
//   - what a cluster's true class is                   (trainingLabel)
//     or, ePIC-style, by generated particle          (sampleLabel, inputCategory)
//
// WHY THIS EXISTS
//
// TMVA's default SplitMode=Random picks the test half with its own internal
// random seed, and nothing outside TMVA can tell afterwards which clusters it
// used. Any efficiency or purity computed later by a separate macro would then
// be measured partly on clusters the model was TRAINED on, which flatters it -
// and a BDT with 600 trees can memorise quite a lot.
//
// So the split is made here, deterministically, and both macros apply the same
// rule to the same feature file.
//
// WHOLE EVENTS, NOT SINGLE CLUSTERS. The two photon clusters of one pi0 are
// correlated - same event, same vertex, related energies. Alternating on the
// cluster index would put one in training and its partner in test, which leaks
// information across the split. Every cluster of an event therefore goes to
// the same side: the event ordinal is counted as the tree is read in order
// (clusters of one event are consecutive in the tree, because the dumpers fill
// them event by event), and even ordinals train, odd ordinals test.
//
// Call isTest() for EVERY entry, in tree order, before any selection skips it
// - otherwise the ordinal count, and with it the split, depends on the cuts.
//
// Header-only, no ROOT, no STAR.
//
// author: generated for Xilin Liang

#ifndef STAR_StFcsTrainTestSplit_HH
#define STAR_StFcsTrainTestSplit_HH

namespace StFcsTrainTestSplit {

class EventSplitter {
  public:
   EventSplitter() : mFirst(true), mRun(0), mEvent(0), mOrdinal(-1) {}

   // 1 = test, 0 = training. Also counts a new event whenever (run, event)
   // differs from the previous entry.
   int isTest(int run, int event) {
      if (mFirst || run != mRun || event != mEvent) {
         mOrdinal++;
         mRun = run;
         mEvent = event;
         mFirst = false;
      }
      return (int)(mOrdinal % 2);
   }

   long nEvents() const { return mOrdinal + 1; }

  private:
   bool mFirst;
   int mRun;
   int mEvent;
   long mOrdinal;
};

// The true class of a cluster, from the truth branches of the feature tree:
// 0 = other, 1 = one photon, 2 = two or more photons, -1 = not usable.
//
// mcLabel (generator level) wins whenever it exists: it counts the generated
// photons projecting onto the cluster and needs no purity cut. truthNPhoton
// (hit level, .fzd only) is the fallback for files written before mcLabel
// existed, and there a cluster whose leading GEANT track carries less than
// purityCut of the energy is too mixed to label.
inline int trainingLabel(int mcLabel, int truthNPhoton, float truthPurity, float purityCut) {
   if (mcLabel >= 0) return (mcLabel > 2) ? 2 : mcLabel;
   if (truthNPhoton < 0) return -1;
   if (truthNPhoton >= 2) return 2;
   if (truthPurity < purityCut) return -1;
   return truthNPhoton;  // 0 or 1
}

// ---------------------------------------------------------------------------
// GENERATED-PARTICLE LABELS - the ePIC-style alternative (labelDef = 1 in
// trainTMVA.C, truthDef = 1 in evalCategory.C). The class is set by which
// single-particle sample the cluster came from (genPid, GEANT3: 1 gamma,
// 7 pi0, 9 pi-), cleaned with mcLabel so that each class holds the object it
// is named after:
//
//   input category (inputCategory)        class trained (sampleLabel)
//   0  gamma input      gamma, mcLabel 1  -> 1 single EM
//   1  pi0 2-cluster    pi0,   mcLabel 1  -> not trained (-1): one photon of a
//                                            RESOLVED pi0, i.e. a single EM
//                                            shower that no feature can tell
//                                            from a gun photon
//   2  pi0 1-cluster    pi0,   mcLabel 2  -> 2 merged pi0
//   3  pi- input        pi-,   any        -> 0 hadronic
//   -1 anything else (gun fragments with no photon inside, other guns)
//
// "1-cluster" / "2-cluster" is decided per CLUSTER from the photons inside it:
// both pi0 photons in this cluster = the pi0 made one cluster; one photon =
// the pi0 was resolved into two. That is the event-level ePIC split without
// being fooled by small fragment clusters.
inline int inputCategory(int genPid, int mcLabel) {
   if (genPid == 1) return (mcLabel == 1) ? 0 : -1;
   if (genPid == 7) {
      if (mcLabel == 1) return 1;
      if (mcLabel >= 2) return 2;
      return -1;
   }
   if (genPid == 9) return 3;
   return -1;
}

inline int sampleLabel(int genPid, int mcLabel) {
   switch (inputCategory(genPid, mcLabel)) {
      case 0: return 1;  // single EM
      case 2: return 2;  // merged pi0
      case 3: return 0;  // hadronic
      default: return -1;
   }
}

}  // namespace StFcsTrainTestSplit

#endif
