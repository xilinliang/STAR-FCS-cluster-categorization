// StFcsTrainTestSplit.h - the training bookkeeping shared by trainTMVA.C and
// evalCategory.C, so the two can never disagree about it:
//   - which clusters are TRAINING and which are TEST   (EventSplitter)
//   - what a cluster's true class is                   (trainingLabel)
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

}  // namespace StFcsTrainTestSplit

#endif
