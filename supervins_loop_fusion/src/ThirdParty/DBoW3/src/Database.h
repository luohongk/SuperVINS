/**
 * File: Database.h
 * Date: March 2011
 * Modified By Rafael Muñoz in 2016
 * Author: Dorian Galvez-Lopez
 * Description:  database of images
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_T_DATABASE__
#define __D_T_DATABASE__

#include "DBoW3.h"
#include <vector>
#include <numeric>
#include <fstream>
#include <string>
#include <list>
#include <set>

#include <cassert>

#include "DBoW3.h"
#include "BowVector.h"
#include <vector>
#include <numeric>
#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include "exports.h"
#include "FeatureVector.h"

#include "ScoringObject.h"
#include <limits>

#include "QueryResults.h"
#include "ScoringObject.h"
#include "BowVector.h"
#include "FeatureVector.h"
#include "exports.h"
// #include "Vocabulary.h"

namespace DBoW3
{

  // For query functions
  static int MIN_COMMON_WORDS = 5;

  class Vocabulary
  {
    friend class FastSearch;

  public:
    /**
     * Initiates an empty vocabulary
     * @param k branching factor
     * @param L depth levels
     * @param weighting weighting type
     * @param scoring scoring type
     */
    Vocabulary(int k = 10, int L = 5,
               WeightingType weighting = TF_IDF, ScoringType scoring = L1_NORM);

    /**
     * Creates the vocabulary by loading a file
     * @param filename
     */
    Vocabulary(const std::string &filename);

    /**
     * Creates the vocabulary by loading a file
     * @param filename
     */
    Vocabulary(const char *filename);

    /**
     * Creates the vocabulary by loading an input stream
     * @param filename
     */
    Vocabulary(std::istream &filename);

    /**
     * Copy constructor
     * @param voc
     */
    Vocabulary(const Vocabulary &voc);

    /**
     * Destructor
     */
    virtual ~Vocabulary();

    /**
     * Assigns the given vocabulary to this by copying its data and removing
     * all the data contained by this vocabulary before
     * @param voc
     * @return reference to this vocabulary
     */
    Vocabulary &operator=(
        const Vocabulary &voc);

    /**
     * Creates a vocabulary from the training features with the already
     * defined parameters
     * @param training_features
     */
    virtual void create(const std::vector<std::vector<cv::Mat>> &training_features);
    /**
     * Creates a vocabulary from the training features with the already
     * defined parameters
     * @param training_features. Each row of a matrix is a feature
     */
    virtual void create(const std::vector<cv::Mat> &training_features);

    /**
     * Creates a vocabulary from the training features, setting the branching
     * factor and the depth levels of the tree
     * @param training_features
     * @param k branching factor
     * @param L depth levels
     */
    virtual void create(const std::vector<std::vector<cv::Mat>> &training_features,
                        int k, int L);

    /**
     * Creates a vocabulary from the training features, setting the branching
     * factor nad the depth levels of the tree, and the weighting and scoring
     * schemes
     */
    virtual void create(const std::vector<std::vector<cv::Mat>> &training_features,
                        int k, int L, WeightingType weighting, ScoringType scoring);

    /**
     * Returns the number of words in the vocabulary
     * @return number of words
     */
    virtual inline unsigned int size() const { return (unsigned int)m_words.size(); }

    /**
     * Returns whether the vocabulary is empty (i.e. it has not been trained)
     * @return true iff the vocabulary is empty
     */
    virtual inline bool empty() const { return m_words.empty(); }

    /** Clears the vocabulary object
     */
    void clear();
    /**
     * Transforms a set of descriptores into a bow vector
     * @param features
     * @param v (out) bow vector of weighted words
     */
    virtual void transform(const std::vector<cv::Mat> &features, BowVector &v)
        const;
    /**
     * Transforms a set of descriptores into a bow vector
     * @param features, one per row
     * @param v (out) bow vector of weighted words
     */
    virtual void transform(const cv::Mat &features, BowVector &v)
        const;
    /**
     * Transform a set of descriptors into a bow vector and a feature vector
     * @param features
     * @param v (out) bow vector
     * @param fv (out) feature vector of nodes and feature indexes
     * @param levelsup levels to go up the vocabulary tree to get the node index
     */
    virtual void transform(const std::vector<cv::Mat> &features,
                           BowVector &v, FeatureVector &fv, int levelsup) const;

    /**
     * Transforms a single feature into a word (without weight)
     * @param feature
     * @return word id
     */
    virtual WordId transform(const cv::Mat &feature) const;

    /**
     * Returns the score of two vectors
     * @param a vector
     * @param b vector
     * @return score between vectors
     * @note the vectors must be already sorted and normalized if necessary
     */
    double score(const BowVector &a, const BowVector &b) const { return m_scoring_object->score(a, b); }

    /**
     * Returns the id of the node that is "levelsup" levels from the word given
     * @param wid word id
     * @param levelsup 0..L
     * @return node id. if levelsup is 0, returns the node id associated to the
     *   word id
     */
    virtual NodeId getParentNode(WordId wid, int levelsup) const;

    /**
     * Returns the ids of all the words that are under the given node id,
     * by traversing any of the branches that goes down from the node
     * @param nid starting node id
     * @param words ids of words
     */
    void getWordsFromNode(NodeId nid, std::vector<WordId> &words) const;

    /**
     * Returns the branching factor of the tree (k)
     * @return k
     */
    inline int getBranchingFactor() const { return m_k; }

    /**
     * Returns the depth levels of the tree (L)
     * @return L
     */
    inline int getDepthLevels() const { return m_L; }

    /**
     * Returns the real depth levels of the tree on average
     * @return average of depth levels of leaves
     */
    float getEffectiveLevels() const;

    /**
     * Returns the descriptor of a word
     * @param wid word id
     * @return descriptor
     */
    virtual inline cv::Mat getWord(WordId wid) const;

    /**
     * Returns the weight of a word
     * @param wid word id
     * @return weight
     */
    virtual inline WordValue getWordWeight(WordId wid) const;

    /**
     * Returns the weighting method
     * @return weighting method
     */
    inline WeightingType getWeightingType() const { return m_weighting; }

    /**
     * Returns the scoring method
     * @return scoring method
     */
    inline ScoringType getScoringType() const { return m_scoring; }

    /**
     * Changes the weighting method
     * @param type new weighting type
     */
    inline void setWeightingType(WeightingType type);

    /**
     * Changes the scoring method
     * @param type new scoring type
     */
    void setScoringType(ScoringType type);

    /**
     * Saves the vocabulary into a file. If filename extension contains .yml, opencv YALM format is used. Otherwise, binary format is employed
     * @param filename
     */
    void save(const std::string &filename, bool binary_compressed = true) const;

    /**
     * Loads the vocabulary from a file created with save
     * @param filename.
     */
    void load(const std::string &filename);

    /**
     * Loads the vocabulary from an input stream created with save
     * @param stream.
     */
    bool load(std::istream &stream);

    /**
     * Saves the vocabulary to a file storage structure
     * @param fn node in file storage
     */
    virtual void save(cv::FileStorage &fs,
                      const std::string &name = "vocabulary") const;

    /**
     * Loads the vocabulary from a file storage node
     * @param fn first node
     * @param subname name of the child node of fn where the tree is stored.
     *   If not given, the fn node is used instead
     */
    virtual void load(const cv::FileStorage &fs,
                      const std::string &name = "vocabulary");

    /**
     * Stops those words whose weight is below minWeight.
     * Words are stopped by setting their weight to 0. There are not returned
     * later when transforming image features into vectors.
     * Note that when using IDF or TF_IDF, the weight is the idf part, which
     * is equivalent to -log(f), where f is the frequency of the word
     * (f = Ni/N, Ni: number of training images where the word is present,
     * N: number of training images).
     * Note that the old weight is forgotten, and subsequent calls to this
     * function with a lower minWeight have no effect.
     * @return number of words stopped now
     */
    virtual int stopWords(double minWeight);

    /** Returns the size of the descriptor employed. If the Vocabulary is empty, returns -1
     */
    int getDescritorSize() const;
    /** Returns the type of the descriptor employed normally(8U_C1, 32F_C1)
     */
    int getDescritorType() const;
    // io to-from a stream
    void toStream(std::ostream &str, bool compressed = true) const throw(std::exception);
    void fromStream(std::istream &str) throw(std::exception);

  protected:
    ///  reference to descriptor
    typedef const cv::Mat pDescriptor;

    /// Tree node
    struct Node
    {
      /// Node id
      NodeId id;
      /// Weight if the node is a word
      WordValue weight;
      /// Children
      std::vector<NodeId> children;
      /// Parent node (undefined in case of root)
      NodeId parent;
      /// Node descriptor
      cv::Mat descriptor;

      /// Word id if the node is a word
      WordId word_id;

      /**
       * Empty constructor
       */
      Node() : id(0), weight(0), parent(0), word_id(0) {}

      /**
       * Constructor
       * @param _id node id
       */
      Node(NodeId _id) : id(_id), weight(0), parent(0), word_id(0) {}

      /**
       * Returns whether the node is a leaf node
       * @return true iff the node is a leaf
       */
      inline bool isLeaf() const { return children.empty(); }
    };

  protected:
    /**
     * Creates an instance of the scoring object accoring to m_scoring
     */
    void createScoringObject();

    /**
     * Returns a set of pointers to descriptores
     * @param training_features all the features
     * @param features (out) pointers to the training features
     */
    void getFeatures(const std::vector<std::vector<cv::Mat>> &training_features,
                     std::vector<cv::Mat> &features) const;

    /**
     * Returns the word id associated to a feature
     * @param feature
     * @param id (out) word id
     * @param weight (out) word weight
     * @param nid (out) if given, id of the node "levelsup" levels up
     * @param levelsup
     */
    virtual void transform(const cv::Mat &feature,
                           WordId &id, WordValue &weight, NodeId *nid, int levelsup = 0) const;
    /**
     * Returns the word id associated to a feature
     * @param feature
     * @param id (out) word id
     * @param weight (out) word weight
     * @param nid (out) if given, id of the node "levelsup" levels up
     * @param levelsup
     */
    virtual void transform(const cv::Mat &feature,
                           WordId &id, WordValue &weight) const;

    /**
     * Returns the word id associated to a feature
     * @param feature
     * @param id (out) word id
     */
    virtual void transform(const cv::Mat &feature, WordId &id) const;

    /**
     * Creates a level in the tree, under the parent, by running kmeans with
     * a descriptor set, and recursively creates the subsequent levels too
     * @param parent_id id of parent node
     * @param descriptors descriptors to run the kmeans on
     * @param current_level current level in the tree
     */
    void HKmeansStep(NodeId parent_id, const std::vector<cv::Mat> &descriptors,
                     int current_level);

    /**
     * Creates k clusters from the given descriptors with some seeding algorithm.
     * @note In this class, kmeans++ is used, but this function should be
     *   overriden by inherited classes.
     */
    virtual void initiateClusters(const std::vector<cv::Mat> &descriptors,
                                  std::vector<cv::Mat> &clusters) const;

    /**
     * Creates k clusters from the given descriptor sets by running the
     * initial step of kmeans++
     * @param descriptors
     * @param clusters resulting clusters
     */
    void initiateClustersKMpp(const std::vector<cv::Mat> &descriptors,
                              std::vector<cv::Mat> &clusters) const;

    /**
     * Create the words of the vocabulary once the tree has been built
     */
    void createWords();

    /**
     * Sets the weights of the nodes of tree according to the given features.
     * Before calling this function, the nodes and the words must be already
     * created (by calling HKmeansStep and createWords)
     * @param features
     */
    void setNodeWeights(const std::vector<std::vector<cv::Mat>> &features);

    /**
     * Writes printable information of the vocabulary
     * @param os stream to write to
     * @param voc
     */
    DBOW_API friend std::ostream &operator<<(std::ostream &os, const Vocabulary &voc);

    /**Loads from ORBSLAM txt files
     */
    void load_fromtxt(const std::string &filename) throw(std::runtime_error);

  protected:
    /// Branching factor
    int m_k;

    /// Depth levels
    int m_L;

    /// Weighting method
    WeightingType m_weighting;

    /// Scoring method
    ScoringType m_scoring;

    /// Object for computing scores
    GeneralScoring *m_scoring_object;

    /// Tree nodes
    std::vector<Node> m_nodes;

    /// Words of the vocabulary (tree leaves)
    /// this condition holds: m_words[wid]->word_id == wid
    std::vector<Node *> m_words;

  public:
    // for debug (REMOVE)
    inline Node *getNodeWord(uint32_t idx) { return m_words[idx]; }
  };

  ///   Database
  class DBOW_API Database
  {
  public:
    /**
     * Creates an empty database without vocabulary
     * @param use_di a direct index is used to store feature indexes
     * @param di_levels levels to go up the vocabulary tree to select the
     *   node id to store in the direct index when adding images
     */
    explicit Database(bool use_di = true, int di_levels = 0);

    /**
     * Creates a database with the given vocabulary
     * @param T class inherited from Vocabulary
     * @param voc vocabulary
     * @param use_di a direct index is used to store feature indexes
     * @param di_levels levels to go up the vocabulary tree to select the
     *   node id to store in the direct index when adding images
     */

    explicit Database(const Vocabulary &voc, bool use_di = true,
                      int di_levels = 0);

    /**
     * Copy constructor. Copies the vocabulary too
     * @param db object to copy
     */
    Database(const Database &db);

    /**
     * Creates the database from a file
     * @param filename
     */
    Database(const std::string &filename);

    /**
     * Creates the database from a file
     * @param filename
     */
    Database(const char *filename);

    /**
     * Destructor
     */
    virtual ~Database(void);

    /**
     * Copies the given database and its vocabulary
     * @param db database to copy
     */
    Database &operator=(
        const Database &db);

    /**
     * Sets the vocabulary to use and clears the content of the database.
     * @param T class inherited from Vocabulary
     * @param voc vocabulary to copy
     */
    void setVocabulary(const Vocabulary &voc);

    /**
     * Sets the vocabulary to use and the direct index parameters, and clears
     * the content of the database
     * @param T class inherited from Vocabulary
     * @param voc vocabulary to copy
     * @param use_di a direct index is used to store feature indexes
     * @param di_levels levels to go up the vocabulary tree to select the
     *   node id to store in the direct index when adding images
     */

    void setVocabulary(const Vocabulary &voc, bool use_di, int di_levels = 0);

    /**
     * Returns a pointer to the vocabulary used
     * @return vocabulary
     */
    const Vocabulary *getVocabulary() const;

    /**
     * Allocates some memory for the direct and inverted indexes
     * @param nd number of expected image entries in the database
     * @param ni number of expected words per image
     * @note Use 0 to ignore a parameter
     */
    void allocate(int nd = 0, int ni = 0);

    /**
     * Adds an entry to the database and returns its index
     * @param features features of the new entry
     * @param bowvec if given, the bow vector of these features is returned
     * @param fvec if given, the vector of nodes and feature indexes is returned
     * @return id of new entry
     */
    EntryId add(const std::vector<cv::Mat> &features,
                BowVector *bowvec = NULL, FeatureVector *fvec = NULL);
    /**
     * Adds an entry to the database and returns its index
     * @param features features of the new entry, one per row
     * @param bowvec if given, the bow vector of these features is returned
     * @param fvec if given, the vector of nodes and feature indexes is returned
     * @return id of new entry
     */
    EntryId add(const cv::Mat &features,
                BowVector *bowvec = NULL, FeatureVector *fvec = NULL);

    /**
     * Adss an entry to the database and returns its index
     * @param vec bow vector
     * @param fec feature vector to add the entry. Only necessary if using the
     *   direct index
     * @return id of new entry
     */
    EntryId add(const BowVector &vec,
                const FeatureVector &fec = FeatureVector());

    /**
     * Empties the database
     */
    void clear();

    /**
     * Returns the number of entries in the database
     * @return number of entries in the database
     */
    unsigned int size() const { return m_nentries; }

    /**
     * Checks if the direct index is being used
     * @return true iff using direct index
     */
    bool usingDirectIndex() const { return m_use_di; }

    /**
     * Returns the di levels when using direct index
     * @return di levels
     */
    int getDirectIndexLevels() const { return m_dilevels; }

    /**
     * Queries the database with some features
     * @param features query features
     * @param ret (out) query results
     * @param max_results number of results to return. <= 0 means all
     * @param max_id only entries with id <= max_id are returned in ret.
     *   < 0 means all
     */
    void query(const std::vector<cv::Mat> &features, QueryResults &ret,
               int max_results = 1, int max_id = -1) const;
    /**
     * Queries the database with some features
     * @param features query features,one per row
     * @param ret (out) query results
     * @param max_results number of results to return. <= 0 means all
     * @param max_id only entries with id <= max_id are returned in ret.
     *   < 0 means all
     */
    void query(const cv::Mat &features, QueryResults &ret,
               int max_results = 1, int max_id = -1) const;

    /**
     * Queries the database with a vector
     * @param vec bow vector already normalized
     * @param ret results
     * @param max_results number of results to return. <= 0 means all
     * @param max_id only entries with id <= max_id are returned in ret.
     *   < 0 means all
     */
    void query(const BowVector &vec, QueryResults &ret,
               int max_results = 1, int max_id = -1) const;

    /**
     * Returns the a feature vector associated with a database entry
     * @param id entry id (must be < size())
     * @return const reference to map of nodes and their associated features in
     *   the given entry
     */
    const FeatureVector &retrieveFeatures(EntryId id) const;

    /**
     * Stores the database in a file
     * @param filename
     */
    void save(const std::string &filename) const;

    /**
     * Loads the database from a file
     * @param filename
     */
    void load(const std::string &filename);

    /**
     * Stores the database in the given file storage structure
     * @param fs
     * @param name node name
     */
    virtual void save(cv::FileStorage &fs,
                      const std::string &name = "database") const;

    /**
     * Loads the database from the given file storage structure
     * @param fs
     * @param name node name
     */
    virtual void load(const cv::FileStorage &fs,
                      const std::string &name = "database");

    // --------------------------------------------------------------------------

    /**
     * Writes printable information of the database
     * @param os stream to write to
     * @param db
     */
    DBOW_API friend std::ostream &operator<<(std::ostream &os,
                                             const Database &db);

  protected:
    /// Query with L1 scoring
    void queryL1(const BowVector &vec, QueryResults &ret,
                 int max_results, int max_id) const;

    /// Query with L2 scoring
    void queryL2(const BowVector &vec, QueryResults &ret,
                 int max_results, int max_id) const;

    /// Query with Chi square scoring
    void queryChiSquare(const BowVector &vec, QueryResults &ret,
                        int max_results, int max_id) const;

    /// Query with Bhattacharyya scoring
    void queryBhattacharyya(const BowVector &vec, QueryResults &ret,
                            int max_results, int max_id) const;

    /// Query with KL divergence scoring
    void queryKL(const BowVector &vec, QueryResults &ret,
                 int max_results, int max_id) const;

    /// Query with dot product scoring
    void queryDotProduct(const BowVector &vec, QueryResults &ret,
                         int max_results, int max_id) const;

  protected:
    /* Inverted file declaration */

    /// Item of IFRow
    struct IFPair
    {
      /// Entry id
      EntryId entry_id;

      /// Word weight in this entry
      WordValue word_weight;

      /**
       * Creates an empty pair
       */
      IFPair() {}

      /**
       * Creates an inverted file pair
       * @param eid entry id
       * @param wv word weight
       */
      IFPair(EntryId eid, WordValue wv) : entry_id(eid), word_weight(wv) {}

      /**
       * Compares the entry ids
       * @param eid
       * @return true iff this entry id is the same as eid
       */
      inline bool operator==(EntryId eid) const { return entry_id == eid; }
    };

    /// Row of InvertedFile
    typedef std::list<IFPair> IFRow;
    // IFRows are sorted in ascending entry_id order

    /// Inverted index
    typedef std::vector<IFRow> InvertedFile;
    // InvertedFile[word_id] --> inverted file of that word

    /* Direct file declaration */

    /// Direct index
    typedef std::vector<FeatureVector> DirectFile;
    // DirectFile[entry_id] --> [ directentry, ... ]

  protected:
    /// Associated vocabulary
    Vocabulary *m_voc;

    /// Flag to use direct index
    bool m_use_di;

    /// Levels to go up the vocabulary tree to select nodes to store
    /// in the direct index
    int m_dilevels;

    /// Inverted file (must have size() == |words|)
    InvertedFile m_ifile;

    /// Direct file (resized for allocation)
    DirectFile m_dfile;

    /// Number of valid entries in m_dfile
    int m_nentries;
  };

  // --------------------------------------------------------------------------

} // namespace DBoW3

#endif
