package me.foldl.corenlp_summarizer;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.PropertiesUtils;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Properties;

/**
 * @author Jon Gauthier
 */
public class Summarizer {

  private static final StanfordCoreNLP pipeline;
  static {
    Properties props = PropertiesUtils.fromString(
      "annotators=tokenize,ssplit,pos\n" +
        "tokenize.language=es\n" +
        "pos.model=edu/stanford/nlp/models/pos-tagger/spanish/spanish-distsim.tagger");
    pipeline = new StanfordCoreNLP(props);
  }

  private final Counter<String> dfCounter;
  private final int numDocuments;

  public Summarizer(Counter<String> dfCounter) {
    this.dfCounter = dfCounter;
    this.numDocuments = (int) dfCounter.getCount("__all__");
  }

  private static Counter<String> getTermFrequencies(List<CoreMap> sentences) {
    Counter<String> ret = new ClassicCounter<String>();

    for (CoreMap sentence : sentences)
      for (CoreLabel cl : sentence.get(CoreAnnotations.TokensAnnotation.class))
        ret.incrementCount(cl.get(CoreAnnotations.TextAnnotation.class));

    return ret;
  }

  private class SentenceComparator implements Comparator<CoreMap> {
    private final Counter<String> termFrequencies;

    public SentenceComparator(Counter<String> termFrequencies) {
      this.termFrequencies = termFrequencies;
    }

    @Override
    public int compare(CoreMap o1, CoreMap o2) {
      return (int) Math.round(tfIDFWeights(o1) - tfIDFWeights(o2));
    }

    private double tfIDFWeights(CoreMap sentence) {
      double total = 0;
      for (CoreLabel cl : sentence.get(CoreAnnotations.TokensAnnotation.class))
        if (cl.get(CoreAnnotations.PartOfSpeechAnnotation.class).startsWith("n"))
          total += tfIDFWeight(cl.get(CoreAnnotations.TextAnnotation.class));

      return total;
    }

    private double tfIDFWeight(String word) {
      double tf = Math.sqrt(termFrequencies.getCount(word));
      double idf = 1 + Math.log(numDocuments / (1.0 + dfCounter.getCount(word)));
      return tf * Math.pow(idf, 2);
    }
  }

  private List<CoreMap> rankSentences(List<CoreMap> sentences, Counter<String> tfs) {
    Collections.sort(sentences, new SentenceComparator(tfs));
    return sentences;
  }

  public String summarize(String document) {
    Annotation annotation = pipeline.process(document);
    List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);

    Counter<String> tfs = getTermFrequencies(sentences);
    sentences = rankSentences(sentences, tfs);

    System.err.println(sentences);
    return null;
  }

  private static final String DF_COUNTER_PATH = "idf-counter.ser";

  @SuppressWarnings("unchecked")
  private static Counter<String> loadDfCounter(String path)
    throws IOException, ClassNotFoundException {
    ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path));
    return (Counter<String>) ois.readObject();
  }

  public static void main(String[] args) throws IOException, ClassNotFoundException {
    String filename = args[0];
    String content = IOUtils.slurpFile(filename);

    Counter<String> dfCounter = loadDfCounter(DF_COUNTER_PATH);

    Summarizer summarizer = new Summarizer(dfCounter);
    summarizer.summarize(content);
  }

}
