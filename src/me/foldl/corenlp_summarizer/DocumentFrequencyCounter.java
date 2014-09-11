package me.foldl.corenlp_summarizer;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.io.ReaderInputStream;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.PropertiesUtils;
import edu.stanford.nlp.util.XMLUtils;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Reader;
import java.io.StringReader;
import java.io.StringWriter;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.regex.Pattern;

public class DocumentFrequencyCounter {

  private static final StanfordCoreNLP pipeline;
  static {
    Properties props = PropertiesUtils.fromString(
      "annotators=tokenize,ssplit,pos\n" +
        "tokenize.language=es\n" +
        "pos.model=edu/stanford/nlp/models/pos-tagger/spanish/spanish-distsim.tagger");
    pipeline = new StanfordCoreNLP(props);
  }

  private static final Pattern headingSeparator = Pattern.compile("[-=]{3,}");

  /**
   * Get an IDF map for the given document string.
   *
   * @param document
   * @return
   */
  private static Counter<String> getIDFMapForDocument(String document) {
    // Clean up -- remove some Gigaword patterns that slow things down
    // / don't help anything
    document = headingSeparator.matcher(document).replaceAll("");

    Annotation annotation = pipeline.process(document);
    List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);

    Counter<String> idfMap = new ClassicCounter<String>();
    for (CoreMap sentence : sentences) {
      List<CoreLabel> cls = sentence.get(CoreAnnotations.TokensAnnotation.class);

      for (CoreLabel cl : cls) {
        String pos = cl.get(CoreAnnotations.PartOfSpeechAnnotation.class);
        if (pos.startsWith("n"))
          idfMap.incrementCount(cl.get(CoreAnnotations.TextAnnotation.class));
      }
    }

    return idfMap;
  }

  private static final String TAG_DOCUMENT = "DOC";
  private static final String TAG_TEXT = "TEXT";

  private static String getFullTextContent(Element e) throws TransformerException {
    TransformerFactory transFactory = TransformerFactory.newInstance();
    Transformer transformer = transFactory.newTransformer();
    StringWriter buffer = new StringWriter();
    transformer.setOutputProperty(OutputKeys.OMIT_XML_DECLARATION, "yes");
    transformer.transform(new DOMSource(e),
                          new StreamResult(buffer));
    String str = buffer.toString();

    // Remove paragraph markers
    str = str.replaceAll("</?(?:TEXT|P)>(\n|$)", "");

    return str;
  }

  /**
   * Get an IDF map for all the documents in the given file.
   * @param file
   * @return
   */
  private static Counter<String> getIDFMapForFile(Reader file)
    throws SAXException, IOException, TransformerException {

    DocumentBuilder parser = XMLUtils.getXmlParser();
    Document xml = parser.parse(new ReaderInputStream(file));
    NodeList docNodes = xml.getDocumentElement().getElementsByTagName(TAG_DOCUMENT);

    Element doc;
    Counter<String> idfMap = new ClassicCounter<String>();
    for (int i = 0; i < docNodes.getLength(); i++) {
      doc = (Element) docNodes.item(i);
      NodeList texts = doc.getElementsByTagName(TAG_TEXT);
      assert texts.getLength() == 1;

      Element text = (Element) texts.item(0);
      String textContent = getFullTextContent(text);

      idfMap.addAll(getIDFMapForDocument(textContent));

      // Increment magic counter
      idfMap.incrementCount("__all__");
    }

    return idfMap;
  }

  private static final class FileIDFBuilder implements Callable<Counter<String>> {
    private final File file;

    public FileIDFBuilder(File file) {
      this.file = file;
    }

    /**
     * Computes a result, or throws an exception if unable to do so.
     *
     * @return computed result
     * @throws Exception if unable to compute a result
     */
    @Override
    public Counter<String> call() throws Exception {
      // We need to hallucinate some overarching document tag.. because the Gigaword files don't
      // have them :/
      String fileContents = IOUtils.slurpFile(file);
      fileContents = "<docs>" + fileContents + "</docs>";

      return getIDFMapForFile(new StringReader(fileContents));
    }
  }

  private static final String OUT_FILE = "df-counts.ser";
  private static final DateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss");

  public static void main(String[] args) throws InterruptedException, ExecutionException,
    IOException {
    ExecutorService pool = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
    List<Future<Counter<String>>> futures = new ArrayList<Future<Counter<String>>>();

    for (String filePath : args)
      futures.add(pool.submit(new FileIDFBuilder(new File(filePath))));

    int finished = 0;
    Counter<String> overall = new ClassicCounter<String>();

    for (Future<Counter<String>> future : futures) {
      System.err.printf("%s: Polling future #%d / %d%n",
          dateFormat.format(new Date()), finished + 1, args.length);
      Counter<String> result = future.get();
      finished++;
      System.err.printf("%s: Finished future #%d / %d%n",
          dateFormat.format(new Date()), finished, args.length);

      System.err.printf("\tMerging counter.. ");
      overall.addAll(result);
      System.err.printf("done.%n");
    }
    pool.shutdown();

    System.err.printf("\n%s: Saving to '%s'.. ", dateFormat.format(new Date()),
        OUT_FILE);
    ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(OUT_FILE));
    oos.writeObject(overall);
    System.err.printf("done.%n");
  }

}
