package it.corvallis.verificafirme;

import it.corvallis.acquisizionefirme.TrainingExecution;
import it.corvallis.acquisizionefirme.UserUtils;
import it.corvallis.common.PropertyUtils;
import it.corvallis.common.SortedProperties;
import it.corvallis.verificafirme.features.Features;
import it.corvallis.verificafirme.preprocess.PreprocessingType;

import java.awt.GraphicsEnvironment;
import java.awt.Rectangle;
import java.awt.TextArea;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StreamTokenizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Properties;
import java.util.Random;
import java.util.LinkedHashMap;
import java.util.List;
import javax.swing.JFrame;
import javax.swing.JOptionPane;

import org.im4java.process.ProcessStarter;
import com.milowski.hmm.Engine;
import com.milowski.hmm.Engine.Prediction;
import com.milowski.hmm.Model;
import com.milowski.hmm.XMLModelWriter;

import be.ac.ulg.montefiore.run.jahmm.ObservationVector;
import be.ac.ulg.montefiore.run.jahmm.io.FileFormatException;
import be.ac.ulg.montefiore.run.jahmm.io.ObservationVectorReader;

/** Classe che racchiude i metodi necessari per la ricerca/massimizzazione del Hidden Markov Model
 * che descrive le firme di un firmatario.
 * @author alex, scritta sulla falsa riga della classe Training del prototipo Acquisizione Firme
 *
 */

public class HmmTraining {
	private JFrame executionDialog;
	private TextArea textArea;
	private boolean interrotto = false;
	private boolean batchMode = false;

	private static String extractedDctDirectoryPath;

	private Class<? extends Writer> writerClass;

	public HmmTraining(Class<? extends Writer> writerClass) {
		this.writerClass = writerClass;
		// TODO Spostare in un file di properties
		ProcessStarter.setGlobalSearchPath("C:/Programmi/ImageMagick-6.8.2-Q16");

		// Ci assicuriamo di poter accedere al database
		File dbFolder = new File(UserUtils.getSignaturesDatabasePath());
		if (dbFolder.exists()) {
			extractedDctDirectoryPath = UserUtils.getSignaturesDatabasePath();
		} else
			UserUtils.erroreLetturaCartella(dbFolder);
	}

	class CheckThread extends Thread {
		private LinkedHashMap<String, Writer> trainedUsers = new LinkedHashMap<String, Writer>();
		private TrainingExecution executionParams = new TrainingExecution();
		Properties a=PropertyUtils.getHmmTraining();

		public CheckThread(String threadName) {
			super(threadName);
		}

		@Override
		public void run() {
			try {
				trainedUsers = startTraining(executionParams);
			}
			catch (Exception e) {
				e.printStackTrace();
			}
		}

		public LinkedHashMap<String, Writer> getTrainedUsers() {
			return trainedUsers;
		}
	}

	public CheckThread executeTraining() {

		executionDialog = new JFrame("Training in esecuzione");
		textArea = new TextArea(">>> Analisi della lista degli utenti...\n");
		executionDialog.add(textArea);
		executionDialog.setResizable(true);
		executionDialog.pack();
		Rectangle maximumWindowBounds = GraphicsEnvironment.getLocalGraphicsEnvironment().getMaximumWindowBounds();
		executionDialog.setLocation((maximumWindowBounds.width - executionDialog.getWidth()) / 2,
				(maximumWindowBounds.height - executionDialog.getHeight()) / 2);

		final CheckThread execution = new CheckThread("Thread training");

		WindowListener l = new WindowAdapter() {
			public void windowClosing(WindowEvent e) {
				super.windowClosing(e);
				int answer = JOptionPane.showConfirmDialog(null, "Interrompere il training?");

				if (answer == JOptionPane.YES_OPTION) {
					execution.interrupt();
					interrotto = true;
					executionDialog.dispose();
				}

			}
		};

		executionDialog.addWindowListener(l);
		executionDialog.setVisible(true);

		execution.start();

		return execution;
	}
	
	//variabile in cui salvo le persone e il loro HMM associato
	private LinkedHashMap<Writer, HMMJscience> initialHmms=new LinkedHashMap<Writer, HMMJscience>();
	private LinkedHashMap<Writer, Model> trainedHmms=new LinkedHashMap<Writer, Model>();

	
	/** metodo che cerca i HMM che meglio descrivono le firme di un firmatario,
	 * attraverso la Cross-Validation,
	 * a partire da modelli iniziali pseudorandom.
	 * @param executionParams racchiude dei parametri
	 *  che rilassano o irrigidiscono la ricerca del modello 
	 * @return gli utenti il cui Hidden Markov Model Ã¨ stato addestrato
	 * @throws Exception
	 */
	private LinkedHashMap<String, Writer> startTraining(TrainingExecution executionParams) throws Exception {

		LinkedHashMap<String, Writer> trainedUsers = new LinkedHashMap<String, Writer>();
		
		
		//un paio di verifiche sul file di hmmTrainingProperties
		if (executionParams.getLogMinValAccepted()>-1) {
			JOptionPane.showMessageDialog(null, "Errore nella selezione del logMinValAccepted! \n"
					+ "Controllare il file testParameters.properties...", "", JOptionPane.ERROR_MESSAGE);
			interrotto = true;
		}

		if (executionParams.getNumberOfProtInitTrainingSet()<executionParams.getNumberOfProtToTrainWith()||
				executionParams.getNumberOfProtInitTrainingSet()<executionParams.getNumberOfProtToKMeansL()) {
			JOptionPane.showMessageDialog(null, "Errore nella selezione del numero di prototipi! \n"
					+ "Controllare il file testParameters.properties...", "", JOptionPane.ERROR_MESSAGE);
			interrotto = true;
		}
		
		if (executionParams.getCycles()<1) {
			JOptionPane.showMessageDialog(null, "Errore nella selezione del numero di cicli interni! \n"
					+ "Controllare il file testParameters.properties...", "", JOptionPane.ERROR_MESSAGE);
			interrotto = true;
		}

		if (executionParams.getStepsKMeansLearner()<1) {
			JOptionPane.showMessageDialog(null, "Errore nella selezione del numero step per il KMeansLearner! \n"
					+ "Controllare il file testParameters.properties...", "", JOptionPane.ERROR_MESSAGE);
			interrotto = true;
		}
		
		if (interrotto)
			executionDialog.dispose();

		File signaturesDatabaseFolder = new File(UserUtils.getSignaturesDatabasePath());

		String[] writerFolders = signaturesDatabaseFolder.list(UserUtils.userFilter);
		Arrays.sort(writerFolders);
		
		StatisticalEngine statEngine = new StatisticalEngine();
		statEngine.useFeature(Features.SLANT, PreprocessingType.SKELETON);
//		statEngine.useFeature(Features.DISCRETECOSINETRANFORM, PreprocessingType.SKELETON);


		int writerCounter = 0;
		
		for (String writerFolder : writerFolders) {
			
			//se esiste il file con le observation sequences posso continuare
			File writerSequences = new File(extractedDctDirectoryPath + File.separator + writerFolder + File.separator + "obsSeqGenuine.seq");
			//File writerHmm = new File(extractedDctDirectoryPath + File.separator + writerFolder + File.separator + "hmm.dot");
			//spostare writerHmm nel test
			if(writerSequences.exists()){//&&(writerFolder.compareToIgnoreCase("pietrogrande2_alberto")==0||writerFolder.compareToIgnoreCase("pietrogrande_alberto")==0)){//&&writerHmm.exists()) {
				
				File currentWriter=new File(signaturesDatabaseFolder.getAbsoluteFile() + File.separator
						+ writerFolder);
				System.out.println(currentWriter.getName());
				String[] currentWriterSignatures = currentWriter.list(UserUtils.imageFilter);
				Arrays.sort(currentWriterSignatures);
				
				// TODO bisogna scegliere quale classe Writer usare
				Writer tmpWriter = writerClass.getConstructor(String.class).newInstance(writerFolder);
	
				textArea.append("\n>>> Calcolo HMM per " + writerFolder + "...");
				textArea.revalidate();
	
				//signer
				System.out.println("persona:" + writerFolder); 
				
//////////////////////////////////////////////////////////////////////////////////////////////////////
				///////////////////////jahmmm////////////////////////////////////////////
				//lettura file sequencesOfObs usando la libreria jahmm
				//lettura sequences of observation da file
				//variabile in cui salvo le sequence of observation di una persona
				List<List<ObservationVector>> sequencesOfObs = new ArrayList<List<ObservationVector>>();
		    	StreamTokenizer st=new StreamTokenizer(new FileReader(writerSequences));
		    	ObservationVectorReader read=new ObservationVectorReader(16);
		    	for(int i=0; i<10; i++) { //5 sequences of observation
		    		//variabile in cui salvo i vettori d'osservazione di una sequenza
		    		List <ObservationVector> oneSequence=new ArrayList<ObservationVector>();
		    		for(int j=0; j<4; j++) { //4 states - 4 observation vector per sequence
		    			try {
							oneSequence.add(read.read(st));
						} catch (FileFormatException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
		    		}
		    		sequencesOfObs.add(oneSequence);
		    	}
				////////////////////////start jscience/////////////////////////////////////////////
				//trasformo la lista (sequencesOfObs) di liste di vettori d'osservazione in double [nrFirme][nrVettori][dimVettore]
		    	double[][][] data=new double[sequencesOfObs.size()][sequencesOfObs.get(0).size()][sequencesOfObs.get(0).get(0).dimension()];
		    	int itSequences=0;
		    	int itSequence=0;
		    	for(List<ObservationVector> seq:sequencesOfObs) {
		    		for(ObservationVector vector: seq) {
		    			for(int i=0; i<vector.dimension(); i++) {
		    				data[itSequences][itSequence][i]=vector.value(i);
		    			}
		    			itSequence++;
		    		}
		    		itSequences++;
		    		itSequence=0;
		    	}
		    	//crea una lista lunga quanto il numero di firme, per poter selezionare indici pseudorandom
		    	ArrayList<Integer> num0to9=new ArrayList<Integer>();
		    	for(int i=0; i<tmpWriter.getNumberOfGenuines(); i++) num0to9.add(i);
		    	
		    	//leggo il numero di firme da utilizzare nel trainig set per la cross validation
	    		int tSetSize=executionParams.getNumberOfProtToTrainWith();//5; //lettura file properties
	    		
	    		//numero di firme da utilizzare nel validation set per la cross validation
	    		int vSetSize=executionParams.getNumberOfProtInitTrainingSet()-tSetSize;//5;//5-tSetSize;
		    	
		    	int minWanted=vSetSize>0?vSetSize:tSetSize; //cross validation for at least
		    	boolean end=false;
		    	
		    	int cyclesDone=0;
		    	//cross validation start
		    	while(!end) {
		    	
		    	//parametro per la cross validation, generare almeno atLeast firme, properties?	
		    	int atLeast=vSetSize>0?vSetSize:tSetSize;

		    	while(atLeast>0&&!end){//11) {//con th 10 Ã¨ arrivato a 84.93%
		    	int cycles=executionParams.getCycles(); //parametro da file properties
		    	int sum=20;
		    	boolean maggiore=false;

		    	while(cycles>0&&!maggiore){ 
		    		cyclesDone++;
		    		cycles--;
		    		//shuffle indici firme in modo da prenderne 5 pseudorandom
		    		Collections.shuffle(num0to9);
		    		//scelgo 5 firme consecutive a partire da una posizione pseudorandom
//		    		Random c=new Random();
//		    		int rand=c.nextInt(executionParams.getNumberOfProtInitTrainingSet());
		    		//variabile in cui tengo i prototipi usatip er il training
		    		HashSet <String> usedAsPrototype = new HashSet<String>();
		    		
		    		//signature training set for cross validation (dimensione=tSetSize/5 firme)
		    		double[][][] dataTSet=new double[tSetSize][4][16];
		    		for(int i=0; i<tSetSize; i++) {
		    			dataTSet[i]=/*data[i]*/data[num0to9.get(i)];
		    			usedAsPrototype.add(currentWriterSignatures[tmpWriter.getNumberOfForgeries()+num0to9.get(i)]);
//		    			System.out.print(" " + currentWriterSignatures[tmpWriter.getNumberOfForgeries()+num0to9.get(i)]);
		    		}
		    		//create pseudorandom signatures from input
//		    		dataTSet=randomVectorsFromSequences(dataTSet, dataTSet.length);
		    		
		    		tmpWriter.setUsedAsPrototypes(usedAsPrototype);
		    		
		    		//signature validation set for cross validation (dimensione=5-tSetSize/5 firme)
		    		double[][][] dataVSet=new double[vSetSize>0?vSetSize:tSetSize][4][16];
		    		if(vSetSize!=0) {
		    			for(int i=0; i<vSetSize; i++) {
		    				dataVSet[i]=/*data[i+tSetSize]*/data[num0to9.get(i+tSetSize)];
		    				usedAsPrototype.add(currentWriterSignatures[tmpWriter.getNumberOfForgeries()+num0to9.get(i+tSetSize)]);
//		    				System.out.print(" " + currentWriterSignatures[tmpWriter.getNumberOfForgeries()+num0to9.get(i+tSetSize)]);
		    			}
		    		}
		    		else {
		    			dataVSet=dataTSet;
		    		}
		    		
		    		ArrayList<Integer> num0to4=new ArrayList<Integer>();
		    		for(int i=0; i<tSetSize; i++) num0to4.add(i);
		    		Collections.shuffle(num0to4);

		    		//rendiamo pseudorandom anche i dati in input al kmeansLearner
		    		double[][][] dataKMean=new double[executionParams.getNumberOfProtToKMeansL()][4][16];
		    		for(int i=0; i<executionParams.getNumberOfProtToKMeansL(); i++) {
		    			dataKMean[i]=dataTSet[num0to4.get(i)];
		    		}
//		    		for(int i=tSetSize; i<tSetSize+vSetSize; i++) {
//		    			dataKMean[i]=dataVSet[i-tSetSize];
//		    		}
		    		//proviamo con un metodo che randomizza i vector stessi "creando" "nuove" firme (sequences of observation)
//		    		double[][][] dataKMean=randomVectorsFromSequences(dataTSet, executionParams.getNumberOfProtToKMeansL());
//		    		HMMJscience initHmm=HMMJscience.kmeansLearner(new double[][][]{dataTSet[num0to4.get(0)]
//		    				,dataTSet[num0to4.get(1)],dataTSet[num0to4.get(2)],dataTSet[num0to4.get(3)],dataTSet[num0to4.get(4)]}, 4, 1000); 
		    		//hidden markov model iniziale, utile per la matrice di probabilitÃ  di emissione

		    		HMMJscience initHmm=HMMJscience.kmeansLearner(dataKMean, 4, executionParams.getStepsKMeansLearner());

		    		//ragionamento training hmm, ora che ho il modello iniziale
		    		//ho bisogno di sequences of observation per addestrarlo
		    		List<short[]> sequences=new ArrayList<short[]>();
		    		//prendo le firme usate per il partizionamento kmeans (training set per il cross validation)
		    		//e le trasformo in simboli
			    	for(int j=0; j<tSetSize; j++) {
			    		short[] sequence=initHmm.convert(dataTSet[j], 16, 4);
			    		sequences.add(new short[]{sequence[0],sequence[1],sequence[2],sequence[3],sequence[3]});
			    	}
			    					    
			    	//jhmm
				    //lettura precision of the model dal file di properties
				    double modelPrecision=0.000000001; //lettura file properties TODO
				    
				    //inizializzo un nuovo modello, che sarÃ  addestrato, usa la matrice di emissione di probabilitÃ  di hmmInit
			    	Model jhmm=new Model("first", initHmm.sigmaSize, 4, modelPrecision);
			    	jhmm.setStateName(1, "s1");
			    	jhmm.setStateName(2, "s2");
			    	jhmm.setStateName(3, "s3");
			    	jhmm.setStateName(4, "s4");
			    	
			    	//alfabeto/simboli del modello jhmm
			    	List<Character> lexiconSymbols=new ArrayList<Character>();
			    	char[] alphabet=new char[]{'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
			    			'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9'};
			    	for(int i=0; i<initHmm.sigmaSize; i++) {
			    		lexiconSymbols.add(alphabet[i]);
			    	}
			    	
			    	jhmm.setLexicon(lexiconSymbols);

			    	//inizializzazione matrice di probabilitÃ  transizione del modello jhmm
			    	//secondo il discrete LtR senza salti di stato
			    	for(int i=1; i<5; i++) {
			    		for(int j=1; j<5; j++) {
			    			if(j==i) jhmm.setStateTransition(i, j, 0.95);//499999999998);
			    			else if(j==i+1) jhmm.setStateTransition(i, j, 0.05);
			    			else jhmm.setStateTransition(i, j, 0);//.0000000000001);
			    		}
			    	}
			    	jhmm.setStateTransition(4, 4, 1);//0.9999999999997);
			    	jhmm.setStateTransition(0, 1, 1);
			    	jhmm.setStateTransition(4, 0, 1);
			    	
			    	//inizializzazione della matrice di probabilitÃ  di emissione
			    	//secondo i valori calcolati da initHmm
			    	for(int i=0; i<4; i++) {
			    		for(int j=0; j<initHmm.b[0].length; j++) {
			    			//System.out.print(" " + initHmm.b[i][j]);
			    			jhmm.setStateEmission(i+1, j, initHmm.b[i][j]==0?0.0000000000001:initHmm.b[i][j]);
			    		}
			    		//System.out.print("\n");
			    	}
			    	
			    	try {
			    		//verifico se le matrici del modello sono stocastiche
			    		jhmm.check();
			    	}
			    	catch(IllegalStateException ise) {
			    		//System.out.println("exception caught: " + ise);
			    		continue;
			    	}
			    	
			    	//salvo il modello jhmm prima di addestrarlo
//			    	BufferedWriter output=null;
//					output = new BufferedWriter(new FileWriter(UserUtils.getSignaturesDatabasePath() + File.separator + tmpWriter.getName()
//		    				+ File.separator + "jhmmBeforeTrain.xml"));
//				    XMLModelWriter wjhmm=new XMLModelWriter(output);
//			    	wjhmm.write(jhmm);
//			    	output.close();
			    	
			    	Engine ejhmm=new Engine();
			    	ejhmm.loadModel(jhmm);
			    	//System.out.println(" before training ");

			    	//see prob before training for the training set in cv
//			    	for(int i=0; i<sequences.size(); i++) {
//			    		Prediction mostLikely=ejhmm.mostLikely(sequences.get(i));
//			    		short[] result=mostLikely.getResult();
//			    		for(int k=0; k<result.length; k++) {
//			    			System.out.print(" " + result[k]);
//			    		}
//			    		System.out.println(" score: " + mostLikely.getScore()+"\n");
//			    	}
			    	//see prob before training for the validation set in cv
//			    	for(int i=0; i<vSetSize; i++) {
//			    		short[] seqSimbols=initHmm.convert(dataVSet[i], 16, 4);
//			    		short[] ss=new short[]{seqSimbols[0],seqSimbols[1],seqSimbols[2],seqSimbols[3],seqSimbols[3]};
//			    		System.out.println(" validation set i: " + i);
//			    		Prediction mostLikelyS=ejhmm.mostLikely(ss);
//			    		short[] resultSixthB=mostLikelyS.getResult();
//			    		for(int k=0; k<resultSixthB.length; k++) {
//			    			System.out.print(" " + resultSixthB[k]);
//			    		}
//			    		System.out.println(" score: " + mostLikelyS.getScore()+"\n");
//			    	}

			    	//scelgo le matrici da aggiornare
			    	ejhmm.setTraining(true, true, true);
			    	try {
			    		//addestra modello con le sequences del training set
			    		//TODO vedere se delta lo si legge da file properties, probabilmente si
			    		ejhmm.train(sequences, executionParams.getBWThreshold(), null); // 
			    	}
			    	catch(RuntimeException re) {
			    		//System.out.println("exception caught: " + re);
			    		continue;
			    	}
			    	//aggiorna le matrici del modello jhmm
			    	ejhmm.updateModel(jhmm);
			    	
			    	//see the probability on the signatures of the training set after training
			    	//System.out.println("after training ");
//			    	for(int i=0; i<sequences.size(); i++) {
//			    		Prediction mostLikely=ejhmm.mostLikely(sequences.get(i));
//			    		short[] result=mostLikely.getResult();
//			    		for(int k=0; k<result.length; k++) {
//			    			System.out.print(" " + result[k]);
//			    		}
//			    		System.out.println(" score: " + mostLikely.getScore()+ "\n");
//			    	}

			    	//see the probability on the signatures of the validation set after training
			    	sum=0;
			    	atLeast=0;
//			    	System.out.println("new cycle ");
			    	for(int i=0; i<(vSetSize>0?vSetSize:tSetSize); i++) {
			    		short[] seqSimbols=initHmm.convert(dataVSet[i], 16, 4);
			    		short[] ss=new short[]{seqSimbols[0],seqSimbols[1],seqSimbols[2],seqSimbols[3],seqSimbols[3]};
			    	//	System.out.println(" validation set i: " + i);
			    		Prediction mostLikelyS=ejhmm.mostLikely(ss);
//			    		short[] resultSixthB=mostLikelyS.getResult();
//			    		for(int k=0; k<resultSixthB.length; k++) {
//			    			System.out.print(" " + resultSixthB[k]);
//			    		}
//			    		System.out.println(" score: " + mostLikelyS.getScore()+"\n");
			    		if(mostLikelyS.getScore()>executionParams.getMaxAvgLog()) { 
			    			atLeast++;
//			    			sum+=Math.abs((int)(mostLikelyS.getScore()));
			    		}
			    	}
			    	if(atLeast==0) sum=100;
			    	else sum=sum/atLeast;
		    		
			    	initialHmms.put(tmpWriter, initHmm);
			    	trainedHmms.put(tmpWriter, jhmm);
				    
				    if(atLeast>=minWanted||minWanted<0) { end=true; maggiore=true; System.out.println("atLeast: " + atLeast + " minWanted: " + minWanted); }

		    	}
			
		    	if(cycles!=0&&atLeast>=minWanted) {end=true;}

		    	} if(cyclesDone-1001>0) { System.out.println("done"); minWanted--; cyclesDone=0; } if(minWanted<0) { end=true;}
		    	}
		    	System.out.println("cycles done: " + cyclesDone);

		    	// Prepariamo un file di properties in cui salviamo dei dati relativi il training:
				File trainingFile = new File(UserUtils.getSignaturesDatabasePath()
						+ File.separator + writerFolder + File.separator + "trainingHmm.properties");
				
				Properties prop= new SortedProperties();
				
				// Salviamo i nomi delle firme usate come prototipo...
				prop.setProperty( "Prototypes", tmpWriter.getUsedAsPrototypes().toString());
				try {
					prop.store(new PrintWriter(trainingFile), "");
					} catch (IOException e) {
					e.printStackTrace();
				}
				writerCounter++;
		    }
			else {
				textArea.append("\n>>> Impossibile eseguire il training per " + writerFolder);
				textArea.revalidate();
			}
			
		}
		//compare 2 hmm models
//		System.out.println(hmmArray[0].hmmDistance(hmmArray[1]) + " " + hmmArray[0].hmmDistance(hmmArray[2])
//		+ " " +hmmArray[1].hmmDistance(hmmArray[2]));
//		
//		System.out.println(hmmArray[0].hmmDistance(hmmArray[0]) + " " + hmmArray[1].hmmDistance(hmmArray[1])
//				+ " " +hmmArray[2].hmmDistance(hmmArray[2]));
		
	
		textArea.append("\n\n>>> Fatto!");
		textArea.append("\n\n>>> Salvataggio dei risultati del training in corso...");
		textArea.revalidate();
	
		// A questo punto bisogna salvare in un file di properties per
		// ciascun utente, il suo modello HMM
		for (Writer tmpWriter : initialHmms.keySet()) {
			
			if (interrotto)
				return null;
			
			try {
				//salvataggio su file XML del trained model usando la libreria jhmm
				BufferedWriter outputA=null;
				outputA = new BufferedWriter(new FileWriter(UserUtils.getSignaturesDatabasePath() + File.separator + tmpWriter.getName()
	    				+ File.separator + "jhmmAfterTrain.xml"));
			    XMLModelWriter wjhmmA=new XMLModelWriter(outputA);
		    	wjhmmA.write(trainedHmms.get(tmpWriter));
		    	outputA.close();
		    	
		    	//salvataggio su file .dot del modello iniziale usando il metodo HMMJscience.write()
			    initialHmms.get(tmpWriter).write(UserUtils.getSignaturesDatabasePath() + File.separator + tmpWriter.getName()
	    				+ File.separator + "hmmJscienceBeforBW.dot");
			    
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		// Salviamo in un unico file di properties per l'intero database
		// le features calcolate e i rispettivi pre-processing
		File trainingFeatureFile = new File(UserUtils.getSignaturesDatabasePath()
				+ File.separator + "trainingFeaturesHmm.properties");
		Properties prop= new SortedProperties();
		
		prop.setProperty("UsedFeatures", statEngine.getUsedFeaturesWithPreprocessing().toString());
				
		try {
			prop.store(new PrintWriter(trainingFeatureFile), "");
			} catch (IOException e) {
			e.printStackTrace();
		}
	
		textArea.append("\n\n>>> Fatto!");
	
		if (!batchMode)
			JOptionPane.showMessageDialog(executionDialog, "Training completato per " + writerCounter + " utenti su "
					+ writerFolders.length);
	
		executionDialog.dispose();
	
		return trainedUsers;
	}

	public boolean isBatchMode() {
		return batchMode;
	}

	public void setBatchMode(boolean batchMode) {
		this.batchMode = batchMode;
	}

	public boolean isInterrotto() {
		return interrotto;
	}

	public void setInterrotto(boolean interrotto) {
		this.interrotto = interrotto;
	}
// try con pdfs, probability density function
//	private OpdfMultiGaussian createOpdfMultiGaussian(double mean, double covariance, int dim) {
//		double[] means = new double[dim];
//		Arrays.fill(means, mean);
//		double[][] covs = new double[dim][dim];
//		for (int i = 0; i < covs.length; i++) {
//			for (int j = 0; j < covs.length; j++) {
//				covs[i][j] = covariance;
//			}
//		}
//		return new OpdfMultiGaussian(means, covs);
//	}
//	private OpdfMultiGaussian createOpdfMultiGaussian(double mean, double covariance, int dim) {
//		BigDecimal[] means = new BigDecimal[dim];
//		Arrays.fill(means, BigDecimal.valueOf(mean));
//		BigDecimal[][] covs = new BigDecimal[dim][dim];
//		for (int i = 0; i < covs.length; i++) {
//			for (int j = 0; j < covs.length; j++) {
//				covs[i][j] = BigDecimal.valueOf(covariance);
//			}
//		}
//		return new OpdfMultiGaussian(means, covs);
//	}
//	private MultiGaussianDistribution createMultiGaussianDistribution(double mean, double covariance, int dim) {
//        double[] means = new double[dim];
//        Arrays.fill(means, mean);
//        double[][] covs = MatrixUtils.createRealIdentityMatrix(dim).scalarMultiply(covariance).getData();
//        return new MultiGaussianDistribution(means, covs);
//}
//
//private MultiGaussianMixtureDistribution createMultiGaussianMixtureDistribution(double mean, double covariance,
//                int dim, int mixtures, double[] mixtureProps) {
//
//        if (mixtureProps == null) {
//                mixtureProps = new double[mixtures];
//                Arrays.fill(mixtureProps, 1. / mixtures);
//        }
//        mixtures = mixtureProps.length;
//        MultiGaussianDistribution[] mgds = new MultiGaussianDistribution[mixtures];
//        for (int i = 0; i < mgds.length; i++) {
//                mgds[i] = createMultiGaussianDistribution(mean, covariance, dim);
//        }
//        return new MultiGaussianMixtureDistribution(mgds, mixtureProps);
//}
//
//private OpdfMultiGaussianMixture<ObservationVector> createOpdfMultiGaussianMixture(double mean, double covariance,
//                int dim, int mixtures, double[] mixtureProps) {
//        return new OpdfMultiGaussianMixture<ObservationVector>(createMultiGaussianMixtureDistribution(mean, covariance,
//                        dim, mixtures, mixtureProps));
//}
	/** metodo che dato in input double[x][y][z], randomizza le x, mantenendo inalterate le posizioni delle y
	 * assemblaggio di "nuove" sequences of observation 
	 * @return numberInInput x
	 */
	public static double[][][] randomVectorsFromSequences(double[][][] dataTSet, int numberInOutput) {
		double[][][] output=new double[numberInOutput][dataTSet[0].length][dataTSet[0][0].length];
		//randomize [x][y], una x Ã¨ composta da "4" y ordinati, avendo piÃ¹ x posso scambiare gli y che sono nella stessa posizione
		ArrayList<Integer> xies=new ArrayList<Integer>();
		//fill xies
		for(int i=0; i<dataTSet.length; i++) {
			xies.add(i);
		}
		
		//create
		for(int j=0; j<dataTSet[0].length; j++) {
			Collections.shuffle(xies);
			for(int i=0; i<output.length; i++) {
				output[i][j]=dataTSet[xies.get(i)][j];
			}
		}
		//little test, veriify the input and the output...
		//the input
//		System.out.println("input: ");
//		for(int i=0; i<dataTSet.length; i++) {
//			for(int j=0; j<dataTSet[0].length; j++) {
//				for(int z=0; z<dataTSet[0][0].length; z++) {
//					System.out.print(" " + dataTSet[i][j][z]);
//				}
//				System.out.println("");
//			}
//			System.out.println("\n");
//		}
//		//the output
//		System.out.println("output: ");
//		for(int i=0; i<output.length; i++) {
//			for(int j=0; j<output[0].length; j++) {
//				for(int z=0; z<output[0][0].length; z++) {
//					System.out.print(" " + output[i][j][z]);
//				}
//				System.out.println("");
//			}
//			System.out.println("\n");
//		}
		return output;
	}
}