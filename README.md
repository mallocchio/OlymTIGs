# Progetto_Tirocinio
Repository per il progetto di tirocinio

---Schema Logico del Sistema:

1. Configurazione Iniziale:

Carica le informazioni di configurazione da un file (Config.txt), includendo il percorso del modello addestrato, 
il numero di immagini da generare e il tipo di immagini da generare (MNIST o CREATE).

2. Inizializzazione dei Classificatori:

In base alla configurazione, inizializza due classificatori: uno basato su TensorFlow (TensorflowClassifier) 
e l'altro su PyTorch (TorchClassifier).

3. Caricamento dei Modelli:

Carica i modelli preaddestrati per entrambi i classificatori utilizzando il percorso specificato nella configurazione.

4. Ciclo di Generazione e Classificazione delle Immagini:

Esegue un ciclo per generare un numero specificato di immagini (come definito nella configurazione).
Per ogni immagine generata, la passa ai due classificatori per ottenere le previsioni.

5. Visualizzazione dei Risultati:

Per ogni immagine e le relative previsioni ottenute dai due classificatori, crea un grafico visuale che include l'immagine stessa, 
un grafico dei logit e un grafico delle previsioni di probabilità.
Mostra il grafico risultante per ciascuna immagine generata.

6. Terminazione del Programma:

Una volta che tutte le immagini sono state elaborate e visualizzate, il programma termina.

---Caratteristiche delle Immagini Accettate dai Classificatori:

Per essere accettate dai due classificatori, le immagini devono rispettare determinate caratteristiche:

Dimensioni: Le immagini devono essere di dimensioni 28x28 pixel.
Scala di Grigi: Le immagini devono essere in scala di grigi, ovvero ciascun pixel deve avere un valore tra 0 (nero) e 255 (bianco).
Normalizzazione: Per entrambi i classificatori, l'intensità dei pixel deve essere normalizzata nell'intervallo [0, 1] o [-1, 1], a seconda della normalizzazione specifica applicata durante la fase di addestramento dei modelli.
Formato dei Dati: Per il classificatore TensorFlow, le immagini devono essere in formato NumPy con le dimensioni (1, 28, 28, 1), dove 1 rappresenta il numero di batch e 28x28x1 rappresenta le dimensioni dell'immagine con un singolo canale (scala di grigi). Per il classificatore PyTorch, le immagini devono essere in formato Tensor con le dimensioni (1, 784), dove 784 rappresenta il vettore piatto delle immagini 28x28.
Tipo di Immagine: Le immagini possono essere di due tipi:
MNIST: Le immagini possono essere campioni del dataset MNIST, che sono cifre scritte a mano in scala di grigi.
CREATE: Le immagini possono essere generate artificialmente utilizzando un generatore di immagini (ImageGenerator). Queste immagini contengono cifre disegnate casualmente su uno sfondo nero.


