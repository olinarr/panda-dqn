# Panda-DQN

* Dentro alla folder soloNeurale/ si trova una versione del progetto in cui è stata utilizzata solamente la rete neurale per raggiungere la zona verde evitando l'ostacolo, raggiungendo poi a distanza millimetrica il target con un controllore a basso livello
* Dentro alla folder conBassoLivello/ si trova una versione del progetto in cui è stata utilizzato in aggiunta alla rete neurale un sistema di previsione delle collisioni

In entrambe le cartelle, abbiamo i file relativi al train, il controllore (con il modello in .h5) allo stato attuale delle cose ed infine un file che contiene le statistiche su un test di 10k operazioni.

Sottolineo che entrambe le versioni sono state trainate con i rispettivi controllori attivi: nella prima versione durante il train era attivo soltanto il controllore finale per chiudere il gap, nella seconda versione anche quello per rilevare le collisioni. Non è stato attivato il controllore per la rilevazione dei blocchi durante il train.
