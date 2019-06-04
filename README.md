# Panda-DQN

## Versioni

* Dentro alla folder *soloNeurale/* si trova una versione del progetto in cui è stata utilizzata solamente la rete neurale per raggiungere la zona verde evitando l'ostacolo, raggiungendo poi a distanza millimetrica il target con un controllore a basso livello
* Dentro alla folder *conBassoLivello/* si trova una versione del progetto in cui è stata utilizzato in aggiunta alla rete neurale un sistema di previsione delle collisioni e dei blocchi

## In entrambe le cartelle, abbiamo:

* i file relativi al train (il vero e proprio *training_std.py* e *l'env_1.py*)
* il controllore (con il modello in .h5) allo stato attuale delle cose
* un file che contiene le statistiche su un test di 10k operazioni ed un link ad un video dimostrativo
* una folder *trainData/* contenente statistiche raccolte durante la fase di train:
   - *failures_std.txt* contiene statistiche sul tipo di fail riscontrati
   - *reward_std.txt* contiene la lista di ricompense ricevute
   - *success_std.txt* contiene la lista di success_rate registrati

## Note

Sottolineo che entrambe le versioni sono state trainate con i rispettivi controllori attivi: nella prima versione durante il train era attivo soltanto il controllore finale per chiudere il gap, nella seconda versione anche quello per rilevare le collisioni. Non è stato attivato il controllore per la rilevazione dei blocchi durante il train.
