# Remove warning and base import
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
from collections import deque
import math
import random

# Import Numnpy
import numpy as np

# Import Keras
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, InputLayer, Dropout

# Import environment
from env_1 import env_1
env_1 = env_1()

# Setting up the ANN

# Questa funzione aggiorna il target model, mettendo i pesi del modello "normale"
def update_target_model():
	target_model.set_weights(model.get_weights())

# out_node = 10 perche' le azioni possibili sono 10 (o avanti o indietro per ogni giuntura mobile, che sono 5)
# come input ho 12 valori: i 5 angoli (che uso) del braccio, xyz del target, xyz e raggio dell'ostacolo.

out_node = 10
model = Sequential()
model.add(Dense(128, input_shape = (12, ), activation='tanh'))
model.add(Dense(128, activation='tanh'))
model.add(Dense(128, activation='tanh'))
model.add(Dense(out_node, activation = 'linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

target_model = Sequential()
target_model.add(Dense(128, input_shape = (12, ), activation='tanh'))
target_model.add(Dense(128, activation='tanh'))
target_model.add(Dense(128, activation='tanh'))
target_model.add(Dense(out_node, activation = 'linear'))
target_model.compile(loss='mse', optimizer='adam', metrics=['mae'])

# Contatore per aggiornare il model
update_model = 0

# Imposto i due modelli come uguali inizialmente
update_target_model()

# Setting up variables
num_episodes = 5000000
# Fattore di decadimento
y = 0.40
y_max = 0.99
growth_factor = 1.000125
# Percentuale di scelta di una mossa casuale
eps = 1.0
# Minima possibilita' di scelta di una mossa casuale
eps_min = 0.01
# Fattore di decadenza della scelta casuale
decay_factor = 0.9995

# Memorizzo tutti i vari risultati. La memoria memorizza tuple di tipo
# stato-azione-next_state-reward-done, in modo da poterle ri-usare. E' stato mostrato che questo
# stabilizza fortemente l'uso del DQN.

# Sul memory generico andranno tutti i passi che non hanno concluso nulla di particolare
memory = deque(maxlen=10000)
# Qua solo quelli che hanno portato alla vittoria
memory_win = deque (maxlen = 10000)
# Qua solo quelli che hanno porato ad un fallimento
memory_lose = deque (maxlen = 10000)

# Struttura dati che memorizza i successi e le ricompense
success_queue = deque(maxlen = 100)
success_list = []
reward_list = []

# Learning Function
def learn():
	# Lista degli stati
	state_list = []
	# Lista dei target
	target_list = []
	# Batch di campioni della memoria
	sample_batch = []

	# Assegno le dimensioni massime ai vari batch
	batch_size = 64
	batch_size_win = 32
	batch_size_lose = 32

	# Se per ora ho meno cose in memoria, le dimensioni effettive saranno minori: 
	if(len(memory) < batch_size):
		batch_size = len(memory)

	if(len(memory_win) < batch_size_win):
		batch_size_win = len(memory_win)

	if(len(memory_lose) < batch_size_lose):
		batch_size_lose = len(memory_lose)

	# Prendo esattamente batch_size* campioni dalle varie memorie
	sample_batch.append(random.sample(memory, batch_size))
	sample_batch.append(random.sample(memory_win, batch_size_win))
	sample_batch.append(random.sample(memory_lose, batch_size_lose))

	# i=0 generici, i=1 vittorie, i=2 perdite
	for i in range(3):
		# per ogni tupla di memoria (quindi stato-azione-ricompensa-stato_prossimo-fatto?)
		# dei campioni ora considerati (generici vittorie e perdite)
		for state, action, reward, next_state, done in sample_batch[i]:
			# Se avevo finito, la ricompensa e' -1 o 1
			target = reward
			# Altrimenti valuto la bonta' di una mossa "non finale" come segue:
			if not done:
				# lo stato prossimo viene scelto selezionando il massimo risultato della previsione del model normale passandogli lo stato prossimo
				n_s = np.argmax(model.predict(np.array([next_state])))
				# Applico la funzione di Q-learn sul TARGET MODEL (gli passo lo stato prossimo e dell'array ricevuto scelgo l'azione selezionata in pracedenza)
				target = reward + y * target_model.predict(np.array([next_state]))[0][n_s]
			# passo da rappresentazione a (1, 10) a un array uni-dimensionale di 10 elementi
			target_f = model.predict(np.array([state]))[0]
			# nella posizione espressa da action sto agendo: e' li che vado a mettere il nuovo valore.
			target_f[action] = target
			# aggiungi alla lista degli stati (input ANN per il fit)
			state_list.append(state)
			# aggiungi alla lista di posizioni-giunture attese per fittare il modello
			target_list.append(target_f)

	# fittiamo il modello con quello che abbiamo recuperato dalla memoria, quindi:
	# input stato (12)
	# output cambiamento sulle giunture (10)
	model.fit(np.array(state_list), np.array(target_list), epochs=1, verbose=0)



for i in range(num_episodes):
	# Riparto da uno stato casuale 
	s = env_1.reset_random()

	# Aggiorno il contatore per aggiornare il model
	update_model += 1

	# Finche' non ho finito (esaurito i passi o raggiunto l'obiettivo)...
	done = False
	while not done:

		# Con probabilita' eps, scelgo un'azione casuale
		if np.random.random() < eps:
			a = np.random.randint(0, out_node)
		else:
		# Altrimenti, la prossima azione e' scelta utilizzando l'ANN per prevedere la mossa migliore
			a = np.argmax(model.predict(np.array([s])))

		# Eseguo uno step: in base all'azione scelta (e l'errore...),
		# avro' un nuovo stato, una ricompensa, sapro' se ho finito e sapro' quante mosse mi rimangono.
		new_s, r, done, moves = env_1.step(a)
		
		# Se la mia ricompensa e' 1, allora questa cosa va nella memoria delle vittorie,
		# Se e' -1 allora delle perdite, in alternativa in quele generiche
		if(r > 0):
			memory_win.append([s, a, r, new_s, done])
		elif(r < 0):
			memory_lose.append([s, a, r, new_s, done])
		else:
			memory.append([s, a, r, new_s, done])

		# Aggiorno lo stato
		s = new_s
	
	# Ok, terminata la sessione di addestramento (e avendo memorizzato le cose), faccio imparare l'ANN
	learn()

	# Ogni 2000 episodi aggiorno i pesi del target model
	if update_model % 2000 == 0:
		update_target_model()
	if update_model % 8000 == 0:
		model.save("periodic_backup.h5")

	# Faccio decadere (se e' il caso) la probabilita' di azione casuale
	if(eps > eps_min):
		eps *= decay_factor
	else:
		eps = eps_min

	if(y < y_max):
		y *= growth_factor
	else:
		y = y_max

	# Qua ci sono robe di debug
	success_queue.append(r)
	success = int(success_queue.count(1)/(len(success_queue)+0.0)*100)
	success_list.append(success)

	reward_list.append(r)

	print("Generation " + str(i) + ", Reward: " + str(r) + ", Success: " + str(success) + ", Eps: " + str(eps) + ", y: " + str(y))

	if(i % 100 == 0):
		np.savetxt("success_std.txt", success_list, fmt='%3i')
		np.savetxt("reward_std.txt", reward_list, fmt='%3i')

	if(success >= 75):
		model.save("backup_std_" + str(success ) + ".h5")
