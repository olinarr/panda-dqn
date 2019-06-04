import math
import numpy as np
from random import uniform as rnd

# Di quanto modifico ogni giuntura con un passo?
# globale perche' serve ogni tanto come argomento
step_size = 5

class env_1:
	# Costruttore di classe
	def __init__(self):


		# errore
		self.epsilon = 0.10
		
		# Minimi e massimi valori degli angoli delle giunture del robot
		self.minRange =  [-50,   -10,   -50,   -95,   -50,   90,   0]
		self.maxRange =  [ 50,    90,    50,    -5,    50,   90,   0]
		
		# Posizione iniziale. Uno stato e' rappresentato dagli angoli delle giunture
		self.state = np.array([0, 40, 0, -40, 0, 90, 0])
		
		# utilizzo questi valori per contare i vari tipi di fallimenti
		self.obstacle_failures = 0
		self.timeout_failures = 0

		# aumento questo indice ogni step, in modo da stampare solo i failure rate ogni 20000 passi.
		# e' diverso dal contatore generale: questo non e' mai riavviato
		self.total_steps = 0

		# raggio del braccio, calcolo anche il quadrato per non ricalcolarlo
		self.arm_radius = 0.02
		self.arm_radius_squared = self.arm_radius * self.arm_radius

		# definisco i parametri necessari a fare FK
		# inoltre calcolo gia' tutte le moltiplicazioni che serviranno nella matrice,
		# per ottimizzare i conti
		self.r_vect = np.array([0, 0, 0, 0.0825, -0.0825, 0, 0.088])
		self.d_vect = np.array([0.333, 0, 0.316, 0, 0.384, 0, 0])
		alpha_vect = [0, -math.pi/2, math.pi/2, math.pi/2, -math.pi/2, math.pi/2, math.pi/2]
		self.sin_alpha_vect = np.sin(alpha_vect)
		self.cos_alpha_vect = np.cos(alpha_vect)
		self.neg_sin_alpha_vect = -self.sin_alpha_vect
		self.neg_d_times_sin_alpha_vect = np.multiply(self.d_vect, self.neg_sin_alpha_vect)
		self.d_times_cos_alpha_vect = np.multiply(self.d_vect, self.cos_alpha_vect)

	# funzione accessorio per fare FK
	def getTn(self, joint_states, n):

		r = self.r_vect[n]
		d = self.d_vect[n]
		sin_alpha = self.sin_alpha_vect[n]
		neg_sin_alpha = self.neg_sin_alpha_vect[n]
		neg_d_times_sin_alpha = self.neg_d_times_sin_alpha_vect[n]
		cos_alpha = self.cos_alpha_vect[n]
		d_times_cos_alpha = self.d_times_cos_alpha_vect[n]
		sin_theta = math.sin(math.radians(joint_states[n]))
		cos_theta = math.cos(math.radians(joint_states[n]))

		M = np.array([
			[cos_theta, -sin_theta, 0, r],
			[sin_theta*cos_alpha, cos_theta*cos_alpha, neg_sin_alpha, neg_d_times_sin_alpha],
			[sin_theta*sin_alpha, cos_theta*sin_alpha, cos_alpha, d_times_cos_alpha],
			[0, 0, 0, 1]]);

		return M

	# applica l'algoritmo per fare FK della giuntura n-esima
	def getJointPos(self, joint_states, n):
		DH = self.getTn(joint_states, 0)

		for i in range(1, n):
			DH = np.dot(DH, self.getTn(joint_states, i))

		# ritorna la posizione xyz
		return DH[0:3, 3]

	# come sopra, ma cumulativo: ritorna un vettore di tutte le posizioni.
	# ho scritto questa funzione semplicemente per risparmiare calcoli!
	# NOTA IMPORTANTE: inizia da 0,0,0!!
	def vectGetJointPos(self, joint_states):
		DH = self.getTn(joint_states, 0)

		res = np.array([[0, 0, 0], DH[0:3, 3]])

		for i in range(1, 7):
			DH = np.dot(DH, self.getTn(joint_states, i))
			res = np.vstack([res, [DH[0:3, 3]]])

		# ritorna la posizione xyz
		return res

# genera un ostacolo casuale tra il braccio e il target
	def generateObstacle(self):

		# eps controlla la variazione casuale che sfasa leggermente il punto risultante dal segmento braccio-target 
		eps = 0.02

		# Posizione xyz del braccio
		arm = self.endEffectorPos(self.state)
		target_pos = self.target

		# la formula e': P - mQ
		# m = (Q-P)
		# il rnd davanti ad m e' t, compare nella formula di riferimento. Dovrebbe essere [0..1]
		# ma lo metto a .1 .9 per non farlo troppo vicino al resto
		# alla fine aggiungo una variazione casuale ai tre assi

		point = arm + rnd(.3, .7)*(target_pos - arm) + [rnd(-eps, eps), rnd(-eps, eps), rnd(-eps, eps)]

		#normalizza di modo che non esca dai bounds
		for i in range(len(point)):
			if point[i] < min(arm[i], target_pos[i]):
				point[i] = min(arm[i], target_pos[i])
			else:
				if point[i] > max(arm[i], target_pos[i]):
					point[i] = max(arm[i], target_pos[i])

		return point


	# Ritorna una nuova "sfida" generica creando un nuovo target (non serve rispostare il braccio, al massimo lo normalizzo)
	def reset_random(self):
		# Azzera il contatore. Abbiamo ricominciato
		self.counter = 0

		# while true per evitare collisioni target-ostacolo
		while True:
			# Genera degli angoli casuali, notare che non usiamo gli ultimi due quindi sono fissi
			t1 = np.random.randint(-50, 50)
			t2 = np.random.randint(-10, 90)
			t3 = np.random.randint(-50, 50)
			t4 = np.random.randint(-95, -5)
			t5 = np.random.randint(-50, 50)
			t6 = 90
			t7 = 0
			
			# Normalizzo per essere raggiungibile
			t1 -= (t1 % 5)
			t2 -= (t2 % 5)
			t3 -= (t3 % 5)
			t4 -= (t4 % 5)
			t5 -= (t5 % 5)
			
			# E questa posizione e' quindi quella che devo raggiungere.
			self.target_angle = [t1, t2, t3, t4, t5, t6, t7]
			
			# Genero le coordinate x-y-z del target a partire dagli angoli di giuntura: si veda la funzione endEffectorPos per i dettagli
			self.target = self.endEffectorPos( self.target_angle )

			# Genero un ostacolo casuale tra il braccio e il target
			self.obstacle = self.generateObstacle()
			self.obstacle_radius = np.array([0.03 + rnd(0, 0.03)])

			
			# we continue ONLY IF the obstacle DOES NOT intersecate the target. Else, we retry.
			if math.sqrt( math.pow((self.obstacle[0]-self.target[0]), 2) + math.pow((self.obstacle[1]-self.target[1]), 2) + math.pow((self.obstacle[2]-self.target[2]), 2) ) > self.obstacle_radius + self.epsilon:
				break
			


		# Preparo quello moltiplicato, per usarlo nelle distanze
		self.obstacle_radius_squared = self.obstacle_radius * self.obstacle_radius
		# vero valore che confronto: deriva dal fatto che a+b^2 = a^2 + b^2 + 2ab
		# così da risparmiarmi di fare la radice quadrata
		self.distance_check = self.obstacle_radius_squared + self.arm_radius_squared + 2*self.arm_radius*self.obstacle_radius
		
		# Quindi ritorno la configurazione formata dalla tripla stato del braccio - posizione xyz del target - posizione xyz dell'ostacolo e raggio
		# Notare che non e' necessario riposizionare il braccio, basta creare un nuovo target
		return np.concatenate((self.normalizeState(), self.target, self.obstacle, self.obstacle_radius ))

	# Controllo se nella catena del braccio, tocco l'ostacolo
	# source: http://geomalgorithms.com/a02-_lines.html
	def hasHitObstacle(self, pos_vect):
		# cicla su tutte le righe tranne l'ultima, che verrai raggiunta comunque essendo che mi sposto avanti di 1
		# in pratica ogni volta genero il segmento P0->P1, e con una formula trovata sul link
		# qui sopra calcolo la distanza minima tra questo segmento e l'ostacolo. Se questa e' minore del raggio, ho colpito, ritorno true.
		# se per nessun segmento ho verificato la condizione, non ho colpito l'ostacolo: bene!
		for i in range(len(pos_vect)-1):
	  		P0 = pos_vect[i]
			P1 = pos_vect[i+1]

			v = P1-P0
			w = self.obstacle - P0

			# uso la tolleranza pre-calcolata per risparmiare conti
			c1 = np.dot(w,v)
			if c1 <= 0:
				if math.pow((self.obstacle[0]-P0[0]), 2) + math.pow((self.obstacle[1]-P0[1]), 2) + math.pow((self.obstacle[2]-P0[2]), 2) <= self.distance_check:
					return True
			else:
				c2 = np.dot(v,v)
				if c2 <= c1:
					if math.pow((self.obstacle[0]-P1[0]), 2) + math.pow((self.obstacle[1]-P1[1]), 2) + math.pow((self.obstacle[2]-P1[2]), 2) <= self.distance_check:
						return True
				else:
					b = c1/c2
					Pb = P0 + b*v
					if math.pow((self.obstacle[0]-Pb[0]), 2) + math.pow((self.obstacle[1]-Pb[1]), 2) + math.pow((self.obstacle[2]-Pb[2]), 2) <= self.distance_check:
						return True

		# Sono soppravvissuto a tutti i test, Bene!
		return False

	def compute_new_state(self, state, action, step = step_size):
		# solo se fattibile
		if ( (action%2) == 0 and (state[int(action/2)] + step ) < self.maxRange[int(action/2)] ):
			state[int(action/2)] += step

		if ( (action%2) == 1 and (state[int(action/2)] - step ) > self.minRange[int(action/2)] ):
			state[int(action/2)] -= step

		return state

		# given a certain state, a certain action, a certain obstacle, is it safe? with a certain step, also!
	def isItSafe(self, joint_states, action, step = step_size):
		return not self.hasHitObstacle(self.vectGetJointPos(self.compute_new_state(joint_states, action, step)))

	# returns a tuple: (bool, list). List is the new joint states, bool says if we're done or not.
	def aux_reach(self):
	# if only one joint is different from the 
	# target state, we will set done to False.
	# increses/decreases one degree at a time.
	# original_vect is used in case, with the "fastest way", we would have a collision. See after
		original_vect = self.state
		done = True
		for i in range(len(self.target_angle)):
			if (self.target_angle[i] - self.state[i]) > step_size:
				self.state[i] += step_size
				done = False
			elif (self.target_angle[i] - self.state[i]) < -step_size:
				self.state[i] -= step_size
				done = False
			else:
				self.state[i] = self.target_angle[i]
	# if with this disposition we hit...
		if self.hasHitObstacle(self.vectGetJointPos(self.state)):
			# try again: done = False!!
			done = False
			# take a random action
			action = np.random.randint(0, 10)
			# and verify if, ont the "Original" state, we would have hit
			while not self.isItSafe(original_vect, action):
				action = np.random.randint(0, 10)
			# when we found a safe action, we perform it
			self.state = self.compute_new_state(original_vect, action)

		return done

	# iteratively tries to reach the target angles from the
	# joint states, using that publisher
	def reach(self):
		# every joint has a workingspace of 100, so 150 is more than enough
		timeout = 500
		count = 0
		# do one step and check if we're done
		done = self.aux_reach()
		# ...and do the same untile we're done or timeouted
		while (not done) and (count < timeout):
			done = self.aux_reach()
			count += 1
		return


	# Eseguo un passo, ovvero un'azione
	def step(self, actions):
		# Numero di passi massimi consentiti
		timeout = 500
		# Errore concesso, ovvero, quando e' considerabile giusto: metto al quadrato sicche' non devo ogni volta calcolare la radice quadrata
		error_square = self.epsilon * self.epsilon
		# Inizializzo questo test
		done = False
		# Ho fatto un passo quindi aumento il contatore per il timeout
		self.counter += 1
		# Ho fatto un passo quindi aumento il contatore per totale
		self.total_steps += 1
		
		# sceglo un'azione senza rischi. Nota! Le azioni devono essere ordinate per previsione
		action = actions[0]
		for i in range(len(actions)):
			if self.isItSafe(self.state, actions[i], step = step_size*2):
				action = actions[i]
				break

		# Le azioni sono codificate cosi'. Se e' pari aumenta, se e' dispari diminuisce, e poi a seconda di che numero e'
		# Decido su che giunzione lavorare. Se l'azione non mi fa uscire dal range, modifico la giuntura dello step
		# Ricorda che state = gli angoli del braccio!
		self.state = self.compute_new_state(self.state, action)
		
		# Calcolo la nuova posizione xyz a partire dagli angoli; il vettore contiene le 5 posizioni xyz dei nodi
		pos_vect = self.vectGetJointPos(self.state)
		# end effector: e' l'ultimo giunto
		x, y, z = pos_vect[-1, 0:3]
		# Calcolo la distanza braccio-target con una semplice formula euclidea, al quadrato! (non faccio la radice)
		distance_square = math.pow((x-self.target[0]), 2) + math.pow((y-self.target[1]), 2) + math.pow((z-self.target[2]), 2)

		reward = 0

		#############

		# Se ho esaurito i passi disponibili, ho fallito: ho fatto (done = True) e la ricompensa e' -1. Inoltre registro il tipo di fallimento.
		if(self.counter >= timeout):
			reward = -3
			done = True
			self.timeout_failures += 1

		# Se ho toccato l'ostacolo, ho fallito. Inoltre registro il tipo di fallimento.
		if(self.hasHitObstacle(pos_vect)):
			reward = -2
			done = True
			self.obstacle_failures += 1
		
		# Infine, se la distanza dal target e' minore di quella tollerata, allora ho finito e la ricompensa e' 1... se raggiungo!
		if(distance_square < error_square):
			self.reach()
			# recheck distance...
			x, y, z = self.endEffectorPos(self.state)
			tar_distance = math.sqrt( math.pow((x-self.target[0]), 2) + math.pow((y-self.target[1]), 2) + math.pow((z-self.target[2]), 2) )
			# se siamo a distanza millimetrica, bene. Altrimenti -1
			if tar_distance < 0.001:
				reward = 1
			else:
				reward = -1
			done = True
		
		# conto i fallimenti totali (per studiare la distribuzione)
		tot_failures = self.timeout_failures + self.obstacle_failures
		# ogni 20000 passi aggiorno i risultati appendendo
		if self.total_steps == 20000:
			# svuoto il file di fallimenti per riscriverlo
			out_file = open("failures_std.txt", "w")
			out_file.write("")
			out_file.close()
		if self.total_steps % 20000 == 0 and tot_failures != 0:
			out_file = open("failures_std.txt","a")
			out_file.write("==========\nprint number %d...\n" % self.total_steps)
			out_file.write("Obstacle:\t %d (%3.2f%%)\n" % (self.obstacle_failures, self.obstacle_failures*100/tot_failures))
			out_file.write("Timeouts:\t %d (%3.2f%%)\n==========\n\n" % (self.timeout_failures, self.timeout_failures*100/tot_failures))
			out_file.close()

		# Ritorno la tupla: nuovo stato (braccio-target-ostacolo-raggio), ricompensa, fatto?, counter, e l'azione reale che abbiamo eseguito, per fare train.
		return np.concatenate((self.normalizeState(), self.target, self.obstacle, self.obstacle_radius )), reward, done, self.counter, action

	# Questa funzione normalizza la posizione del braccio tra -1 e 1 con delle formule
	def normalizeState(self):
		normalizedState = []
		# 5 perche' usiamo 5 angoli, quindi per ogni angolo:
		for i in range(5):
			# deg: gradi della giuntura
			# space: ampiezza di movimento
			# offset: centro dell'ampiezza
			# deg-offeset: distanza dal centro.
			# Se divido per l'ampiezza, ottengo un valore normalizzato tra -1 e 1
			deg = self.state[i]
			space = math.fabs(self.maxRange[i] - self.minRange[i]) / 2.0
			offset = (self.maxRange[i] - space)
			# Metto nello stato normalizzato il valore tra -1 e 1 ottenuto
			normalizedState.append((deg - offset) / space)

		# Se uno degli angoli non era tra -1 e 1 allora ero fuori range
		if(max(normalizedState) > 1 or min(normalizedState) < -1):
			print ("ERROR: INPUT OUT OF RANGE state " + str(max(normalizedState)) + ", " + str(min(normalizedState)))
		
		# Ritorno cio' che ho ottenuto
		return np.array(normalizedState)

	def endEffectorPos(self, joint_states):
		return self.getJointPos(joint_states, 7)
