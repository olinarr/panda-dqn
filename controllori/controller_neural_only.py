# contiene:
# script per chiudere il gap
# randomizer se mi stucko
# selettore della migliore azione so far
# seleziona azione casuale MIGLIORE (i.e. non collidere), ma solo se rimango stuckato
# nella chiusura del gap, evitare ostacoli
# impedire agli ostacoli di generarsi dentro il target

import rospy
from std_msgs.msg import String
from std_msgs.msg import Header
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA

import numpy as np
import math
import os

from random import uniform as rnd

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, InputLayer, Dropout

from collections import deque

publish_count = 0
rate = 30
episodes = 10

pub = rospy.Publisher('joint_states', JointState, queue_size=10)
rospy.init_node('joint_state_publisher')
rate = rospy.Rate(rate) # 10hz (1 slow - 100 fast)

# Normalization funciton for the NN input
def normalized_state(deg_vec):
	used_joints = 5 # uso solo 5 angoli

	min_range =	[-50, -10, -50, -95, -50, 90, 0]
	max_range =	[ 50, 90, 50, -5, 50, 90, 0]

	normalizedState = []
	for i in range(used_joints):
		deg = deg_vec[i]
		space = math.fabs(max_range[i] - min_range[i]) / 2.0
		delay = (max_range[i] - space)
		normalizedState.append((deg - delay) / space)

		if(max(normalizedState) > 1 or min(normalizedState) < -1):
			print ("ERROR: INPUT OUT OF RANGE state " + str(max(normalizedState)) + ", " + str(min(normalizedState)))

	return normalizedState

# Support function for the degree to radians conversion
def deg2rad(deg_vec):
	rad_vec = []
	for i in range(9):
		rad_vec.append(math.radians(deg_vec[i]))
	return rad_vec

# the default change of joint angles
default_step = 2.5


# Support function for the next state calculation
# by default, use default_step, but this can be changed. For instance,
# in collision detecting, we use a greater step, because, empirically,
# this means less hits.
def compute_new_state(state, action, step = default_step):
	min_range = [-50, -10, -50, -95, -50, 90, 0]
	max_range = [ 50, 90, 50, -5, 50, 90, 0]
	
	# solo se fattibile
	if ( (action%2) == 0 and (state[int(action/2)] + step ) < max_range[int(action/2)] ):
		state[int(action/2)] += step

	if ( (action%2) == 1 and (state[int(action/2)] - step ) > min_range[int(action/2)] ):
		state[int(action/2)] -= step

	return state

# Support function for the target random generation
def generate_target():
	t1 = np.random.randint(-50, 50)
	t2 = np.random.randint(-10, 90)
	t3 = np.random.randint(-50, 50)
	t4 = np.random.randint(-95, -5)
	t5 = np.random.randint(-50, 50)
	t6 = 90
	t7 = 0
	t8 = 0
	t9 = 0

	target_angle = [t1, t2, t3, t4, t5, t6, t7, t8, t9]

	x, y, z = endEffectorPos(target_angle)
	return ([x, y, z], target_angle)

def generate_obstacle(state_deg, target):

	# eps controlla la variazione casuale che sfasa leggermente il punto risultante dal segmento braccio-target 
	eps = 0.02

	# Posizione xyz del braccio
	arm = endEffectorPos(state_deg)
	target_pos = target

	# la formula e': P - mQ
	# m = (Q-P)
	# il rnd davanti ad m e' t, compare nella formula di riferimento. Dovrebbe essere [0..1]
	# ma lo metto a .1 .9 per non farlo troppo vicino al resto
	# alla fine aggiungo una variazione casuale ai tre assi

	point = arm + rnd(.3, .7)*(target_pos - arm) + np.array([rnd(-eps, eps), rnd(-eps, eps), rnd(-eps, eps)])

	#normalizza di modo che non esca dai bounds
	for i in range(len(point)):
		if point[i] < min(arm[i], target_pos[i]):
			point[i] = min(arm[i], target_pos[i])
		else:
			if point[i] > max(arm[i], target_pos[i]):
				point[i] = max(arm[i], target_pos[i])

	return point

# Support function for the marker printer
def marker_spawn(marker_publisher, position, radius, colors, marker_id):
	robotMarker = Marker()
	robotMarker.header.frame_id = 'panda_link0'
	robotMarker.header.stamp		= rospy.get_rostime()
	robotMarker.type = 2 # sphere

	robotMarker.id = marker_id

	robotMarker.pose.position.x = position[0]
	robotMarker.pose.position.y = position[1]
	robotMarker.pose.position.z = position[2]

	robotMarker.scale.x = radius
	robotMarker.scale.y = radius
	robotMarker.scale.z = radius

	robotMarker.color.r = colors[0]
	robotMarker.color.g = colors[1]
	robotMarker.color.b = colors[2]
	robotMarker.color.a = 1

	marker_publisher.publish(robotMarker)

# Get end effector position with the Franka forward kinematics
def endEffectorPos(joint_states):
	return getJointPos(joint_states, 7)

# Loading the keras model
def create_model():
	out_node = 10
	model = Sequential()
	model.add(Dense(128, input_shape = (12, ), activation='tanh'))
	model.add(Dense(128, activation='tanh'))
	model.add(Dense(128, activation='tanh'))
	model.add(Dense(out_node, activation = 'linear'))
	model.compile(loss='mse', optimizer='adam', metrics=['mae'])

	model.load_weights("model_neural_only.h5")

	return model


# definisco i parametri necessari a fare FK
r_vect = np.array([0, 0, 0, 0.0825, -0.0825, 0, 0.088])
d_vect = np.array([0.333, 0, 0.316, 0, 0.384, 0, 0])
alpha_vect = [0, -math.pi/2, math.pi/2, math.pi/2, -math.pi/2, math.pi/2, math.pi/2]
sin_alpha_vect = np.sin(alpha_vect)
cos_alpha_vect = np.cos(alpha_vect)
neg_sin_alpha_vect = -sin_alpha_vect
neg_d_times_sin_alpha_vect = np.multiply(d_vect, neg_sin_alpha_vect)
d_times_cos_alpha_vect = np.multiply(d_vect, cos_alpha_vect)

# calcola la n-esima matrice necessaria a fare FK
def getTn(joint_states, n):
	r = r_vect[n]
	d = d_vect[n]
	sin_alpha = sin_alpha_vect[n]
	neg_sin_alpha = neg_sin_alpha_vect[n]
	neg_d_times_sin_alpha = neg_d_times_sin_alpha_vect[n]
	cos_alpha = cos_alpha_vect[n]
	d_times_cos_alpha = d_times_cos_alpha_vect[n]
	sin_theta = math.sin(math.radians(joint_states[n]))
	cos_theta = math.cos(math.radians(joint_states[n]))

	M = np.array([
		[cos_theta, -sin_theta, 0, r],
		[sin_theta*cos_alpha, cos_theta*cos_alpha, neg_sin_alpha, neg_d_times_sin_alpha],
		[sin_theta*sin_alpha, cos_theta*sin_alpha, cos_alpha, d_times_cos_alpha],
		[0, 0, 0, 1]]);

	return M

# applica l'algoritmo per fare FK della giuntura n-esima
def getJointPos(joint_states, n):
	DH = getTn(joint_states, 0)

	for i in range(1, n):
		DH = np.dot(DH, getTn(joint_states, i))

	# ritorna la posizione xyz
	return DH[0:3, 3]

# come sopra, ma cumulativo: ritorna un vettore di tutte le posizioni.
# ho scritto questa funzione semplicemente per risparmiare calcoli!
# NOTA IMPORTANTE: inizia da 0,0,0!!
def vectGetJointPos(joint_states):
	DH = getTn(joint_states, 0)

	res = np.array([[0, 0, 0], DH[0:3, 3]])

	for i in range(1, 7):
		DH = np.dot(DH, getTn(joint_states, i))
		res = np.vstack([res, [DH[0:3, 3]]])

	# ritorna la posizione xyz
	return res

def hasHitObstacle(pos_vect, obstacle, distance_check):
	# cicla su tutte le righe tranne l'ultima, che verrai raggiunta comunque essendo che mi sposto avanti di 1
	# in pratica ogni volta genero il segmento P0->P1, e con una formula trovata sul link
	# qui sopra calcolo la distanza minima tra questo segmento e l'ostacolo. Se questa e' minore del raggio, ho colpito, ritorno true.
	# se per nessun segmento ho verificato la condizione, non ho colpito l'ostacolo: bene!
	for i in range(len(pos_vect)-1):
		P0 = pos_vect[i]
		P1 = pos_vect[i+1]

		v = P1-P0
		w = obstacle - P0

		# uso il raggio squared per risparmiare conti
		c1 = np.dot(w,v)
		if c1 <= 0:
			if math.pow((obstacle[0]-P0[0]), 2) + math.pow((obstacle[1]-P0[1]), 2) + math.pow((obstacle[2]-P0[2]), 2) <= distance_check:
				return True
		else:
			c2 = np.dot(v,v)
			if c2 <= c1:
				if math.pow((obstacle[0]-P1[0]), 2) + math.pow((obstacle[1]-P1[1]), 2) + math.pow((obstacle[2]-P1[2]), 2) <= distance_check:
					return True
			else:
				b = c1/c2
				Pb = P0 + b*v
				if math.pow((obstacle[0]-Pb[0]), 2) + math.pow((obstacle[1]-Pb[1]), 2) + math.pow((obstacle[2]-Pb[2]), 2) <= distance_check:
					return True

	# Sono soppravvissuto a tutti i test, Bene!
	return False

def publish_message(joint_states):
	global publish_count
	publish_count += 1
	state_rad = deg2rad(joint_states)

	#Prepare the ROS msg
	msg = JointState()
	msg.header = Header()
	msg.header.stamp = rospy.Time.now()
	msg.name = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2']
	msg.position = state_rad
	msg.velocity = []
	msg.effort = []

	#Publish message on the correct ros topic
	#rospy.loginfo(msg)
	pub.publish(msg)

# returns a tuple: (bool, list). List is the new joint states, bool says if we're done or not.
def aux_reach(target_angles, joint_states, obstacle, distance_check, step = default_step):
# if only one joint is different from the 
# target state, we will set done to False.
# increses/decreases one step at a time.
# original_vect is used in case, with the "fastest way", we would have a collision. See after
	original_vect = joint_states
	done = True
	for i in range(len(target_angles)):
		if (target_angles[i] - joint_states[i]) > step:
			joint_states[i] += step
			done = False
		elif (target_angles[i] - joint_states[i]) < -step:
			joint_states[i] -= step
			done = False
		else:
			joint_states[i] = target_angles[i]
# if with this disposition we hit...
	if hasHitObstacle(vectGetJointPos(joint_states), obstacle, distance_check):
		# try again: done = False!!
		done = False
		# take a random action
		action = np.random.randint(0, 10)
		# and verify if, ont the "Original" state, we would have hit
		while not isItSafe(original_vect, action, obstacle, distance_check, step = step):
			action = np.random.randint(0, 10)
		# when we found a safe action, we perform it
		joint_states = compute_new_state(original_vect, action, step = step)

	return (done, joint_states)

# iteratively tries to reach the target angles from the
# joint states, using that publisher
def reach(target_angles, joint_states, obstacle, distance_check):
	# every joint has a workingspace of 100, so 150 is more than enough
	timeout = 500
	count = 0
	# do one step and check if we're done
	done, joint_states = aux_reach(target_angles, joint_states, obstacle, distance_check)
	# publish the new position
	publish_message(joint_states)
	# rospy sleep
	rate.sleep()
	# ...and do the same untile we're done or timeouted
	while (not done) and (count < timeout):
		done, joint_states = aux_reach(target_angles, joint_states, obstacle, distance_check)
		publish_message(joint_states)
		rate.sleep()
		count += 1

	return joint_states

# given a certain state, a certain action, a certain obstacle, is it safe? with a certain step, also!
def isItSafe(state_deg, action, obstacle, distance_check, step = default_step):
	return not hasHitObstacle(vectGetJointPos(compute_new_state(state_deg, action, step = step)), obstacle, distance_check)

# Main function
# subscription to the visualziation topic and the franka joint topic
def talker():
	marker_publisher = rospy.Publisher('visualization_marker', Marker, queue_size=10)

	model = create_model()

	reaches = 0
	success = 0
	obs = 0

	arm_radius = 0.02
	arm_radius_squared = arm_radius*arm_radius

	tar_radius = 0.10

	for sample in range (episodes):

		# Set a random target and obs
		state_deg = [0, 40, 0, -40, 0, 90, 0, 0, 0]
		x, y, z = endEffectorPos( state_deg[0:7] )
		while True:
			target, target_angle = generate_target()
			obstacle = generate_obstacle(state_deg, target)
			obs_radius = np.array([0.03 + rnd(0, 0.03)]) #randomized radius!
			# we continue ONLY IF the obstacle DOES NOT intersecate the target. Else, we retry.
			if math.sqrt( math.pow((obstacle[0]-target[0]), 2) + math.pow((obstacle[1]-target[1]), 2) + math.pow((obstacle[2]-target[2]), 2) ) > obs_radius + tar_radius:
				break

		# we use a distance already squared because it is faster
		distance_check = obs_radius*obs_radius + arm_radius_squared + 2*obs_radius*arm_radius

		#while not rospy.is_shutdown():
		# Loop for the trajectory (max 150 step before shutdown)
		for iter in range(150):
			# Set NN input and get the output
			input_layer = np.concatenate((normalized_state(state_deg[0:7]), target, obstacle, obs_radius)) #input: the normalized (5!!!) joints, xyz of target, xyz of obs, radius= 12 elemnts!!

			# best action avaiable
			action = np.argmax(model.predict(np.array([input_layer])))

			# Compute new state, degrees and rad
			state_deg = compute_new_state(state_deg, action)

			publish_message(state_deg)

			# Draw marker graphic
			marker_spawn(marker_publisher, target, tar_radius, [0, 1, 0], 0)
			# Draw obs graphic
			marker_spawn(marker_publisher, obstacle, obs_radius, [1, 0, 0], 1) #another id!

			# Compute distances
			x, y, z = endEffectorPos( state_deg[0:7] )
			tar_distance = math.sqrt( math.pow((x-target[0]), 2) + math.pow((y-target[1]), 2) + math.pow((z-target[2]), 2) )
			
			if(hasHitObstacle(vectGetJointPos(state_deg), obstacle, distance_check)):
				print ("FAIL: OBS REACHED!!")
				obs += 1
				break
	
			# Check if the target is reached
			if(tar_distance < tar_radius):
				# we reached it. we'll us this number for a stat
				reaches += 1
				state_deg = reach(target_angle, state_deg, obstacle, distance_check)
				# recheck distance...
				x, y, z = endEffectorPos( state_deg[0:7] )
				tar_distance = math.sqrt( math.pow((x-target[0]), 2) + math.pow((y-target[1]), 2) + math.pow((z-target[2]), 2) )
				if tar_distance < 0.001:
					success += 1
				break

			rate.sleep()

		# Debug print, get the success rate of the test
		print ("Success :" + str(success) + "/" + str(sample + 1) + " with " + str(obs) + " obs hits and " + str((sample + 1)-success-obs) + " timeouts.")
		print (target)
		print (state_deg)
	
	print ("Of the %d times we reached the green space, we had %d successes." % (reaches, success))
	print ("Average: %.2f message published per iteration" % (publish_count*1.0/episodes))

if __name__ == '__main__':
	try:
		talker()
	except rospy.ROSInterruptException:
		pass
