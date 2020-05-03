import numpy as np
import matplotlib.pyplot as plt

class k_bandit_epsilon_greedy:
	"""
	A multi-armed epsilon greedy bandit, where rewards are drawn 
	from a normal distribution with defined mean and variance.

	We will define this as a stationary problem: reward values do not change over time. 

	Author: Olga Bane, 2020 
	"""

	def __init__(self, k: int, epsilon: float, iters: int, initial_reward: float=0.0, 
		reward_means: np.array = None, reward_variances: np.array = None):

		# Number of arms (actions)
		self.k = k
		# Fraction of time that we explore
		self.epsilon = epsilon
		# Number of iterations
		self.iters = iters
		# Initialize reward for each action to initial reward
		self.action_values = np.repeat(initial_reward, k)
		# Store number of times action chosen
		self.action_chosen_counter = np.repeat(0, k)
		# Store total reward
		self.total_reward = 0.0

		# Reward for each action is drawn from some normal disribution with mean and 
		# This means that the reward is not always the same for a given action
		if reward_means is None:
			# Reward mean drawn from normal distribution with mean 0, variance 1
			self.reward_means = np.random.normal(0, 1, k)
		else:
			 self.reward_means = reward_means

		if reward_variances is None: 
			self.reward_variances = np.repeat(1, k)
		else:
			 self.reward_variances = reward_variances


	def choose_action(self):
		"""
		Choose to explore or exploit.
		"""

		# Explore or exploit
		rand_val = np.random.ranf(1)[0]
		if rand_val <= self.epsilon: # Explore
			# Explore: Pick action at random (low inclusive high exclusive)
			chosen_action = np.random.randint(0, self.k)
		else: 
			# Exploit: Pick action with highest reward
			chosen_action = np.argmax(self.action_values)

		return chosen_action
	

	def get_reward(self, chosen_action):
		"""
		Get reward from normally distributed reward outcomes
		Update total reward
		"""
		reward = np.random.normal(self.reward_means[chosen_action], 
			self.reward_variances[chosen_action])

		# Store total reward
		self.total_reward = self.total_reward + reward

		return reward


	def update_action_values(self, chosen_action, reward):
		"""
		Update action values.
		"""

		# Record that action was chosen
		self.action_chosen_counter[chosen_action] = self.action_chosen_counter[chosen_action] + 1
		# Update action_value
		self.action_values[chosen_action] = self.action_values[chosen_action] + 1/(self.action_chosen_counter[chosen_action]) * (reward - self.action_values[chosen_action])


	def run_bandit(self) -> float:
		"""
		Function that runs the bandit for the specified number of iterations.

		Returns:
			self.total_reward: sum of all rewards received.

		"""
		iteration_counter = 0

		while iteration_counter <= self.iters:
			chosen_action = self.choose_action()
			reward = self.get_reward(chosen_action)
			self.update_action_values(chosen_action, reward)

			# Update iteration counts
			iteration_counter = iteration_counter + 1

		return self.total_reward


	# Methods not required to run bandit, but useful for exploration.
	def plot_reward_means(self):
		"""
		# Plot reward means for some exploration.
		"""

		fig, ax = plt.subplots(1, 1, figsize=(8, 3))
		ax.scatter(range(len(self.reward_means)), self.reward_means, color='black')
		ax.set_xticks(range(len(self.reward_means)))
		ax.set_xlabel("Action")
		ax.set_ylabel("Reward Mean")
		plt.show()



	def get_max_reward(self):
		"""
		A function to identify the best action. 
		Of course, the bandit would not be able to do this, as the mean and variance of the 
		action-values is unknown to the bandit. This function can be used to compare bandit 
		performance to optimal outcome.
		"""

		# TODO: write function to determine the maximom possible reward.


def plot_average_reward(avg_reward_dict: dict, epsilon_val=float):
	"""
	Plot of average reward from bandit as number of iterations changes
	"""

	fig, ax = plt.subplots(1, 1, figsize=(8, 5))

	ax.scatter(avg_reward_dict.keys(), avg_reward_dict.values(), color='black')
	ax.set_xlabel("Number of iterations")
	ax.set_ylabel("Average reward")
	ax.set_title(f"\u03B5 greedy armed bandit, \u03B5 = {epsilon_val}")

	plt.show()

	return






