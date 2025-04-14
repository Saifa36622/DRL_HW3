# **Cartpole**

## **Review algorithm**

### Linear Q learning

Linear Q-Learning is a type of Q-Learning where instead of using a table to store the Q-values, we use a linear function (a weighted sum of features) to approximate the Q-values

So instead of keeping the q table like this 

$$Q(s,a) = value$$

and the value will be size of number of action ,So we can visualize it like this 

<p align="center">
  <img src="image/image.png" alt="alt text">
</p>

<br>

**but**

<br>

linear q learning will approaximate Q(s,a) using Linear function <br>

$$Q(s,a) = f(s,a) · w_a $$

Where: 

- **$f(s,a)$** is feature vector each function extracts the value of a feature for state-action pair , the size of this matrix will depend on  nunmber of action * observation space

<p align="center">
  <img src="image/image7.png" alt="alt text">
</p>


-> in this cart pole problem there wil be 4 obs term that is [cart_pos, pole_angle, cart_vel, pole_vel] 

<p align="center">
  <img src="image/image3.png" alt="alt text">
</p>

- **$w_a$** is weight matrix for each action on each state 

such as if I have number of action = 5 and 4 observation term the $w_a$ will represent the weight of each action on each state

<p align="center">
  <img src="image/image2.png" alt="alt text">
</p>


then when the Calculate Q-value for all action from the equation

<p align="center">
  <img src="image/image8.png" alt="alt text">
</p>

or we can write it in this form 

<p align="center">
  <img src="image/image4.png" alt="alt text">
</p>

then the output the Q-value will be Vector size 1 x 5 → 5 Q-value → each for action 5 action 

<p align="center">
  <img src="image/image5.png" alt="alt text">
</p>

(if u want to Q-value of specific action a only ,we can use dot product between obs and W column of action a)

<p align="center">
  <img src="image/image6.png" alt="alt text">
</p>

To update the weights (update rule) on inti all weights will be 0 then update the weight in every step according to 

<p align="center">
  <img src="image/image10.png" alt="alt text">
</p>

where 

$f_i(s,a)$ = current feature state vector <br>
$a$ = learning rate <br>
$δ$ = TD error from equation 

$$δ = target - prediction$$

where : 

$target$ = $r + γ ⋅ maxQ(s′,a′)$ <br>

&emsp;where : 
<br>
    &emsp;&emsp;&emsp; 
    $γ$  = Discount Factor <br>
    &emsp;&emsp;&emsp; 
    $Q(s′,a′)$ = The predicted Q-value from the next state 𝑠′ ,across all possible actions 𝑎′ ( max -> Best possible &emsp;&emsp;&emsp;action in next state )


$prediction$ = $Q(s,a)$ or Q-value from the this state

or in full equation as 

$$δ = r + γ ⋅ maxQ(s′,a′) - Q(s,a)$$

So we can write the update rule in full form as this 

<p align="center">
  <img src="image/image11.png" alt="alt text">
</p>

then the rest of the algorithm is the same with q-learning by using the selected policy (in this project will be epsilon greedy) to selected the optimal action and so on


### Deep Q learning (DQN)

Similar to linear q learning but instead of using linear function to approax the q value ,this algorithm use of deep neural networks to approximate value and Q-functions.

In deep Q-learning, Q-functions are represented using deep neural networks. Instead of selecting features and training weights, we learn the parameters $\theta$ to a neural network. The Q-function is $Q(s,a;\theta)$


The deep reinforcement learning TD update (aka Gradient Descent) is:

$$\theta \leftarrow \theta + \alpha \cdot \delta \cdot \nabla_{\theta} Q(s,a; \theta)$$

where : <br>
  $\nabla_{\theta} Q(s,a; \theta)$ = gradient of the Q-function<br>
  $a$ = learning rate <br>
  $δ$ = TD error (Target - Prediction)

or we can write it as 



**component of DQN** 

1. Eval Net (Online Net)



2. Target Net

Target network soft update equation 

$$\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-$$

Hard update (evry N stetp)

$$\theta^- = \theta$$

3. DQN lossfunction 


$$L(v)=E[(Q_{max} ​ − Q_{eval} ​ (s_τ ​ ,a_τ ​ ;v))^2 ]$$


or we can write it as 


$$L(θ)=(y−Q(s,a;θ))^2$$

when 

$$y = r+γ \cdot maxQ_{target} ​ (s^′ ,a^′ ;θ^- )$$

how it work 

<p align="center">
  <img src="image/image14.png" alt="alt text">
</p>
<p align="center">
  Example Q-network (DQN) algorithm flow chart <br>
  (Figure (1) from Wang et al., "Flexible Transmission Network Expansion Planning Based on DQN Algorithm")
</p>





### MC reinforce (Monte Carlo Policy Gradient Algorithm)

This algorithm directly optimize the policy or policy base $𝜋(𝑎∣𝑠; 𝜃)$, by increasing the probability of actions that lead to high rewards.

Update parameter $𝜃$ in the direction that makes good actions increaseing and also similar to Monte Carlo by Wait until the episode ends then Calculate Return $G_t$ Then update

when it policy base it mean there is no value function (Q or V) .Instead of Deterministic Policy that forced randomness from ε like linear q learn or DQN that use ε-greedy to selected the random action this algorithm are using Stochastic Policy that mean the agent will choose probability distribution over actions and we acn achive that by apply Softmax function at the output layer of the Neural Network

**Softmax function**

Softmax is a function that turns any raw score (real numbers) → into probability distribution by Take exp of every score so all values become positive & bigger gap for higher scores then divide by sum of all exp scores to normalize into probability (sum to 1)


<p align="center">
  <img src="image/image12.png" alt="alt text">
</p>

or we can write in DRL term as 

$$\pi_\theta(a|s) = \frac{e^{f(s, a)}}{\sum_{b} e^{f(s, b)}}$$

where : 

$\pi_\theta(a|s)$ = probability of choosing action 𝑎 given state 𝑠 or probability distribution over actions

$e^{f(s, a)}$ = output score from NN

$\sum_{b} e^{f(s, b)}$ = sum of output score

**Example**

if we wan to aply softmax to 3 action score 

NN output for 3 actions at state $s$

- Action 0 -> output 2.0
- Action 1 -> output 1.0
- Action 2 -> output 0.1

to compute we will apply 

$$e^z = [e^2.0,e^1.0,e^0.1] ≈ [7.39 ,2.71,1.10]$$

Sum all 

$$sum = 7.39 + 2.71 +1.10 = 11.2$$

then Compute Probability of each action

- Action 0 -> 7.39 / 11.2 ≈ 0.659 
- Action 1 -> 2.71 / 11.2 ≈ 0.242
- Action 2 -> 1.10 / 11.2 ≈ .098

Now, the agent doesn't pick action 0 directly like argmax but instead interpret the score as % 

- Action 0 → 66% chance
- Action 1 → 24% chance
- Action 2 → 10% chance


**MC REINFORCE Update Rule**

$$\theta \leftarrow \theta + \alpha \cdot G_t \cdot \nabla_\theta \log \pi(a_t|s_t, \theta)$$

where :

$𝜃$ = Policy parameter (Weight of NN) -> [ $\nabla_\theta \log \pi(a_t|s_t; \theta)$ = Gradient of Log Policy Probability ] <br>
$a$ = Learning Rate <br>
$G_t$ = Return (Sum of future rewards from time t) from this equation 

$$G_t = r_{t} + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^{T-t-1}r_{T}$$

or 

$$G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$$

 &emsp;&emsp;where : <br>

  &emsp; &emsp; &emsp; $t$ = Current timestep <br>

  &emsp; &emsp; &emsp; $r_t$ = Immediate reward at timestep $t$	<br>

  &emsp; &emsp; &emsp; $r_{t+1} , r_{t+2} ,...,$  = Future rewards at time step $t + k$ <br>

  &emsp; &emsp; &emsp; $T$ = finale episode before update the parameter <br>

  &emsp; &emsp; &emsp; $γ$ = Discount Factor


**Algorithm flow**


<p align="center">
  <img src="image/image13.png" alt="alt text">
</p>

The flow of the Monte Carlo REINFORCE algorithm begins with the initialization of the policy parameters 𝜃, which represent the weights of the neural network. The environment provides the initial state $s_0$ which is passed into the neural network to produce a raw output or logit for each possible action. This output is then processed through a Softmax function to convert it into a probability distribution $π(a∣s)$, ensuring that actions are chosen stochastically based on their probabilities.

An action is then sampled from this distribution and executed in the environment. The environment responds with feedback in the form of the next state, the reward received, and a termination flag. If the episode is not yet done, the process repeats: passing the new state through the neural network, applying softmax, sampling a new action, and interacting with the environment.

Once the episode ends ( when "done" is True), the algorithm proceeds to update the policy parameters. This update is based on the REINFORCE rule, which adjusts 𝜃 using the return $G_t$(the total discounted reward from timestep $t$ to the end of the episode) multiplied by the gradient of the log probability of the taken action $∇θlogπ(a_t∣s_t,θ)$. This gradient indicates the direction to adjust the policy to make the chosen action more or less likely in similar future situations.

Finally, the process loops back to the beginning, starting a new episode with the updated policy parameters.




### Proximal policy optimization (PPO)