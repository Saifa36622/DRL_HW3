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

then the rest of the algorithm is the same with q-learning by using the selected policy (in this project will be epsilon greedy) to selected the optimal action


### Deep Q learning (DQN)

Similar to linear q learning but instead of using linear function to approax the q value ,this algorithm use of deep neural networks to approximate value and Q-functions.

In deep Q-learning, Q-functions are represented using deep neural networks. Instead of selecting features and training weights, we learn the parameters $\theta$ to a neural network. The Q-function is $Q(s,a;\theta)$

The update rule for deep Q-learning looks similar to that of updating a linear Q-function.

The deep reinforcement learning TD update is:

$$\theta \leftarrow \theta + \alpha \cdot \delta \cdot \nabla_{\theta} Q(s,a; \theta)$$

where : <br>
  $\nabla_{\theta} Q(s,a; \theta)$ = gradient of the Q-function<br>
  $a$ = learning rate <br>
  $δ$ = TD error from same equation as Linear q-learn

how it work 

[flow pic]

exlpain flow 


### MC reinforce (Monte Carlo Policy Gradient Algorithm)

This algorithm directly optimize the policy $𝜋(𝑎∣𝑠; 𝜃)$, by increasing the probability of actions that lead to high rewards.There is no No value function (Q or V) and Update parameter $𝜃$ in the direction that makes good actions more and also similar to Monte Carlo by Wait until the episode ends then Calculate Return $G_t$ Then update

MC REINFORCE Update Rule

$$\theta \leftarrow \theta + \alpha \cdot G_t \cdot \nabla_\theta \log \pi(a_t|s_t; \theta)$$

where :

$𝜃$ = Policy parameter (Weight of NN) -> [ $\nabla_\theta \log \pi(a_t|s_t; \theta)$ = Gradient of Log Policy Probability ] <br>
$a$ = Learning Rate <br>
$G_t$ = Return (Sum of future rewards from time t) from this equation 

$$G_t = r_{t} + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^{T-t-1}r_{T}$$

or 

$$G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$$


 &emsp;&emsp;where : <br>
  &emsp; &emsp; &emsp;$t$ = Current timestep <br>
  &emsp; &emsp; &emsp;$r_t$ = Immediate reward at timestep $t$	<br>
  &emsp; &emsp; &emsp;$r_{t+1} , r_{t+2} ,...,$  = Future rewards at time step $t + k$ <br>
  &emsp; &emsp; &emsp;$T$ = finale episode before update the parameter <br>
  &emsp; &emsp; &emsp;$γ$ = Discount Factor

  [flow pic]

exlpain flow 

### Proximal policy optimization (PPO)