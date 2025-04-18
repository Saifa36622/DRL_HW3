# **Cartpole**

## **Part 1 : Review algorithm**

### Linear Q learning

Linear Q-Learning is a type of Q-Learning where instead of using a table to store the Q-values, we use a linear function (a weighted sum of features) to approximate the Q-values (still **value-based** like q learning -> it mean still use epsilon greedy as policy)

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

In deep Q-learning, Q-functions are represented using deep neural networks. Instead of selecting features and training weights, we learn the parameters $\theta$ to a neural network. The Q-function is $Q(s,a;\theta)$ (still **value-based** like q learning -> it mean still use epsilon greedy as policy)



**component of DQN** 

1. **Eval Net (Online Net)**

"Eval Net" or "Evaluation Network" is the main neural network in Deep Q-Network (DQN) used to **estimate the current Q-value** 𝑄 ( 𝑠 , 𝑎 ; 𝜃 ) given the current state 𝑠  and action 𝑎 .It is trained frequently using the loss between its own prediction and the target Q-value generated using Target Net.

This component other than use to **estimate the current Q-value** it still use as **action selection** ,and in this project we will use Epsilon-Greedy that will selected by $argmax_a Q(s,a;θ)$

The Eval Net use TD update to update the weight $θ$ (aka Gradient Descent equation) that will updated via backpropagation using the loss function:

$$\theta \leftarrow \theta + \alpha \cdot \delta \cdot \nabla_{\theta} Q(s,a; \theta)$$

where : <br>
  $\nabla_{\theta} Q(s,a; \theta)$ = gradient of the Q-function<br>
  $a$ = learning rate <br>
  $δ$ = TD error (Target - Prediction)

or we can write it as 

$$θ← θ - α⋅∇ θ ​ L(θ)$$

where : 

$θ$ = Model Parameters -> Weight of NN

$L(θ)$ = Loss Function -> How bad model prediction is (validate by MSE)

$∇_θL(θ)$ =  Gradient of Loss -> Tells which direction decrease the Loss the most

$α$ = Learning Rate	

**both eqution work the same by Update to move 𝜃 a little bit towards lower loss**

2. **DQN loss function**

aka a mathematical way to measure How wrong your model is.by compares

**What your model thinks → vs → What is the correct answer**

Formula deffinition 

$$L(θ)=f(Prediction,Ground Truth)$$

where :

$𝜃$ = Model weight (to be updated)

Prediction → From model output such as $Q(s,a;θ)$

Ground Truth → Real answer 

So we can write it in DQN term as follow 

$$L(v)=E[(Q_{max} ​ − Q_{eval} ​ (s_τ ​ ,a_τ ​ ;v))^2 ]$$

or we can write it as 


$$L(θ)=(y−Q(s,a;θ))^2$$

when 

$$y = r+γ \cdot maxQ_{target} ​ (s^′ ,a^′ ;θ^- )$$

where :

$L(θ)$ = Loss Function	

$𝑟$ = real reward from environment

$𝛾$ = discount factor

$Q(s,a;θ)$ = Predict from Eval Net

$Q(s^′,a^′;θ^-)$ = Target from Target Net

the target of loss function is to **Minimize Loss**	to improve model accuracy and the purpose of the Gradient of Loss	is to Tells which **direction to change the weight**



3. **Target Net**

Target Network is a separate neural network in the Deep Q-Network (DQN) algorithm that is used to provide a stable and consistent target value **aka $𝑄_{max}$ when calculating the loss** for training the Evaluation Network (Eval Net).

If you directly calculate target using Eval Net (same network that is learning), the target will keep changing every step → very unstable.

Target network soft update equation 

$$\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-$$

Slowly follow Eval Net with small update rate $𝜏$ (ex: 0.005) and $τ$ is Soft Update Rate parameter

Hard update (evry N stetp)

$$\theta^- = \theta$$

Copy Eval Net weight to Target Net every few steps (ex: every 1000 steps).

how it work 

<p align="center">
  <img src="image/image14.png" alt="alt text">
</p>
<p align="center">
  Example Q-network (DQN) algorithm flow chart <br>
  (Figure (1) from Wang et al., "Flexible Transmission Network Expansion Planning Based on DQN Algorithm")
</p>

1. initial state 

pass the state $s_0$ from the enviroment to the eval net

2. Choose Action $a_t$

in eval net ,It use epilon greedy to seleceted the action 
with probability $𝜖$ -> selected random action or probability $1- 𝜖$ -> selected by $argmax_a Q(s,a;θ)$ from the estimate $Q(s,a;θ)$ from NN 

3. Execute and revieve observation feedback 

Environment responds the param such as Next state $s_{t+1}$ ,reward $r_t$ or done flag

4. Store experience in Replay Buffer

store the feedback from the enviroment in the replay buffer 

5. Sample mini-batch from Replay Buffer

Randomly sample multiple past experiences (feedback) at batch size $k$ (such as 32 or 64) back in to eval net for the NN model in eval net to estimate (to calculate on loss funciton in the future)

6. Calculate Target Q-Value (from Target Net)

Randomly sample multiple past experiences (feedback) at batch size $k$ (such as 32 or 64) to target net for the NN model in target net too (to calculate on loss funciton in the future)

7. Calculate Loss Function

calculate loss function from the result of the eval net and target net at the same Replay Buffer batch at time $t$

$$ L(θ)=E[(Q_{target} ​− Q_{eval} ​ (s,a;θ))^2 ]$$

8. Update Eval Net (Gradient Descent)

then update the Eval Net weight using the loss function that we calculate 

$$θ←θ−α∇_θ ​ L(θ)$$

9.  Update Target Net (Slowly Follow Eval Net)

 on soft update or Hard Update every N steps

10. Then repeat the step 

### MC reinforce (Monte Carlo Policy Gradient Algorithm)

This algorithm directly optimize the policy or **policy base** $𝜋(𝑎∣𝑠; 𝜃)$, by increasing the probability of actions that lead to high rewards.

Update parameter $𝜃$ in the direction that makes good actions increaseing and also similar to Monte Carlo by Wait until the episode ends then Calculate Return $G_t$ Then update

when it policy base it mean there is no value function (Q or V) .Instead of Deterministic Policy that forced randomness from ε like linear q learn or DQN that use ε-greedy to selected the random action this algorithm are using **Stochastic Policy** that mean the agent will choose probability distribution over actions and we acn achive that by apply Softmax function at the output layer of the Neural Network

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

Proximal policy optimization (PPO) is an on-policy, policy gradient reinforcement learning method for environments with a discrete or continuous action space. It directly estimates a **stochastic policy** and uses a **value function critic** to estimate the value of the policy. This algorithm alternates between sampling data through environmental interaction and optimizing a clipped surrogate objective function using stochastic gradient descent

The clipped surrogate objective function improves training stability by limiting the size of the policy change at each step. For continuous action spaces, this agent does not enforce constraints set in the action specification; therefore, if you need to enforce action constraints, you must do so within the environment.


**PPO component**

1. Actor $π(A|S;θ)$ — The actor, with parameters $θ$, outputs the conditional probability of taking each action A when in state S as one of the following:

- Discrete action space — The probability of taking each discrete action. The sum of these probabilities across all actions is 1.

- Continuous action space — The mean and standard deviation of the Gaussian probability distribution for each continuous action.

2. Critic $V(S;ϕ)$ — The critic, with parameters $ϕ$, takes observation S and returns the corresponding expectation of the discounted long-term reward.

During training, the agent tunes the parameter values in $θ$.

(Both actor and critic are Nueral network model)

**PPO key method**

1. Clipped Surrogate

key trick used in PPO to prevent the updated policy from moving too far away from the old policy in a single update.(limit policy update)

by using this equation 

$$L^{CLIP} (θ)=E t ​ [min(r_t ​ (θ)A_t \space​ ,\space clip(r_t ​ (θ)\space,\space 1−ϵ \space ,\space 1+ϵ)A_t)]$$

where :

$r_t ​ (θ)A_t$ = Normal Policy Gradient <br>


$clip(r_t ​ (θ)\space,\space 1−ϵ \space ,\space 1+ϵ)A_t$ = Force $𝑟_𝑡$ ​ to stay between $1 − 𝜖$  and $1 + 𝜖$ to limit update size

such as the example pic 

<p align="center">
  <img src="image/image15.png" alt="alt text">
</p>
<p align="center">
  Visualization of the PPO Clipped Surrogate Objective <br>
  (Figure (2) from Schulman et al., "Proximal Policy Optimization Algorithms")
</p>

 The x-axis represents the probability ratio $𝑟$ that come from $r(\theta) = \frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)}$
  or ratio between new and old policies, and the y-axis shows the clipped objective function $𝐿_{CLIP}$.

 The left graph shows the case when the advantage 𝐴 > 0 , meaning the action taken was better than expected. In this case, the function grows linearly with $𝑟$ up to a threshold at $1 + 𝜖$, after which it flattens. This prevents the new policy from becoming too confident in good actions. 
 
 The right graph shows the case when 𝐴 < 0 , indicating a worse than expected action. Here, the loss remains flat until $𝑟 < 1 − 𝜖$ , where it starts to penalize the update. 
 
 In both cases, the clipping mechanism ensures the update remains within a trust region $[1−ϵ,1+ϵ]$, preventing large and destabilizing changes to the policy, thereby improving training stability and efficiency.


 2. Adaptive KL Penalty Coefficient (optional)

 Another approach, which can be used as an **alternative** to the clipped surrogate objective, or in addition to it 


<p align="center">
  <img src="https://latex.codecogs.com/png.latex?L%5E%7B%5Ctext%7BKL-Penalty%7D%7D%28%5Ctheta%29%20%3D%20%5Chat%7B%5Cmathbb%7BE%7D%7D_t%20%5Cleft%5B%20%5Cfrac%7B%5Cpi_%5Ctheta%28a_t%20%5Cmid%20s_t%29%7D%7B%5Cpi_%7B%5Ctheta_%7B%5Ctext%7Bold%7D%7D%7D%28a_t%20%5Cmid%20s_t%29%7D%20%5Chat%7BA%7D_t%20-%20%5Cbeta%20%5Ccdot%20%5Cmathrm%7BKL%7D%28%5Cpi_%7B%5Ctheta_%7B%5Ctext%7Bold%7D%7D%7D%28%5Ccdot%20%5Cmid%20s_t%29%5C%3B%5C%7C%5C%3B%5Cpi_%5Ctheta%28%5Ccdot%20%5Cmid%20s_t%29%29%20%5Cright%5D" alt="KL-Penalty Loss">
</p>




Where:

$L^{KL}(\theta)$ = Loss with KL penalty

$A_t$ = Advantage at timestep $t$

$\beta$ = Adaptive KL penalty coefficient
 
$D_{KL}$ = KL Divergence between old and new policy

Adaptive KL Penalty are adjusts $\beta$ based on how much the new policy diverges from the old policy.

** **It is optional for PPO algorithm so in this project we will not yet implement that** **

 3. Advantage function

advantage function is function helps us understand how much better (or worse) an action is compared to the average action in a particular state.

The advantage function is defined as:

$$A^π (s,a)=Q^π (s,a) − V^π(s)$$

when : 

$Q^π (s,a)$ = The expected return after taking action $𝑎$ in state $𝑠$  and following policy $𝜋$ 

$V^π(s)$ = Expected Return from $𝑠$ without specific action

from the equation it will indicate that If I take action $𝑎$ now, is it better than my average action at state $𝑠$ or not ? ,So the advantage functions help reduce the variance in policy gradient updates, leading to more stable and efficient learning.

**Different Methods for Estimating Advantage**

there are multiple ways to estimate the advantage such as 

- Simple Advantage (One-step TD)

$$A^t ​ =r_t ​ +γV(s_{t+1} ​ )−V(s_t ​)$$

this is quick but can be high variance

- Monte Carlo Advantage

$$A^t ​ =G_t ​ − V(s_t ​)$$

More accurate but high variance and needs to wait until episode ends.


- GAE (Generalized Advantage Estimation)

$$A_t = \sum_{l=0}^{T - t - 1} (\gamma \lambda)^l \delta_{t+l}$$

when 

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$


or we can write it in full form as 

$$A_t = \sum_{l=0}^{T - t - 1} (\gamma \lambda)^l \left( r_{t+l} + \gamma V(s_{t+l+1}) - V(s_{t+l}) \right)$$

where : 

$λ$ = controls bias-variance tradeoff

$γ$ = discount factor

$V(s)$ = value function

$δ_t$ = TD error

in this project we will selected the **GAE method** cause it designed to balance learning speed and stability.

4. Value function for Critic 

In PPO, the actor chooses actions (via the policy), and the critic helps evaluate how good those actions are. The critic is trained to approximate the true return, which we call:

$$V_t^{\text{target}} = G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}
$$

or in Generalized Advantage Estimation (GAE) term

$$V_t^{\text{target}} = \hat{A}_t + V(s_t)$$


This target value tells the critic what the true total reward from state $s_t$ was (either via Monte Carlo or bootstrapped from next state values).

Value Function Loss Formula

To train the critic or to make the value predictions accurate
PPO minimizes the squared difference between:

- The predicted value $V_{\theta}(s_t)$
- The actual return $V_t^{\text{target}}$

by this equation 

<p align="center">
  <img src="https://latex.codecogs.com/png.latex?L%5E%7B%5Ctext%7BVF%7D%7D_t%28%5Ctheta%29%20%3D%20%28V_%5Ctheta%28s_t%29%20-%20V_t%5E%7B%5Ctext%7Btarget%7D%7D%29%5E2" alt="Value Function Loss">
</p>



The output will be mean squared error (MSE), and it tells the critic how much error the critic esmitate was off on this state

5. Entropy Bonus

Entropy Bonus is used in PPO (and other DRL algorithms) to encourage exploration rather than premature convergence to a deterministic (greedy) policy too early.

Equation of Entropy Bonus

$$
S_\pi = - \sum_{a} \pi_\theta(a|s) \cdot \log \pi_\theta(a|s)
$$

where : 

$\pi_\theta(a|s)$ = probability of action $𝑎$  at state $𝑠$ 

then we can write it as in Entropy Bonus Loss added in PPO 


<p align="center">
  <img src="https://latex.codecogs.com/png.latex?L%5E%7B%5Ctext%7Bentropy%7D%7D_t%28%5Ctheta%29%20%3D%20c_2%20%5Ccdot%20S_%5Cpi" alt="Entropy Loss">
</p>



where : 

$c_2$ = entropy coefficient (usually small like 0.01) -> Controls how strong exploration force is


$S_\pi$ = entropy at state $s_t$

6. Loss function for PPO

similar to loss function from DQN ,for PPO loss function is improve the policy (how actions are selected) while preventing updates that are too large, which can destabilize training. So, the loss function in PPO balances Policy improvement , Stability (small policy changes),Value function learning and Entropy for exploration

the loss function equation for PPO is 


$$ L^{PPO} (θ)= \mathbb{E}_t ​ [L_t^{CLIP} ​ (θ)− c_1 ​ L_t^{VF} ​ (θ)+c_2 ​ S[π_θ ​ ](s_t ​ )] $$


Where:

$L_t^{CLIP}$ = Clipped policy surrogate loss

$L_t^{VF}$ = Value function loss (critic)

$S[π_θ ​ ](s_t ​ )$ = Entropy bonus (encourages exploration)

$c_1 , c_2$ = Coefficients to balance the terms

then we write the PPO from this suedo code   

<p align="center">
  <img src="image/image16.png" alt="alt text">
</p>
<p align="center">
  PPO Algorithm Pseudocode Flow (Actor-Critic Style)
  <br>
  (Figure (3) from Schulman et al., "Proximal Policy Optimization Algorithms")
</p>

the we can write the flow as

<p align="center">
  <img src="image/PPO6.png" alt="alt text">
</p>

The process begins with initialization of the actor and critic networks. An initial state $s_0$ is sampled from the environment, and the actor network (parameterized by $θ$) selects an action $a_t$ based on the current policy $π_θ$. The selected action is then executed in the environment, which provides feedback

Next, the Advantage Function is calculated using Generalized Advantage Estimation (GAE) to evaluate how good an action was compared to the expected value. Simultaneously, the critic network (parameterized by 
$𝜙$) estimates the value function $V_ϕ(s)$, and its parameters are updated via gradient descent from loss function such as this equation 

Gradient Descent equation

$$ϕ←ϕ−α⋅∇_ϕ ​ L^VF (ϕ)$$

Where:

$α$ = learning rate

$∇_ϕ$ = gradient of the value loss -> critic weights

Entropy is then calculated to encourage exploration and prevent premature convergence.

Using the advantage estimates and clip ratio, the loss function is formed (including the clipped surrogate loss, entropy bonus, and value function loss). The final step is applying gradient descent to minimize the total loss and update the actor network's parameters $θ$

Then loop it all agian

### Conclusion

So we can conclude that 

| Algorithm       | Approach        | Policy Type   | Observation Space | Action Space   | Exploration vs. Exploitation Strategy |
|----------------|------------------|---------------|-------------------|----------------|----------------------------------------|
| **Linear Q**   | Value-Based      | Deterministic | Discrete          | Discrete       | Epsilon-greedy: starts with high exploration (`ε=1.0`), decays over time. |
| **DQN**        | Value-Based      | Deterministic | Continuous        | Discrete       | Epsilon-greedy + Replay Buffer: balances by decaying ε and learning from diverse past experiences. |
| **MC REINFORCE** | Policy-Based     | Stochastic    | Continuous        | Discrete       | Inherent stochasticity + policy gradient encourages exploration; learns from full episodes. |
| **PPO**        | Actor-Critic     | Stochastic    | Continuous        | Continuous / Discrete    | Uses clipped objective to prevent overly large policy updates; stochastic policy ensures ongoing exploration. |


## Part 4: Evaluate Cart-Pole Agent performance

To evaluted we will propose the base performance result of each hyperparameter fine tune and result of the fine tune of each parameter 

### Linear Q-learning

with this set of parameter ,similar to q learning

"num_of_action": 7,
    "action_range": [
        -12.0,
        12.0
    ],
    "learning_rate": 0.0003,
    "epsilon_decay": 0.9997,
    "discount": 0.99,

The result of the train are 


<p align="center">
  <img src="image/image18.png" alt="alt text">
</p>

or can be show in the video as 



and if we plot the moveement will be shown as 



#### Key take aways 

even thought how many find tune the parameter the result of linear q learn are not that good that can be cuase from 
- Linear q learning use **linear function** to predict the $Q(s, a)$ or assume that value functions change linearly with state variables but cartpole task are **non-linear** motion dynamics .So, the linear model will be struggles to capture the actual complexity of the environment’s transitions and reward structure.


### DQN (Deep q learnig)

with this set of parameter

  "num_of_action": 7,
  "action_range": [
      -12.0,
      12.0
  ],
  "hidden_dim": 128,
  "learning_rate": 0.0003,
  "epsilon_decay": 0.9997,
  "discount": 0.99,
  "buffer size": 10000,
  "batch size": 64,

The result is as follow 


<p align="center">
  <img src="image/image19.png" alt="alt text">
</p>

and the loss function over time 

<p align="center">
  <img src="image/image20.png" alt="alt text">
</p>


or can be show in the video as 



and if we plot the moveement will be shown as 



and if we fin tune some parameter the result can be as follow 

<p align="center">
  <img src="image/image21.png" alt="alt text">
</p>

from the example picture there is 2 more data incldue that are 

- DQN with increase epsilon decay rate from 0.9997 -> 0.9998 (increase the value to slow down the decay rate)

- DQN wih less hidden_dim 


#### Key take away 

##### Normal DQN 

- agent learned a reasonably good policy, as shown by increased rewards and task count over time.
However, the training is not fully stable, and the spiky loss during exploitation that may need Better target network update frequency or soft updates or smaller learning rate to stabilize updates.

- for the loss function the General Trend show according to the teory that loss will be decreases over time, especially in the early phase from explore phase to expliot phase indicate that **The DQN is learning to better predict the Q-values** and The TD-error (Temporal Difference error) is being minimized as expected 

&emsp; &emsp;   but around the exploit phase there are some sharp spikes in the loss (according to theory there should be less loss or trending low)
that may cuase from too much **Replay Buffer** .If replay buffer contains a wide range of past experiences, the agent may sample rare transitions that are from older policies and that morre likely to cause unexpected Q-value updates or we can use smaller learning rate to stabilize updates **but** The loss stays below 0.05 most of the time in the exploit phase which suggests convergence is still happening with some noise .So the performance it not that bad 

##### DQN fine tune 


- DQN_increase_epsilon

  - agent uses a slower epsilon decay, allowing for longer exploration phase but Learns more conservatively in the beginning and  catches up over time

  - Achieves competitive performance later in training but with higher variance in reward and episode length

- DQN_less_hidden

  - Performs the worst in both reward and episode count that cause by using fewer hidden units, resulting in limited model capacity

  - Indicates that model expressiveness is critical for learning good policies, even in relatively simple tasks.

In conclusion Model capacity ( number of hidden units) is essential for learning an effective Q-function and exploration rate (epsilon decay rate) are greatly impacts the learning phase and stability.

### MC reinforce

with this set of parameter

"num_of_action": 7,
"action_range": [
    -12.0,
    12.0
],
"hidden_dim": 256,
"learning_rate": 0.0003,
"discount": 0.99,
"drop_out": 0.3,
"n_obs": 4


<p align="center">
  <img src="image/image22.png" alt="alt text">
</p>

or can be show in the video as 



and if we plot the moveement will be shown as 

#### Key take aways 

-  Unstable learning. Both reward and count remain low and unstable show that the agent does not consistently improve.
- Loss function is highly volatile, indicating unstable policy updates

- the result may cuase from 

1. Poor Neural network setup 

REINFORCE is a high-variance algorithm, and without stabilizers like baselines or critics, it relies heavily on the neural network to approximate the policy well .A weak or poorly structured network will not be able to Represent complex policies effectively and Generalize from noisy, sparse rewards and the cuase of the poor Neural network setup have a lot such as 
- Too small hidden layer 
- Poor activation functions ( such as using no non-linearity or only shallow layers causes underfitting ) , 
- learning rate mismatches if too high it can causes exploding gradients and if too low the agent can be stuck and have no learning.





### PPO

with this set of parameter

"num_of_action": 7,
"action_range": [
    -12.0,
    12.0
],
"hidden_dim": 128,
"learning_rate": 0.0003,
"epsilon_decay": 0.9998,
"discount": 0.99,
"buffer size": 10000,
"batch size": 64 and clipping policy = 0.2



and the result as follow 

<p align="center">
  <img src="image/image23.png" alt="alt text">
</p>

or can be show in the video as 



and if we plot the moveement will be shown as 



and if we fin tune some parameter the result can be as follow 

<p align="center">
  <img src="image/image24.png" alt="alt text">
</p>


from the example picture there is 1 more data incldue that are 

- PPO with more batch size (from 64 to 128)


#### Key take aways 


##### Normal PPO 


- the agent shows initial learning success, followed by sudden divergence in both actor and critic losses around episode 9k, leading to a collapse in performance that may from **local optimum** -> The local optimum is when The agent learns a behavior that consistently gives a moderate reward then it stops exploring and becomes overconfident in this behavior 

- Another thing that incidcate the local optimum problem is from the trajectory in the video that indicate that the agent is trying to balance the pole but did not seem to care about the cart position that cause the agernt to run out of bound and be terminate (might indicate to the problem in reward term)

- another problem is **Network architecture limitations and Hyperparameter sensitivity** .A shallow or poorly initialized neural network might not be expressive enough to model the complex reward landscape, trapping the agent in simple policies and also an inappropriate learning rate or clip range can cause gradients to overshoot, leading to policy collapse and locking the agent into bad strategies
 - from the **loss function of Critic** .we can see that on the initially loss are low and stable during early training but Around step ~10,000, there's a huge spike in loss, reaching values over 7000, then it collapses back down that might cause from value function exploded, possibly due to poor mini-batch sampling (very different next states or rewards) or instability in TD error or temporal difference targets. and that cuase the reward term and the count term to have significantly worse performance

##### PPO fine tune 

- PPO with more batch size are more stable and effective than the Normal one.Larger batch sizes seem to stabilize the learning process, reduce variance, and help the agent escape premature convergence or instability **but**  PPO with more batch still likely fell into a local optimum or suffered from exploding gradients due to poor hyperparameters or poor nueral network setup 

##### Example of local optima
![ab05197e-67ce-4cf4-be4d-17a0ae5afb0d](https://github.com/user-attachments/assets/1c606e91-c1b2-464c-b265-edde1f6c6dd7)

  In Deep Reinforcement Learning (DRL), local optima means the agent finds a way to get some reward, but it’s not the best way possible. The agent gets stuck doing okay actions and doesn’t explore better ones because it thinks it’s already doing well. This often happens when the environment is complicated, and the learning algorithm (like policy gradient) stops improving too early. To fix this, we use tricks like exploration (trying random actions sometimes), adding randomness to learning (like entropy), or smarter methods like PPO that help the agent keep exploring while learning stably. The goal is to help the agent not settle too early and find the best long-term strategy.

### Conclusion

compare all the algorithm 

<p align="center">
  <img src="image/image25.png" alt="alt text">
</p>

and to answer following question 

Which algorithm performs best? 
and 
Why does it perform better than the others? 

refference from only in this project we can conclude that 

- PPO

  - Shows a very steep improvement early on, reaching a count above 6 within just a few thousand episodes.This indicates it learns quickly how to balance the pole effectively.
  - Performance are not stable but remains consistently high, suggesting stable learning with room for optimization

- DQN 
  - Learns more slowly and steadily over time.Eventually statturate around count = 2, much lower than PPO.Indicates decent learning, but less efficient and stable than PPO

- MC reinforce
  - Shows almost no improvement throughout training.Learning stagnates early, likely due to high variance in return estimates or poor hyperparameter/NN design

- Linear q learing 

  - Performs the worst overall, staying flat near count = 0.5. due to the models that lack the performance to capturing CartPole’s dynamics

So I can say that 

Which algorithm performs best -> **PPO**

and why ? 

1. **Policy Gradient + Clipping**: PPO uses policy gradients with clipping, allowing for more stable updates without drastically changing the policy

2. **Actor-Critic base**: Benefits from separate policy (actor) and value (critic) networks, reducing variance and improving learning

3. **Better handling of continuous env**: PPO can naturally optimize within a continuous action space
