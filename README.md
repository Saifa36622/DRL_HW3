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
linear q learning will approaximate Q(s,a) using Linear function

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


then when the Calculate Q-value for all action from the eqation

<p align="center">
  <img src="image/image8.png" alt="alt text">
</p>

or we can write it in this form 

<p align="center">
  <img src="image/image4.png" alt="alt text">
</p>

the n the out put the Q-value will be Vector size 1 x 5 → 5 Q-value → each for action 5 action 

<p align="center">
  <img src="image/image5.png" alt="alt text">
</p>

(if u want to Q-value of specific action a only ,we can use dot product between obs and W column of action a)

<p align="center">
  <img src="image/image6.png" alt="alt text">
</p>

To update the weights (update rule) on all weights will be 0 then update the weight according to 

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

where : 
<br>
    $γ$ = Discount Factor <br>
    $Q(s′,a′)$ = The predicted Q-value from the next state 𝑠′ ,across all possible actions 𝑎′ ( max -> Best possible action in next state )


$prediction$ = $Q(s,a)$ or Q-value from the this state

or in full equation as 

$$δ = r + γ ⋅ maxQ(s′,a′) - Q(s,a)$$

So we can write the update rule in full form as this 

<p align="center">
  <img src="image/image11.png" alt="alt text">
</p>

