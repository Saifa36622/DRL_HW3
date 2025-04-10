# **Cartpole**

## **Review algorithm**

### Linear Q learning

Linear Q-Learning is a type of Q-Learning where instead of using a table to store the Q-values, we use a linear function (a weighted sum of features) to approximate the Q-values

So instead of keeping the q table like this 

$$Q(s,a) = value$$

and the value will be size of number of action ,So we can visualize it like this 

![alt text](image/image.png)

the table size will increase directly by size of <ins>observation space</ins> * <ins>action space</ins>


linear q learning will approaximate Q(s,a) using Linear function

$$Q(s,a) = obs · w_a $$

Where: 

$obs$ is observation space -> in this cart pole problem there wil be 4 obs term that is [cart_pos, pole_angle, cart_vel, pole_vel] 

![alt text](image/image3.png)

$w_a$ is weight matrix for each action on each state 

such as if I have number of action = 5 and 4 observation term the $w_a$ will represent the weight of each action on each state

![alt text](image/image2.png)


then when the Calculate Q-value for all action

![alt text](image/image4.png)

the Q-value will be = Vector size 1 x 5 → 5 Q-value → each for action 5 action 

![alt text](image/image5.png)

(if u want to Q-value of specific action a only ,we can use dot product between obs and W column of action a)

![alt text](image/image6.png)
