---
title: Lesson 6
author: Kalpesh Chavan
description: Lecture notes converted from Jupyter notebooks.
pubDatetime: 2026-05-02T07:43:54Z
modDatetime:
draft: true
tags:
  - notebook
  - math
  - ml
---
# Refining RNNs

As discussed towards the end of last lesson the primary problem with a vanilla recurrent neural network is that as we move across timesteps, the gradients calculated towards the left-end (earlier timesteps) have an exponentially reduced impact compared to the latest calculated gradients.

- This effectively means that an RNN cannot "remember" across long timespans.They struggle with long term memory since they cannot adequetly recall information from much older timesteps.

- Earlier gradients contribute almost nothing to the eventual gradient update.

**We need a mechanism which filters out what is important to recall versus what can be forgotten.**

Most applications for RNNs involve sequential data:

- Next word prediction
- Time series analysis and prediction (ex. weather, stock data, etc)
- Natural language processing and generation
- Video analysis

RNNs need a way to capture data selectively as opposed to treating the data as all equal per timestep.

**One clever solution is to embed and train neural networks within each "unit" of the RNN to *gate* or decide what and when vector information to carry over to the next timestep!**

## Long Short Term Memory

What if we could embed a sort of "contextual" window of data within each RNN that could contain important data from much earlier in the network.

- RNNs are still viable in situations where the majority of data needed to predict the state can be ascertained from a nearby past state (i.e. only a few steps in the past) but otherwise to store much older data **we need to find a way to *artifact* that data in a stateful manner.**

By that logic lets introduce a cell state and denote it as $C$

This cell state will act as a persistent timeline which gets acted upon via other networks.

- Recall the most basic neural network can be thought of as a single neuron that outputs a probability from 0 to 1.
- Any classification neural network which outputs this 0-1 signal can do so by using various activation functions.
- For LSTMs our activation function of choice for the internal decision making networks will be the **sigmoid** function.

### The Anatomy of an LSTM Cell

Within each unrolled LSTM cell, each decision-making neural network can be called a "gate" and each of them will be tasked with a unique responsibility.

#### The Forget Gate $f_t$

- The *forget* gate will take the prior hidden state, $h_{t-1}$, and the current observation, $x_t$, as inputs and outputs 0 to 1 for each value in the most recent cell state $C_{t-1}$. The number it generates for each value in $C_{t-1}$ will indicate whether or not that value is worth forgetting.
- - 0 means lets drop this value completely (not worth recalling)
- - 1 means lets never ever let go of this value (hold onto it for dear life)

The math performed in the forget gate should remind you of a traditional neural network:

$$f_t = \sigma{(W_f \cdot [h_{t - 1}, x_t] + b_f)}$$

Then to apply this gate back onto the existing cell state we perform some multiplication:

$$f_t \odot C_{t - 1}$$

---

####  The Input Gate $i_t$

- The *input* gate will take the prior hidden state, $h_{t - 1}, the current observation, $x_t$, as inputs and outputs 0 to 1 for each value in the most recent cell state $C_{t-1}$. The number it generates for each value in $C_{t - 1}$ determines if a particular state value should be shifted.
- - O means that we should perform no shifts on a state value.
- - 1 means that we should apply a full magnitude shift on that state value.

The math performed in this gate is functionally identical to what occurs in the forget gate but uses a separate set of weights and biases:

$$i_t = \sigma{(W_i \cdot [h_{t - 1}, x_t] + b_i)}$$


Note that this gate is responsible for telling us what values to shift probabilstically. But this output alone is not enough information. Why?

- **Since the sigmoid output cannot be negative, this gate cannot tell us the directionality (or how) we should conduct each change.**

To supplement the sigmoid function we also construct a classifier that utilizes a third set of weights and biases but is activated by the $\tanh$ function as opposed to the sigmoid function!

- We label the output of this gate as $\tilde{C}$ to indicate that it is a subset of candidate values to concatenate into the existing cell state.

$$\tilde{C_t} = \tanh(W_c \cdot [h_{t - 1}, x_t] + b_c)$$

- **$\tanh$ outputs in a range of -1 to 1, which allows $\tilde{C_t}$ to express both the direction and magnitude of the values we are seeking to add in.**

*Every single value in $\tilde{C_t}$​ is nonzero. It's always proposing something to add to every dimension of the cell state.*

- This raw addition proposes a small **non-zero** shift on each dimension, but this is overkill and can result in a bloated cell state quite quickly.
- We can utilize the output of the sigmoid input gate $i_t$ as a mask to determine the necessity and magnitude of each of those updates via simple element wise multiplication.

$$i_t \odot \tilde{C_t}$$

---

#### Assembling the New Cell State $C_t$

After familiarizing yourself with each of the component gates we still have to compose them into a coherent sequence of operations.

Once again, given the following activations:

$$f_t = \sigma{(W_f \cdot [h_{t - 1}, x_t] + b_f)}$$

$$i_t = \sigma{(W_i \cdot [h_{t - 1}, x_t] + b_i)}$$

$$\tilde{C_t} = \tanh(W_c \cdot [h_{t - 1}, x_t] + b_c)$$

We apply them to the prior cell state, $C_{t-1}$, to generate the current cell state, $C_t$ in the following manner:

$$C_t = f_t \odot C_{t - 1} + i_t \odot \tilde{C_t}$$

This is what is logically occuring:

- The forget gate performs element-wise multiplication between $f_t$ and the prior cell state ($C_{t - 1}$) to lower OR maintain the significance of certain state values.
- The input gate performs element-wise multiplication between $i_t$ and the list of candidate updates ($\tilde{C_t}$) to determine how to update the cell state in new various directions.

Then we perform element-wise addition of the two results:
1) the filtered old state 
2) the scaled candidate updates

to produce a new cell state that retains relevant past information while incorporating new information from the current timestep.

Even after determining the new cell state we have to determine the next hidden state.

---

#### The Output Gate $o_t$

If the forget gate is responsible for choosing what information to retain, and the input gate is responsible for choosing what information additionally recall, then the output gate is responsible for taking the current input, $x_t$, and prior hidden state, $h_{t - 1}$, AND the current cell state, $C_t$ to generate the next hidden state $h_t$.

- The idea is to use the prior hidden layer and current input to inform what part of the cell state should be outputted via a sigmoid layer.
- But additionally we will multiply the output of that sigmoid network by the $\tanh$ activation of the cell state ($C_t$) **after the cell state has been processed through the forget and input gates**.

Mathematically the process is:

$$o_t = \sigma{(W_o \cdot [h_{t - 1}, x_t] + b_o)}$$

$$h _t = o_t \odot \tanh(C_t)$$

**The cell state and hidden state are then passed to the next timestep (i.e. the next cell). However, the final state can also be determined directly via the final hidden state. You feed that single final $h_t$ vector into a linear layer and softmax to get a class prediction. The intermediate hidden states are discarded. This is identical to the BPTT process that occurs with RNNs.**

- With LSTMs as is with traditional RNNs, we use many prior hidden input calculations to inform a single, final hidden output and its corresponding predection. But for tasks that require many past inputs to inform many output values we need another mechanism. We won't get there yet, but keep it in mind!

---

Here is a visual of all the operations that were discussed.

Credit: GeekForGeeks

![LSTM Cell](/notebooks/media/gate_of_lstm.webp)




