### 4.1

When we use --default_train, our code sets up a character-level language model with specific parameters that I can understand. It uses 3000 epochs (which means it'll learn from the text 3000 times), a hidden size of 128 (like the model's memory capacity), and 2 layers of LSTM (making it able to understand more complex patterns). The learning rate is set to 0.005, which controls how big the learning steps are - not too big, not too small.

Inside the training loop, something really interesting happens. I can see from the output that every 100 epochs (that's what print_every=100 does), it shows us three things: the time passed, how far along we are in training (like "3.333%"), and the current loss (starting around 2.5839 and eventually getting down to 1.4940). The lower loss means the model is getting better at predicting the next character.

What's fascinating is watching the generated text improve over time. At first (epoch 100), it's mostly gibberish: "Art shid fe whe these..." But by the end (epoch 3000), it's forming more coherent patterns and even using punctuation correctly. Every 10 epochs (plot_every=10), it's also saving the average loss to track how well the model is learning over time.

The output shows the model gradually getting better - the loss decreases from about 2.5 at the start to 1.4 at the end, and the text starts looking more like actual English. It's like watching the model learn to write, starting from random characters and slowly figuring out how English text should look!

### 4.3

#### Plotting the Training Losses

![Training Losses](loss_plot_1.png)


### 4.4 

#### Generated text with different temperatures:

**Prime string:** ith attent

**Temperature = 0.1:**
ith attent of the constion the come to have been the constion the more of the come to have been the constion t


**Temperature = 0.5:**
ith attent the portublishation the had before to hands's the same any come to his very consideming the bation,

**Temperature = 1.0:**
ith attentay:

  'In my look confurant, it on the
father pyetucted an whaty remwning finely-steaty, to be and


**Temperature = 2.0:**
ith attenty,
lome
sw? Bazlyondwanto
!Bleainsuppak hisponsl's-
-EpUyparled!s bjitrrys),S'w
up-b hom,ch?'

**Discussion:**
Temperature acts as a divisor for the output probabilities before we select the next character. When we use a very low temperature of 0.1, the model becomes extremely conservative, which is why we see it repeating "the constion" over and over. This happens because dividing by a small number makes the highest probabilities even more dominant, causing the model to always choose the most likely next character.

As we increase the temperature to 0.5, the division effect becomes more moderate. The model starts to consider other character choices beyond just the most probable one, leading to more varied text with words like "portublishation". It's still relatively structured but allows for some creativity.

At temperature 1.0, we're using the model's original probability distributions without much modification. This creates more natural-looking text with proper sentence structure and punctuation, as seen in the output with dialogue marks and natural-looking phrases.

When we push the temperature up to 2.0, we're dividing the probabilities by a larger number, which makes the differences between probable and improbable choices much smaller. This explains why we get such chaotic output with random-looking words like "Bazlyondwanto" and unusual punctuation patterns. The model is essentially treating all possible characters as nearly equally likely choices.

The temperature parameter fundamentally controls how "spread out" the probability distribution becomes when choosing the next character, determining whether the model plays it safe or takes creative risks in generating text.

---

#### Text Generation Risks and Responsibilities

When working with text generation models like our temperature-based character model, I can see several important risks we need to consider. The main risk is that the model might generate inappropriate or harmful content, especially at higher temperatures where the output becomes more random and less controlled. As we saw in our experiments, even with simple Dickens text, higher temperatures led to unpredictable outputs.

From what I understand about responsibility, it's shared between everyone involved. When I built and tested the model with different temperatures, I was responsible for checking that the outputs made sense and weren't harmful. The person who uses the model (like me adjusting temperatures) needs to be careful about what they do with the generated text. It's like citing sources in an essay - we need to be clear about what's AI-generated and what's human-written.

For safety measures, I think we should be extra careful with temperature settings in our model. As we saw, very high temperatures (like 2.0) can create random, potentially meaningless or inappropriate text, while very low temperatures (like 0.1) might get stuck repeating phrases. A moderate temperature around 0.8-1.0 seems safer as it produces more predictable and controlled output. We should also probably check the generated text before using it, especially since our model was trained on old literature that might contain outdated views.

In conclusion, just like I needed to be careful with temperature settings in our experiments, we need to be careful with how we use these models in real applications. It's better to start with conservative settings and gradually adjust them while monitoring the outputs.

