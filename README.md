Distillation Training Pipeline to Avoid Backdoor Attacks

# Purpose
Final project for ECE572 - Secure Machine Learning Design and Applications

# Motivation
The motivation for this project is as follows: Knowledge distillation is most useful when training smaller models on larger datasets. As mentioned previously, this often underfits the smaller models. In parallel, small models are often deployed on edge, where the presence of a trigger could be most dangerous. If a teacher model upstream can be confirmed to be safe and under the condition that the performance of the model is sufficient on the secondary data to train the student model then the logits can be trusted to train the edge model. Together edge models can be trained to a high level of performance and without concerns of a trigger. Trusted models can be common open source models confirmed to be the same as source or a previously trained model that is sufficiently field tested. It is important to note that data can be corrupted in transit by many types of communication attack such as a man in the middle attack to inject a trigger.

# Results
Given a teacher model of sufficent performance of clean accuarcy and low trigger rate, the pipeline is successful. Future improvements can look to work with 'DLP: towards active defense against backdoor attacks with decoupled learning process' by Z.Ying to apply the distillation from the unlearned trigger step to realign the weights from any residue leftover after the trigger is unlearned. This will prevent the requirement that a teacher model from a clean dataset is needed before training