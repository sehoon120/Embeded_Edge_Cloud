# Server Open Code
- servertest.cmd
- servertest.py
---
# Addition of Embeddings on Model
- han_embedding.npy
---
# Server connect + Model Inference
- one_detect_server.py
- two_detect_server.py

# Client code with using SNN membrane potential method
In Spiking Neural Network, Membrane potential and spike are used to lighten the Neural Network.
On the basis of this idea, we apply Spike system on our Raspberry Pi system.

We process image only in spike occuration time, and server connection also only occurs in spike occuration time.
This will make a lot of power reduction and light progress of Raspberry pi system.
Also, Spike condition can be manually modified by handling hyperparameters.

There are two versions
- spike_client.py
- spike_server_client.py

spike_server_client.py file is one of our final goal in spike client code which is connecting to server only in spike time.
