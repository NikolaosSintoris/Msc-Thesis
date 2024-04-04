Nowadays, many malicious activities on the Internet, e.g. carrying out DDoS attacks, are based on botnets, i.e. networks of computers (bots) infected with malware. The execution of the malicious actions is based on the commands that the bots receive from the Command Control (C&C) servers using various techniques. Domain Generation Algorithms (DGA's) are widely used by bots to identify their C&C servers.

Bots use DGAs to locate C&C servers, whose IP addresses can change frequently. With this technique, malicious traffic can bypass domain name blacklists, while locating C&C and cutting it off from its bots becomes an extremely demanding process.

It has been observed that the format of domain names produced by DGAs usually differs to some extent from that of legit names. Taking advantage of this information, many researchers have turned to the development of classifiers with Deep Learning methods, with the ultimate goal of detecting algorithmically generated domain names.

The purpose of this thesis is the study and implementation of collaborative machine learning models to identify malicious names without sharing data, and the interpretation of these models with eXplainable Artificial Intelligence (XAI) algorithms.

To implement this plan, we use a collaborative learning environment based on the modern architecture of Federated Learning. When training with Federated Learning, the only data exchanged are the local models that are trained on each client and then sent to a central server to be aggregated into a new global model. Training data remains protected on local devices throughout training.

Although machine learning and deep learning methods are becoming increasingly popular in tackling this problem and show excellent accuracy, they remain difficult for researchers to understand how their decisions and predictions are made. Aiming to solve the above problem, we present deep learning models to illustrate and interpret the features that determined the categorization of domain names as benign or malicious.

With tests on popular datasets, the goal is to use XAI algorithms to interpret classification decisions in the individual models of the entities participating in Federated Learning and in the aggregated models that result after aggregating the individual data.
