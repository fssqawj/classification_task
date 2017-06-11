# KSEM Related Work

Bib: MLA

## EMNLP15-Hua He-Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks

Bib: He, Hua, Kevin Gimpel, and Jimmy J. Lin. "Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks." EMNLP. 2015.


He et al. (2015) adopt _ to address  "Modeling sentence similarity is complicated by the ambiguity and variability of linguistic expression", use CNN featuring convolution filters with multiple granurities and window sizes. followed by multiple types of pooling. Multiple distance function: cosine distance, euclidean distance and element wise distance.

> Studies that leverage CNNs for sentence pair modeling include Hu et al. (2014), Yin and Schutze (2015), He et al. (2015), etc.

Next

> For TE, Bowman et al. (2015b) use recursive neural networks to encode entailment on SICK (Marelli et al., 2014b). Rocktaschel et al. (2016) present an attention-based LSTM for the Stanford natural language inference corpus (Bowman et al., 2015a). Our system is the first CNN-based work on TE.


> Some prior work aims to solve a general sentence matching problem. Hu et al. (2014) present two CNN architectures, ARC-I and ARC-II, for sentence matching. ARC-I focuses on sentence representation learning while ARC-II focuses on matching features on phrase level. Both systems were tested on PI, sentence completion (SC) and tweetresponse matching. Yin and Schutze (2015b) propose the MultiGranCNN architecture to model general sentence matching based on phrase matching on multiple levels of granularity and get promising results for PI and SC. Wan et al. (2015) try to match two sentences in AS and SC by multiple sentence representations, each coming from the local representations of two LSTMs. Our work is the first one to investigate attention for the general sentence matching task.