# Neural Graph Composer

This repository hosts a project that aims to explore the idea of representing a music piece 
as a graph neural network and structure and investigate what insights 
we can gain from this approach.

# Overview.

Musical harmony and rhythm involve deep and complex mathematical models, making music generation 
using deep learning techniques an area of interest for the past two decades. We have previously 
experimented with applied transformer and LSTM architectures for music generation tasks. 

Our current proposal takes a novel approach to music generation. If successful, it will 
demonstrate a new method for generative modeling and showcase the ability of Graph Neural 
Networks to extract hidden structures from tasks that may not have an obvious graph structure.

Our inspiration for this work came from Jazz music and the way that Jazz musicians view chord progressions. 
For musicians, chords or chord inversions are seen as progressions from one shape to another, where a 
hand takes on a specific form or shape. Similarly, minor chords form a specific shape on a regular piano. 
If we analyze the music composition process, we can see that it transitions from one shape to another.

What if we reformulate the problem in the following way: each shape of the chord that we represent 
is a sub-graph of a larger graph G, and each note in a chord is a node that is a sub-graph. 
Every transition to the next chord is a message passing operation in the graph. Thus, some chords 
are strongly connected, while others are not. For example, a minor chord is connected to a major 
chord, and other chords are firmly connected to a chord that finishes a musical phrase.

The proposed approach is different from many other music generation techniques as it aims to 
guide a composer during the music creation process. To achieve this, we want to teach an agent 
music phrases or passages. For instance, as a composer, when playing a particular chord, 
the traditional approach involves choosing a scale and a set of chord progressions.

We can see this as Graph Completion task or Generative.

However, each chord has multiple possibilities. Therefore, the idea behind our project is to create 
an agent that learns different chord-to-chord transitions. In our previous experiments, the primary 
goal was to teach the agent long-term dependencies, similar to a sequence-to-sequence generation 
task where a particular note is dependent on the note played before it. To achieve this, we explored 
deep learning techniques such as LSTM, GCN, and GAT for music generation tasks.

## Data representation.

MIDI is a popular file format used to represent digital sheet music and communicate musical performance 
data between electronic musical instruments and computers.  In MIDI, each instrument is represented 
by a sequence of events such as note on/off, pitch, velocity, and time. MIDI files are 
hierarchical in nature, with multiple tracks that can be combined into a single composition.

MIDI consist multiple tracks that can be combined into a single composition. 
A single MIDI file can contain multiple parallel tracks, with each track representing
a separate instrument in the composition. For example, a piano piece might have separate 
tracks for the left and right hands, while an orchestral piece might have separate 
tracks for each instrument in the ensemble. By combining and synchronizing these 
tracks, MIDI files can accurately represent complex musical performances.

Representing music composition as graphs provides several advantages over other formats. 
Graphs are a natural way to represent hierarchical structures, such as the 
multiple tracks in a MIDI file. Nodes in the graph can represent individual notes, and 
edges can capture the relationships between the notes, such as chords and note sequences. 

This graph representation can help capture the underlying structure of the music, 
making it easier to analyze, visualize, and manipulate.

## Data Modeling.

Graph neural networks (GNNs) have shown great promise in the field of machine learning, 
particularly in tasks involving graph-structured data.  By representing MIDI note sequence as graphs, 
we can leverage the power of GNNs to  learn patterns in the music and generate new compositions.

### Motivation

Chords are a fundamental part of music, and they are composed of multiple notes played together.
The notes in a chord create a harmonic relationship that is a crucial aspect of music theory. 
In a musical piece, chords are often used to create a sense of tension and release or 
to provide a stable harmonic foundation.

A graph can be used to encode the relationship between the notes in a chord. Each note can be represented 
as a node in the graph, and the edges between nodes can represent the harmonic relationships 
between the notes. For example, in a major chord, the root note is connected to the third and fifth notes, 
while in a minor chord, the root note is connected to the flattened third and fifth notes.

By representing chords as graphs, we can capture the complex relationships 
between the notes and their harmonies. This can be especially useful in music 
analysis and composition, as it allows us to visualize and manipulate the 
structure of chords and their progressions.

In music theory, the concept of chord progression is central to understanding how 
different chords relate to each other and how they can be used to create harmony. 
Chord progressions are often represented as a series of chords that are connected to 
each other in some way, such as by sharing common notes or by following a certain pattern.

### Relation between nodes

In music theory, a musical phrase is a group of notes that express a complete musical idea, 
often ending in a cadence. A common way to end a musical phrase is to use a chord progression 
that creates a sense of resolution or finality.

For example, the progression from the dominant chord (V) to the tonic chord (I) is a 
common way to end a musical phrase. This progression creates a strong sense of 
resolution because the dominant chord contains a leading tone that wants to
resolve to the tonic chord.

On the other hand, some chords may have weaker connections to other chords in a progression. 
For example, a chromatic passing chord may be used to connect two chords that are not 
closely related harmonically. In this case, the passing chord may have weaker 
connections to the chords that come before and after it, and its weight 
in the chord graph may reflect this.

Overall, using a directed graph with weights allows us to represent the different strengths 
of connections between chords in a musical phrase and can help guide the generation 
of new chord progressions that follow the rules of harmonic progression.

To represent these relationships between chords in a machine learning model,  we can use a directed graph. 
Each node in the graph represents a chord, and the  edges between the nodes represent the relationships 
between the chords. For example,  if chord A can smoothly transition to chord B, we can represent that relationship 
as a directed edge from A to B. We can also assign weights to the edges to represent the strength 
or likelihood of a particular chord transition.

Using a directed graph with weights allows us to model the complex relationships 
between chords and capture the nuances of musical harmony. It also allows us 
to use graph neural networks to analyze and manipulate these relationships, 
enabling us to generate new chord progressions or make recommendations for chord transitions.

## Data representation 

A set of notes form a set and hash of that set is unique hash representation to encode each chord, 
and we never create the same hash twice in the same graph.  Therefore, each chord in one MIDI is 
represented as one node in the graph. The edges between nodes represent the relationships 
between the chords, such as which chords come before or after others, which chords are 
played together, and so on. This allows us to capture the complex relationships 
between chords and create a rich representation of the musical structure of the piece.

## Data processing

* Collect data: First, we would collect MIDI composition containing full music piece. 
  This data would be used to train and evaluate our chord recommendation model.

* Preprocess data: The MIDI data would be preprocessed into a format suitable for our graph-based model. 
This might involve converting the MIDI files to a sequence of chords, then representing 
each chord as a node in a graph  with weighted edges between related chords.

### Midi Reader and Midi Tracks and Sequences

Responsible for converting a MIDI file to an internal representation where each music piece 
is MidiNoteSequences:and then using Graph Builder to process the data can be broken 
down into the following steps:

**midi_sequences.py** and **midi_reader.py** modules for creating and manipulating MIDI files.
midi_reader.py provides a set of functions for reading MIDI files and converting them into a 
data structure that can be used with graph neural networks. Specifically, it uses the pretty_midi 
library to parse the MIDI files and extract the relevant information, such as the time, note, 
and velocity of each event in the MIDI sequence. It then converts this information 
into a dictionary-like structure that represents the MIDI sequence as a graph.

**midi_sequences.py**, on the other hand, provides a set of functions for working with 
the MIDI sequences represented as graphs. For example, it includes a function to 
split a MIDI sequence into overlapping windows, where each window represents a 
subsequence of notes and their associated features. It also includes functions to 
generate batches of training data from the MIDI sequences, which can be used to 
train graph neural networks for music generation.

Together, these modules provide a powerful set of tools for working with MIDI files 
and training graph neural networks for music generation.

MidiReader.read method reads the MIDI file and extracts relevant information, 
such as key signatures, tempo changes, time signatures, notes, pitch bends, 
and control changes. It creates a MidiNoteSequences object, which is an internal 
representation of the MIDI file, with each MidiNoteSequence representing a single instrument.

### Graph Builder Graph Re presentation.

The Graph Builder component is responsible for parsing the MIDI files and constructing a graph 
representation of the music piece. For each instrument, the MIDI file is analyzed and data extract
relevant information such as pitch, velocity, and timing of the notes played. Based on this 
information, the Graph Builder constructs a graph where each node represents a set of notes that
form a chord or just single note, which is a set of notes played together. 
If there are multiple notes played together, then they form a chord.

We use small tolerance to handle imperfection if some note is slitly off.

In addition, the Graph Builder handles the case where one note spans multiple chords. 
For example, if three notes have the same starting point, and two notes have a short duration, 
less while the third note continues, and then two more notes start, the middle note would be present 
in both chord sets. The graph representation allows the model to learn the chord progressions 
and relationships between the chords in the music piece.

Analyze individual instruments: Graph Builder iterates through each instrument in the MIDI
file and examines the notes, pitch bends, and control changes associated with the instrument.
It uses this information to build the graph structure and establish connections between the nodes.

Example how we build a graph.

* C forms an edge to itself with weight 1.
* When the next note played is C, it updates the weight to 2.
* When the next note played is D, D has an edge back to C with edge weight 2.
* So we never create new nodes if the node is already in the graph.
* So if the chord Dm is played 4 times in a row and then Cm

* Why do we believe it is important? Our intuition is as follows:

* If we build a graph as described, it will describe all possible structures for a given music piece.
* Many chords repeat one after another, so we want to capture that.
* The notion of time is a perception of humans, but the underlying structure connects 
* notes and chord is harmony.

Essentially, our representation can reconstruct key musical information even without deep learning. 
For example, if you perform a random walk over the graph that we generate, you will always 
produce some tone or chord. However, it may not necessarily be the correct one but in Key.

### Construct PyG graph: 

Thus, The Graph Builder reads each MIDI file and analyzes the information related to each instrument. 
It processes the data and constructs a graph with nodes and edges representing 
the relationships between the notes and chords and their features. 

The build method of Graph Builder is the main interface class of
MidiDataset(InMemoryDataset) consume. It provides two options for constructing the graph.

The build method of Graph Builder is the main interface class of MidiDataset(InMemoryDataset). 

* It provides two options for constructing the graph:
* Build a graph for each instrument. For example, if we have 2 MIDI files and each contains 2 instruments, 
 we will have 4 graphs in total.

* Map all instruments of a single piece to a single graph. For example, if the piano is playing 
a Dm chord at time T1, then on another track, we might have a bass note.

The midi_dataset.py file defines the MidiDataset class, which is a PyTorch InMemoryDataset 
subclass that is responsible for loading and processing the Procces Data (i,e data in PyG format) 
or convert MIDI files to PyG graph by utilizing Graph Builder.

The midi_graph_builder.py file defines the GraphBuilder class, which is used by the
MidiDataset to construct the  graph representations of the MIDI files.

Graph Builder outputs an iterator that emits PyG data. 

* MidiDataset provides a path to a MIDI file. 
* Graph Builder constructs the graph in the NetworkX format and then converts it to PyG Data. 
* After the graph generation process is complete the Graph Builder converts each graph 
* from NetworkX's Directed Graph representation to PyG. 
* This design choice was made because we wanted to be able to visualize data 
* and convert it to other formats if necessary. NetworkX provides all of these options.

* MidiDataset saves all graphs produced during the processed to two files: 
  * one file contains a graph per instrument. 
  * and the second file that stores each music piece as a single graph.

* MidiDataset also save all mapping hash to index , index to hash etc.

### Data Representation in PyG

During this conversation, all the edge attributes mapped to PyG Data.
The Graph Builder assigns a unique hash value to each chord node in the graph. 
This is done by computing a hash function on the set of pitches that make up the chord. 
The resulting hash value is then attached to the node as an attribute.

The set of pitches [60, 61, 62, ...] forms a feature vector, and a hash of each 
set is used as the label and y value for a corresponding node in the graph. 
Therefore, the number of unique hash values corresponds to the number of 
nodes in the graph. Once the hash values are computed and attached to the nodes, 
the Graph Builder maps each hash to a label and y value in the PyG dataset.

In the PyG Data object, y and label is represented as indices. 

* Note that we construct a dictionary that maps each set (using frozenset) to its respective hash.
* In parallel, we map each hash to its corresponding set, allowing us to recover the vector that represents a 
* list of pitch values for a given node. (By default, we group up to 12 notes into the same node, 
  i.e., simultaneously playing notes.)
* It's important to note that a set is unique and permutation invariant. That is, a chord like C E F  
  and F E C represents the same chord.
* Therefore, each Data.x contains a list of pitch numbers up to 12 pitch values, with each 
  preserving the octave information.
* It's also important to note that the same pitch set in a different octave is a different 
  set and has a separate hash. For example, C0 E0 F0 in octave 0 and C0 E1 F0 are two different sets.

When we convert Graph to PyG.   When converting a graph to PyG, a shared global allocator is used 
to allocate indices for each Data.label, which is then used as the label for the node in the PyG dataset.
This ensures that the number of classes is shared across all graphs in the dataset.

Indices used for  label and y is then used as the label for the node in the PyG dataset.

We also support where the node labels and y converted to one-hot encoding and stored 
in the y attribute of the PyG dataset. 

In summary 
 * Data.x is a feature vector representing a list of pitch values, (optionally velocity) either padded or using one-hot encoding.
 * Data.label contains indices for a given class (i.e., hash).
 * Data.y contains indices for true hash.
 * Data.edge_attr contains weights between nodes.
 
* Mappings.
  * index_to_hash maps indices to hash values.
  * hash_to_notes maps hash values back to a list of pitch values, which is important for generation.

  * i.e During graph decoding, the embedded representation for each node is decoded and its corresponding 
  * index in the PyG dataset is retrieved. The shared global allocator is then used to recover the 
  * original pitch value from the index. Finally, a list of pitch values that correspond to a 
  * node is constructed from the recovered pitch value.

# Mappings. 

The graphs() method, is the generator and creates an index for each unique hash value by checking
if the hash is not already present in the hash_to_index dictionary. Next, the generator iterates 
through the PyGData objects, maps the node_hash of each PyGData object to their corresponding 
index in the hash_to_index dictionary and sets the label and y values for each node.
Finally, the generator yields the PyGData object with updated label and y values.

The label and y values are created by mapping each node_hash of the PyGData object to
their corresponding index in the hash_to_index dictionary. The label tensor represents
the unique index for each node_hash, whereas the y tensor represents the unique index 
for each target node_hash. This mapping is essential for the Graph Neural 
Network to learn the relationships between nodes in the graph.

### Train the model: 

We evaluate different ideas related to  predication task , 
We would train the model using the preprocessed MIDI data. 
This involves optimizing the model's weights and biases to minimize the loss 
function, which measures the difference between the predicted and actual chord sequences.

Evaluate the model: Once the model is trained, we would evaluate its performance by 
testing it on a held-out dataset of MIDI files. This allows us to see how well the 
model can generalize to new examples of chord progressions.

Generate chord recommendations: Once the model is trained and evaluated, we can use it to 
generate chord recommendations for a given sub-graph of chords. This involves inputting 
the sub-graph into the model and using its learned weights and biases to predict the 
most likely next chord(s) in the progression.

Refine the recommendations: Finally, we might refine the chord recommendations based 
on additional criteria, such as musical rules or user preferences. For example, 
we might ensure that the recommended chord fits within a particular key or 
has a certain harmonic function within the progression. This can help to 
ensure that the recommendations are musically coherent and pleasing to the listener.

The process of converting a MIDI file to an internal representation and then using Graph Builder 
to process the data can be broken down into the following steps:

Convert MIDI to internal representation: The MidiReader.read method reads the MIDI file 
and extracts relevant information, such as key signatures, tempo changes, time signatures, 
notes, pitch bends, and control changes. It creates a MidiNoteSequences object, which is an 
internal representation of the MIDI file, with each MidiNoteSequence representing a single instrument.

Graph Builder processing: Graph Builder reads each MIDI file and analyzes the information 
related to each instrument. It processes the data and builds a graph where each node represents 
a set of notes played together, forming a chord.

Analyze individual instruments: Graph Builder iterates through each instrument in the MIDI
file and examines the notes, pitch bends, and control changes associated with the instrument.
It uses this information to build the graph structure and establish connections between the nodes.

Construct graph: As Graph Builder processes the notes and other data in the MIDI file, 
it constructs a graph with nodes and edges representing the relationships between the notes and
their features. Each node in the graph is a set of notes that form a chord, and the edges indicate 
how the chords are connected in the music piece.

Generate feature vectors: Graph Builder generates feature vectors for each node in the graph.
These feature vectors can be used as input to various machine learning models, 
such as Graph Autoencoders (GAE), which can then generate embeddings or perform other 
tasks like music generation, classification, or analysis.

Graph Autoencoder (GAE) processing: GAE takes the graph structure generated from the 
MIDI dataset and processes it to generate embeddings or perform other tasks like music generation 
or classification. The GAE model learns the relationships and patterns in the graph structure,
capturing the essence of the musical piece and its structure.

In summary, the process starts with reading MIDI files and extracting relevant information. 
Graph Builder processes the data and constructs a graph representation, which is then used by models 
like Graph Autoencoders for further analysis, generation, or classification tasks.

In the context of the model you have shown, the Graph Autoencoder (GAE) works with the
MIDI dataset and GraphBuilder to learn patterns and relationships in the musical data. 
Here's a high-level overview of the process:

MIDI to internal representation: The MidiReader class reads the MIDI files and converts 
them into an internal representation (MidiNoteSequences). It extracts information about tempo, 
key signature, time signature, and notes (including pitch, start time, end time, and other properties).

Constructing the graph: GraphBuilder takes the MidiNoteSequences and constructs a graph where each 
node represents a set of notes that form a chord. Edges are created based on the relationships between 
these chords, such as their temporal proximity or harmonic similarity.

Node features: In this case, the features associated with each node are the properties of the 
notes that form a chord. These features can include pitch, velocity, duration, and other note attributes.

Graph Autoencoder (GAE): The GAE model takes the graph created by GraphBuilder as input and 
learns a latent representation of the nodes using neighborhood aggregation techniques. 
The GAE consists of an encoder that generates a low-dimensional embedding for each node, 
and a decoder that reconstructs the graph structure based on these embeddings. The model is
trained by minimizing the reconstruction loss between the original graph and the reconstructed one.

Learning patterns and relationships: As the GAE processes the graph, it learns the underlying 
patterns and relationships between the chords (nodes) in the MIDI dataset. By encoding the 
features of each node and its neighbors, the GAE can discover higher-order relationships and 
structures, such as chord progressions, harmonies, and other musical patterns.

Overall, the GAE model, in combination with the GraphBuilder and MidiReader, 
works to extract meaningful information and relationships from the MIDI dataset 
by processing the graph representation of the musical compositions.


Regarding the permutation invariance of a group of notes, this property means that the order 
of the notes in the group does not matter when considering them as a single entity (e.g., a chord). 
In other words, a set of notes {A, B, C} is considered the same as {B, A, C} or {C, B, A}. 
This is because chords are defined by the combination of notes played together, rather than the 
specific order in which they are played.

Permutation invariance is a useful property for the MIDI dataset and the graph-based neural network model. 
It ensures that the model treats different permutations of the same set of notes as equivalent, 
which simplifies the graph representation and reduces the complexity of the problem. Additionally,
it allows for more efficient processing and generalization, as the model does not need to learn 
different representations for different permutations of the same group of notes.

Example:

Notes in the chord: (60, 0.5), (64, 0.5), (67, 0.5)
Concatenate notes: "60_0.5_64_0.5_67_0.5"
Hash value (using MD5): "c7eef5e51fbd34e21335ff93c0db87cd"
Mapping: To efficiently represent chords as indices in the graph, 
you can create two dictionaries:

notes_to_hash: Maps the chord's notes (as a frozenset) to the unique hash value.
hash_to_notes: Maps the unique hash value back to the chord's notes (as a frozenset).
This allows you to convert chords to hash values and vice versa efficiently.
Feature vector: To use chords as input to a graph-based neural network (like GAE), you need to convert
each chord into a feature vector. One approach is to use a binary representation where the feature 
vector's length is equal to the maximum MIDI note number (128 for standard MIDI) and each element 
corresponds to a specific MIDI note number. If a note is present in the chord, its corresponding element 
in the feature vector is set to 1; otherwise, it is set to 0.

Example:

Chord: (60, 0.5), (64, 0.5), (67, 0.5)
Feature vector (length 128): [0, 0, ..., 1, ..., 1, ..., 1, ...]
(1s at indices 60, 64, and 67, and 0s elsewhere)
By following these steps, you can represent a set of notes as a hash and a feature vector, 
which can be used as input to a graph-based neural network. The mappings (notes_to_hash and hash_to_notes) 
allow you to convert between the chord's notes and the hash value efficiently.


### Graph Geneartion.

the second model is called GraphGenerationModel. It is a neural network that is used to generate new nodes and edges for the 
graph of chords in the MIDI dataset.

The model takes the MIDI dataset and processes it using GraphBuilder to create a graph representation. 
The graph is then used as input to the model, which learns to generate new nodes and edges that fit with 
the existing graph structure.

The model has several hyperparameters, including the number of epochs, batch size, learning rates, and model type.
It uses a Graph Convolutional Network (GCN) or Graph Attention Network (GAT) to learn node embeddings and a 
Long Short-Term Memory (LSTM) network to generate new nodes and edges based on the learned embeddings.

The GCN or GAT takes the graph structure and node features as input and learns to generate a low-dimensional
embedding for each node in the graph. The LSTM takes the sequence of node embeddings generated by the 
GCN or GAT and uses it to generate new nodes and edges that fit with the existing graph structure.

The model is trained by minimizing the reconstruction loss between the original graph and the reconstructed one. 
After training, the model can generate new nodes and edges that fit with the existing graph structure, 
allowing for music generation and other tasks.


Great! So this model is a Graph Generation Model that generates new graphs using MIDI files as input.
It uses the MidiDataset class that is responsible for creating a dataset from a list of MIDI files. 
The MidiDataset class consumes the GraphBuilder class, which is responsible for creating graphs 
from MIDI files. It also has several optional parameters, such as whether to treat each instrument 
as a separate graph, whether to split the dataset into graphs, and the default node attribute name.

The GraphGenerationModel class initializes several parameters, such as the epochs, batch size, 
embeddings and LSTM learning rates, the model type, and the GCN hidden dimension. It also 
creates a GCN3 or GAT model based on the model type, an instance of the Decoder class, an
instance of the GraphLSTM class, and optimizers for the GCN and LSTM models.

The GCN3 and GAT models are used for node classification tasks and consist of several 
Graph Convolutional Network (GCN) or Graph Attention Network (GAT) layers that learn 
embeddings for each node in the graph. The Decoder class is responsible for 
decoding the LSTM output into a graph structure, while the GraphLSTM class is 
responsible for generating the LSTM output based on the input graph.


this model is a Graph Generation Model that generates new graphs using MIDI files as input. 
It uses the MidiDataset class that is responsible for creating a dataset from a list of MIDI files.
The MidiDataset class consumes the GraphBuilder class, which is responsible for creating graphs from MIDI files.
It also has several optional parameters, such as whether to treat each instrument as a separate graph, whether to
split the dataset into graphs, and the default node attribute name.

The GraphGenerationModel class initializes several parameters, such as the epochs, batch size, embeddings 
and LSTM learning rates, the model type, and the GCN hidden dimension. It also creates a GCN3 or GAT model 
based on the model type, an instance of the Decoder class, an instance of the GraphLSTM class, and optimizers 
for the GCN and LSTM models.

The GCN3 and GAT models are used for node classification tasks and consist of several Graph Convolutional 
Network (GCN) or Graph Attention Network (GAT) layers that learn embeddings for each node in the graph. 
The Decoder class is responsible for decoding the LSTM output into a graph structure, while the GraphLSTM 
class is responsible for generating the LSTM output based on the input graph.




example03a_lstm.py is an example script that demonstrates how to use a graph LSTM to generate music. 
The script first loads a dataset of MIDI files and then creates a GraphGenerationModel instance, which is responsible for training and generating new music.

The GraphGenerationModel class uses a graph LSTM to generate new music. The graph LSTM takes as input a graph represented as a PyTorch Geometric data object, which consists of a graph structure and node and edge features. The graph structure is represented by the adjacency matrix and the node features represent the properties of each node in the graph.

The graph LSTM processes the input graph sequentially, updating the hidden state of each node based on its current state and the states of its neighbors. At each timestep, the graph LSTM generates a new output vector, which is then used to predict the class of the next note in the generated music sequence.

During training, the GraphGenerationModel class trains the graph LSTM using a cross-entropy loss function. The model is trained using backpropagation through time (BPTT), which involves computing gradients for all timesteps in the input sequence and updating the model parameters accordingly.

After training, the GraphGenerationModel class can be used to generate new music sequences. To generate new music, the class takes as input a seed graph, which is used to initialize the hidden state of the graph LSTM. The class then iteratively generates new notes by updating the hidden state of the graph LSTM and using the output vector to predict the class of the next note. This process continues until a predetermined length of the generated music sequence is reached.

Overall, example03a_lstm.py demonstrates how graph neural networks can be used for music generation tasks, specifically using a graph LSTM to generate new music sequences.


## Inductive.

In the inductive setting, the model is trained on a fixed graph and then applied to new graphs 
with the same node and edge features. The node and edge features in the new graph can 
be completely different from the original graph, but the structure of the graph remains the same.

Example
Music genre classification:
  Given a MIDI file, we can use the graph neural network to classify the genre 
  of the music piece based on the patterns of notes and chords used in the file.

Melody extraction: We can use the graph neural network to extract 
                   the melody from a given MIDI file. The model can learn to differentiate between 
                   the melody and the accompanying chords or other instruments.

Music recommendation: Using the graph neural network, we can learn the preferences of a 
                      listener and recommend similar pieces of music based on the patterns 
                       of notes and chords in the MIDI files.

Music transcription: We can use the graph neural network to transcribe a MIDI file to sheet music. 
                      The model can learn to recognize the patterns of notes and chords used in the music 
                      and translate them to the appropriate symbols on the sheet music.

Transductive 

In contrast, in the transductive setting, the model is trained on a graph and then  applied to new nodes 
or edges that were not present in the training graph.  This means that the node and edge features 
in the new graph can be different, and the structure of the new graph can also be different from the original graph.

# Graph Completion and Generation

* We consider two main task.
  * We are given a graph in which each node represents a set of musical notes or chords, and we want a model 
    that can suggest a set of chords or notes that can be played based on the input graph
  
  * We are given initial input and model generate set of notes.

The the graph network is trained in the inductive setting because it is trained on 
a fixed graph structure, and the same graph structure is used to generate new nodes and edges.
However, the LSTM component of the model is trained in the transductive setting because it generates 
new nodes and edges that were not present in the training graph.

# Model Arhitecture: Generative LSTM Based model 

The example uses a LSTM-based model to generate new sub-graph sequences of chords (nodes) 
based on a given input sequence.  The model is trained to predict the next sequence of 
chords given the previous sequence, which can be used to generate new sequences of chords.

Given start node or sub-graph. We first compute use Dijkstra's algorithm to compute 
the shortest path from a starting node to all other nodes in the input graph, 
(note at moment we don't bound but idea use weight directed graph) and this produces 
a node sequence and edge sequence. We then feed these sequences to a GCN Encoder layer 
that is trained using a shared Decoder with an LSTM. The idea here is to train the 
GCN Encoder to produce embeddings that we can use to train the LSTM for sequence-to-sequence generation.

The GCN layer takes the input feature matrix (node features) and the edge indices of the input graph as inputs, 
and applies a series of graph convolution operations to produce the node embeddings.  

GCN model is trained to produce node embeddings that can be used for downstream tasks i.e.
as node classification or sequence generation.

The decoder layer is a feedforward neural network that is trained to decode the 
node embeddings produced by the GCN model into the output sequence. 

In this implementation, the decoder layer is used in conjunction with an LSTM-based model 
i.e. Shared decoder, to generate new sub-graph sequences based on a given input sequence. 

The LSTM model takes the node embeddings produced by the GCN layer as inputs, 
and generates a new sequence of node embeddings that can be decoded by the decoder layer 
to produce the output sequence.  The LSTM model is trained using teacher forcing, 
where the input sequence fed to the model at each time step is the ground truth output sequence
from the previous time step.

### LSTM Layer.

The LSTM-based model consists of an input layer, an LSTM layer, and an output layer. 
The input layer takes the graph representation of the MIDI as input and passes 
it through an embedding layer to obtain a fixed-size representation of the graph. 
This representation is then passed through the LSTM layer, which generates a sequence 
of hidden states. Finally, the output layer maps each hidden state to a probability 
distribution over the possible notes. The LSTM takes a sequence of node embeddings 
as input and produces a sequence of output embeddings, one for each time step in
the input sequence. These output embeddings can be used to generate new sub-graph sequences. 
The LSTM output embeddings are then passed through a linear layer (shared decoder) 
to produce the final node embeddings.

The model is trained using a combination of binary cross-entropy loss and KL-divergence loss.
The binary cross-entropy loss is used to train the model to predict the presence or absence 
of a note at each time step, while the KL-divergence loss is used to encourage the model
to generate sequences that are similar to the training data.

After training, the model is used to generate new music sequences by feeding it with an 
initial graph representation of a MIDI file and iteratively generating new notes until 
the desired length is reached. The generated sequences can then be converted 
back into MIDI files using a MidiConverter class.

Second variation.

In the context of GCN, we can use the second variation of combining GCN and LSTM 
and optimizing both loss terms to generate new graphs that are similar to the target 
graphs in the training data. By minimizing the joint loss that combines the GCN and 
LSTM loss terms, we encourage the model to learn node embeddings that capture the 
underlying structure and patterns in the training data. This can result in generated 
graphs that have similar node embeddings as the target graphs in the training data, and 
can be used to generate new graphs that follow similar patterns and styles 
as the training data.

The model architecture consists of two main components: a Graph Convolutional Network (GCN) 
and a Long Short-Term Memory (LSTM) network.

The GCN takes as input a graph represented as a sparse adjacency matrix and learns to generate 
node embeddings, which represent the structural features of the nodes in the graph. 
The GCN uses a series of graph convolutional layers to aggregate information from the 
neighboring nodes in the graph and update the node embeddings. The final node embeddings 
are concatenated and fed into the LSTM network.

The LSTM network takes the concatenated node embeddings as input and learns to generate a 
sequence of tokens that represent the music. The LSTM network is trained using teacher forcing,
which means that during training, the input to the LSTM at each time step is the actual token from
the ground-truth sequence at the previous time step. During inference, the generated token from the 
previous time step is used as the input to the LSTM at the current time step.

The model is trained using two loss terms: the GCN loss and the LSTM loss. The GCN loss measures the
difference between the generated node embeddings and the ground-truth node embeddings, 
while the LSTM loss measures the difference between the generated music sequence a
nd the ground-truth music sequence. The final loss is a weighted sum of the two 
losses and is used to update the model parameters using backpropagation.


## Graph Autoencoder

##
Another model we evaluate on our representation modified GAE.

GAE Graph Autoencoder is another type of graph neural network model that also aims to learn low-dimensional 
embeddings of nodes in a graph. Like the model we discussed earlier, GAE also uses an autoencoder architecture.
However, there are some differences between the two models. 

GAE uses a matrix factorization technique to learn the graph embeddings, whereas the model we 
described earlier uses a combination of graph convolutional networks and LSTMs. 
Additionally, GAE is designed specifically for unsupervised learning, whereas the model we discussed can 
be used for both supervised and unsupervised learning tasks.
