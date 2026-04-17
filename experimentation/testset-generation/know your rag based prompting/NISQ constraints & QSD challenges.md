This contain "NISQ constraints & QSD challenges" subdomain question preparation

# Theme extraction:

## Theme text:
```
## 3 The potential of quantum computing 

## 3.1 Why we think quantum computing is powerful 

An ordinary computer processes bits, where each bit could be, say, a switch which is either one or off. But to build highly complex entangled quantum systems, the fundamental information-carrying components of a quantum computer must be quantum bits, what we call _qubits_ . A qubit can be realized physically in many different ways. It can be carried by a single atom, or a single electron, or a single photon (a particle of light). Or a qubit can be carried by a more complicated system, like a very cold superconducting electrical circuit in which many electrons are moving. 

When we speak of quantum complexity, what springs to mind is the staggering complexity of using ordinary classical data to describe highly entangled quantum states of many qubits. Giving a complete description of all the correlations among just a few hundred qubits may require more bits than the number of atoms in the visible universe. It will never be possible, even in principle, to write that description down, or to faithfully describe the processing of a few hundred qubits using classical language. 

This apparently extravagant complexity of the quantum world, though highly suggestive, does not by itself ensure that quantum computers are more powerful than classical ones. But we have at least three good reasons for thinking that quantum computers have capabilities surpassing what classical computers can do. 

- (1) _Quantum algorithms for classically intractable problems._ First, we know of problems that are believed to be hard for classical computers, but for which quantum algorithms have been discovered that could solve these problems easily. The best known example is the problem of finding the prime factors of a large composite integer. We believe factoring is hard because many smart people have tried for many decades to find better factoring algorithms and haven’t succeeded. Perhaps a fast classical factoring algorithm will be discovered in the future, but that would be a big surprise. 

- (2) _Complexity theory arguments._ The theoretical computer scientists have provided arguments, based on complexity theory, showing (under reasonable assumptions) that quantum states which are easy to prepare with a quantum computer have superclassical properties; specifically, if we measure all the qubits in such a state we are sampling from a correlated probability distribution that can’t be sampled from by any efficient classical means. 

- (3) _No known classical algorithm can simulate a quantum computer._ But perhaps the most persuasive argument we have that quantum computing is powerful is simply that we don’t know how to simulate a quantum computer using a digital computer; that remains true even after many decades of effort by physicists to find better ways to simulate quantum systems. 

It’s a remarkable claim — one of the most amazing ideas I’ve encountered in my scientific life — that there is a distinction between problems that are classically hard and problems that are quantumly hard. And it is a compelling challenge to understand better what problems are classically hard but quantumly easy. We should recognize in particular that the power of a quantum computer is not unlimited. We don’t expect, for example, that a quantum computer will be able to solve efficiently the hard instances of NPhard problems like the traveling salesman problem. For such hard combinatorial search problems we probably can’t do much better than exhaustively searching for a solution. Quantum computers can speed up exhaustive search, but only modestly, so NPhard problems are likely to be quantumly hard as well as classically hard. 

For a physicist seeking problems which are classically hard and quantumly easy, the natural place to look is the task of simulating a many-particle quantum system. As two great physicists, Bob Laughlin and David Pines, put it some years ago, we have a “theory of everything that is relevant to ordinary life.” We have high confidence this theory is correct, and we can write down the equations precisely — they are the equations that describe how atomic nuclei and electrons interact electromagnetically. But we can’t solve those equations. And so as Laughlin and Pines put it: “We have a theory of everything only to discover that it has revealed exactly nothing about many things of great importance.” Those things of importance they envisaged are the situations in the quantum world where entanglement has profound consequences. Dramatizing the futility of the task they proclaimed: “No computer existing, or that will ever exist, can break this barrier” of solving the equations describing many entangled particles. 

But in fact, years before Laughlin and Pines wrote these words, the physicist Richard Feynman had articulated a rebuttal. As Feynman put it: “Nature isn’t classical dammit, and if you want to make a simulation of Nature you better make it quantum mechanical, and by golly it’s a wonderful problem because it doesn’t look so easy.” Feynman had envisioned using a quantum computer to solve the quantum physics problems that physicists and chemists had failed to solve using digital computers. Laughlin and Pines knew well that Feynman had made this proposal years earlier, but had dismissed his idea as impractical. Now, some 35 years after Feynman’s proposal, we’re just beginning to reach the stage where quantum computers can provide useful solutions to hard quantum problems. 

## 3.2 Why quantum computing is hard 

So why is it taking so long? What is it about quantum computing that’s so difficult? The core of the problem stems from a fundamental feature of the quantum world — that we cannot observe a quantum system without producing an uncontrollable disturbance in the system. That means that if we want to use a quantum system to store and reliably process information, then we need to keep that system nearly perfectly isolated from the outside world. At the same time, though, we want the qubits to strongly interact with one another so we can process the information; we also need to be able to control the system from the outside, and eventually read out the qubits so we can find the result of our computation. It is _very_ challenging to build a quantum system that satisfies all of these desiderata. It has taken many years of development in materials and control and fabrication to get where we are now. 

Eventually we expect to be able to protect quantum systems and scale up quantum computers using the principle of quantum error correction. The essential idea of quantum error correction is that if we want to protect a quantum system from damage then we should encode it in a very highly entangled state; like that 100-page book I described earlier, this entangled state has the property that the environment, interacting with parts of the system one at a time, is unable to glimpse the encoded information and therefore can’t damage it. Furthermore, we’ve understood in principle how to process quantum information which is encoded in a highly entangled state. Unfortunately, there is a significant overhead cost for doing quantum error correction — writing the protected quantum information into a highly entangled book requires many additional physical qubits — so reliable quantum computers using quantum error correction are not likely to be available very soon. 


## 4 The NISQ era unfolds 

## 4.1 The 50-qubit barrier 

Even with fault-tolerant quantum computing still a rather distant dream, we are now entering a pivotal new era in quantum technology. For this talk, I needed a name to describe this impending new era, so I made up a word: _NISQ_ . This stands for _Noisy IntermediateScale Quantum_ . Here “intermediate scale” refers to the size of quantum computers which will be available in the next few years, with a number of qubits ranging from 50 to a few hundred.[1] 50 qubits is a significant milestone, because that’s beyond what can be simulated by brute force using the most powerful existing digital supercomputers.[2] “Noisy” emphasizes that we’ll have imperfect control over those qubits; the noise will place serious limitations on what quantum devices can achieve in the near term. 

Physicists are excited about this NISQ technology, which gives us new tools for exploring the physics of many entangled particles. It might also have useful applications of interest to the business community, but we’re not sure about that. We shouldn’t expect NISQ is to change the world by itself; instead it should be regarded as a step toward more powerful quantum technologies we’ll develop in the future. I do think that quantum computers will have transformative effects on society eventually, but these may still be decades away. We’re just not sure how long it’s going to take. 

## 4.2 Qubit “quality” 

I’ve emphasized the number of qubits as a measure of how difficult it is to do the simulation of a quantum computer on a classical device, but the number of qubits isn’t the only thing we care about. We also care about the _quality_ of the qubits, and in particular the accuracy with which we can perform quantum gates — well-controlled entangling operations acting on pairs of qubits. With the best hardware we have now for controlling trapped ions [14] or superconducting circuits [15], the error rate per gate for two-qubit gates is above the _._ 1% level (and often much worse). Furthermore, we don’t yet know whether error rates that low can be maintained in larger devices with many qubits; perhaps we’ll find out soon. Naively, then, and as I’ll say later this might be too naive, with these noisy devices we don’t expect to be able to execute a circuit that contains many more than about 1000 gates — that is, 1000 fundamental two-qubit operations — because the noise will overwhelm the signal in a circuit much larger than that. That limitation on circuit size imposes a ceiling on the computational power of NISQ technology. Eventually we’ll do better, using quantum error correction to scale up to larger circuits. But as I’ve already emphasized, since quantum error correction imposes a heavy overhead cost in number of qubits and number of gates, scaling up using quantum error correction is a more distant goal. When I speak of the NISQ era, I’m imagining quantum computers with noisy gates unprotected by quantum error correction. 

There are other things we care about, too, aside from the number of qubits and the gate error rate. The time it takes to execute a single gate is also important for setting the time scale needed for a quantum computer to solve a problem, and it is noteworthy that superconducting circuits are about a thousand times faster than ion trap quantum processors. We need to be able to prepare and measure qubits accurately; currently the measurement error probability is about 1% for superconducting qubits, and much better than that for trapped ions. We care about the connectivity among the qubits — for which pairs of qubits in our device can we perform an accurate two-qubit gate? It is also important to know how reliably we can fabricate qubits — if we attempt to construct a many-qubit device, how many of the qubits actually perform well enough to be useful? These considerations and others, too, should be kept in mind when we make comparisons across different quantum computing platforms.
```

## Theme Prompt:
```
In a few words, extract the main theme behind the following passage: [[{ context}]]
```

## Theme Prompt Output:
Model used: Gemini 3.1 Pro
```
The main theme is the immense potential, practical challenges, and current transitional "NISQ" era of quantum computing.
```

# Statement Extraction

## Factual statements

### Factual statements prompt

```
Extract at most five factual statements based on the following passage and its theme. You need to strictly comply with the following guidelines: 

- Each statement must contain a single unit of factual information.  
- Each statement must be written in the style of an answer to a factual question. 
- Each statement must be understandable without the aid of any other source of information. 
- Each statement must include contextual information derived from the passage theme. 
- Each statement must only contain information that exists in the original passage and theme. 
- Each statement must be independent from the other statements.  

Generate the statements as a bullet list with the following format: 
> Statement 
> Statement etc  

Theme: [[The main theme is the immense potential, practical challenges, and current transitional "NISQ" era of quantum computing.]] 
Passage: [[{context}]]
```

### Factual Statement prompt Output
Model used: Gemini 3.1 Pro
```
Quantum bits, or qubits, are the fundamental information-carrying components of a quantum computer and can be physically realized through entities such as single atoms, electrons, photons, or superconducting electrical circuits.

The current "NISQ" era of quantum computing is defined by Noisy Intermediate-Scale Quantum devices that possess between 50 and a few hundred qubits but lack the protection of quantum error correction.

Quantum computers are expected to solve certain classically intractable problems, such as finding the prime factors of large composite integers, more efficiently than any known classical algorithm.

A significant practical challenge in quantum computing is the requirement to keep qubits nearly perfectly isolated from the outside world to prevent uncontrollable disturbances while still allowing them to interact for information processing.

Quantum computers are not expected to efficiently solve all difficult problems, as hard combinatorial search tasks like the traveling salesman problem are likely to remain quantumly hard despite the potential for modest speedups.
```

## Summary statements

### Summary statements prompt
```
Merge the following sentences into three summary statements. 
Each summary statement must summarise information contained in more than one sentence. 
Each summary statement must be independent and non-overlapping. 
Each summary statement should be a complete sentence. 
Each summary statement can include contextual information contained in the theme below. 
Each summary statement must be understandable without the aid of any other source of information.  

Generate the statements as a bullet list with the following format: 
> Summary statement 
> Summary statement 
> Summary statement  

Theme: [[The main theme is the immense potential, practical challenges, and current transitional "NISQ" era of quantum computing.]]

Sentences:[[ Quantum bits, or qubits, are the fundamental information-carrying components of a quantum computer and can be physically realized through entities such as single atoms, electrons, photons, or superconducting electrical circuits.

The current "NISQ" era of quantum computing is defined by Noisy Intermediate-Scale Quantum devices that possess between 50 and a few hundred qubits but lack the protection of quantum error correction.

Quantum computers are expected to solve certain classically intractable problems, such as finding the prime factors of large composite integers, more efficiently than any known classical algorithm.

A significant practical challenge in quantum computing is the requirement to keep qubits nearly perfectly isolated from the outside world to prevent uncontrollable disturbances while still allowing them to interact for information processing.

Quantum computers are not expected to efficiently solve all difficult problems, as hard combinatorial search tasks like the traveling salesman problem are likely to remain quantumly hard despite the potential for modest speedups. ]]
```

### Summary statements prompt Output
Model used: Gemini 3.1 Pro
```
The current transitional "NISQ" era of quantum computing is defined by devices that utilize between 50 and a few hundred physically realized qubits—such as single atoms, electrons, photons, or superconducting circuits—as their fundamental information-carrying components.
A major practical challenge in quantum computing is the current absence of quantum error correction, requiring the nearly perfect isolation of these sensitive systems from uncontrollable outside disturbances without preventing the interactions needed for information processing.
Although the technology possesses the immense potential to efficiently overcome classically intractable problems like discovering the prime factors of large composite integers, it is not anticipated to provide efficient solutions for hard combinatorial search tasks like the traveling salesman problem.
```