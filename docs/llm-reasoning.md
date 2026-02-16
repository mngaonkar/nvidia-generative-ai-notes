# LLM Reasoning

## Introduction to LLM Reasoning
LLM reasoning refers to the ability of large language models to perform complex cognitive tasks that require understanding, logical thinking, and multi-step problem-solving. Unlike simple pattern matching or text completion, reasoning involves the capacity to:

- **Break down complex problems** into manageable steps
- **Apply logical rules** and inference to reach conclusions
- **Maintain consistency** across multiple reasoning steps
- **Draw connections** between seemingly unrelated concepts
- **Evaluate and verify** the correctness of intermediate steps

Modern LLMs exhibit emergent reasoning capabilities that improve with scale, training methods, and prompting techniques. These capabilities enable applications in mathematical problem-solving, code generation, scientific analysis, and decision-making tasks that require more than surface-level language understanding. 

## Types of Reasoning
Following are the popular types of LLM reasoning frameworks:

### Chain of Thought (CoT)
#### How Chain of Thought Works
Chain of Thought (CoT) is a prompting technique that improves LLM reasoning by encouraging the model to break down complex problems into intermediate steps before arriving at a final answer.

#### Two Main Approaches
1. Few-Shot CoT - Provide examples with reasoning steps in the prompt:

Include 2-3 example problems showing step-by-step reasoning. The model learns the pattern and applies it to new problems

2. Zero-Shot CoT - Simply add "Let's think step by step" to your prompt:

- Remarkably effective with just this simple phrase
- No examples needed
- Works across many problem types

[CoT Examples](cot-examples.md)

#### Key Benefits
- Improved accuracy - Especially on math, logic, and multi-step reasoning tasks. Performance gains of 20-50% on complex problems.
- Error detection - Intermediate steps make it easier to spot where reasoning went wrong
- Interpretability - You can see exactly how the model arrived at its answer
- Handles complexity - Breaks down problems that would overwhelm direct answering

### Reason and Act (ReAct)
#### How ReAct Works
In traditional approaches, LLMs either reason internally (chain-of-thought) or take actions (like using tools) separately. ReAct combines both by having the model alternate between:

1. Thought - The model reasons about what to do next
2. Action - The model takes an action (like calling a tool, searching, or executing code)
3. Observation - The model receives feedback from that action
4. Repeat - This cycle continues until the task is complete

![ReAct Example](react-example.png)

[ReAct: Synergizing Reasoning and Acting in Language Models](https://react-lm.github.io)

CoT is best for problems solvable through pure reasoning (math, logic puzzles), while ReAct shines when you need to interact with external systems (searching, calculating, accessing databases).

### Tree of Thought (ToT)
#### How ToT Works
4 Key Components:

- Thought Decomposition - Break the problem into steps
- Thought Generation - Create multiple candidate thoughts at each step
- State Evaluation - Assess how promising each path is
- Search Algorithm - Decide which paths to explore (BFS, DFS, or beam search)

Visual Comparison
Chain of Thought (Linear):
```
Start → Step 1 → Step 2 → Step 3 → Answer
```

Tree of Thoughts (Branching):
```
Start
                   /  |  \
                  /   |   \
             Step1a Step1b Step1c
               /  \     |      \
              /    \    |       \
          Step2a Step2b Step2c Step2d
            |      |      ✓       ✗
            |      |   Answer   Dead-end
            ✗      ✗
         Dead-end Dead-end
```
#### Implementation Approaches
1. Breadth-First Search (BFS)

Explore all options at each level before going deeper
Good for finding shortest solution path

2. Depth-First Search (DFS)

Explore one path fully before trying others
More efficient for deep problems

3. Beam Search

Keep top-k most promising paths at each level
Balance between exploration and efficiency

#### ToT Prompt Template
```
Problem: [Your problem]
Objective: [Objective with clear goals]
Let's explore multiple solution paths:

Possible Approaches:
a) [Approach A]
b) [Approach B]  
c) [Approach C]

Evaluate each:
- Approach A: [Likelihood of success: High/Medium/Low]
- Approach B: [Likelihood of success: High/Medium/Low]
- Approach C: [Likelihood of success: High/Medium/Low]

Select most promising: [Best approach]

Continue with [selected approach]:
[Develop this path further...]

If this path fails, backtrack and try [alternative approach].
```
#### When to Use ToT
✓ Best for:

Creative problem-solving with multiple valid approaches
Planning tasks (travel itineraries, project plans)
Strategic games (chess, puzzles)
Problems where dead-ends are likely
Situations requiring exploration of trade-offs

✗ Overkill for:

Simple arithmetic (use CoT)
Factual questions
Problems with one clear solution path
Tasks requiring external information (use ReAct)

### Graph of Thought (GoT)
Graph of Thoughts (GoT) is the most advanced reasoning framework for LLMs, extending Tree of Thoughts by allowing thoughts to form arbitrary graph structures where different reasoning paths can merge, reference each other, and be combined—not just branch out.

- Tree of Thoughts: Thoughts can only branch (split) and never merge
- Graph of Thoughts: Thoughts can branch, merge, loop back, and interconnect
This enables more sophisticated reasoning where insights from different approaches can be synthesized together.

#### Graph Structure
```
Start
        /  |  \
       /   |   \
      A    B    C     ← Generate different approaches
      |    |    |
      D    E    F     ← Develop each path
       \   |   /
        \  |  /
          G           ← Merge insights from D, E, F
          |
        Refine
          |
        Answer
```

#### ToT Example
```
Start: Business Plan
                   /      |      \
                  /       |       \
            Market    Operations  Financial
            Analysis  Strategy    Model
               |          |          |
         Competitors  Suppliers  Revenue
         Customers   Equipment   Costs
         Location    Staffing    Funding
               |          |          |
               └──────────┴──────────┘
                         ↓
                  AGGREGATE INSIGHTS
                         ↓
              Identify Conflicts:
              - Market wants low prices
              - Sustainable suppliers cost more
              - Location rent is high
                         ↓
                    REFINE EACH
                   /      |      \
                  /       |       \
            Adjust      Optimize    Recalculate
            Market      Operations  Financial
            Strategy    for cost    Projections
                  \       |       /
                   \      |      /
                    \     |     /
                  FINAL SYNTHESIS
                         ↓
                  Integrated Plan
```
#### Tree of Thoughts (ToT) vs Graph of Thoughts (GoT): Key Differences

| Feature | Tree of Thoughts (ToT) | Graph of Thoughts (GoT) |
|--------|------------------------|--------------------------|
| Structure | Strict hierarchy | Arbitrary connections |
| Path merging | ❌ No | ✅ Yes |
| Cross-referencing | ❌ No | ✅ Yes |
| Loops / iteration | ❌ No | ✅ Yes |
| Synthesis | Pick best branch | Combine multiple branches |
| Complexity | Moderate | High |


### LLM Compiler
LLM Compiler (also called LLMCompiler) is an advanced framework that optimizes LLM agent execution by treating function calling like a compilation problem—planning all tool calls upfront and executing them in parallel when possible, rather than calling them sequentially.

Traditional Agent Approach (ReAct-style):
```
1. LLM decides → Call Tool A → Wait for result
2. LLM decides → Call Tool B → Wait for result  
3. LLM decides → Call Tool C → Wait for result
4. LLM generates final answer

Total time: Sum of all calls (sequential)
```

LLM Compiler Approach:
```
1. LLM plans ALL tool calls upfront as a DAG
2. Execute independent calls in parallel
3. Execute dependent calls when dependencies ready
4. LLM generates final answer with all results

Total time: Much faster (parallel execution)
```
The Compiler Analogy
Traditional compilers optimize code execution:

1. Parse source code into components
2. Analyze dependencies between operations
3. Optimize by reordering, parallelizing, eliminating redundancy
4. Execute efficiently

LLM Compiler does the same for agent tasks:

1. Parse user query into required function calls
2. Analyze dependencies between calls
3. Optimize execution plan (parallelize independent calls)
4. Execute according to plan


### Plan and Execute
Plan and Execute is an LLM agent framework that separates planning from execution—first creating a complete action plan upfront, then systematically executing each step, rather than figuring things out on the fly.

#### Core Philosophy
ReAct approach: "Think about next step → Act → See result → Think again..." \
Plan and Execute: "Think about ALL steps → Execute step 1 → Execute step 2 → ..."It's like the difference between:

Phase 1: Planning
The LLM creates a comprehensive, numbered plan
```
User Query: "Research competitors and create a comparison report"

Planning LLM Output:
─────────────────────────
PLAN:
1. Identify top 3 competitors in the industry
2. For each competitor, gather: pricing, features, market share
3. Search for recent news about each competitor
4. Compile findings into comparison table
5. Analyze strengths and weaknesses
6. Generate executive summary
7. Create final report document
─────────────────────────
```

Phase 2: Execution
Execute each step sequentially.
```
Step 1: Identify competitors
  → Tool: web_search("top competitors in [industry]")
  → Result: CompanyA, CompanyB, CompanyC ✓

Step 2: Gather competitor data
  → Tool: web_search("CompanyA pricing features")
  → Tool: web_search("CompanyB pricing features")  
  → Tool: web_search("CompanyC pricing features")
  → Results: [collected data] ✓

Step 3: Search recent news
  → Tool: web_search("CompanyA news 2024")
  → [continue for B and C]
  → Results: [news articles] ✓

[Continue through all steps...]

Step 7: Create report
  → Tool: create_document(data)
  → Result: Report.docx ✓
```

### Language Agent Tree Search (LATS)
Language Agent Tree Search (LATS) is a sophisticated framework that combines Monte Carlo Tree Search (MCTS) with LLM agents, enabling them to explore reasoning paths strategically, learn from failures through reflection, and improve their approach iteratively—similar to how AlphaGo masters chess.

#### Core Innovation
LATS brings reinforcement learning concepts to language agents:

- Explores multiple solution paths like ToT
- Learns from both successes and failures
- Uses value estimation to guide exploration
- Employs reflection to improve future attempts
- Implements strategic search rather than exhaustive exploration

The MCTS Algorithm Applied to Language
Traditional MCTS (e.g., AlphaGo)
```
1. Selection: Pick most promising node to explore
2. Expansion: Add new child nodes  
3. Simulation: Play out game to end
4. Backpropagation: Update values of parent nodes
```

LATS for Language Tasks
```
1. Selection: Choose promising reasoning path
2. Expansion: Generate new thoughts/actions
3. Evaluation: Assess quality (LLM or external feedback)
4. Reflection: Learn from mistakes
5. Backpropagation: Update path values
```

### Think Critique Improve
Think Critique Improve (also called Self-Critique or Critique and Refine) is a reflective reasoning framework where the LLM generates an initial response, critically evaluates it, identifies flaws, and then produces an improved version—essentially having the model be its own editor.

Core PhilosophyInstead of generating a response and stopping, the LLM engages in self-reflection:

1. Think: Generate initial solution
2. Critique: Evaluate what's wrong with it
3. Improve: Create better version based on critique

It's like writing a draft, reading it critically, and revising—a natural human process now automated for LLMs.

#### Complete Workflow Diagram
```
User Query
    ↓
┌─────────────────┐
│  THINK Phase    │ → Generate initial response
└─────────────────┘
    ↓
┌─────────────────┐
│ CRITIQUE Phase  │ → Identify problems:
└─────────────────┘   • What's missing?
    ↓                 • What's wrong?
    │                 • What could be better?
    ↓
┌─────────────────┐
│ IMPROVE Phase   │ → Regenerate with fixes
└─────────────────┘
    ↓
  Better Output

Optional: Loop back to CRITIQUE if still not good enough
```

## Test Time Compute
Test-Time Compute refers to using additional computational resources during inference (when the model is generating responses) rather than during training, to improve output quality. The core insight: sometimes generating multiple candidates and picking the best one yields better results than a single pass.

### Best of N
Generate N independent completions, evaluate each one, and return the single best result.

#### Evaluation Methods
1. Model-Based Scoring
```
def evaluate_completion(completion, query):
    score_prompt = f"""
    Rate this response from 1-10:
    
    Query: {query}
    Response: {completion}
    
    Consider: accuracy, helpfulness, clarity
    Score (1-10):
    """
    
    score = LLM(score_prompt)
    return float(score)
```
2. Reward Model
```
# Pre-trained reward model (like in RLHF)
def evaluate_with_reward_model(completion):
    embedding = embed(completion)
    score = reward_model(embedding)
    return score
```
3. Rule-Based / Heuristic
```
def evaluate_code(code):
    score = 0
    
    # Runs without errors?
    if code_runs(code): score += 3
    
    # Passes test cases?
    score += count_passing_tests(code)
    
    # Good style?
    if lint_score(code) > 8: score += 2
    
    # Efficient?
    if complexity(code) < O(n²): score += 2
    
    return score
```

4. User Preference (A/B Testing)
```
def evaluate_via_user():
    # Show multiple options to user
    # Track which they select
    # Use historical click-through rates
    return historical_preference_score
```


### Beam Search

```
def beam_search(prompt, beam_width=5, max_length=100):
    # Start with prompt
    beams = [(prompt, 0.0)]  # (sequence, cumulative_log_prob)
    
    for step in range(max_length):
        candidates = []
        
        # Expand each beam
        for sequence, score in beams:
            # Get next token probabilities
            probs = LLM.get_next_token_probs(sequence)
            
            # Consider top tokens for this beam
            for token, prob in probs.top_k(beam_width):
                new_sequence = sequence + token
                new_score = score + log(prob)
                candidates.append((new_sequence, new_score))
        
        # Keep top beam_width candidates
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Check if all beams ended
        if all(seq.endswith('[EOS]') for seq, _ in beams):
            break
    
    # Return best beam
    return beams[0][0]
```

#### Visual Example: Story Generation
```
Query: "Continue the story: The explorer entered the cave and..."

Beam Width = 3

                    "The explorer entered the cave and"
                                    |
        ┌──────────────────────────┼──────────────────────────┐
        |                          |                          |
    "discovered"             "heard"                    "felt"
    (p=0.4)                  (p=0.35)                  (p=0.25)
        |                          |                          |
    ┌───┴───┐               ┌──────┴──────┐           ┌──────┴──────┐
    |       |               |             |           |             |
"ancient" "a"          "strange"    "footsteps"   "cold"      "trembling"
(p=0.5) (p=0.3)        (p=0.4)      (p=0.35)    (p=0.45)     (p=0.3)

Cumulative scores:
"...discovered ancient" = 0.4 × 0.5 = 0.20 ✓ KEEP
"...heard strange" = 0.35 × 0.4 = 0.14 ✓ KEEP
"...felt cold" = 0.25 × 0.45 = 0.11 ✓ KEEP

Drop all others, continue expanding top-3...
```

### Lookahead Search


#### Example: Story Writing with Lookahead
```
Prompt: "Write a mystery story opening"

Current: "The phone rang at midnight."

Option 1: "She picked it up immediately."
  └─ Lookahead: "Hello?" she said. The line was silent...
  └─ Evaluation: Boring, predictable
  └─ Score: 5/10

Option 2: "She ignored it, but it kept ringing."
  └─ Lookahead: After the 7th ring, she finally answered...
  └─ Evaluation: Builds tension well
  └─ Score: 8/10 ✓

Option 3: "She was expecting this call."
  └─ Lookahead: "Is it done?" she whispered into the receiver...
  └─ Evaluation: Intriguing, mysterious
  └─ Score: 9/10 ✓✓

Select: Option 3 (highest score)
```

```
def lookahead_generate(prompt, lookahead_depth=2):
    # Generate K initial continuations
    candidates = LLM.generate(prompt, n=5, max_tokens=20)
    
    best_score = -inf
    best_candidate = None
    
    for candidate in candidates:
        # Look ahead: continue each candidate further
        future_continuations = LLM.generate(
            prompt + candidate, 
            n=3, 
            max_tokens=50
        )
        
        # Evaluate complete sequences
        future_scores = [
            evaluate(prompt + candidate + continuation)
            for continuation in future_continuations
        ]
        
        # Score this candidate by its best possible future
        candidate_score = max(future_scores)
        
        if candidate_score > best_score:
            best_score = candidate_score
            best_candidate = candidate
    
    return best_candidate
```

### Comparison Table

| Strategy | Cost | Parallelizable | Best For | Main Idea |
|----------|------|---------------|----------|-----------|
| **Greedy** | 1× | N/A | Speed | Pick most likely token each step |
| **Best-of-N** | N× | ✓✓✓ Yes | Clear metrics | Generate N complete sequences, pick best |
| **Beam Search** | K×L | △ Partial | Structured tasks | Keep top-K partial sequences at each step |
| **Lookahead** | B^D | △ Partial | Strategic tasks | Explore future consequences before choosing |
